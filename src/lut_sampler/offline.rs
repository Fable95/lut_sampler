use std::iter;
use std::usize;
use itertools::{izip, repeat_n, Itertools};


use maestro::rep3_core::share::HasZero;
use maestro::{
    rep3_core::
    {
        network::{task::Direction, NetSerializable}, party::{error::{MpcError, MpcResult}, MainParty, Party}, 
        share::{RssShare, RssShareVec}
    }, 
    chida::online::mul_no_sync,
    share::{bs_bool16::BsBool16, Field}
};

use crate::{
    share::{gf2p64::GF2p64, gf_template::{AllowedTypes, GFTTrait, Share, GFT}},
    util::mul_triple_vec::{BitStringMulTripleRecorder, MulTripleRecorder}
};
use super::{CubeOhv, OhvVec, RndOhvOutput};

type GF128_8  = GFT<u128,u8,16>;

// Takes two vectors of bitsliced values and returns a vector of byte-wise RSS shares 
// At the moment not dynamic with respect to Embedded
fn decompose<T: GFTTrait>(v1: Vec<T::Wrapper>, v2: Vec<T::Wrapper>) -> RssShareVec<T::Embedded>{
    let amount = v1.len();
    let new_len = std::mem::size_of::<T::Embedded>();
    let mut res = Vec::with_capacity(T::RATIO*amount);
    let vec1 = <T::Wrapper as NetSerializable>::as_byte_vec(v1, amount);
    let vec2 = <T::Wrapper as NetSerializable>::as_byte_vec(v2, amount);
    for (ci, cii) in vec1.chunks(new_len).zip(vec2.chunks(new_len)){
        res.push(RssShare{
            si:  T::Embedded::pack_bytes(ci),
            sii: T::Embedded::pack_bytes(cii)
        });
    }
    res
}

fn rss_decompose<T: GFTTrait>(rss_in: &RssShare<T::Wrapper>) -> RssShareVec<T::Embedded>{
    let mut res = Vec::with_capacity(T::RATIO);
    let vec1 = T::new(rss_in.si).unpack();
    let vec2 = T::new(rss_in.sii).unpack();
    for (si, sii) in vec1.into_iter().zip(vec2){
        res.push(RssShare{si, sii});
    }
    res
}


fn rss_vec_decompose<T: GFTTrait>(slice_in: &RssShareVec<T::Wrapper>) -> RssShareVec<T::Embedded>{
    let res = slice_in.iter().flat_map(|x| rss_decompose::<T>(x)).collect();
    res
}

fn vec_to_rss<T: GFTTrait>(v1: &[T::Wrapper], v2: &[T::Wrapper]) -> RssShareVec<T::Wrapper>{
    let mut res = Vec::with_capacity(v1.len());
    for (si, sii) in v1.iter().zip(v2){
        res.push(RssShare{si: *si, sii: *sii});
    }
    res
}

pub fn compute_0_offsets<T: GFTTrait>(party: &mut MainParty, amount: usize) -> MpcResult<RssShareVec<T::Embedded>>{
    let si: Vec<T::Embedded> = party.generate_alpha::<T::Embedded>(amount * 3).collect();
    let mut sii = vec![T::Embedded::default(); amount*3];
    party.send_field::<T::Embedded>(Direction::Previous, si.iter().cloned(), amount * 3);
    
    party.receive_field_slice(Direction::Next, &mut sii).rcv()?;
    let mut res = Vec::with_capacity(amount*3);

    for (ai, aii) in si.into_iter().zip(sii){
        res.push(RssShare{
            si: ai,
            sii: aii
        });
    }
    Ok(res)
}

fn print_many(ai_vec: &[GF2p64], aii_vec: &[GF2p64], bi_vec: &[GF2p64], bii_vec: &[GF2p64], ci_vec: &[GF2p64], cii_vec: &[GF2p64]){
    println!("ai:  {:?},\naii: {:?},\nbi:  {:?},\nbii: {:?},\nci:  {:?},\ncii: {:?},\n", 
            ai_vec, aii_vec, bi_vec, bii_vec, ci_vec, cii_vec);
}

fn check_len(ai_vec: &[GF2p64], aii_vec: &[GF2p64], bi_vec: &[GF2p64], bii_vec: &[GF2p64], ci_vec: &[GF2p64], cii_vec: &[GF2p64]){
    debug_assert!(ai_vec.len() == aii_vec.len());
    debug_assert!(ai_vec.len() == bi_vec.len());
    debug_assert!(ai_vec.len() == bii_vec.len());
    debug_assert!(ai_vec.len() == ci_vec.len());
    debug_assert!(ai_vec.len() == cii_vec.len());
    println!("Adding {} mul triples", ai_vec.len());
}

fn compute_product<
REC: MulTripleRecorder<GF2p64>,
T: GFTTrait
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    values_a: &Vec<RssShare<T::Wrapper>>,
    values_b: &Vec<RssShare<T::Wrapper>>,
    _mask: T::Wrapper
) -> MpcResult<Vec<RssShare<T::Wrapper>>> {
    let len = values_a.len();
    
    let mut ci= Vec::with_capacity(len);
    let mut cii = vec![T::Wrapper::default(); len];
    let mut ai_vec;
    let mut aii_vec;
    let mut bi_vec;
    let mut bii_vec;
    let mut ci_vec;
    let mut cii_vec;
    
    if mal_sec {
        ai_vec = Vec::with_capacity (len * T::RATIO);
        aii_vec = Vec::with_capacity(len * T::RATIO);
        bi_vec = Vec::with_capacity (len * T::RATIO);
        bii_vec = Vec::with_capacity(len * T::RATIO);
        ci_vec = Vec::with_capacity (len * T::RATIO);
        cii_vec = Vec::with_capacity(len * T::RATIO);
    } else {
        ai_vec = Vec::new();
        aii_vec = Vec::new();
        bi_vec = Vec::new();
        bii_vec = Vec::new();
        ci_vec = Vec::new();
        cii_vec = Vec::new();
    }

    let alphas = party.generate_alpha(len);
    for ((alpha_i, a), b) in alphas.zip(values_a).zip(values_b){
        let ai =  a.si;
        let aii = a.sii;
        let bi =  b.si;
        let bii = b.sii;
        
        let ai_bi  =  ai * bi ;// * mask;
        let ai_bii =  ai * bii;// * mask;
        let aii_bi = aii * bi ;// * mask;
        let current_ci = ai_bi + aii_bi + ai_bii + alpha_i;
        
        if mal_sec {
            ai_vec.append(&mut ai.bit_embed_gf2p64());
            aii_vec.append(&mut aii.bit_embed_gf2p64());
            bi_vec.append(&mut bi.bit_embed_gf2p64());
            bii_vec.append(&mut bii.bit_embed_gf2p64());
            ci_vec.append(&mut current_ci.bit_embed_gf2p64());
        }
        ci.push(current_ci);
    }
    party.send_field::<T::Wrapper>(Direction::Previous, ci.iter(), ci.len());
    party.receive_field_slice(Direction::Next, &mut cii).rcv()?;
    
    if mal_sec {
        for cii_val in cii.iter() {
            cii_vec.append(&mut cii_val.bit_embed_gf2p64());
        }
        // print_many(&ai_vec, &aii_vec, &bi_vec, &bii_vec, &ci_vec, &cii_vec);
        // check_len(&ai_vec, &aii_vec, &bi_vec, &bii_vec, &ci_vec, &cii_vec);
        mul_triple_recorder.record_mul_triple(
            &ai_vec, &aii_vec, &bi_vec, &bii_vec, &ci_vec, &cii_vec,
        );
    }
    
    let mut res = vec_to_rss::<T>(&ci, &cii);
    res.truncate(len);
    Ok(res)

}

fn compute_bias_4<
REC: MulTripleRecorder<GF2p64>,
T: GFTTrait
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    len: usize,
    _mask: T::Wrapper
) -> MpcResult<Vec<RssShare<T::Wrapper>>> {
    
    let vals: Vec<RssShareVec<T::Wrapper>> = (0..2)
        .map(|_| party.generate_random(len)).collect_vec();
    compute_product::<_,T>(party, mul_triple_recorder, mal_sec, &vals[0], &vals[1], _mask)
}

fn compute_bias_16<
REC: MulTripleRecorder<GF2p64>,
T: GFTTrait
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    len: usize,
    _mask: T::Wrapper
) -> MpcResult<Vec<RssShare<T::Wrapper>>> {
    
    let vals: MpcResult<Vec<RssShareVec<T::Wrapper>>> = (0..2)
        .map(|_| {
            let res = compute_bias_4::<_,T>(party, mul_triple_recorder, mal_sec, len, _mask);
            res
        }).collect();
    let vals = vals?;
    compute_product::<_,T>(party, mul_triple_recorder, mal_sec, &vals[0], &vals[1], _mask)
}

pub fn compute_biased_offsets<
REC: MulTripleRecorder<GF2p64>, 
T: GFTTrait
>(
    party: &mut MainParty, 
    mul_triple_recorder: &mut REC, 
    amount: usize, 
    mal_sec: bool,
    skew: usize,
    _mask: T::Wrapper
) -> MpcResult<RssShareVec<T::Embedded>>{
    assert!(skew > 0, "Skew of 0 would always lead to 2^n-1");
    assert!(skew < 8, "Skew value must be below 8");
    let n_elts = if amount % T::RATIO == 0 {
        amount / T::RATIO
    } else {
        amount / T::RATIO + 1
    };
    let len = 3 * n_elts;
    let mut init = false;
    let mut vals= Vec::new();
    println!("Computing bias with skew: {}", skew);
    if (skew & 0b001) != 0{
        println!("Generating initial with prob 1/2");
        vals = party.generate_random::<T::Wrapper>(len);
        init = true;
    } 
    if (skew & 0b010) != 0{
        let tmp = compute_bias_4::<_,T>(party, mul_triple_recorder, mal_sec, len, _mask)?;
        if !init{
            println!("Generating initial with prob 1/4");
            init = true;
            vals = tmp;
        } else{
            println!("Multiplying with prob 1/4");
            vals = compute_product::<_,T>(party, mul_triple_recorder, mal_sec, &vals, &tmp, _mask)?;
        }
    } 
    if (skew & 0b100) != 0{
        let tmp = compute_bias_16::<_,T>(party, mul_triple_recorder, mal_sec, len, _mask)?;
        if !init{
            println!("Generating initial with prob 1/16");
            vals = tmp;
        } else{
            println!("Multiplying with prob 1/16");
            vals = compute_product::<_,T>(party, mul_triple_recorder, mal_sec, &vals, &tmp, _mask)?;
        }
    }

    let mut res_small = rss_vec_decompose::<T>(&vals);

    res_small.truncate(amount*3);
    Ok(res_small)
}

pub fn compute_binomial_offsets<T: GFTTrait, const SIZE: usize>(party: &mut MainParty, amount: usize, d: usize) -> MpcResult<RssShareVec<T::Embedded>>{
    let usize_len = 8*std::mem::size_of::<usize>();
    let n_elts_per_value = SIZE.div_ceil(usize_len);
    let mask = if n_elts_per_value * usize_len > SIZE{
        usize::MAX >> (n_elts_per_value * usize_len - SIZE)
    }  else {
        usize::MAX 
    };

    let n_elts = d * n_elts_per_value * amount;
    
    println!("Generating {} indices with {} bits and {} u128 per value", d*amount, SIZE, n_elts_per_value);
    println!("Applying mask: {:x}", mask);

    let mut vals = party.generate_random::<T::Wrapper>(n_elts);
    let mut res=  Vec::with_capacity(d*amount);
    for _ in 0..d*amount{
        let mut si  = T::Embedded::default();
        let mut sii = T::Embedded::default();
        for i in 0..n_elts_per_value{
            let mut v = vals.pop().ok_or(MpcError::Broadcast)?;
            println!("Before mask: {:x?}", v);
            if i == n_elts_per_value - 1 {
                v.si  =  v.si * T::Wrapper::from_usize(mask);
                v.sii = v.sii * T::Wrapper::from_usize(mask);
            }
            println!("After mask: {:x?}", v);
            si  += T::new(v.si)
                .unpack()
                .into_iter()
                .fold(T::Embedded::default(), 
                |acc, x| acc + x.count_ones()
            );
            sii += T::new(v.sii)
                .unpack()
                .into_iter()
                .fold(T::Embedded::default(), 
                |acc, x| acc + x.count_ones()
            );

        }
        res.push(RssShare{si,sii});
    }
    println!("Resulting shares {:x?}", res);
    Ok(res)
}

pub fn un_bitslice_generic<
    T: GFTTrait,
    const SIZE: usize,
    const SIZE_RED: usize
>(bs: &[Vec<RssShare<BsBool16>>]) 
-> 
Vec<(OhvVec<T,SIZE,SIZE_RED>,OhvVec<T,SIZE,SIZE_RED>)>{
    let mut vec_i = vec![[false; SIZE]; bs[0].len()*16];
    let mut vec_ii = vec![[false; SIZE]; bs[0].len()*16];
    // debug_assert!(bs.len() == SIZE);
    
    for (i, ohvi) in bs.iter().enumerate(){
    // the i'th bit in the OHV -- iterates from 0 to SIZE
        for (j, ohvj) in ohvi.iter().enumerate() {     
            // the j'th 16 bit element -- iterates from 0 to ceil(amount/16)
            let si = ohvj.si.as_u16();
            let sii = ohvj.sii.as_u16();
            let offset = 16*j;
            for k in 0..16 {
                let mask = 1 << k;
                vec_i[offset + k][i] = (si & mask) != 0;
                vec_ii[offset + k][i] = (sii & mask) != 0;
            }
        }
    }
    vec_i.into_iter()
        .zip(vec_ii)
        .map(|(si, sii)| 
        (OhvVec::<T,SIZE,SIZE_RED>::new(si),OhvVec::<T,SIZE,SIZE_RED>::new(sii)))
        .collect()
}

pub fn un_bitslice_generic_index<
T: GFTTrait,
const SIZE: usize,
const SIZE_RED: usize
>  (bs: &[Vec<RssShare<BsBool16>>]) -> Vec<RssShare<T::Embedded>> {
    debug_assert!(bs.len() <= 8*std::mem::size_of::<T::Embedded>());
    let mut res = vec![(T::Embedded::default(), T::Embedded::default()); bs[0].len()*16];
    for (i, bsboolvec) in bs.iter().enumerate(){
        for (j, bsbool) in bsboolvec.iter().enumerate(){
            let offset = 16*j;
            for k in 0..16 {                
                *res[offset + k].0.inner_mut() |= 
                    T::Embedded::from_u8(((bsbool.si.as_u16() >> k) & 0x1) as u8).inner() << i;
                *res[offset + k].1.inner_mut() |= 
                    T::Embedded::from_u8(((bsbool.sii.as_u16() >> k) & 0x1) as u8).inner() << i;
            }
        }
    }
    res.into_iter().map(|(si, sii)| RssShare{si, sii}).collect()
}

/// bits are in lsb-first order
pub fn generate_ohv<P: Party, Rec: BitStringMulTripleRecorder>(
    party: &mut P,
    mul_triple_recorder: &mut Rec,
    mut bits: Vec<Vec<RssShare<BsBool16>>>,
    n: usize,
) -> MpcResult<Vec<Vec<RssShare<BsBool16>>>> {
    if n == 2 {
        debug_assert_eq!(bits.len(), 1);
        let b = bits[0].clone();
        let b_prime = b
            .iter()
            .map(|rss| *rss + party.constant(BsBool16::ONE))
            .collect();
        Ok(vec![b_prime, b])
    } else {
        let msb = bits.remove(bits.len() - 1);
        let f = generate_ohv(party, mul_triple_recorder, bits, n / 2)?;
        // Mult
        let e_rest = simple_mul(party, mul_triple_recorder, &msb, &f[..=f.len() - 2])?;
        let mut sum_e = Vec::with_capacity(msb.len());
        for i in 0..msb.len() {
            let mut sum = RssShare::from(BsBool16::ZERO, BsBool16::ZERO);
            e_rest.iter().for_each(|v| sum += v[i]);
            sum_e.push(sum);
        }
        let mut e_last = sum_e;
        e_last
            .iter_mut()
            .zip(msb)
            .for_each(|(e_sum, v_k)| *e_sum = v_k - *e_sum);
        let mut res = Vec::with_capacity(n);
        izip!(f, e_rest.iter().chain(iter::once(&e_last))).for_each(|(f, e)| {
            debug_assert_eq!(f.len(), e.len());
            res.push(
                f.into_iter()
                    .zip(e)
                    .map(|(el_f, el_e)| el_f - *el_e)
                    .collect_vec(),
            );
        });
        res.extend(e_rest.into_iter().chain(iter::once(e_last)));
        Ok(res)
    }
}

fn simple_mul<P: Party, Rec: BitStringMulTripleRecorder>(
    party: &mut P,
    mul_triple_recorder: &mut Rec,
    msb: &Vec<RssShare<BsBool16>>,
    other: &[Vec<RssShare<BsBool16>>],
) -> MpcResult<Vec<Vec<RssShare<BsBool16>>>> {
    let ai_bit = msb.iter().map(|rss| rss.si).collect_vec();
    let aii_bit = msb.iter().map(|rss| rss.sii).collect_vec();
    let ai = repeat_n(&ai_bit, other.len())
        .flat_map(|vec| vec.iter().copied())
        .collect_vec();
    let aii = repeat_n(&aii_bit, other.len())
        .flat_map(|vec| vec.iter().copied())
        .collect_vec();
    let bi = other
        .iter()
        .flat_map(|rss_vec| rss_vec.iter().map(|rss| rss.si))
        .collect_vec();
    let bii = other
        .iter()
        .flat_map(|rss_vec| rss_vec.iter().map(|rss| rss.sii))
        .collect_vec();
    let mut ci = vec![BsBool16::ZERO; other.len() * msb.len()];
    let mut cii = vec![BsBool16::ZERO; other.len() * msb.len()];
    mul_no_sync(party, &mut ci, &mut cii, &ai, &aii, &bi, &bii)?;
    mul_triple_recorder.record_bit_bitstring_triple(msb.len(), &ai_bit, &aii_bit, &bi, &bii, &ci, &cii);
    let drain_ci = ci.into_iter();
    let drain_cii = cii.into_iter();
    let res = izip!(
        drain_ci.chunks(msb.len()).into_iter(),
        drain_cii.chunks(msb.len()).into_iter()
    )
    .map(|(ci, cii)| {
        izip!(ci, cii)
            .map(|(si, sii)| RssShare::from(si, sii))
            .collect_vec()
    })
    .collect_vec();
    Ok(res)
}


pub fn generate_rndohv_k<
    Rec: BitStringMulTripleRecorder, 
    T: GFTTrait,
    const SIZE: usize,
    const SIZE_RED: usize
    > (party: &mut MainParty, mul_triple_recorder: &mut Rec, k: usize, amount: usize) -> 
    MpcResult<Vec<RndOhvOutput<T,SIZE,SIZE_RED>>>{
        let n_blocks = amount.div_ceil(16);
        
        
        let bits = (0..k).map(|_| party.generate_random(n_blocks)).collect_vec();
        let indices = un_bitslice_generic_index::<T,SIZE,SIZE_RED>(&bits);
        let ohv = generate_ohv(party, mul_triple_recorder, bits, 1<<k)?;
        // println!("Generated OHV {:?}", ohv.len());
        // for (i, ohvi) in ohv.iter().enumerate(){
        //     println!("OHVi {}: {}", i, ohvi.len());
        //     for (j, ohvj) in ohvi.iter().enumerate() {
        //         println!("{}: OHVII {:x?}", j, ohvj.si);
        //     }
        // }
        let un = un_bitslice_generic::<T,SIZE,SIZE_RED>(&ohv);
        let res: Vec<RndOhvOutput<T, SIZE, SIZE_RED>> = un.into_iter().zip(indices).map(|(ohv, index)| 
            RndOhvOutput::new(ohv, index)).collect();
        // println!("Result: {:?}", res[0]);
        Ok(res)
}



pub fn compute_ohv_vectors<
REC: BitStringMulTripleRecorder,
T: GFTTrait,
const SIZE:usize,
const SIZE_RED: usize
>(party: &mut MainParty, mul_triple_recorder: &mut REC, k: usize, amount: usize) -> MpcResult<Vec<CubeOhv<T,SIZE,SIZE_RED>>>{

    let ohv1 = generate_rndohv_k::<_,T,SIZE,SIZE_RED>(party, mul_triple_recorder, k, 3*amount)?;
    let mut res = Vec::with_capacity(amount);
    for i in 0..amount {
        let matrix = CubeOhv{
            row_ohv: ohv1[3 * i],
            col_ohv: ohv1[3 * i + 1],
            lay_ohv: ohv1[3 * i + 2],
            _marker: std::marker::PhantomData,
        };
        res.push(matrix);
    }
    Ok(res)    
}

pub fn extract_byte_from_table<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>(
    row: usize, 
    col: usize, 
    lay: usize,
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE]
) -> T::Embedded {
    let lay_large = lay / SIZE_RED;
    let lay_small = lay % SIZE_RED;
    let mat = &lut_table[row];
    let vec = &mat[col];
    let val_large = <T as GFTTrait>::Wrapper::new(vec[lay_large]).to_usize();
    let val_small = (val_large >> (lay_small * size_of::<T::Embedded>() * 8)) as usize;
    println!("before reduce 0x{:16x}", val_large);
    T::Embedded::from_usize(val_small)
}

pub fn compare_result<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>(
    indices: Vec<T::Embedded>, 
    value: T::Embedded,
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE]
) {
    let row = indices[0].inner().to_usize();
    let col = indices[1].inner().to_usize();
    let lay = indices[2].inner().to_usize();
    let expected = extract_byte_from_table::<T, SIZE, SIZE_RED>(row, col, lay, lut_table);
    // if expected != value {
        // assert!(expected == value, "Extract pos [{},{},{}]:\t expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
        println!("Extract pos [{:03},{:03},{:03}]: expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
    // }
    // assert!(expected.to_u8() == value.to_u8(), "Extract pos [{},{},{}]:\t expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
}
