use std::iter;
use std::usize;
use itertools::{izip, repeat_n, Itertools};


use maestro::rep3_core::share::HasZero;
use maestro::{
    rep3_core::
    {
        network::{task::Direction, NetSerializable}, party::{error::{MpcResult}, MainParty, Party}, 
        share::{RssShare, RssShareVec}
    }, 
    chida::online::mul_no_sync,
    share::{bs_bool16::BsBool16, Field}
};

use crate::lut_sampler::Network;
use crate::lut_sampler::ohv_container::MatrixOhv;
// use crate::lut_sampler::online::open_rss_many;
use crate::share::gf_template::ShareType;
use crate::{
    share::{gf2p64::GF2p64, helper_types::AllowedTypes, gf_template::{GFTTrait, Share, GFT}},
    util::mul_triple_vec::{BitStringMulTripleRecorder, MulTripleRecorder}
};
use super::{CubeOhv, ohv_container::{OhvVec, RndOhvOutput}};


type GF128_8  = GFT<u128,u8,16>;

// Takes two vectors of bitsliced values and returns a vector of byte-wise RSS shares 
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
    let vec1 = T::unpack(&rss_in.si);
    let vec2 = T::unpack(&rss_in.sii);
    for (si, sii) in vec1.into_iter().zip(vec2){
        res.push(RssShare{si, sii});
    }
    res
}


fn rss_vec_decompose<T: GFTTrait>(slice_in: &RssShareVec<T::Wrapper>) -> RssShareVec<T::Embedded>{
    let res = slice_in.iter().flat_map(|x| rss_decompose::<T>(x)).collect();
    res
}


fn vec_to_rss<T: Share>(v1: &[T], v2: &[T]) -> RssShareVec<T>{
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

pub fn print_many(ai_vec: &[GF2p64], aii_vec: &[GF2p64], bi_vec: &[GF2p64], bii_vec: &[GF2p64], ci_vec: &[GF2p64], cii_vec: &[GF2p64]){
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
T: Share
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    values_a: &Vec<RssShare<T>>,
    values_b: &Vec<RssShare<T>>
) -> MpcResult<Vec<RssShare<T>>> {
    let len = values_a.len();
    
    let mut ci= Vec::with_capacity(len);
    let mut cii = vec![T::default(); len];
    let mut ai_vec;
    let mut aii_vec;
    let mut bi_vec;
    let mut bii_vec;
    let mut ci_vec;
    let mut cii_vec;
    
    if mal_sec {
        ai_vec = Vec::with_capacity (len);
        aii_vec = Vec::with_capacity(len);
        bi_vec = Vec::with_capacity (len);
        bii_vec = Vec::with_capacity(len);
        ci_vec = Vec::with_capacity (len);
        cii_vec = Vec::with_capacity(len);
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
        
        let ai_bi  =  ai * bi ;     // * mask;
        let ai_bii =  ai * bii;     // * mask;
        let aii_bi = aii * bi ;     // * mask;
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
    party.send_field::<T>(Direction::Previous, ci.iter(), ci.len());
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
IndexType: AllowedTypes,
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    len: usize
) -> MpcResult<Vec<RssShare<ShareType::<IndexType>>>> {

    let vals: Vec<RssShareVec<ShareType::<IndexType>>> = 
        (0..2).map(|_| party.generate_random::<ShareType::<IndexType>>(len)).collect_vec();
    compute_product::<_,ShareType::<IndexType>>(party, mul_triple_recorder, mal_sec, &vals[0], &vals[1])
}

fn compute_bias_16<
REC: MulTripleRecorder<GF2p64>,
IndexType: AllowedTypes,
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    len: usize
) -> MpcResult<Vec<RssShare<ShareType::<IndexType>>>> {
    
    let vals: MpcResult<Vec<RssShareVec<ShareType::<IndexType>>>> = (0..2)
        .map(|_| compute_bias_4(party, mul_triple_recorder, mal_sec, len)).collect();
    let vals = vals?;
    compute_product::<_,ShareType::<IndexType>>(party, mul_triple_recorder, mal_sec, &vals[0], &vals[1])
}

fn compute_bias_256<
REC: MulTripleRecorder<GF2p64>,
IndexType: AllowedTypes,
>(
    party: &mut MainParty,
    mul_triple_recorder: &mut REC,
    mal_sec: bool,
    len: usize
) -> MpcResult<Vec<RssShare<ShareType::<IndexType>>>> {

    let vals: MpcResult<Vec<RssShareVec<ShareType::<IndexType>>>> = (0..2)
        .map(|_| compute_bias_16(party, mul_triple_recorder, mal_sec, len)).collect();
    let vals = vals?;
    compute_product::<_,ShareType::<IndexType>>(party, mul_triple_recorder, mal_sec, &vals[0], &vals[1])
}

pub fn compute_biased_offsets<
REC: MulTripleRecorder<GF2p64>, 
IndexType: AllowedTypes,
>(
    network: &mut Network, 
    mul_triple_recorder: &mut REC, 
    amount: usize, 
    mal_sec: bool,
    skew: usize,
    k: &[usize],
    l: usize,
) -> MpcResult<Vec<RssShareVec<ShareType::<IndexType>>>>{
    assert!(skew > 0, "Skew of 0 would always lead to 2^n-1");
    assert!(skew < 16, "Skew value must be below 16");
    assert!(k.iter().all(|&ki| ki <= 8* std::mem::size_of::<IndexType>()), "IndexType not large enough to contain lut size");
    let total = k.iter().sum();
    assert!(l <= total);
    let party = network.chida.as_party_mut();
    let mut init = false;
    let mask: Vec<ShareType<IndexType>> = k.iter().map(|&ki| ShareType::<IndexType>::from_usize((1 << ki) - 1)).collect();
    let d = k.len();
    // (i,(i-1)//8) 
    let final_index = if l <= k[0] {
        0
    } else if l <= (k[0] + k[1]){
        1
    } else { // This can only happen in 3D case.
        2
    };
    let skewed_dimensions = if l == 0 {
        0 
    } else {
        final_index + 1
    };

    let mf: IndexType = if l == 0{
        IndexType::from_usize(0)
    } else if l == k[0] || l == k[0] + k[1] || l == total {
        IndexType::from_usize((1 << k[final_index]) - 1)
    } else {
        let mut l_counter = l;
        for index in 0..final_index{
            if l_counter < k[index]{
                panic!("final index computation is flawed");
            }
            l_counter -= k[index];
        }
        IndexType::from_usize((1 << l_counter) - 1)
    };
    let inverse_final_mask = !IndexType::from(0) ^ mf;
    let final_mask = ShareType(mf);
    let final_mask_inverse = ShareType(inverse_final_mask);
    let len = skewed_dimensions * amount;
    // println!("Computing bias with skew: {}", skew);
    // println!("skew {}, l {}, k {} (max {}), skewed dims {}, mf {:b}, imf {:b}",
    //     skew, l, k, (1<<k)-1, skewed_dimensions, mf, inverse_final_mask,
    // );

    // println!("Fill everything with prob 1/2");
    // println!("Generating {} offsets with {} dimensions", amount, d);
    let mut vals = (0..amount).map(|_| party.generate_random::<ShareType::<IndexType>>(d)).collect::<Vec<_>>();
    vals.iter_mut().for_each(|v| 
        v.iter_mut().zip(mask.iter()).for_each(|(x, mask)|
            *x = *x * *mask
    ));

    // let open = open_rss_many(party, &mut network.broadcast_context, &vals[0])?;
    // println!("Unbiased coins: {:?}", open);

    if l == 0{
        return Ok(vals)
    }

    let mut skewed_values = Vec::new();

    if (skew & 0b001) != 0{
        // println!("Generating initial with prob 1/2");
        skewed_values = party.generate_random::<ShareType::<IndexType>>(len);
        skewed_values
            .chunks_mut(skewed_dimensions)
            .for_each(|v| 
                v.iter_mut()
                .zip(mask.iter())
                .for_each(|(v,mask)|
                *v = *v * *mask
            ));
        init = true;
    } 

    if (skew & 0b010) != 0{
        let tmp = compute_bias_4(party, mul_triple_recorder, mal_sec, len)?;
        if !init{
            // println!("Generating initial with prob 1/4");
            init = true;
            skewed_values = tmp;
            skewed_values
            .chunks_mut(skewed_dimensions)
            .for_each(|v| 
                v.iter_mut()
                .zip(mask.iter())
                .for_each(|(v,mask)|
                *v = *v * *mask
            ));
        } else{
            // println!("Multiplying with prob 1/4");
            skewed_values = compute_product::<_,ShareType::<IndexType>>(party, mul_triple_recorder, mal_sec, &skewed_values, &tmp)?;
        }
    } 
    if (skew & 0b100) != 0{
        let tmp = compute_bias_16(party, mul_triple_recorder, mal_sec, len)?;
        if !init{
            // println!("Generating initial with prob 1/16");
            init = true;
            skewed_values = tmp;
            skewed_values
            .chunks_mut(skewed_dimensions)
            .for_each(|v| 
                v.iter_mut()
                .zip(mask.iter())
                .for_each(|(v,mask)|
                *v = *v * *mask
            ));
        } else {
            // println!("Multiplying with prob 1/16");
            skewed_values = compute_product::<_,ShareType::<IndexType>>(party, mul_triple_recorder, mal_sec, &skewed_values, &tmp)?;
        }
    }
    if (skew & 0b1000) != 0{
        let tmp = compute_bias_256(party, mul_triple_recorder, mal_sec, len)?;
        if !init{
            // println!("Generating initial with prob 1/256");
            skewed_values = tmp;
            skewed_values
            .chunks_mut(skewed_dimensions)
            .for_each(|v| 
                v.iter_mut()
                .zip(mask.iter())
                .for_each(|(v,mask)|
                *v = *v * *mask
            ));
        } else {
            // println!("Multiplying with prob 1/16");
            skewed_values = compute_product::<_,ShareType::<IndexType>>(party, mul_triple_recorder, mal_sec, &skewed_values, &tmp)?;
        }
    }

    for (value, sources) 
        in vals.iter_mut()
        .zip(skewed_values.chunks(skewed_dimensions)){
        for (dimension, source) in sources.iter().enumerate(){
            value[dimension] = if dimension == final_index{
                *source * final_mask + value[dimension] * final_mask_inverse
            } else {
                *source * mask[dimension]
            }
        }
    }
    // let open = open_rss_many(party, &mut network.broadcast_context, &vals[0])?;
    // let open_biased = open_rss_many(party, &mut network.broadcast_context, &skewed_values)?;
    // println!("Biased cois: {:?}", open_biased);
    // println!("Final coins: {:?}", open);

    Ok(vals)
}

pub fn un_bitslice_generic<
    T: GFTTrait,
    IndexType: AllowedTypes,
    const SIZE: usize,
>(bs: &[Vec<RssShare<BsBool16>>]) 
-> 
Vec<(OhvVec<T, IndexType, SIZE>,OhvVec<T, IndexType, SIZE>)>{
    let mut vec_i = vec![[false; SIZE]; bs[0].len()*16];
    let mut vec_ii = vec![[false; SIZE]; bs[0].len()*16];
    debug_assert!(bs.len() == SIZE, "bit string has {} vs SIZE {}", bs.len(), SIZE);
    
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
        (OhvVec::<T, IndexType, SIZE>::new(si),OhvVec::<T, IndexType, SIZE>::new(sii)))
        .collect()
}

pub fn un_bitslice_generic_index<
T: GFTTrait,
IndexType: AllowedTypes,
>  (bs: &[Vec<RssShare<BsBool16>>]) -> Vec<RssShare<ShareType<IndexType>>> {
    debug_assert!(bs.len() <= 8*std::mem::size_of::<ShareType<IndexType>>());
    let mut res = vec![(ShareType::<IndexType>::default(), ShareType::<IndexType>::default()); bs[0].len()*16];
    for (i, bsboolvec) in bs.iter().enumerate(){
        for (j, bsbool) in bsboolvec.iter().enumerate(){
            let offset = 16*j;
            for k in 0..16 {                
                *res[offset + k].0.inner_mut() |= 
                    ShareType::<IndexType>::from_u8(((bsbool.si.as_u16() >> k) & 0x1) as u8).inner() << i;
                *res[offset + k].1.inner_mut() |= 
                    ShareType::<IndexType>::from_u8(((bsbool.sii.as_u16() >> k) & 0x1) as u8).inner() << i;
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
    IndexType: AllowedTypes,
    const SIZE: usize
    > (party: &mut MainParty, mul_triple_recorder: &mut Rec, k: usize, amount: usize) -> 
    MpcResult<Vec<RndOhvOutput<T,IndexType,SIZE>>>{
        let n_blocks = amount.div_ceil(16);
        
        // TODO: replace uniform with sampled bits
        let bits = (0..k).map(|_| party.generate_random(n_blocks)).collect_vec();
        let indices = un_bitslice_generic_index::<T,IndexType>(&bits);
        let ohv = generate_ohv(party, mul_triple_recorder, bits, 1<<k)?;
        // println!("Generated OHV {:?}", ohv.len());
        // for (i, ohvi) in ohv.iter().enumerate(){
        //     println!("OHVi {}: {}", i, ohvi.len());
        //     for (j, ohvj) in ohvi.iter().enumerate() {
        //         println!("{}: OHVII {:x?}", j, ohvj.si);
        //     }
        // }
        let un = un_bitslice_generic::<T,IndexType,SIZE>(&ohv);
        let res: Vec<RndOhvOutput<T, IndexType, SIZE>> = un.into_iter().zip(indices).map(|(ohv, index)| 
            RndOhvOutput::new(ohv, index)).collect();
        // println!("Result: {:?}", res[0]);
        Ok(res)
}



pub fn compute_ohv_vectors_cube<
REC: BitStringMulTripleRecorder,
T: GFTTrait,
IndexType: AllowedTypes,
const SIZE1: usize,
const SIZE2: usize,
const SIZE3: usize,
>(
    party: &mut MainParty, 
    mul_triple_recorder: &mut REC, 
    amount: usize,
    k: &[usize;3],
) -> MpcResult<Vec<CubeOhv<T,IndexType,SIZE1,SIZE2,SIZE3>>>{

    let ohv1 = generate_rndohv_k::<_,T,IndexType,SIZE1>(party, mul_triple_recorder, k[0], amount)?;
    let ohv2 = generate_rndohv_k::<_,T,IndexType,SIZE2>(party, mul_triple_recorder, k[1], amount)?;
    let ohv3 = generate_rndohv_k::<_,T,IndexType,SIZE3>(party, mul_triple_recorder, k[2], amount)?;
    // ohv1[0].print();
    let mut res = Vec::with_capacity(amount);
    for i in 0..amount {
        let cube = CubeOhv{
            row_ohv: ohv1[i],
            col_ohv: ohv2[i],
            lay_ohv: ohv3[i],
            _marker: std::marker::PhantomData,
        };
        res.push(cube);
    }
    Ok(res)    
}

pub fn compute_ohv_vectors_matrix<
REC: BitStringMulTripleRecorder,
T: GFTTrait,
IndexType: AllowedTypes,
const SIZE1:usize,
const SIZE2:usize,
>(
    party: &mut MainParty, 
    mul_triple_recorder: &mut REC, 
    amount: usize,
    k: &[usize; 2],
) -> MpcResult<Vec<MatrixOhv<T,IndexType,SIZE1,SIZE2>>>{
    let ohv1 = generate_rndohv_k::<_,T,IndexType,SIZE1>(party, mul_triple_recorder, k[0], amount)?;
    let ohv2 = generate_rndohv_k::<_,T,IndexType,SIZE2>(party, mul_triple_recorder, k[1], amount)?;
    let mut res = Vec::with_capacity(amount);
    for i in 0..amount {
        let matrix = MatrixOhv {
            row_ohv: ohv1[i],
            col_ohv: ohv2[i],
            _marker: std::marker::PhantomData,
        };
        res.push(matrix);
    }
    Ok(res)    
}

pub fn extract_byte_from_cube<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>(
    indices_usize: &Vec<usize>,
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]]
) -> T::Embedded {
    let row = indices_usize[0];
    let col = indices_usize[1];
    let lay = indices_usize[2];
    let lay_large = lay / T::RATIO;
    let lay_small = lay % T::RATIO;
    let mat = &lut_table[row];
    let vec = &mat[col];
    let val_large = <T as GFTTrait>::Wrapper::new(vec[lay_large]).to_usize();
    let val_small = (val_large >> (lay_small * size_of::<T::Embedded>() * 8)) as usize;
    // println!("before reduce 0x{:16x}", val_large);
    T::Embedded::from_usize(val_small)
}

pub fn extract_byte_from_matrix<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>(
    indices_usize: &Vec<usize>,
    lut_table: &[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]
) -> T::Embedded {
    let row = indices_usize[0];
    let col = indices_usize[1];
    let col_large = col / T::RATIO;
    let col_small = col % T::RATIO;
    let vec = &lut_table[row];
    let val_large = <T as GFTTrait>::Wrapper::new(vec[col_large]).to_usize();
    let val_small = (val_large >> (col_small * size_of::<T::Embedded>() * 8)) as usize;
    // println!("before reduce 0x{:16x}", val_large);
    T::Embedded::from_usize(val_small)
}

pub fn compare_result<
T: GFTTrait, 
IndexType: AllowedTypes,
const SIZE: usize, 
const SIZE_RED: usize>(
    indices: Vec<ShareType<IndexType>>, 
    value: T::Embedded,
    dim: usize,
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]]
) {
    let indices_usize: Vec<usize> = indices.iter().map(|x| x.inner().to_usize()).collect();
    let expected = match dim {
        3 => extract_byte_from_cube::<T, SIZE, SIZE_RED>(&indices_usize, lut_table),
        2 => extract_byte_from_matrix::<T, SIZE, SIZE_RED>(&indices_usize, &lut_table[0]),
        _ => panic!("Unsupported dimension: {}", dim),
    };
        
    if expected.to_usize() != value.to_usize() {
        // assert!(expected == value, "Extract pos [{},{},{}]:\t expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
        println!("Extract pos {:?}: expected: {:x}, got: {:x}", indices_usize, expected.to_usize(), value.to_usize());
    }
    // assert!(expected.to_u8() == value.to_u8(), "Extract pos [{},{},{}]:\t expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
}


