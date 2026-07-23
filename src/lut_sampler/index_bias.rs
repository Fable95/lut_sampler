use itertools::Itertools;

use maestro::rep3_core::{
    party::{error::MpcResult, MainParty, Party},
    share::{RssShare, RssShareVec},
};

use rss_lut::offline::compute_product;
use rss_lut::party::Network;
use rss_lut::share::gf_template::ShareType;
use rss_lut::share::{gf2p64::GF2p64, gf_template::Share, helper_types::AllowedTypes};
use rss_lut::util::mul_triple_vec::MulTripleRecorder;

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
