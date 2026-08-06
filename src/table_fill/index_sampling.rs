// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
use std::collections::HashMap;

use fastnum::decimal::Decimal;
use tracing::{instrument, span, Level};

use crate::lut_sampler::IndexSampling;


fn bernoulli_skewed<const S: usize>(k: usize, l: usize, p: &Decimal<S>) -> Vec<Decimal<S>> {
    assert!(k >= l, "more skewed bits requested than available");
    let p_0 = Decimal::<S>::ONE - *p;
    let mut res = Vec::with_capacity(1 << k);

    // each value has the probability factor of 2^-(k-l)
    let factor = Decimal::<S>::TWO.powi((l as i32) - (k as i32));
    let mask = (1 << l) - 1;
    for i in 0..(1 << k){
        let relevant_bits = i & mask;
        let n1 = (relevant_bits as usize).count_ones();
        let n0 = (l as u32) - n1;
        let power0 = p_0.powi(n0 as i32);
        let power1 = p.powi(n1 as i32);
        let prob = power0 * power1 * factor;
        res.push(prob);
    }
    res
}

pub fn get_probability_vector<const S: usize>(sampling: IndexSampling, k: usize, p: Decimal<S>) -> Vec<Decimal<S>>{
    let one = Decimal::<S>::ONE;
    match sampling {
        IndexSampling::Uniform => {
            let p = one / (1 << k);
            vec![p; 1 << k]
        },
        IndexSampling::Biased => bernoulli_skewed(k, k, &p),
    }
}

// Computes all probability products from multidimensional index combinations
#[instrument(name = "Compute Available probs", skip_all)]
pub fn get_probability_table<const S: usize>(k: &Vec<usize>, l: usize, p: &Decimal<S>) -> (HashMap<Decimal<S>, Vec<Vec<usize>>>, Vec<Decimal<S>>){
    let d = k.len();
    let total_k: usize = k.iter().sum();
    assert!(total_k >= l, "SUM({:?}) = {} < l {}", k, total_k, l);
    let mut dimensions = Vec::with_capacity(d);
    
    let span = span!(Level::INFO, "Compute Probability Vectors").entered();
    
    let mut remaining: i64 = l.try_into().unwrap();
    for &ki in k.iter(){
        let k_i = i64::try_from(ki).unwrap();
        let active = remaining.clamp(0, k_i) ;
        remaining -= active;
        dimensions.push(bernoulli_skewed(ki, active as usize, p));
    }

    span.exit();
    let mut map = HashMap::new();
    let mut indices = Vec::with_capacity(dimensions.len());
    let mut sorted_keys = Vec::new();
    let span_n =  span!(Level::INFO, "fill map recursive and sort").entered();
    fill_map(dimensions.as_slice(), 0, &mut indices, Decimal::<S>::ONE, &mut map, &mut sorted_keys);
    sorted_keys.sort_by(|a,b| b.cmp(a));
    span_n.exit();
    (map, sorted_keys)
}

pub fn fill_map<const S: usize>(
    dimensions: &[Vec<Decimal<S>>], 
    depth: usize,  
    indices: &mut Vec<usize>,
    accum: Decimal<S>,
    map: &mut HashMap<Decimal<S>, Vec<Vec<usize>>>,
    sorted_keys: &mut Vec<Decimal<S>>
){
    if depth == dimensions.len() {
        let entry = map.entry(accum).or_insert_with(||{
            sorted_keys.push(accum);
            Vec::new()
        });
        entry.push(indices.clone());
        return;
    }

    for (i, &prob) in dimensions[depth].iter().enumerate() {
        indices.push(i);
        fill_map(dimensions, depth + 1, indices, accum * prob, map, sorted_keys);
        indices.pop();
    }
}


#[cfg(test)]
mod tests {
    use fastnum::dec256;

    use super::*;

    #[test]
    fn test_skewed(){
        let p = dec256!(0.25);
        let (map, sorted_keys) = get_probability_table(&vec![8,8,8], 24, &p);
        let mut count = 0;
        for key in sorted_keys{
            println!("Key: {}", key.to_scientific_notation());
            let vec = map.get(&key).unwrap();
            count += vec.len();
            // for indices in vec{
            //     println!("{:?}", indices);
            // }
        }
        println!("Total: {}", count);
    }


    // #[test]
    // fn test_uniform(){
    //     let p = dec256!(0);
    //     let (map, sorted_keys) = get_probability_table(&vec![8,8,8], 0, &p);
    //     let mut count = 0;
    //     for key in sorted_keys{
    //         println!("Key: {}", key.to_scientific_notation());
    //         let vec = map.get(&key).unwrap();
    //         count += vec.len();
    //         // for indices in vec{
    //         //     println!("{:?}", indices);
    //         // }
    //     }
    //     println!("Total: {}", count);
    // }

    // #[test]
    // fn test_binomial(){
    //     let indices = &[IndexSampling::Binomial; 2];
    //     let p = &[dec512!(0.5); 2];
    //     let (map, sorted_keys) = get_probability_table(6, indices, p);
    //     let mut count = 0;
    //     for key in sorted_keys{
    //         println!("Key: {}", key.to_scientific_notation());
    //         let vec = map.get(&key).unwrap();
    //         count += vec.len();
    //         // for indices in vec{
    //         //     println!("{:?}", indices);
    //         // }
    //     }
    //     println!("Total: {}", count);
    // }

}