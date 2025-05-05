use std::collections::HashMap;

use fastnum::decimal::UnsignedDecimal;
use num_integer::IterBinomial;
use tracing::{instrument, span, Level};

use crate::lut_sampler::IndexSampling;

fn bin_k<const S: usize>(n: u128, p: UnsignedDecimal<S>) -> Vec<UnsignedDecimal<S>> {
    assert!(n < 129, "At the momement max n is set to 128");

    let binomial_iterator = IterBinomial::new(n);
    let p_inv = UnsignedDecimal::<S>::ONE - p;
    
    let mut res = Vec::with_capacity((n + 1) as usize);
    for (k, coefficient) in binomial_iterator.enumerate(){
        let prob = p.powi(k as i32) * p_inv.powi(((n as usize) - k) as i32) * coefficient;
        res.push(prob);
    }
    res
}


fn bernoulli_skewed<const S: usize>(k: usize, p: UnsignedDecimal<S>) -> Vec<UnsignedDecimal<S>> {
    let p_0 = UnsignedDecimal::<S>::ONE - p;
    let mut res = Vec::with_capacity(1 << k);

    for i in 0..(1 << k){
        let n1 = (i as usize).count_ones();
        let n0 = (k as u32) - n1;
        let power0 = p_0.powi(n0 as i32);
        let power1 = p.powi(n1 as i32);
        let prob = power0 * power1;
        res.push(prob);
    }
    res
}

pub fn get_probability_vector<const S: usize>(sampling: IndexSampling, k: usize, p: UnsignedDecimal<S>) -> Vec<UnsignedDecimal<S>>{
    let one = UnsignedDecimal::<S>::ONE;
    match sampling {
        IndexSampling::Uniform => {
            let p = one / (1 << k);
            vec![p; 1 << k]
        },
        IndexSampling::Binomial => bin_k((1<<k)-1, p),
        IndexSampling::Biased => bernoulli_skewed(k, p),
    }
}

// Computes all probability products from multidimensional index combinations
#[instrument(name = "Compute Available probs", skip_all)]
pub fn get_probability_table<const S: usize>(k: usize, sampling: &[IndexSampling], p: &[UnsignedDecimal<S>]) -> (HashMap<UnsignedDecimal<S>, Vec<Vec<usize>>>, Vec<UnsignedDecimal<S>>){
    let d = sampling.len();
    assert!(p.len() == d, "not enough probs for each dist");
    let mut dimensions = Vec::with_capacity(d);
    
    let span = span!(Level::INFO, "Compute Probability Vectors").entered();
    for (sample_strat, p_val) in sampling.iter().zip(p) {
        dimensions.push(get_probability_vector(*sample_strat, k, *p_val));
    }
    span.exit();
    let mut map = HashMap::new();
    let mut indices = Vec::with_capacity(dimensions.len());
    let mut sorted_keys = Vec::new();
    let span_n =  span!(Level::INFO, "fill map recursive and sort").entered();
    fill_map(dimensions.as_slice(), 0, &mut indices, UnsignedDecimal::<S>::ONE, &mut map, &mut sorted_keys);
    sorted_keys.sort_by(|a,b| b.cmp(a));
    span_n.exit();
    (map, sorted_keys)
}

pub fn fill_map<const S: usize>(
    dimensions: &[Vec<UnsignedDecimal<S>>], 
    depth: usize,  
    indices: &mut Vec<usize>,
    accum: UnsignedDecimal<S>,
    map: &mut HashMap<UnsignedDecimal<S>, Vec<Vec<usize>>>,
    sorted_keys: &mut Vec<UnsignedDecimal<S>>
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
    use fastnum::{udec256, udec512};

    use super::*;

    #[test]
    fn test_skewed(){
        let indices = &[IndexSampling::Biased; 3];
        let p = &[udec256!(0.25); 3];
        let (map, sorted_keys) = get_probability_table(8, indices, p);
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


    #[test]
    fn test_uniform(){
        let indices = &[IndexSampling::Uniform; 3];
        let p = &[udec256!(0); 3];
        let (map, sorted_keys) = get_probability_table(8, indices, p);
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

    #[test]
    fn test_binomial(){
        let indices = &[IndexSampling::Binomial; 2];
        let p = &[udec512!(0.5); 2];
        let (map, sorted_keys) = get_probability_table(6, indices, p);
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

}