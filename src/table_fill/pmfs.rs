use std::collections::HashMap;

use fastnum::decimal::UnsignedDecimal;
use tracing::instrument;

// Iterates from 0 to n, computes the probability of DLap(L = k, p) for 0 and 2 * DLap(L = k, p) else
// This distribution is accurate if the sign is sampled uniformly at random
#[instrument(name = "Compute DLap probabilities and Delta", skip_all)]
pub fn d_lap<const S: usize>(n: u8, p: UnsignedDecimal<S>, debug: bool) -> (HashMap<u8, UnsignedDecimal<S>>, UnsignedDecimal<S>) {
    let mut delta = UnsignedDecimal::<S>::ONE;
    let p_factor = (UnsignedDecimal::<S>::ONE - p) / (UnsignedDecimal::<S>::ONE + p);
    
    let mut res = HashMap::new();
    let p0 = p_factor;
    delta -= p0;
    res.insert(0, p0);
    if debug {
        println!("DLap(p= {}), p_factor: {}", 
            p.rescale(10).to_scientific_notation(), 
            p_factor.rescale(10).to_scientific_notation()
        );
        println!("0 {} ", p0.to_scientific_notation());
    }
    let mut prob = p_factor.clone() * UnsignedDecimal::<S>::TWO;
    for i in 1..=n{
        prob *= p;
        if debug {
            println!("{} {} -> delta: {}", i, prob.to_scientific_notation(), delta.to_scientific_notation());
        }
        delta -= prob;
        res.insert(i, prob);
    }
    if delta.is_nan() {
        delta = UnsignedDecimal::<S>::ZERO;
    }
    (res, delta)
}


#[cfg(test)]
mod tests {
    use fastnum::udec1024;

    use super::d_lap;


    #[test]
    fn test_dlap(){
        let n = 255;
        let p = udec1024!(0.2);
        let (map, delta) = d_lap(n, p, true);
        println!("DLap with p: {} and bound: {} has delta: {}", p.to_scientific_notation(), n, delta.to_scientific_notation());
        
        for index in 0..=n{
            let value = map.get(&index).unwrap();
            println!("{}: {}",index, value.to_scientific_notation());
        }
    }

}