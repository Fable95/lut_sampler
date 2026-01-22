use std::collections::HashMap;

use fastnum::decimal::Decimal;
use tracing::{instrument, span, Level};

use super::LookupTable;


fn debug_input_probs<const S: usize>(map: &HashMap<Decimal<S>, Vec<Vec<usize>>>, sorted_keys: &Vec<Decimal<S>>){
    let mut count = 0;
    for key in sorted_keys{
        let vec = map.get(&key).unwrap();
        println!("{} of: {}", vec.len(), key.to_scientific_notation());
        count += vec.len();
        // for indices in vec{
        //     println!("{:?}", indices);
        // }
    }
    println!("Total: {}", count);
}

pub fn neg_power_two<const S: usize>(p: &Decimal<S>) -> String {
    if p.is_one(){
        return String::from("0")
    }
    let one = Decimal::<S>::ONE;
    if !p.is_normal(){
        return String::from("NaN");
    }
    assert!(one > *p, "1 > p, not satisfied");
    assert!(Decimal::<S>::ZERO < *p, "p > 0, not satisfied");
    let mut res = 1;
    let mut copy = p.clone();
    while copy < one{
        copy *= Decimal::<S>::TWO;
        res -= 1
    }
    res.to_string()
}

#[instrument(name = "Fill LUT", skip_all)]
pub fn fill_d_lut<
const S: usize
>(
    k: &Vec<usize>,
    n: u16,
    index_map: &HashMap<Decimal<S>, Vec<Vec<usize>>>,
    index_prob_vec: &Vec<Decimal<S>>,
    target_prob_map: &HashMap<u16, Decimal<S>>,
    truncation_delta: &Decimal<S>,
    debug: bool
) 
-> (LookupTable<u16>, Decimal<S>)
{
    let mut map = index_map.clone();
    let prob_vec = index_prob_vec.clone();
    if debug{
        debug_input_probs(&map, &prob_vec);
    }
    // Get target probabilities and initial delta
    let mut target_map = target_prob_map.clone();
    let mut delta = truncation_delta.clone();
    let mut counted_deltas = 0;
    // Create instance of final lookup table
    let mut table = LookupTable::<u16>::new(&k, 0u16);

    // Placeholder for potentially unused probabilities
    let mut unused_probabilities = Vec::new();

    let mut sum_unused = Decimal::<S>::ONE;

    // Iterate over the available sorted probabilities
    if debug{
        println!("First loop underfitting target pmf");
    }
    let span_first = span!(Level::INFO, "Underfitting Loop").entered();
    for probability in prob_vec{
        let source_vec = map.get_mut(&probability).unwrap();
        // Iterate over the entire range of the target
        for i in 0..=n{
            let target = target_map.get_mut(&(i as u16)).unwrap(); 

            // As long as a target has higher probability it is entered into the table
            while *target >= probability{
                *target -= probability;
                sum_unused -= probability;
                // Get the vector of indices and set the respective value in the table
                let indices = source_vec.pop().unwrap();
                table.set(&indices, i);

                if source_vec.is_empty() {
                    break;
                }
            }
            if source_vec.is_empty() {
                break;
            }
        }
        if !source_vec.is_empty(){
            unused_probabilities.push(probability);
        }
    }
    span_first.exit();
    // (table, delta, unused_probabilities)
    if debug {
        println!("Remaining mass: {}\nSecond loop, filling remainders", sum_unused.to_scientific_notation());
        println!("Number of available probabilities in table {}", unused_probabilities.len());
    }
    // Now the target is underfitted and we will fill the table with minimal "damage"
    let span_second = span!(Level::INFO, "Second Loop, filling remainders").entered();
    for probability in unused_probabilities{
        let source_vec = map.get_mut(&probability).unwrap();
        for indices in source_vec{
            let mut max_value = Decimal::<S>::ZERO;
            let mut max_index = 0u16;
            for (index, value) in target_map.iter(){
                if *value > max_value{
                    max_value = *value;
                    max_index = *index;
                }
            }
            if max_value > probability{
                println!("In the second loop there should not be a positive overshoot, but found {} > {}", 
                    max_value.to_scientific_notation(), 
                    probability.to_scientific_notation()
                );
                return (table, Decimal::<S>::ZERO);
            } 
            
            // Summing up half the negative overshoot
            // println!("{}: {} - {}", max_index, probability.to_scientific_notation(), max_value.to_scientific_notation());
            delta += probability - max_value;
            counted_deltas += 1;
            table.set(indices, max_index);

            target_map.remove(&max_index);
        }
    }
    span_second.exit();
    let span_final = span!(Level::INFO, "Compute delta and count values").entered();
    // Summing up all remaining positive probabilities
    if debug {
        println!("Remaining targets: {}\n", target_map.keys().len());
    }
    for (i, value) in target_map.iter(){
        if debug{
            println!("{}: {}", i, value.to_scientific_notation());
        }
        delta += *value;
        counted_deltas += 1;
    }
    delta /= Decimal::<S>::TWO;
    span_final.exit();
    if debug {
        println!("Counted deltas: {}", counted_deltas);
    }
    // println!("SD(Z, Pi_Z) = {} < 2^{} considered range: [0,{}]\n", 
    //     delta.to_scientific_notation(), 
    //     neg_power_two(&delta),    
    //     n
    // );
    (table, delta)
}


#[cfg(test)]
mod tests{

    use fastnum::{D256, D512, dec256, decimal::Decimal};

    use crate::{table_fill::{TableParams, greedy_fill::neg_power_two, index_sampling::get_probability_table, pmfs::{d_lap, get_prob_bits}}};

    use super::fill_d_lut;






    #[test]
    fn test_greedy_bit_fill(){
        let k: Vec<Vec<usize>> = vec![vec![6],vec![7],vec![8],vec![9],vec![10],vec![11],vec![12]]; 
        // let k: Vec<usize> = vec![10,12,14]; 
        // let l: usize = 5;
        let p = D512::from_f64(0.9);
        let bits = 40;
        let map_vec = get_prob_bits(&p, bits);

        let p_fourth = D512::HALF / D512::TWO;

        let vals: Vec<D512> = std::iter::successors(Some(p_fourth), |x| {
            Some(*x / D512::TWO)
        }).take(12).collect();
        
        
        let mut deltas = vec![TableParams::new(D512::from_i16(-130).exp2(), (1 << bits) - 1 as usize); map_vec.len()];

        for (c_ber, p) in vals.iter().enumerate(){
            println!("Testing p: {}", p.to_scientific_notation());
            for k in k.iter(){
                let total_k = k.iter().sum();
                if TableParams::all_set(&deltas){
                    break;
                }
                // print!("{} ", k);
                for l in 0..total_k{
                    for (b, target_map) in map_vec.iter().enumerate(){
                        let (index_map, index_probs) = get_probability_table(k, l, &p);
                        // println!("\tTesting d={} k={} l={} index p=2^-{} for target bit {}", d, k, l, c_ber+2, b);
                        let (table, delta) = fill_d_lut(k,  1, &index_map, &index_probs, &target_map, &Decimal::ZERO, false);
                        if delta < deltas[b].delta {
                            deltas[b].update(vec![table], &k, l, (c_ber + 2) as u32, delta, false, false);
                        }
                    }
                }
            }
            if TableParams::all_set(&deltas){
                break;
            }
            // print!("\n");
        }
        let mut total_delta = D512::ZERO;
        for (b, tc) in deltas.iter().enumerate() {
            println!("Bit {}: has {:?}", b, tc);
            total_delta += tc.delta;
        }
        println!("SD(Z, Pi_Z) = {} < 2^{}\n", total_delta.to_scientific_notation(), neg_power_two(&total_delta));        
    }

    #[test]
    fn test_greedy_fill(){
        
        let k: Vec<usize> = vec![8,8,8]; 
        // const D: usize = 3;
        let d_lap_p = [
            dec256!(0.1),dec256!(0.2),dec256!(0.3),
            dec256!(0.4),dec256!(0.5),dec256!(0.6),
            dec256!(0.7),dec256!(0.8),dec256!(0.9)
        ];

        let p_half = D256::ONE / D256::TWO;
        let p_sixteenth = p_half / D256::EIGHT;
    
        for p in d_lap_p{
            let (index_map, index_probs) = get_probability_table(&k, 24, &p_sixteenth);
            let (target_map, delta, _) = d_lap(255, &p, true);
            let (_, _) = fill_d_lut(&k, 255, &index_map, &index_probs, &target_map, &delta, true);
        }
        
    }
}