use std::collections::HashMap;

use fastnum::decimal::UnsignedDecimal;
use tracing::{instrument, span, Level};


use crate::lut_sampler::IndexSampling;

use super::{index_sampling::get_probability_table, pmfs::d_lap, LookupTable};


fn debug_input_probs<const S: usize>(map: &HashMap<UnsignedDecimal<S>, Vec<Vec<usize>>>, sorted_keys: &Vec<UnsignedDecimal<S>>){
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

fn neg_power_two<const S: usize>(p: &UnsignedDecimal<S>) -> String {
    let one = UnsignedDecimal::<S>::ONE;
    if !p.is_normal(){
        return String::from("NaN");
    }
    assert!(one > *p, "1 > p, not satisfied");
    assert!(UnsignedDecimal::<S>::ZERO < *p, "p > 0, not satisfied");
    let mut res = 1;
    let mut copy = p.clone();
    while copy < one{
        copy *= UnsignedDecimal::<S>::TWO;
        res -= 1
    }
    res.to_string()
}

#[instrument(name = "Fill LUT", skip_all)]
pub fn fill_d_lut<
const S: usize
>(
    k: usize, 
    index_samplings: &[IndexSampling],
    index_sampling_probabilities: &[UnsignedDecimal<S>],
    d_lap_p: UnsignedDecimal<S>,
    debug: bool
) 
-> (LookupTable<u8>, UnsignedDecimal<S>)
{
    let d = index_samplings.len();
    assert!(index_sampling_probabilities.len() == d, "each sampling needs a probabilies");

    // Get available probabilities table
    let (mut map, prob_vec) = get_probability_table(k, index_samplings, &index_sampling_probabilities);
    if debug{
        debug_input_probs(&map, &prob_vec);
    }
    // Get target probabilities and initial delta
    let (mut target_map, mut delta) = d_lap(255, d_lap_p, debug);
    let mut counted_deltas = 0;
    // Create instance of final lookup table
    let mut table = LookupTable::<u8>::new(d, k, 0u8);

    // Placeholder for potentially unused probabilities
    let mut unused_probabilities = Vec::new();

    let mut sum_unused = UnsignedDecimal::<S>::ONE;

    // Iterate over the available sorted probabilities
    if debug{
        println!("First loop underfitting target pmf");
    }
    let span_first = span!(Level::INFO, "Underfitting Loop").entered();
    for probability in prob_vec{
        let source_vec = map.get_mut(&probability).unwrap();
        // Iterate over the entire range of the target
        for i in 0..=255{
            let target = target_map.get_mut(&(i as u8)).unwrap(); 

            // As long as a target has higher probability it is entered into the table
            while *target > probability{
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
        println!("unused elements {}, remaining targets {}", unused_probabilities.len(), target_map.keys().len());
    }
    // Now the target is underfitted and we will fill the table with minimal "damage"
    let span_second = span!(Level::INFO, "Second Loop, filling remainders").entered();
    for probability in unused_probabilities{
        let source_vec = map.get_mut(&probability).unwrap();
        for indices in source_vec{
            let mut max_value = UnsignedDecimal::<S>::ZERO;
            let mut max_index = 0u8;
            for (index, value) in target_map.iter(){
                if *value > max_value{
                    max_value = *value;
                    max_index = *index;
                }
            }
            assert!(max_value < probability, "In the second iteration there may never be a larger probability");
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
    delta /= UnsignedDecimal::<S>::TWO;
    span_final.exit();
    println!("SD(DLap({}), Pi_Z) = {} < 2^{} considered range: [0,{}]\n", 
        d_lap_p.rescale(10).to_scientific_notation(),  
        delta.to_scientific_notation(), 
        neg_power_two(&delta),    
        counted_deltas-1
    );
    (table, delta)
}


#[cfg(test)]
mod tests{
    use fastnum::{udec256, UD256};

    use crate::lut_sampler::IndexSampling;

    use super::fill_d_lut;



    #[test]
    fn test_greedy_fill(){
        
        let k: usize = 8; 
        let index_samplings =  &[IndexSampling::Biased; 3]; 
        let d_lap_p = [
            udec256!(0.1),udec256!(0.2),udec256!(0.3),
            udec256!(0.4),udec256!(0.5),udec256!(0.6),
            udec256!(0.7),udec256!(0.8),udec256!(0.9)
        ];

        let p_half = UD256::ONE / UD256::TWO;
        let p_sixteenth = p_half / UD256::EIGHT;
    
        let mut p_vec = Vec::with_capacity(index_samplings.len());
        for sampling in index_samplings{
            p_vec.push( match sampling {
                IndexSampling::Uniform => UD256::ZERO,
                IndexSampling::Binomial => p_half,
                IndexSampling::Biased => p_sixteenth,
            })
        }

        for p in d_lap_p{
            let (_, _) = fill_d_lut(k, index_samplings, &p_vec, p, true);
        }
        
    }
}