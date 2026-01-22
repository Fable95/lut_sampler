use std::{collections::HashMap, fmt::Error};
use fastnum::decimal::Decimal;
use tracing::{instrument};

use crate::table_fill::{LookupTable, greedy_fill::neg_power_two};

#[instrument(name = "Fill LUT", skip_all)]
pub fn fill_many_luts<
const S: usize
>(
    k: usize,
    n: u16,
    target_prob_map: &HashMap<u16, Decimal<S>>,
    truncation_delta: &Decimal<S>,
    accuracy_breakpoint: &Decimal<S>,
    max_tables: usize,
    debug: bool
) 
-> (Vec<LookupTable<u16>>, Vec<usize>, Decimal<S>)
{

    let num_elts = Decimal::<S>::from_i64(1<<k);
    

    // Get target probabilities and initial delta
    let mut target_map = target_prob_map.clone();

    let mut res_table_vec = Vec::with_capacity(max_tables);
    let mut maximum_indices = Vec::with_capacity(max_tables);
    let index_prob = Decimal::ONE.div(num_elts);
    let mut out_pmf = vec![Decimal::<S>::ZERO; (n+1) as usize];
    let mut sum_unused = Decimal::ONE;
    let mut current_index_prob = index_prob;
    for table_index in 0..max_tables{
        // Create instance of final lookup table
        if debug {
            println!("Table {} has index prob 2^{}", table_index, neg_power_two(&current_index_prob));
        }
        let mut table = LookupTable::<u16>::new(&vec![k], 0u16);
        let mut current_index = 0;
        let mut num_unused = num_elts;
        for i in 0..=n{
            let target = target_map.get_mut(&(i as u16)).unwrap(); 
    
            // As long as a target has higher probability it is entered into the table
            while *target >= current_index_prob{
                *target -= current_index_prob;
                sum_unused -= current_index_prob;
                num_unused -= 1;
                out_pmf[i as usize] += current_index_prob;
                // Get the vector of indices and set the respective value in the table
                table.set(&[current_index], i);
                current_index += 1;
                assert!(current_index <= (1 << k), "index must not be larger than elements in table");
            }
        }
        let remaining_prob = num_unused * current_index_prob;
        current_index_prob = remaining_prob * index_prob;
        res_table_vec.push(table);
        maximum_indices.push(current_index);
        if num_unused == Decimal::from_i64(1 << k) {
            panic!("\t {} elts remaining, approximation accuracy cannot be reached, increase table size", num_unused);    
        }
        println!("\t and {} elts remaining", num_unused);

        if sum_unused.mul(Decimal::TWO) < *accuracy_breakpoint{
            break;
        }

    }

    let mut sd_check = Decimal::ZERO;
    for i in 0..=n{
        sd_check += (out_pmf[i as usize] - *target_prob_map.get(&i).unwrap()).abs();
    }

    let delta = *truncation_delta + sum_unused;

    println!("SD(Z, Pi_Z) = {} < 2^{} considered range: [0,{}]\n", 
        delta.to_scientific_notation(), 
        neg_power_two(&delta),    
        n
    );
    println!("SD-CHECK = {} < 2^{} considered range: [0,{}]\n", 
        sd_check.to_scientific_notation(), 
        neg_power_two(&sd_check),    
        n
    );
    println!("Final number of tables: {}", res_table_vec.len());
    
    (res_table_vec, maximum_indices, delta)
}

#[instrument(name = "Fill LUT", skip_all)]
pub fn fill_many_luts_biased<
const S: usize
>(
    k: usize,
    n: u16,
    target_prob_map: &HashMap<u16, Decimal<S>>,
    truncation_delta: &Decimal<S>,
    index_map: &HashMap<Decimal<S>, Vec<Vec<usize>>>,
    index_prob_vec: &Vec<Decimal<S>>,
    accuracy_breakpoint: &Decimal<S>,
    max_tables: usize,
    debug: bool
) 
-> Result<(Vec<LookupTable<u16>>, Vec<Vec<usize>>, Decimal<S>), Error>
{
    // Copy input data - for now only accept single table bias
    let mut target_map = target_prob_map.clone();
    

    let mut res_table_vec = Vec::with_capacity(max_tables);
    let mut remaining_indices_all = Vec::with_capacity(max_tables);
    let mut out_pmf = vec![Decimal::<S>::ZERO; (n+1) as usize];
    let mut sum_unused = Decimal::ONE;
    let mut current_index_prob = Decimal::ONE;
    for table_index in 0..max_tables{
        // Create instance of final lookup table
        if debug {
            println!("Table {} has index prob factor 2^{}", table_index, neg_power_two(&current_index_prob));
        }
        let mut map = index_map.clone();
        let mut table = LookupTable::<u16>::new(&vec![k], 0u16);
        let mut unassigned_probability = Decimal::ONE;
        let mut remaining_indices = Vec::with_capacity(1 << k);
        for &probability in index_prob_vec.iter(){
            let source_vec = map.get_mut(&probability).unwrap();
            let index_probability = probability * current_index_prob;
            for i in 0..=n{

                let target = target_map.get_mut(&(i as u16)).unwrap(); 
        
                // As long as a target has higher probability it is entered into the table
                while *target >= index_probability{
                    *target -= index_probability;
                    sum_unused -= index_probability;
                    unassigned_probability -= index_probability;
                    out_pmf[i as usize] += index_probability;
                    // Get the vector of indices and set the respective value in the table
                    let indices = source_vec.pop().unwrap();
                    table.set(&indices, i);

                    if source_vec.is_empty() {
                        break;
                    }
                }
            }
            if !source_vec.is_empty(){
                for index in source_vec{
                    remaining_indices.push(index[0]);
                }
            }
        }

        current_index_prob *= unassigned_probability;
        res_table_vec.push(table);
        println!("\t and {} elts remaining", remaining_indices.len());
        if remaining_indices.len() == (1 << k) {
            return Err(Error)
        }
        remaining_indices_all.push(remaining_indices);
        
        if sum_unused.mul(Decimal::TWO) < *accuracy_breakpoint{
            break;
        }

    }

    let mut sd_check = Decimal::ZERO;
    for i in 0..=n{
        sd_check += (out_pmf[i as usize] - *target_prob_map.get(&i).unwrap()).abs();
    }

    let delta = *truncation_delta + sum_unused;

    println!("SD(Z, Pi_Z) = {} < 2^{} considered range: [0,{}]\n", 
        delta.to_scientific_notation(), 
        neg_power_two(&delta),    
        n
    );
    println!("SD-CHECK = {} < 2^{} considered range: [0,{}]\n", 
        sd_check.to_scientific_notation(), 
        neg_power_two(&sd_check),    
        n
    );
    println!("Final number of tables: {}", res_table_vec.len());
    
    Ok((res_table_vec, remaining_indices_all, delta))
}





#[cfg(test)]
mod tests{

    use fastnum::{D256, dec256};

    use crate::table_fill::{TableParams, index_sampling::get_probability_table, many_fill::fill_many_luts_biased, pmfs::d_gauss};

    use super::fill_many_luts;

    #[test]
    fn test_many_fill(){
        
        let k: usize = 14; 
        let lambda = 40;
        let break_point = D256::TWO.powi(-lambda);
        let variance = dec256!(1);
        
        
        let (target_map, delta, bound) = d_gauss(65535, &variance, false);
        let (_, _, _) = fill_many_luts(k, bound, &target_map, &delta, &break_point, lambda as usize, true);
        
    }

    
    #[test]
    fn test_many_fill_biased(){
        
        let d = 1;
        let k: usize = 16; 
        let lambda = 40;
        let break_point = D256::TWO.powi(-lambda);
        let variance = dec256!(935089);
        let p_fourth = D256::HALF / D256::TWO;

        let vals: Vec<D256> = std::iter::successors(Some(p_fourth), |x| {
            Some(*x / D256::TWO)
        }).take(12).collect();
        
        let (target_map, delta, bound) = d_gauss(65535, &variance, false);
        let mut best_delta = TableParams::new(D256::from_i16(-45).exp2(), bound as usize);

        for (c_ber, p) in vals.iter().enumerate(){
            for l in 0..=k{    
                println!("\tTesting d={} k={} l={} index p=2^-{}", d, k, l, c_ber+2);
                let (index_map, index_probs) = get_probability_table(&vec![16], l, &p);
                let res
                    = fill_many_luts_biased(k, bound, &target_map, &delta, &index_map, &index_probs, &break_point, lambda as usize, true);
                if res.is_ok(){
                    let (table, _, delta) = res.unwrap();
                    best_delta.update(table, &vec![k], l, (c_ber + 2) as u32, delta, false, true);
                }
            }
        }
        println!("Best delta: {:?}", best_delta);
    }
}