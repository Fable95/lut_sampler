// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
#![allow(long_running_const_eval)]
use std::collections::HashMap;

use fastnum::decimal::Decimal;
use tracing::instrument;

// Iterates from 0 to n, computes the probability of DLap(L = k, p) for 0 and 2 * DLap(L = k, p) else
// This distribution is accurate if the sign is sampled uniformly at random
#[instrument(name = "Compute DLap probabilities and Delta", skip_all)]
pub fn d_lap<const S: usize>(n: u16, p: &Decimal<S>, debug: bool) -> (HashMap<u16, Decimal<S>>, Decimal<S>, u16) {
    let mut delta = Decimal::<S>::ONE;
    let p_factor = (Decimal::<S>::ONE - *p) / (Decimal::<S>::ONE + *p);
    
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
    let mut prob = p_factor.clone() * Decimal::<S>::TWO;
    let limit = Decimal::<S>::from_i16(-512).exp2();
    let mut bound = n;
    for i in 1..=n{
        prob *= *p;
        if debug {
            println!("{} {} -> delta: {}", i, prob.to_scientific_notation(), delta.to_scientific_notation());
        }
        delta -= prob;
        res.insert(i, prob);
        if (i+1).is_power_of_two() && prob < limit {
            bound = i;
            break;
        }
    }
    println!("Truncation error at bound {}: {}", bound, delta.to_scientific_notation());
    if delta.is_nan() {
        delta = Decimal::<S>::ZERO;
    }
    (res, delta, bound)
}

fn check_invalid<const S: usize>(val: &Decimal<S>, name: &str, debug: bool) -> bool {
    let mut invalid = false;
    if val.is_op_underflow(){
        if debug {
            println!("{} has underflowed", name);
        }
        invalid = true;
    }
    if val.is_op_subnormal(){
        if debug {
            println!("{} is subnormal", name);
        }
        invalid = true;
    }
    if val.is_nan() {
        if debug {
            println!("{} is NaN", name);
        }
        invalid = true;
    }
    if val.is_infinite() {
        if debug {
            println!("{} is infinite", name);
        }
        invalid = true;
    }
    if val.is_negative() {
        if debug {
            println!("{} is negative", name);
        }
        invalid = true;
    }
    invalid
}

fn check_inexact<const S: usize>(val: &Decimal<S>, name: &str, debug: bool) -> bool {
    let mut invalid = false;
    if val.is_op_inexact(){
        if debug {
            println!("{} is inexact", name);
        }
        invalid = true;
    }
    if val.is_op_rounded(){
        if debug {
            println!("{} is rounded", name);
        }
        invalid = true;
    }
    invalid
}

#[instrument(name = "Compute DGauss probabilities and Delta", skip_all)]
pub fn d_gauss<const S: usize>(n: u16, v: &Decimal<S>, debug: bool) -> (HashMap<u16, Decimal<S>>, Decimal<S>, u16) {
    let limit = Decimal::<S>::from_i16(-130).exp2();

    let mut delta = Decimal::<S>::ONE;
    let mut res = HashMap::new();
    let var = *v * Decimal::<S>::TWO; // Compute the double variance
    let mut denominator = Decimal::<S>::ONE; // Init with the 0th element
    let mut y_vec = Vec::with_capacity(n as usize);
    let mut bound = n;
    let mut truncation_error = Decimal::<S>::ZERO;
    for i in 1..=n {
        let mut y = Decimal::<S>::from_u16(i);
        y = y.powi(2);
        y = -y / var;
        
        // assert!(!check_inexact(&y, "y", debug));
        y = y.exp();
        y *= Decimal::<S>::TWO; // Multiply by 2
        // println!("Current d_i: {}", y.to_scientific_notation());
        if check_invalid(&y, "di", debug){
            bound = i-1;
            // println!("Bound B = {}", bound);
            break;
        }
        assert!(!check_invalid(&y, "di", debug), "invalid di at i: {}", i);
        
        denominator += y;
        if check_invalid(&denominator, "denom", debug){
            bound = i-1;
            // println!("Bound B = {}", bound);
            break;
        }
        y_vec.push(y);
        truncation_error = y;
        if (i+1).is_power_of_two() && truncation_error < limit {
            bound = i;
            break;
        }
    }
    println!("Truncation error at {}: {}", bound, truncation_error.to_scientific_notation());
    let p0 = Decimal::<S>::ONE / denominator;
    res.insert(0, p0);

    if debug {
        println!("Denominator: {}", denominator.to_scientific_notation());
        println!("{} {} -> delta: {}", 0, p0.to_scientific_notation(), delta.to_scientific_notation());
    }
    delta -= p0;
    
    for i in 1..=bound {
        let mut y = y_vec[(i-1) as usize];
        y /= denominator; // Divide by the denominator
        // if debug {
        //     println!("{} {} -> delta: {}", i, y.to_scientific_notation(), delta.to_scientific_notation());
        // }
        delta -= y;
        if check_invalid(&y, "y", debug) {
            // println!("y or delta is invalid");
            panic!("Invalid value encountered in DGauss computation at i: {}", i);
            // break;
        }
        res.insert(i, y);
    }
    if delta.is_nan() || delta.is_negative() {
        delta = Decimal::<S>::ZERO;
    }
    if debug{
        println!("Truncation Delta: {}", delta.to_scientific_notation());
        
        println!("DGauss, first 10 probs");
        println!("{{\n\t0: {},", res.get(&0).unwrap_or(&Decimal::<S>::ZERO).to_scientific_notation());
        for i in 1..=100.min(bound){
            let val = res.get(&i).unwrap_or(&Decimal::<S>::ZERO).div(Decimal::<S>::TWO);
            println!("\t{}: {}, ", i, val.to_scientific_notation());
        }
        println!("}}");
    }

    (res, delta, bound)
}

pub fn get_prob_bits<const S: usize>(p: &Decimal<S>, bits: usize) -> Vec<HashMap<u16, Decimal<S>>> {
    let mut map_vec = Vec::with_capacity(bits);
    
    for i in 0..bits{
        let beta_i = (Decimal::ONE + p.powi(-2 * i as i32)).powi(-1);
        println!("Beta_{} = {}", i, beta_i.to_scientific_notation());
        let mut target_map = HashMap::new();
        target_map.insert(1, beta_i);
        target_map.insert(0, Decimal::<S>::ONE - beta_i);
        map_vec.push(target_map);
    }

    map_vec    
}




#[cfg(test)]
mod tests {
    #![allow(long_running_const_eval)]
    use fastnum::{D256, decimal::Decimal};
    use super::d_lap;


    #[test]
    fn test_dlap_sums_to_one(){
        let n = 512;
        const S: usize = 1024/64;
        let p = Decimal::<S>::from_f64(0.4);
        // let p = D2048::from_f64(0.996);
        let (_map, delta, bound) = d_lap(n, &p, false);
        println!("DLap with p: {} and bound: {} has delta: {}", p.to_scientific_notation(), bound, delta.to_scientific_notation());
        let mut accum = Decimal::<S>::ZERO;
        for index in 0..=bound{
            let value = _map.get(&index).unwrap();
            accum = accum.add(*value);
        }
        println!("Accumulated probability: {}", accum.to_scientific_notation());
    }

    #[test]
    fn test_dgauss_sums_to_one(){
        let n = 64;
        // let v = 0.2;
        const S: usize = 1024/64;
        let var = Decimal::<S>::from_i64(1);
        let (map, delta, bound) = super::d_gauss(n, &var, true);
        println!("DGauss with var: {} and bound: {} has delta: {}", var.to_scientific_notation(), bound, delta.to_scientific_notation());
        let mut accum = Decimal::<S>::ZERO;
        for index in 0..=bound{
            let value = map.get(&index).unwrap();
            accum = accum.add(*value);
            // println!("{}: {}",index, value.to_scientific_notation());
        }
        println!("Accumulated probability: {}", accum.to_scientific_notation());
    }

    #[test]
    fn test_fdl_high(){
        // let v = 0.2;
        let var = D256::from_f64(0.8);
        let vec = super::get_prob_bits(&var, 16);
        println!("FDL with p: {} and bound: {} ", var.to_scientific_notation(), ((1 << vec.len()) - 1));
        for (i, val) in vec.iter().enumerate(){
            println!("{}: {}", i, val.get(&1).unwrap_or(&D256::ZERO).to_scientific_notation());
        }
    }
}