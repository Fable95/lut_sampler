// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
#![allow(long_running_const_eval)]

use std::io::{self, Write};

use fastnum::{decimal::{Context, Decimal}};
use lut_sampler::table_fill::{LookupTable, TableParams, export::{EmbeddedType, write_cube_lut_to_rust_file, write_matrix_lut_to_rust_file}, greedy_fill::{fill_d_lut}, index_sampling::get_probability_table, pmfs::{d_gauss, d_lap}};
use tracing_forest::{util::LevelFilter, ForestLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};
use clap::Parser;

#[derive(Clone)]
enum TargetDistribution{
    L(Vec<String>),
    G(Vec<String>),
    F(Vec<String>)
}

impl TargetDistribution {
    fn new(eps: Option<Vec<String>>, sig: Option<Vec<String>>, p: Option<Vec<String>>) -> Self {
        if eps.is_some() {
            if sig.is_some() || p.is_some() {
                panic!("Both eps and sig or p are provided, please provide only one to determine the target distribution");
            }
            TargetDistribution::L(eps.unwrap())
        } else if sig.is_some() {
            if p.is_some() {
                panic!("Both sig and p are provided, please provide only one to determine the target distribution");
            }
            TargetDistribution::G(sig.unwrap())
        } else if p.is_some(){
            TargetDistribution::F(p.unwrap())
        } else {
            panic!("Either eps or sig must be provided to determine the target distribution");
        }
    }

    fn vec(&self) -> &Vec<String> {
        match self {
            TargetDistribution::L(eps) => eps,
            TargetDistribution::G(sig) => sig,
            TargetDistribution::F(p) => p,
        }
    }

    fn len(&self) -> usize {
        self.vec().len()
    }

    fn name(&self) -> &str {
        match self {
            TargetDistribution::L(_) => "Laplace with parameter P_Lap",
            TargetDistribution::G(_) => "Gaussian with Variance parameter",
            TargetDistribution::F(_) => "FDL with parameter P_FDL"
        }
    }
}



#[derive(Parser, Clone)]
struct Cli{
    #[arg(
        long, 
        help = "The log length of the vectors", 
        num_args=1..,
        value_delimiter=' ',
        default_values = ["8", "8", "8"])]
    k: Vec<usize>,

    #[arg(long, help = "The domain of the target dist, setting to 1 creates bit-wise tables", default_value_t = 255)]
    n: u16,

    #[arg(long, help = "Search stops at 2^-{error}.", default_value_t = 128)]
    error: i16,

    #[arg(long, help="If set, all up to k vec sizes will be tested (starting from 4)", default_value_t = false)]
    search: bool,

    #[arg(
        long, 
        help = "The range of number of active bits to test for each Bernoulli exponent", 
        num_args=1..,
        value_delimiter=' ',
        default_values = ["7","9"],
    )]
    l: Vec<usize>,

    #[arg(long, help = "The Bernoulli Exponent", num_args=1.., value_delimiter = ' ', default_values=["1"])]
    ber: Vec<u32>,

    #[arg(
        long, 
        help = "Epsilon value such that p = e^(-epsilon)", 
        num_args=1.., 
        value_delimiter=' ')]
    eps: Option<Vec<String>>,

    #[arg(
        long, 
        help = "Sigma value such that sigma = epsilon^2", 
        num_args=1.., 
        value_delimiter=' ')]
    sig: Option<Vec<String>>,

    #[arg(long, help="If set, runtime breakdown is recorded", default_value_t = false)]
    bench_info: bool,

    #[arg(long, help="If set, verbose debug output is activated", default_value_t = false)]
    v: bool,

    #[arg(long, help="If set, creates debug matrix vor checking", default_value_t = false)]
    debug: bool,

    #[arg(long, help="File path to store the LUT")]
    path: Option<String>,

}

fn get_p_vec_lap<const S: usize>(eps: &[String]) -> Vec<Decimal<S>>{
    let mut res = Vec::with_capacity(eps.len());
    for epsilon in eps.iter(){
        let e = -Decimal::<S>::from_str(epsilon.as_str(), Context::default()).unwrap();
        res.push(e.exp());
    }
    res
}

fn get_decimal_from_strings<const S: usize>(eps: &[String]) -> Vec<Decimal<S>> {
    eps.iter()
        .map(|epsilon| {
            Decimal::<S>::from_str(epsilon.as_str(), Context::default())
                .expect("Failed to parse variance as Decimal")
        })
        .collect()
}

fn approximate_direct<const S: usize>(cli: Cli){
    let k: Vec<usize> = cli.k.clone(); 
    let d = k.len();
    // let l = cli.l;
    let target = TargetDistribution::new(cli.eps.clone(), cli.sig.clone(), None);
    let num_targets = target.len();
    let num_sources = cli.ber.len();
    let mut p_vec = Vec::with_capacity(num_sources);
    let mut index_parameters = cli.ber.clone();
    index_parameters.sort();
    for ber in index_parameters.iter(){
        let p = Decimal::<S>::TWO.powi(-(*ber as i32));
        p_vec.push(p);
    }

    // Writes out best table if only one target is given, a path is given and matrix dimension is at least 2
    let write_out = num_targets == 1 && cli.path.is_some() && (d == 2 || d == 3);
    let embedded_type = EmbeddedType::from_n(cli.n);


    let target_p;
    
    let mut target_maps = Vec::with_capacity(num_targets);
    let mut deltas = Vec::with_capacity(num_targets);
    let mut bounds = Vec::with_capacity(num_targets);
                
    // Precomputing all target distributions
    match target{
        TargetDistribution::L(ref eps) => {
            target_p = get_p_vec_lap(&eps);
            println!("\tWith DLap Probabilies");
            for p_d in target_p.iter(){
                println!("\t\t{:?}",p_d.rescale(3).to_scientific_notation());
                let (target_map, delta, bound) = d_lap(cli.n, p_d, cli.v);
                target_maps.push(target_map);
                deltas.push(delta);
                bounds.push(bound);
                if bound > 255{
                    println!("######### requires bound > 255 #########");
                }
            }
        },
        TargetDistribution::G(ref sig) => {
            target_p = get_decimal_from_strings(&sig);
            println!("\tWith Variance Parameters");
            for p_d in target_p.iter(){
                println!("\t\t{:?}",p_d.rescale(3).to_scientific_notation());
                let (target_map, delta, bound) = d_gauss(cli.n, p_d, cli.v);
                target_maps.push(target_map);
                deltas.push(delta);
                bounds.push(bound);
                if bound > 255{
                    println!("######### requires bound > 255 #########");
                }
            }
        },
        TargetDistribution::F(_) => panic!("Currently FDL is only supported for bit-wise shares")
    }

    // Precomputing all index distributions
    // println!("\tWith index Probabilities");
    let total_k = k.iter().sum();
    assert!(total_k > 4, "k must be larger than 4");
    let start = if cli.search { 12 } else { total_k };
    


    // Execute all settings: For all target distributions, run all index distributions
    // for (((tm, delta),p), bound) in target_maps.iter().zip(deltas.iter()).zip(target_p.iter()).zip(bounds.into_iter()){
        // let mut prev_delta = Decimal::<S>::ONE;    
        // let mut prev_table = LookupTable::<u16>::new(d, k, 0u16);
        // let mut prev_ber = 0;
        // println!("\tRunning {} {} and B = {}", target.name(), p.rescale(3).to_scientific_notation(), bound);
        let mut table_parms = bounds.iter().map(|b| TableParams::new(Decimal::<S>::from_i16(-cli.error).exp2(), *b as usize)).collect::<Vec<_>>();
        for k_total_i in start..=total_k {
            table_parms = bounds.iter().map(|b| TableParams::new(Decimal::<S>::from_i16(-cli.error).exp2(), *b as usize)).collect::<Vec<_>>();
            let k_vec = if cli.search { vec![k_total_i] } else { k.clone() };
            print!("\n\t\tWith k: {} d: {} and index probabilities: ", k_total_i, k_vec.len());
            for (p, ber) in p_vec.iter().zip(index_parameters.iter()){
                print!("2^-{} ", ber);
                io::stdout().flush().unwrap();
                let mut improved = false;
                let l_set = if cli.search { (0..=k_total_i).collect() } else { cli.l.clone() };
                for &l in l_set.iter() {
                    let (index_map, index_prob) = get_probability_table(&k_vec, l, p);
                    for (index, ((tm, truncation_delta), &bound)) in target_maps.iter().zip(deltas.iter()).zip(bounds.iter()).enumerate(){
                        let (current_table, delta) = fill_d_lut(&k, bound, &index_map, &index_prob, tm, truncation_delta, cli.v);
                        if delta.is_zero(){
                            // Index distribution is too concentrated
                            continue;
                        }
                        improved |= table_parms[index].update(vec![current_table], &k_vec, l, *ber, delta, true, false);
                    }
                }
                if !improved {
                    print!("Increasing index bias has no further impact");
                    io::stdout().flush().unwrap();
                    break;
                }
            }
            print!("\n");
            if TableParams::all_set(&table_parms) {
                println!("\t\tBest Approximations for k: {:?} and target: {}", k_vec, target.name());
                for (parms, p) in table_parms.iter().zip(target_p.iter()){
                    println!("\t\t\tv {}: {:?}", p.rescale(3).to_scientific_notation(), parms);
                }
                println!("Sufficient accuracy reached at k: {:?}", k_vec);
                break;
            }
            println!("\t\tBest Approximations for k: {:?} and target: {}", k_vec, target.name());
            for (parms, p) in table_parms.iter().zip(target_p.iter()){
                println!("\t\t\tv {}: {:?}", p.rescale(3).to_scientific_notation(), parms);
            }
        }
        // println!("Best approximation = {:?}", table_parms);

        if write_out{
            let best_params = &table_parms[0];
            if d == 3 {
                // Write the 3D LUT to a file
                // print_dimensions(&current_table);
                write_cube_lut_to_rust_file(best_params.table[0].clone(), &cli.path.clone(), embedded_type, best_params.c_ber, best_params.l, &best_params.delta, best_params.bound as u16).unwrap();
            } else if d == 2 {
                // Write the 2D LUT to a file
                write_matrix_lut_to_rust_file(best_params.table[0].clone(), &cli.path.clone(), embedded_type, best_params.c_ber, best_params.l, &best_params.delta, best_params.bound as u16).unwrap();
            }
        }
    // }
}


fn main() -> Result<(), String>{
    const S: usize = 512 / 64;
    let cli = Cli::parse();
    let d = cli.k.len();
    let n = cli.n;
    let embedded_type = EmbeddedType::from_n(n);
    if cli.bench_info{
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();
        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }
    println!("Greedy LUT fill algorithm");
    
    if !cli.debug{
        println!("Generating approximations with type: {}", embedded_type.to_str());
        approximate_direct::<S>(cli.clone());
    } else {
        if d == 2{
            println!("Generating debug matrix LUT");
            let l = LookupTable::generate_pseudo_deterministic_matrix(&vec![8,8], |vec| {
                (vec[0] % 256) + 256 * (vec[1] % 256)
            });
            write_matrix_lut_to_rust_file(l, &Some(String::from("debug_table.rs")), EmbeddedType::U16, 1, 0, &Decimal::<S>::ZERO, 0xFFFF).unwrap();
        } else if d == 3{
            println!("Generating debug cube LUT");
            let l = LookupTable::generate_pseudo_deterministic_cube(&vec![5, 5, 5], |vec| {
                (vec[0] % 32) + 32 * (vec[1] % 32) + 1024 * (vec[0] % 32)
            });
            write_cube_lut_to_rust_file(l, &Some(String::from("debug_cube.rs")), EmbeddedType::U16, 1, 0, &Decimal::<S>::ZERO, 0xFFFF).unwrap();
        } else {
            return Err(String::from("Debug LUT generation only supports d=2 or d=3"));
        }
    }
    return Ok(())
}