#![allow(dead_code)]
use std::{path::PathBuf, time::Duration};
use clap::Parser;
use itertools::Itertools;

use lut_sampler::{
    share::gf_template::GFT, 
    lut_sampler::{debug_table::DEBUG_TABLE, lut_8192_2_2, lut_8192_3_2, lut_8192_4_2, lut_8192_6_2, lut_sampler_benchmark, IndexSampling}
}; 
use maestro::rep3_core::network::{self, ConnectedParty};
use tracing_forest::{util::LevelFilter, ForestLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};


#[derive(Parser)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    config: PathBuf,

    #[arg(
        long,
        value_name = "N_THREADS",
        help = "The number of worker threads. Set to 0 to indicate the number of cores on the machine. Optional, default single-threaded"
    )]
    threads: Option<usize>,

    #[arg(long, help = "The number of parallel sampling calls to benchmark. You can pass multiple values.", num_args = 1.., default_values=["1"])]
    simd: Vec<usize>,

    #[arg(long, help = "The number repetitions of the protocol execution", default_value_t = 1)]
    rep: usize,

    #[arg(long, help = "The Hypercube dimension for LUT sampling", default_value_t = 3)]
    dim: usize,

    #[arg(long, help = "The bit length of the LUT", default_value_t = 8)]
    k: usize,

    #[arg(long, help = "Path to write benchmark result data as CSV. Default: result.csv", default_value = "result.csv")]
    csv: PathBuf,
    
    #[arg(long, help="If set, benchmark all protocol variants and ignore specified targets.", default_value_t = false)]
    all: bool,

    #[arg(long, help="If set, the protocol is run in mal_sec.", default_value_t = false)]
    mal_sec: bool,

    #[arg(value_enum, help="List of sampling strategies for the index distribution", num_args = 1.., default_values=["s", "s", "s"])]
    indexdist: Vec<String>,

    #[arg(long, help="Skewing parameter for the Bernoulli bits in the 's' distribution", default_value_t = 2)]
    skew: usize,
}

type GFT128_8 = GFT<u128, u8, 16>;
type GFT64_8 = GFT<u64, u8, 8>;
const SIZE: usize = 256;
const SIZE_RED_128: usize = 16;
const SIZE_RED_64: usize = 32;


fn main() -> Result<(), String> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();


    let cli = Cli::parse();

    let (party_index, config) = network::Config::from_file(&cli.config).unwrap();

    if !cli.simd.iter().all_unique() {
        return Err(format!("Duplicate simd values in argument {:?}", cli.simd));
    }

    let mut skew = cli.skew;

    let selected_table= match cli.skew {
        2 => &lut_8192_2_2::LUT_TABLE,
        3 => &lut_8192_3_2::LUT_TABLE,
        4 => &lut_8192_4_2::LUT_TABLE,
        6 => &lut_8192_6_2::LUT_TABLE,
        _ => {
            println!("Invalid skew value. Using debug table.");
            skew = 2;
            &DEBUG_TABLE
        }
    };

    let index_samplings = IndexSampling::get_sample_vec(cli.indexdist);

    for i in 0..cli.rep{
        println!("Iteration {}",i);
        
        for simd in cli.simd.iter(){
            let connected = ConnectedParty::bind_and_connect(
                party_index, 
                config.clone(), 
                Some(Duration::from_secs(60))
            ).unwrap();
            
            
            lut_sampler_benchmark::<GFT64_8, SIZE, SIZE_RED_64>(connected, *simd, &index_samplings, skew, cli.mal_sec, cli.threads, selected_table);
        }
        
    }

    return Ok(())
}