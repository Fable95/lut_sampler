#![allow(dead_code)]
use std::{path::PathBuf, time::Duration};
use clap::{Parser, ValueEnum};

use maestro::rep3_core::network::{self, ConnectedParty};
use tracing::{span, Level};
use tracing_forest::{util::LevelFilter, ForestLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

use lut_sampler::lut_sampler::{LUTSamplerPartyCube, LUTSamplerPartyMatrix, Network, lut_sampler_benchmark, tables}; 

#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "kebab_case")] // accepted: size, table4, table3, variance, lambda, all

enum BenchType{
    Size,
    Table4,
    Table3,
    Variance,
    Lambda,
    All
}

#[derive(Parser, Clone)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    config: PathBuf,

    #[arg(long, help = "The number of parallel sampling calls to benchmark.", default_value_t = 1)]
    simd: usize,

    #[arg(long, help = "The number repetitions of the protocol execution", default_value_t = 1)]
    rep: usize,

    #[arg(long, help="If set, the protocol is run in mal_sec.", default_value_t = false)]
    mal_sec: bool,

    #[arg(long, help="If set, the protocol output is revealed, composed and checked", default_value_t = false)]
    debug: bool,

    #[arg(long, help="If set, the network cost is benchmarked.", default_value_t = false)]
    network: bool,

    #[arg(long, value_enum, help="Determines which benchmark suites are run.", default_value_t = BenchType::Table3)]
    bench: BenchType,

}

fn run_matrix<
    M: tables::Matrix<SIZE1,SIZE2,SIZE2_RED>, 
    const SIZE1: usize, 
    const SIZE2: usize, 
    const SIZE2_RED: usize
    >(network: &mut Network, cli: Cli) -> Result<(), String> {
        // let mut network = Network::setup(connected).unwrap();
        let table = &M::LUT_TABLE;
        let mut party: LUTSamplerPartyMatrix<M::GF, SIZE1, SIZE2, SIZE2_RED> = 
            LUTSamplerPartyMatrix::setup(cli.mal_sec, M::SKEW, &M::K, M::L, table);
        let span = span!(Level::INFO, "All repetitions").entered();
        for _ in 0..cli.rep{
            lut_sampler_benchmark::<
                M::GF, 
                _, 
                >(cli.simd, network, &mut party, cli.network, cli.debug);
                
        }
        span.exit();
        Ok(())
}

fn run_cube<
    C: tables::Cube<SIZE1,SIZE2,SIZE3,SIZE3_RED>, 
    const SIZE1: usize, 
    const SIZE2: usize, 
    const SIZE3: usize, 
    const SIZE3_RED: usize
    >(network: &mut Network, cli: Cli) -> Result<(), String> {
        // let mut network = Network::setup(connected).unwrap();
        let table = &C::LUT_TABLE;
        let mut party: LUTSamplerPartyCube<C::GF, SIZE1, SIZE2, SIZE3, SIZE3_RED> = 
            LUTSamplerPartyCube::setup(cli.mal_sec, C::SKEW, &C::K, C::L, table);
        let span = span!(Level::INFO, "All repetitions").entered();
        for _ in 0..cli.rep{
            lut_sampler_benchmark::<
                C::GF, 
                _, 
                >(cli.simd, network, &mut party, cli.network, cli.debug);
        }
        span.exit();
        Ok(())
}


fn print_name(lambda: usize, k: usize, d: usize, epsilon: f64){
    println!("\n---------------------------------------------------------------------------------");
    println!(
        "Running benchmark with {} dimensions, total k of {}, with maximum SD = λ = {}, and ε: {}",
        d, k, lambda, epsilon);
}

// Benchmark tables with c = 12 as a worst case index sampling (all tables approximate sigma = 1)
fn benchmark_sizes(network: &mut Network, cli: Cli) -> Result<(), String> {
    println!("Benchmarking Size suite going from k = 12 to k = 24");
    print_name(40, 12, 2, 1.0);
    run_matrix::<tables::K12MatBench,_,_,_>(network, cli.clone())?;
    print_name(40, 14, 3, 1.0);
    run_cube::<tables::K14CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 16, 3, 1.0);
    run_cube::<tables::k16_bench::K16CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 18, 3, 1.0);
    run_cube::<tables::K18CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 20, 3, 1.0);
    run_cube::<tables::K20CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 22, 3, 1.0);
    run_cube::<tables::K22CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 24, 3, 1.0);
    run_cube::<tables::K24CubeBench,_,_,_,_>(network, cli.clone())?;
    Ok(())
}

// Approximations for (Table 3)
fn benchmark_table3(network: &mut Network, cli: Cli) -> Result<(), String> {
    println!("Benchmarking Data for table 3");
    print_name(128, 24, 3, 1.0);
    run_cube::<tables::K24CubeBench,_,_,_,_>(network, cli.clone())?;
    Ok(())
}

// Approximations for (Table 4)
fn benchmark_table4(network: &mut Network, cli: Cli) -> Result<(), String> {
    println!("Benchmarking Data for table 4");
    print_name(40, 14, 3, 1.0);
    run_cube::<tables::K14Sig1Cube,_,_,_,_>(network, cli.clone())?;
    Ok(())
}
// Different Gauss approximations with varying variance: (Figure 8)
fn benchmark_variances(network: &mut Network, cli: Cli) -> Result<(), String> {
    println!("Benchmarking Variance suite going from ε = 10 to ε = 0.001");
    print_name(40, 10, 2, 10.0);
    run_matrix::<tables::K10Sig01Mat,_,_,_>(network, cli.clone())?;
    print_name(40, 14, 3, 1.0);
    run_cube::<tables::K14Sig1Cube,_,_,_,_>(network, cli.clone())?;
    print_name(40, 15, 3, 0.1);
    run_cube::<tables::K15Sig10Cube,_,_,_,_>(network, cli.clone())?;
    print_name(40, 19, 3, 0.01);
    run_cube::<tables::K19Sig100Cube,_,_,_,_>(network, cli.clone())?;
    print_name(40, 22, 3, 0.001);
    run_cube::<tables::K22Sig1000Cube,_,_,_,_>(network, cli.clone())?;
    println!("Benchmarking Laplace suite going from ε = 10 to ε = 0.01");
    print_name(40, 12, 3, 10.0);
    run_matrix::<tables::K12MatBench,_,_,_>(network, cli.clone())?;
    print_name(40, 14, 3, 1.0);
    run_cube::<tables::K14CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 16, 3, 0.1);
    run_cube::<tables::k16_bench::K16CubeBench,_,_,_,_>(network, cli.clone())?;
    print_name(40, 20, 3, 0.01);
    run_cube::<tables::K20CubeBench,_,_,_,_>(network, cli.clone())?;
    Ok(())
}

// Different Gauss Accuracies for sig = 1 (Figure 7)
fn benchmark_accuracies(network: &mut Network, cli: Cli) -> Result<(), String> {
    println!("Benchmarking Accuracy suite going from λ = 40 to λ = 128");
    println!("Starting with Gauss");
    print_name(40, 12, 2, 1.0);
    run_matrix::<tables::K12Lam40Mat,_,_,_>(network, cli.clone())?;
    print_name(80, 17, 3, 1.0);
    run_cube::<tables::K17Lam80Cube,_,_,_,_>(network, cli.clone())?;
    print_name(128, 22, 3, 1.0);
    run_cube::<tables::K22Lam128Cube,_,_,_,_>(network, cli.clone())?;
    println!("Starting with Laplace");
    print_name(40, 14, 3, 1.0);
    run_cube::<tables::K14Lam40CubeLap,_,_,_,_>(network, cli.clone())?;
    print_name(80, 18, 3, 1.0);
    run_cube::<tables::K18Lam80CubeLap,_,_,_,_>(network, cli.clone())?;
    print_name(128, 23, 3, 1.0);
    run_cube::<tables::K23Lam128CubeLap,_,_,_,_>(network, cli.clone())?;
    Ok(())
}

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
            

    let repetitions = if cli.network { 1 } else { cli.rep };
    let simd = cli.simd;
    
    let span = span!(Level::INFO, "Setup Connections").entered();
    let connected = ConnectedParty::bind_and_connect(
        party_index, 
        config.clone(), 
        Some(Duration::from_secs(60))
    ).unwrap();
    span.exit();

    let mut network = Network::setup(connected).unwrap();

    match cli.bench {
        BenchType::Size => {
            benchmark_sizes(&mut network, cli.clone())?;
        },
        BenchType::Table4 => {
            benchmark_table4(&mut network, cli.clone())?;
        },
        BenchType::Table3 => {
            benchmark_table3(&mut network, cli.clone())?;
        },
        BenchType::Variance => {
            benchmark_variances(&mut network, cli.clone())?;
        },
        BenchType::Lambda => {
            benchmark_accuracies(&mut network, cli.clone())?;
        },
        BenchType::All => {
            // Table3 and Table4 settings are covered by the other tests
            benchmark_sizes(&mut network, cli.clone())?;
            benchmark_variances(&mut network, cli.clone())?;
            benchmark_accuracies(&mut network, cli.clone())?;

        },
    }

    network.teardown().unwrap();

    println!("All repetitions ({} = simd {} * rep {}) finished.", 
        repetitions * simd, 
        simd, 
        repetitions
    );

    return Ok(())
}