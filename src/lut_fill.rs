#![allow(long_running_const_eval)]
use fastnum::{decimal::{Context, Decimal, UnsignedDecimal}, udec1024, udec256, udec512, UD1024, UD2048, UD256, UD4096, UD512, UD8192};
use lut_sampler::{lut_sampler::IndexSampling, table_fill::{export::{print_dimensions, write_lut_to_rust_file}, greedy_fill::fill_d_lut, LookupTable}};
use tracing_forest::{util::LevelFilter, ForestLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};
use clap::Parser;


#[derive(Parser)]
struct Cli{
    #[arg(long, help = "The log length of the vectors", default_value_t = 8)]
    k: usize,

    #[arg(
        long, 
        help = "The sampling strategies, either of s, b, u, for skewed (Bernoulli Bits), binomial or uniform", 
        num_args=1..,
        value_delimiter=' ',
        default_values = ["s","s","s"],
    )]
    samplings: Vec<String>,

    #[arg(long, help = "The Bernoulli Exponent", default_value_t = 1)]
    ber: u32,

    #[arg(
        long, 
        help = "Epsilon value such that p = e^epsilon", 
        num_args=1.., 
        value_delimiter=' ',
        default_values=["0.1","1","5"])]
    eps: Vec<String>,

    #[arg(long, help="If set, runtime breakdown is recorded", default_value_t = false)]
    bench_info: bool,

    #[arg(long, help="If set, verbose debug output is activated", default_value_t = false)]
    v: bool,

    #[arg(long, help="If set, creates debug matrix vor checking", default_value_t = false)]
    debug: bool,

    #[arg(long, help="File path to store the LUT")]
    path: Option<String>,

}

fn get_p_vec<const S: usize>(eps: Vec<String>) -> Vec<UnsignedDecimal<S>>{
    let base = UnsignedDecimal::<S>::E;
    let mut res = Vec::with_capacity(eps.len());
    for epsilon in eps.iter(){
        let e = Decimal::<S>::from_str(&epsilon, Context::default()).unwrap();
        res.push(base.pow(-e));
    }
    res

}

fn main() -> Result<(), String>{
    let cli = Cli::parse();

    if cli.bench_info{
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();
        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }

    let k: usize = cli.k; 
    let d: usize = cli.samplings.len();
    let ber = -(cli.ber as i32);
    let p = UD8192::TWO.powi(ber);
    let p_vec = vec![p ; d];
    // let index_samplings =  vec![IndexSampling::Biased; d]; 
    let index_samplings =  IndexSampling::get_sample_vec(cli.samplings); 
    
    let num_iter = cli.eps.len();

    let d_lap_p = get_p_vec(cli.eps);

    
    println!("Greedy LUT fill algorithm");
    // println!("\tWith Sampling");
    // println!("\t\t{:?}",index_samplings);
    // println!("\tWith vector lenght:");
    // println!("\t\t{:?}",1<<k);
    // println!("\tWith Dimensions");
    // println!("\t\t{:?}",d);
    println!("\tWith index Probability");
    println!("\t\t{:?}",p.to_scientific_notation());
    println!("\tWith DLap Probabilies");
    for p_d in d_lap_p.iter(){
        println!("\t\t{:?}",p_d.rescale(3).to_scientific_notation());
    }
    if !cli.debug{
        for p in d_lap_p{
            let (current_table, _) = fill_d_lut(k, &index_samplings, &p_vec, p, cli.v);

            if num_iter == 1{
                if let Some(path) = cli.path.as_deref(){
                    print_dimensions(&current_table);
                    write_lut_to_rust_file(current_table, path).unwrap();
                }
            }
        }
    } else {
        let l = LookupTable::generate_pseudo_deterministic(3, 8, |row, col, lay| {
            (row % 8) + 8 * (col % 8) + 64 * (lay % 4)
        });
        print_dimensions(&l);
        write_lut_to_rust_file(l, "debug_table.rs").unwrap();
    }
    return Ok(())
}