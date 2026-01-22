use crate::share::gf_template::{GFTTrait, Share};

// Benchmark tables with c = 12 as a worst case index sampling (all tables approximate sigma = 1)
pub mod k12_mat_bench;
pub mod k14_bench;
pub mod k16_bench;
pub mod k18_bench;
pub mod k20_bench;
pub mod k22_bench;
pub mod k24_bench;


// Approximations for (Table 4)
pub mod k14_sig1;

// Different Gauss approximations with varying variance: (Figure 8)
// sig = 0.1
pub mod k10_sig01_lam_40;
// sig = 1
// Included below
// sig = 10
pub mod k15_sig10_lam_40;
// sig = 100
pub mod k19_sig100_lam_40;
// sig = 1000
pub mod k22_sig1000_lam_40;



// Different Gauss Accuracies for sig = 1 (Figure 7)
// 40: k = 12 (Matrix)
pub mod k12_sig1_lam_40_mat;
pub mod k14_eps1_lam_40;
// 80: k =
pub mod k17_sig1_lam_80;
pub mod k18_eps1_lam_80;
// 128: k = 
pub mod k22_sig1_lam_128;
pub mod k23_eps1_lam128;

// Benchmark tables with c = 12 as a worst case index sampling (all tables approximate sigma = 1)
pub struct K12MatBench;
pub struct K14CubeBench;
// pub struct K16CubeBench;
pub struct K18CubeBench;
pub struct K20CubeBench;
pub struct K22CubeBench;
pub struct K23CubeBench;
pub struct K24CubeBench;

// Different Gauss approximations with varying variance: (all are Figure 8 and K14 is Table 4)
pub struct K10Sig01Mat;
pub struct K14Sig1Cube;
pub struct K15Sig10Cube;
pub struct K19Sig100Cube;
pub struct K22Sig1000Cube;

// Different Gauss Accuracies for sig = 1 (Figure 7)
pub struct K12Lam40Mat;
pub struct K17Lam80Cube;
pub struct K22Lam128Cube;
pub struct K14Lam40CubeLap;
pub struct K18Lam80CubeLap;
pub struct K23Lam128CubeLap;

pub trait Matrix<const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize> {
    // type Wrapper: Share;
    // type Embedded: Share;
    const RATIO: usize;
    type GF: GFTTrait;
    const K: [usize; 2];
    const SKEW: usize;
    const L: usize;
    const D: usize;
    const SIZE1: usize = SIZE1;
    const SIZE2: usize = SIZE2;
    const SIZE2_RED: usize = SIZE2_RED;
    const LUT_TABLE: [[<<Self::GF as GFTTrait>::Wrapper as Share>::InnerType; SIZE2_RED]; SIZE1];
}

pub trait Cube<const SIZE1: usize, const SIZE2: usize, const SIZE3: usize, const SIZE3_RED: usize> {
    // type Wrapper: Share;
    // type Embedded: Share;
    const RATIO: usize;
    type GF: GFTTrait;
    const K: [usize; 3];
    const SKEW: usize;
    const L: usize;
    const D: usize;
    const SIZE1: usize = SIZE1;
    const SIZE2: usize = SIZE2;
    const SIZE3: usize = SIZE3;
    const SIZE3_RED: usize = SIZE3_RED;
    const LUT_TABLE: [[[<<Self::GF as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2]; SIZE1];
}