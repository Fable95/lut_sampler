use offline::{compute_biased_offsets, compute_ohv_vectors_cube};//, compute_ohv_vectors};
use online::open_rss_many;
use clap::ValueEnum;
use maestro::{
        chida::ChidaParty, rep3_core::{
        network::{ConnectedParty}, 
        party::{CombinedCommStats, broadcast::BroadcastContext, error::MpcResult}, 
        share::RssShareVec
    }, share::Field, util::ArithmeticBlackBox
};
use crate::{
    lut_sampler::{mult_verification::TripleVector, offline::{compute_ohv_vectors_matrix, extract_byte_from_matrix}, ohv_container::{CubeOhv, MatrixOhv}, online::{sample_many_cube, sample_many_matrix}}, share::{gf_template::{GFTTrait, Share, ShareType}, gf2p64::GF2p64, helper_types::AllowedTypes}, util::mul_triple_vec::{GF2p64DotEncoder, GF2p64Encoder, MulTripleVector, NoMulTripleRecording}
};
use tracing::{instrument, span, Level};
pub mod offline;
pub mod online;
pub mod tables;
pub mod mult_verification;
pub mod ohv_container;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Hash)]
pub enum IndexSampling {
    Uniform,
    Biased
}

impl IndexSampling{
    pub fn from_literal(str: String) -> Self {
        match str {
            val if val == "u".to_owned() => Self::Uniform,
            val if val == "s".to_owned() => Self::Biased,
            _ => panic!("undefined literal, please select: s = skewed (Bernoulli bits), u = uniform"),
        }
    }

    pub fn get_sample_vec(in_vec: Vec<String>) -> Vec<Self>{
        in_vec.into_iter()
            .map(Self::from_literal)
            .collect()
    }
}

pub trait LUTSamplerParty {
    fn rotate_ohvs(&mut self, network: &mut Network) -> MpcResult<()>;
    fn sample_ohvs(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<()>;
    
    #[instrument(name = "Run index sampling", level = "trace", skip_all)]
    fn sample_indices(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<()>{
        let mal_sec = self.mal_sec();
        let skew = self.skew();
        let k: Vec<usize> = self.k().iter().map(|v| *v).collect();
        let l = self.l();
        let offsets = if self.mal_sec(){ 
            compute_biased_offsets(network, self.prep_check_vec(), n_samples, mal_sec, skew, &k, l)?
        } else{
            compute_biased_offsets(network, &mut NoMulTripleRecording, n_samples, mal_sec, skew, &k, l)?
        };
        *self.prep_offsets() = offsets;
        Ok(())
    }

     #[instrument(name = "Do preprocessing", level = "trace", skip_all)]
    fn do_preprocessing(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<()>{
        self.sample_ohvs(n_samples, network)?;
        self.sample_indices(n_samples, network)?;
        self.rotate_ohvs(network)?;
        Ok(())
    }

    fn verify_triples(&mut self, network: &mut Network) -> MpcResult<bool>  {
        if !self.mal_sec(){
            return Ok(true)
        }
        let d = self.k().len();
        let size = self.sizes();
        let mut prep = self.prep_check_vec().clone();
        let dot_products = self.dot_check_vec();
        let mut d0 = dot_products[0].clone();
        let dot1 = &mut GF2p64DotEncoder::new(size[1], d0.get_mut_triple_vector());
        
        if d == 2{
            return mult_verification::verify_multiplication_triples(
                &mut network.chida.as_party_mut(), 
                &mut network.broadcast_context, 
                &mut[
                    &mut GF2p64Encoder(&mut prep), 
                    dot1,
                    ], 
                false
            )
        } else {
            let dot2 = &mut GF2p64DotEncoder::new(size[2], dot_products[1].get_mut_triple_vector());
            return mult_verification::verify_multiplication_triples(
                &mut network.chida.as_party_mut(), 
                &mut network.broadcast_context, 
                &mut[
                    &mut GF2p64Encoder(&mut prep), 
                    dot1,
                    dot2
                    ], 
                false
            )
        }    
    }

    fn sample_with_lut(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<RssShareVec<<Self::T as GFTTrait>::Embedded>>;
    fn get_coordinates(&mut self, index: usize, network: &mut Network) -> MpcResult<Vec<ShareType<Self::IndexType>>>;
    fn open_many(&mut self, shares: &RssShareVec<<Self::T as GFTTrait>::Embedded>, network: &mut Network) -> MpcResult<Vec<<Self::T as GFTTrait>::Embedded>>{
        open_rss_many::<<Self::T as GFTTrait>::Embedded>(network.chida.as_party_mut(), &mut network.broadcast_context, shares)
    }
    fn compare_result(&self, indices: Vec<ShareType<Self::IndexType>>, value: <Self::T as GFTTrait>::Embedded);
    fn print_ohvs(&self);
    fn mal_sec(&self) -> bool;
    fn dot_check_vec(&mut self) -> &mut [TripleVector];
    fn prep_check_vec(&mut self) -> &mut MulTripleVector<GF2p64>;
    fn prep_offsets(&mut self) -> &mut Vec<RssShareVec<ShareType<Self::IndexType>>>;
    fn clear_self(&mut self);
    fn skew(&self) -> usize;
    fn k(&self) -> &[usize];
    fn l(&self) -> usize;
    fn dim(&self) -> usize;
    fn sizes(&self) -> Vec<usize> {
        self.k().iter().map(|ki| 1 << ki).collect()
    }
    // const D: usize;
    type T: GFTTrait;
    type IndexType: AllowedTypes;
}

pub struct Network{
    pub chida: ChidaParty,
    pub broadcast_context: BroadcastContext,
}

impl Network{
    pub fn setup(connected: ConnectedParty) -> MpcResult<Self>{
        let chida = ChidaParty::setup(connected, None, None)?;
        let broadcast_context = BroadcastContext::new();
        Ok(Self { chida, broadcast_context })
    }
    pub fn party_index(&self) -> usize {
        self.chida.party_index()
    }
    pub fn reset_comm_stats<T: Field>(&mut self) -> CombinedCommStats {
        <ChidaParty as ArithmeticBlackBox<T>>::io(&self.chida).reset_comm_stats()
    }
    pub fn teardown(&mut self) -> MpcResult<()> {
        self.chida.as_party_mut().teardown()
    }
}

pub struct LUTSamplerPartyCube<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE3: usize, const SIZE3_RED: usize>{
    lut_table: &'a [[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2];SIZE1],
    mal_sec: bool,
    dot_check_vec: [TripleVector; 2],
    prep_check_vec: MulTripleVector<GF2p64>,
    prep_cubes: Vec<CubeOhv<T, <Self as LUTSamplerParty>::IndexType, SIZE1, SIZE2, SIZE3>>,
    prep_offsets: Vec<RssShareVec<ShareType<<Self as LUTSamplerParty>::IndexType>>>,
    skew: usize, 
    l: usize,
    k: [usize; 3],
}

pub struct LUTSamplerPartyMatrix<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize>{
    lut_table: &'a [[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE2_RED]; SIZE1],
    mal_sec: bool,
    dot_check_vec: [TripleVector;1],
    prep_check_vec: MulTripleVector<GF2p64>,
    prep_matrices: Vec<MatrixOhv<T, <Self as LUTSamplerParty>::IndexType, SIZE1, SIZE2>>,
    prep_offsets: Vec<RssShareVec<ShareType<<Self as LUTSamplerParty>::IndexType>>>,
    skew: usize,
    l: usize,
    k: [usize; 2],
}

impl<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE3: usize, const SIZE3_RED: usize> LUTSamplerPartyCube<'a, T, SIZE1, SIZE2, SIZE3,SIZE3_RED>{
    pub fn setup(mal_sec: bool, skew: usize, k: &[usize], l: usize, lut_table: &'a [[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2];SIZE1]) -> Self{
        debug_assert!(k.len() == 3, "not enough k parameters for cube setting");
        debug_assert!(1 << k[0] == SIZE1, "k must be such that 2^K == SIZE");
        debug_assert!(1 << k[1] == SIZE2, "k must be such that 2^K == SIZE");
        debug_assert!(1 << k[2]== SIZE3, "k must be such that 2^K == SIZE");
        debug_assert!((1 << k[2]) / T::RATIO == SIZE3_RED, "k must be such that 2^K == SIZE");
        Self { 
            lut_table,
            mal_sec,
            dot_check_vec: [TripleVector::new(mal_sec), TripleVector::new(mal_sec)],
            prep_check_vec: MulTripleVector::<GF2p64>::new(),
            prep_cubes: Vec::new(), 
            prep_offsets: Vec::new(),
            skew,
            k: [k[0], k[1], k[2]],
            l
        }
    }
}

impl<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize> LUTSamplerPartyMatrix<'a, T, SIZE1, SIZE2,SIZE2_RED>{
    pub fn setup(mal_sec: bool, skew: usize, k: &[usize], l: usize, lut_table: &'a [[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE2_RED]; SIZE1]) -> Self{
        debug_assert!(k.len() == 2 , "need two k values for matrix setting.");
        debug_assert!(1 << k[0] == SIZE1, "k must be such that 2^K == SIZE");
        debug_assert!(1 << k[1] == SIZE2, "k must be such that 2^K == SIZE");
        debug_assert!((1 << k[1]) / T::RATIO == SIZE2_RED, "k must be such that 2^K == SIZE");
        Self { 
            lut_table,
            mal_sec,
            dot_check_vec: [TripleVector::new(mal_sec)],
            prep_check_vec: MulTripleVector::<GF2p64>::new(),
            prep_matrices: Vec::new(), 
            prep_offsets: Vec::new(),
            skew,
            k: [k[0], k[1]],
            l
        }
    }
}

impl<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize> LUTSamplerParty for LUTSamplerPartyMatrix<'a, T, SIZE1, SIZE2, SIZE2_RED>{
    // const D: usize = 2;

    fn clear_self(&mut self) {
        self.prep_check_vec.clear();
        self.prep_matrices.clear();
        self.prep_offsets.clear();
    }

    #[instrument(name = "Rotate one-hot vectors", level = "trace", skip_all)]
    fn rotate_ohvs(&mut self, network: &mut Network) -> MpcResult<()> {
        debug_assert_eq!(self.prep_matrices.len()*self.dim(),self.prep_offsets.len()*self.prep_offsets[0].len());
        let mut offsets_public = Vec::with_capacity(self.prep_offsets.len());
        for (offsets, matrix) in 
            self.prep_offsets.iter()
            .zip(self.prep_matrices.iter_mut())
        {
            offsets_public.append(&mut matrix.compute_offset(offsets));
        }

        let offsets_p = open_rss_many::<ShareType::<Self::IndexType>>(network.chida.as_party_mut(), &mut network.broadcast_context, &offsets_public)?;
        for (offsets, matrix) in offsets_p
            .chunks(self.dim())
            .zip(self.prep_matrices.iter_mut()){
            matrix.rotate(offsets);
        }
        Ok(())
    }

    #[instrument(name = "Generate one-hot vectors for matrix", level = "trace", skip_all)]
    fn sample_ohvs(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<()> {
        if self.mal_sec {
            self.prep_matrices = compute_ohv_vectors_matrix::<_,T,Self::IndexType,SIZE1,SIZE2>(network.chida.as_party_mut(), &mut self.prep_check_vec, n_samples, &self.k)?;
        } else {
            self.prep_matrices = compute_ohv_vectors_matrix::<_,T,Self::IndexType,SIZE1,SIZE2>(network.chida.as_party_mut(), &mut NoMulTripleRecording, n_samples, &self.k)?;
        }
        Ok(())
    }


    #[instrument(name = "Sigma LUT sampling", level = "trace", skip_all)]
    fn sample_with_lut(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<RssShareVec<<T as GFTTrait>::Embedded>> {
        debug_assert_eq!(self.prep_matrices.len(), n_samples);
        sample_many_matrix(network.chida.as_party_mut(), &self.prep_matrices, &self.lut_table, &mut self.dot_check_vec[0],  n_samples)
    }

    fn get_coordinates(&mut self, index: usize, network: &mut Network) -> MpcResult<Vec<ShareType<Self::IndexType>>> {
        assert!(index < self.prep_matrices.len());
        self.prep_matrices[index].get_coordinates(network.chida.as_party_mut(), &mut network.broadcast_context)
    }
    
    fn print_ohvs(&self) {
        for (i, ohv_mat) in self.prep_matrices.iter().enumerate() {
            println!("Matrix Ohv {}:", i);
            ohv_mat.print();
        }
    }

    fn compare_result(&self, indices: Vec<ShareType<Self::IndexType>>, value: <T as GFTTrait>::Embedded) {
        let indices_usize: Vec<usize> = indices.iter().map(|x| x.inner().to_usize()).collect();
        let expected = extract_byte_from_matrix::<T, SIZE1, SIZE2_RED>(&indices_usize, self.lut_table);
        if expected != value {
            // assert!(expected == value, "Extract pos [{},{},{}]:\t expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
            println!("Extract pos {:?}: expected: {:?}, got: {:?}", indices_usize, expected, value);
        }
        // assert!(expected.to_u8() == value.to_u8(), "Extract pos [{},{},{}]:\t expected: {}, got: {}", row, col, lay, expected.to_u8(), value.to_u8());
    }
    
    fn mal_sec(&self) -> bool {
        self.mal_sec
    }
    
    fn dot_check_vec(&mut self) -> &mut [TripleVector] {
        &mut self.dot_check_vec
    }
    
    fn prep_check_vec(&mut self) -> &mut MulTripleVector<GF2p64> {
        &mut self.prep_check_vec
    }
    
    fn prep_offsets(&mut self) -> &mut Vec<RssShareVec<ShareType<Self::IndexType>>> {
        &mut self.prep_offsets
    }
    
    fn skew(&self) -> usize {
        self.skew
    }
    
    fn k(&self) -> &[usize] {
        &self.k
    }
    
    fn l(&self) -> usize {
        self.l
    }
    
    type T = T;
    
    fn dim(&self) -> usize {
        2
    }
    
    type IndexType = u16;
}

impl<
    'a, 
    T: GFTTrait, 
    const SIZE1: usize, 
    const SIZE2: usize, 
    const SIZE3: usize, 
    const SIZE3_RED: usize
> LUTSamplerParty for LUTSamplerPartyCube<'a, T, SIZE1, SIZE2, SIZE3, SIZE3_RED>{

    // const D: usize = 3;
    fn clear_self(&mut self) {
        self.prep_check_vec.clear();
        self.prep_cubes.clear();
        self.prep_offsets.clear();
    }

    #[instrument(name = "Rotate one-hot vectors", level = "trace", skip_all)]
    fn rotate_ohvs(&mut self, network: &mut Network) -> MpcResult<()>{
        debug_assert_eq!(self.prep_cubes.len()*self.dim(),self.prep_offsets.len()*self.prep_offsets[0].len());
        let mut offsets_public = Vec::with_capacity(self.prep_offsets.len());
        for (offsets, cube) in 
            self.prep_offsets.iter()
            .zip(self.prep_cubes.iter_mut())
        {
            offsets_public.append(&mut cube.compute_offset(offsets));
        }

        let offsets_p = open_rss_many::<ShareType<Self::IndexType>>(network.chida.as_party_mut(), &mut network.broadcast_context, &offsets_public)?;
        for (offsets, cube) in offsets_p
            .chunks(self.dim())
            .zip(self.prep_cubes.iter_mut()){
            cube.rotate(offsets);
        }
        Ok(())
    }
    #[instrument(name = "Generate one-hot vectors for cubes", level = "trace", skip_all)]
    fn sample_ohvs(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<()>{
        if self.mal_sec {
            self.prep_cubes = compute_ohv_vectors_cube::<_,T,Self::IndexType,SIZE1,SIZE2,SIZE3>(network.chida.as_party_mut(), &mut self.prep_check_vec, n_samples, &self.k)?;
        } else {
            self.prep_cubes = compute_ohv_vectors_cube::<_,T,Self::IndexType,SIZE1,SIZE2,SIZE3>(network.chida.as_party_mut(), &mut NoMulTripleRecording, n_samples, &self.k)?;
        }
        Ok(())
    }

    #[instrument(name = "Sigma LUT sampling", level = "trace", skip_all)]
    fn sample_with_lut(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<RssShareVec<T::Embedded>>{
        debug_assert_eq!(self.prep_cubes.len(), n_samples);
        sample_many_cube(network.chida.as_party_mut(), &self.prep_cubes, self.lut_table, &mut self.dot_check_vec,  n_samples)
    }

    fn get_coordinates(&mut self, index: usize, network: &mut Network) -> MpcResult<Vec<ShareType<Self::IndexType>>>{
        assert!(index < self.prep_cubes.len());
        self.prep_cubes[index].get_coordinates(network.chida.as_party_mut(), &mut network.broadcast_context)
    }

    fn print_ohvs(&self) {
        for (i, ohv_cube) in self.prep_cubes.iter().enumerate() {
            println!("Cube Ohv {}:", i);
            ohv_cube.print();
        }
    }
    
    fn compare_result(&self, indices: Vec<ShareType<Self::IndexType>>, value: <T as GFTTrait>::Embedded) {
        offline::compare_result::<T, Self::IndexType, SIZE2, SIZE3_RED>(indices, value, self.dim(), self.lut_table);
    }

        fn mal_sec(&self) -> bool {
        self.mal_sec
    }
    
    fn dot_check_vec(&mut self) -> &mut [TripleVector] {
        &mut self.dot_check_vec
    }
    
    fn prep_check_vec(&mut self) -> &mut MulTripleVector<GF2p64> {
        &mut self.prep_check_vec
    }
    
    fn prep_offsets(&mut self) -> &mut Vec<RssShareVec<ShareType<Self::IndexType>>> {
        &mut self.prep_offsets
    }
    
    fn skew(&self) -> usize {
        self.skew
    }
    
    fn k(&self) -> &[usize] {
        &self.k
    }
    
    fn l(&self) -> usize {
        self.l
    }

    type T = T;
    
    fn dim(&self) -> usize {
        3
    }
    
    type IndexType = u16;
}


/// This function implements the LUT sampling benchmark.
///
/// The arguments are
/// - `connected` - the local party
/// - `simd` - number of parallel samples calls
/// - `n_worker_threads` - number of worker threads
#[instrument(name = "Run LUT benchmark", level = "trace", skip_all)]
pub fn lut_sampler_benchmark<
T: GFTTrait,
P: LUTSamplerParty,
>(
    simd: usize, 
    net: &mut Network,
    party: &mut P,
    network: bool,
    debug: bool
) {
    let span_tot = span!(Level::TRACE, "Total runtime").entered();
    let span = span!(Level::TRACE, "Preprocessing").entered();
    // println!("preprocessing");
    //let total_semi = net.
    party.sample_ohvs(simd, net).unwrap();
    let ohv_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    party.sample_indices(simd, net).unwrap();
    let index_sampling_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    party.rotate_ohvs(net).unwrap();
    let rotation_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    span.exit();

    // println!("Sample LUT");
    let _output = party.sample_with_lut(simd, net).unwrap();
    let online_comm_stats = net.reset_comm_stats::<T::Wrapper>();

    // println!("Verifying mult triples products");
    let span_verify = span!(Level::TRACE, "Verifying dot products").entered();
    let valid = party.verify_triples(net).unwrap();
    span_verify.exit();
    let verify_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    if !valid{
        panic!("Error Triple Verification failed!");
    }
    span_tot.exit();

    if debug{
        let span_end = span!(Level::TRACE, "Checking and Teardown").entered();
        println!("Checking the result with the cube coordinates");
        let samples = party.open_many(&_output, net).unwrap();
        for i in 0..samples.len(){
            let coordinates = party.get_coordinates(i, net).unwrap();
            party.compare_result(coordinates, samples[i]);
        }
        println!("Samples {:?}", samples.iter().map(|x| x.inner()).collect::<Vec<_>>());
        span_end.exit();
    }

    
    if network {
        println!("Finished benchmark");
        println!("One-hot Vectors:");
        ohv_comm_stats.print_comm_statistics(net.party_index());
        println!("\nIndex Sampling:");
        index_sampling_comm_stats.print_comm_statistics(net.party_index());
        println!("\nRotations:");
        rotation_comm_stats.print_comm_statistics(net.party_index());
        println!("\nOnline Phase:");
        online_comm_stats.print_comm_statistics(net.party_index());
        println!("\nVerify Triples:");
        verify_comm_stats.print_comm_statistics(net.party_index());
        // party.main_party_mut().print_statistics();   
    }
}
