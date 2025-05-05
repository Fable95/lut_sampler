use std::marker::PhantomData;

use offline::{compare_result, compute_biased_offsets, compute_binomial_offsets, compute_ohv_vectors};//, compute_ohv_vectors};
use online::{open_rss_many, sample_many};
use clap::ValueEnum;
use maestro::{
        chida::ChidaParty, rep3_core::{
        network::{task::IoLayerOwned, ConnectedParty, NetSerializable}, 
        party::{broadcast::BroadcastContext, error::MpcResult, MainParty}, 
        share::{RssShare, RssShareVec}
    }, util::ArithmeticBlackBox
};
use crate::{
    share::{gf2p64::{GF2p64, GF2p64Subfield}, gf_template::{GFTTrait, Share}},
    util::mul_triple_vec::{DotProdRecorder, DotProdVector, GF2p64DotEncoder, GF2p64Encoder, MulTripleVector, NoMulTripleRecording}
};
use tracing::{instrument, span, Level};
pub mod offline;
pub mod online;
pub mod lut_8192_2_2;
pub mod lut_8192_3_2;
pub mod lut_8192_4_2;
pub mod lut_8192_6_2;
pub mod debug_table;
pub mod mult_verification;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Hash)]
pub enum IndexSampling {
    Uniform,
    Binomial,
    Biased
}


pub enum TripleVector{
    SEMI(NoMulTripleRecording),
    MAL(DotProdVector<GF2p64>)
}

impl TripleVector{
    pub fn new(mal: bool) -> Self{
        if mal {
            return Self::MAL(DotProdVector::new());
        } else {
            return Self::SEMI(NoMulTripleRecording);
        }
    }

    pub fn get_mut_triple_vector(&mut self) -> &mut DotProdVector<GF2p64>{
        match self {
            TripleVector::SEMI(_) => panic!("Malicious setting expected"),
            TripleVector::MAL(vec) => vec,  
        }
    }

    pub fn get_triple_vector(&self) -> &DotProdVector<GF2p64>{
        match self {
            TripleVector::SEMI(_) => panic!("Malicious setting expected"),
            TripleVector::MAL(vec) => vec,  
        }
    }

    pub fn get_len(&self) -> usize {
        match self {
            TripleVector::SEMI(_) => panic!("Malicious setting expected"),
            TripleVector::MAL(vec) => vec.len(),  
        }
    }

    pub fn append(&mut self, mut other: TripleVector) {
        let o = other.get_mut_triple_vector();
        let i = self.get_mut_triple_vector();
        i.append(o);
    }

}


impl DotProdRecorder<GF2p64> for TripleVector{

    fn reserve_for_more_dotprods(&mut self, n: usize) {
        match self{
            TripleVector::SEMI(no_mul_triple_recording) => 
                DotProdRecorder::<GF2p64>::reserve_for_more_dotprods(no_mul_triple_recording, n),
            TripleVector::MAL(mul_triple_vector) => 
                mul_triple_vector.reserve_for_more_dotprods(n),
        }
    }

    fn record_dot_prod(&mut self, a_i: &[Vec<GF2p64>], a_ii: &[Vec<GF2p64>], b_i: &[Vec<GF2p64>], b_ii: &[Vec<GF2p64>], c_i: &Vec<GF2p64>, c_ii: &Vec<GF2p64>) {
        match self{
            TripleVector::SEMI(no_mul_triple_recording) => 
                no_mul_triple_recording.record_dot_prod(a_i, a_ii, b_i, b_ii, c_i, c_ii),
            TripleVector::MAL(mul_triple_vector) => 
                mul_triple_vector.record_dot_prod(a_i, a_ii, b_i, b_ii, c_i, c_ii),
        }
    }
    
    fn record_dot_in(&mut self, a_i: &[Vec<GF2p64>], a_ii: &[Vec<GF2p64>], b_i: &[Vec<GF2p64>], b_ii: &[Vec<GF2p64>]) {
        match self{
            TripleVector::SEMI(no_rec) => DotProdRecorder::<GF2p64>::record_dot_in(no_rec, a_i, a_ii, b_i, b_ii),
            TripleVector::MAL(rec) => rec.record_dot_in(a_i, a_ii, b_i, b_ii),
        }
    }
    
    fn record_dot_out(&mut self, c_i: &Vec<GF2p64>, c_ii: &Vec<GF2p64>) {
        match self{
            TripleVector::SEMI(no_rec) => DotProdRecorder::<GF2p64>::record_dot_out(no_rec, c_i, c_ii),
            TripleVector::MAL(rec) => rec.record_dot_out(c_i, c_ii),
        }
    }

}

impl IndexSampling{
    pub fn from_literal(str: String) -> Self {
        match str {
            val if val == "u".to_owned() => Self::Uniform,
            val if val == "s".to_owned() => Self::Biased,
            val if val == "b".to_owned() => Self::Binomial,
            _ => panic!("undefined literal, please select: s = skewed (Bernoulli bits), b = binomial, u = uniform"),
        }
    }

    pub fn get_sample_vec(in_vec: Vec<String>) -> Vec<Self>{
        in_vec.into_iter()
            .map(Self::from_literal)
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OhvVec<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>{
    pub inner: [bool; SIZE],
    _marker: PhantomData<T>
}

pub struct CubeOhv<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>{
    pub row_ohv: RndOhvOutput<T, SIZE, SIZE_RED>,
    pub col_ohv: RndOhvOutput<T, SIZE, SIZE_RED>,
    pub lay_ohv: RndOhvOutput<T, SIZE, SIZE_RED>,
    pub _marker: PhantomData<T>,
}

#[derive(Copy, Clone, Debug)]
pub struct RndOhvOutput<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize> {
    /// share i of one-hot vector
    pub si: OhvVec<T, SIZE, SIZE_RED>,
    /// share i+1 of one-hot vector
    pub sii: OhvVec<T, SIZE, SIZE_RED>,
    /// (2,3) sharing of the position of the 1 in the vector
    pub random_offset: RssShare<T::Embedded>,
    _marker: PhantomData<T>
}


pub struct LUTSamplerParty<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize>{
    pub inner: ChidaParty,
    pub broadcast_context: BroadcastContext,
    mal_sec: bool,
    dot_check_vec: TripleVector,
    prep_check_vec: MulTripleVector<GF2p64>,
    prep_cubes: Vec<CubeOhv<T, SIZE, SIZE_RED>>,
    prep_offsets: RssShareVec<T::Embedded>,
    skew: usize,
}

impl<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize> LUTSamplerParty<T, SIZE, SIZE_RED>{
    pub fn setup(connected: ConnectedParty, n_worker_threads: Option<usize>, prot_str: Option<String>, mal_sec: bool, skew: usize) -> MpcResult<Self>{
        ChidaParty::setup(connected, n_worker_threads, prot_str).map(|party| Self { 
            inner: party, 
            broadcast_context: BroadcastContext::new(),
            mal_sec,
            dot_check_vec: TripleVector::new(mal_sec),
            prep_check_vec: MulTripleVector::<GF2p64>::new(),
            prep_cubes: Vec::new(), 
            prep_offsets: Vec::new(),
            skew
        })
    }
    
    pub fn io(&self) -> &IoLayerOwned {
        <ChidaParty as ArithmeticBlackBox<T::Wrapper>>::io(&self.inner)
    }

    pub fn main_party_mut(&mut self) -> &mut ChidaParty {
        &mut self.inner
    }
    #[instrument(name = "Rotate One hot vectors", skip_all)]
    fn rotate_ohvs(&mut self) -> MpcResult<()>{
        debug_assert_eq!(self.prep_cubes.len()*3,self.prep_offsets.len());
        let mut offsets_public = Vec::with_capacity(self.prep_offsets.len());
        for (offsets, cube) in 
            self.prep_offsets
            .chunks(3)
            .zip(self.prep_cubes.iter_mut())
        {
            offsets_public.append(&mut cube.compute_offset(offsets));
        }

        let offsets_p = open_rss_many::<T>(self.inner.as_party_mut(), &mut self.broadcast_context, offsets_public)?;
        for (offsets, cube) in offsets_p
            .chunks(3)
            .zip(self.prep_cubes
            .iter_mut())
        {
            cube.rotate(offsets);
        }
        Ok(())
    }
    #[instrument(name = "Generate one-hot vectors for cubes", skip_all)]
    pub fn sample_ohvs(&mut self, n_samples: usize) -> MpcResult<()>{
        if self.mal_sec {
            self.prep_cubes = compute_ohv_vectors::<_,T,SIZE,SIZE_RED>(self.inner.as_party_mut(), &mut self.prep_check_vec, 8, n_samples)?;
        } else {
            self.prep_cubes = compute_ohv_vectors::<_,T,SIZE,SIZE_RED>(self.inner.as_party_mut(), &mut NoMulTripleRecording, 8, n_samples)?;
        }
        Ok(())
    }
    #[instrument(name = "Run index sampling", skip_all)]
    pub fn sample_indices(&mut self, n_samples: usize, dist: &[IndexSampling]) -> MpcResult<()>{
        assert!(dist.len() == 3, "Currently only cube evaluation supported");
        match dist[0] {
            IndexSampling::Uniform => {
                
            },
            IndexSampling::Binomial => {
                self.prep_offsets = compute_binomial_offsets::<T,SIZE>(self.inner.as_party_mut(), n_samples, 3)?;
            },
            IndexSampling::Biased => {
                let mut mask = SIZE-1;
                for _ in 0..T::RATIO{
                    mask = mask | (mask << (std::mem::size_of::<T::Embedded>()*8));
                }
                if self.mal_sec{ // Not the nicest way to do this
                    self.prep_offsets = compute_biased_offsets::<_,T>(self.inner.as_party_mut(), &mut self.prep_check_vec, n_samples, self.mal_sec, self.skew, T::Wrapper::from_usize(mask))?;
                } else{
                    self.prep_offsets = compute_biased_offsets::<_,T>(self.inner.as_party_mut(), &mut NoMulTripleRecording, n_samples, self.mal_sec, self.skew, T::Wrapper::from_usize(mask))?;
                }
            },
        }
        
        Ok(())
    }
    #[instrument(name = "Do preprocessing", skip_all)]
    pub fn do_preprocessing(&mut self, n_samples: usize, dist: &[IndexSampling]) -> MpcResult<()>{
        self.sample_ohvs(n_samples)?;
        self.sample_indices(n_samples, dist)?;
        self.rotate_ohvs()?;
        Ok(())
    }
    #[instrument(name = "Sigma LUT sampling", skip_all)]
    pub fn sample_with_lut(&mut self, lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE], n_samples: usize) -> MpcResult<RssShareVec<T::Embedded>>{
        debug_assert_eq!(self.prep_cubes.len(), n_samples);
        sample_many(self.inner.as_party_mut(), &self.prep_cubes, lut_table, &mut self.dot_check_vec,  n_samples)
        
    }

    pub fn get_coordinates(&mut self, index: usize) -> MpcResult<Vec<T::Embedded>>{
        assert!(index < self.prep_cubes.len());
        self.prep_cubes[index].get_coordinates(self.inner.as_party_mut(), &mut self.broadcast_context)
    }

    pub fn verify_triples(&mut self) -> MpcResult<bool> {
        match &mut self.dot_check_vec{
            TripleVector::SEMI(_) => {
                println!("No verification in Semi-honest setting");
                Ok(true)
            },
            TripleVector::MAL(mal_vec) => {
                mult_verification::verify_multiplication_triples(
                    &mut self.inner.as_party_mut(), 
                    &mut self.broadcast_context, 
                    &mut[&mut GF2p64Encoder(&mut self.prep_check_vec), &mut GF2p64DotEncoder(mal_vec)], 
                    false)
            },
        }
    }

}

impl<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize> CubeOhv<T, SIZE, SIZE_RED>{

    // collapses columns
    pub fn collapse_columns_local(&self, lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE]) -> [[[T::Wrapper; SIZE_RED]; SIZE];2]{
        let ohv = &self.col_ohv;
        [ohv.si.collapse_columns(lut_table), ohv.sii.collapse_columns(lut_table)]
    }

    // Collapses Rows
    pub fn collapse_rows(&self, matrices: &[[[T::Wrapper; SIZE_RED]; SIZE]; 2], rec: &mut TripleVector) -> [T::Wrapper; SIZE_RED]{
        let mut res_vec = [T::Wrapper::default(); SIZE_RED];
        let ohv = &self.row_ohv;
        match rec{
            TripleVector::SEMI(_) => {
                for(i, (layer_i, layer_ii)) in matrices[0].iter().zip(matrices[1]).enumerate(){
                    for (j, r) in res_vec.iter_mut().enumerate(){
                        if ohv.si.inner[i]{
                            *r += layer_i[j];
                            *r += layer_ii[j];
                        } if ohv.sii.inner[i] {
                            *r += layer_i[j];
                        }
                    }
                }
            },
            TripleVector::MAL(recorder) => {
                // The boolean values
                let mut ai  = vec![Vec::with_capacity(SIZE);SIZE];
                let mut aii = vec![Vec::with_capacity(SIZE);SIZE];
                // The LUT values
                let mut bi  = vec![Vec::with_capacity(SIZE);SIZE];
                let mut bii = vec![Vec::with_capacity(SIZE);SIZE];
                for(i, (layer_i, layer_ii)) in matrices[0].iter().zip(matrices[1]).enumerate(){
                    let ohvi = ohv.si.inner[i];
                    let ohvii = ohv.sii.inner[i];
                    let bool_si =  GF2p64::new(ohvi);
                    let bool_sii = GF2p64::new(ohvii);
                    for (inner_ai, inner_aii) in ai.iter_mut().zip(&mut aii){
                        inner_ai.push(bool_si);
                        inner_aii.push(bool_sii);
                    }
                    
                    
                    for (j, r) in res_vec.iter_mut().enumerate(){                        
                        let offset = j * T::RATIO;
                        let embedded_layer_i = T::new(layer_i[j]).embed(); 
                        let embedded_layer_ii = T::new(layer_ii[j]).embed();
                        debug_assert!(embedded_layer_i.len() == embedded_layer_ii.len());
                        debug_assert!(embedded_layer_i.len() == T::RATIO);
                        for (k, (e_i, e_ii)) in embedded_layer_i.into_iter().zip(embedded_layer_ii).enumerate() {
                            bi[offset + k].push(e_i);
                            bii[offset + k].push(e_ii);
                        }
                        
                        if ohvi{
                            *r += layer_i[j];
                            *r += layer_ii[j];
                        } 
                        if ohvii {
                            *r += layer_i[j];
                        }
                    }
                }
                for b_i in bi.iter(){
                    debug_assert!(b_i.len() == SIZE, "expected: 256, actual: {}", b_i.len());
                }
                // println!("lengths: ai {}, aii {}, bi {}, bii {}", ai[0].len(), aii[0].len(), bi[0].len(), bii[0].len());
                recorder.record_dot_in(&ai, &aii, &bi, &bii);
            }
        }
        
        res_vec
    }

    // #[instrument(name = "Collapse Columns", skip_all)]
    // Collapses Layers
    pub fn collapse_layers(&self, v_i: &[T::Wrapper], v_ii: &[T::Wrapper], rec: &mut TripleVector) -> T::Embedded{
        let new_len= std::mem::size_of::<T::Embedded>();
        let mut res = T::Embedded::default();
        let ohv = &self.lay_ohv;
        let val_i = <T::Wrapper as NetSerializable>::as_byte_vec(v_i.iter(), v_i.len());
        let val_ii = <T::Wrapper as NetSerializable>::as_byte_vec(v_ii.iter(), v_ii.len());
        debug_assert!(val_i.len() == v_i.len()*std::mem::size_of::<T::Wrapper>());
        debug_assert!(val_ii.len() == v_ii.len()*std::mem::size_of::<T::Wrapper>());
        
        for (i, (vi, vii)) in val_i.chunks(new_len).zip(val_ii.chunks(new_len)).enumerate(){
            if ohv.si.inner[i]{
                res += T::Embedded::pack_bytes(vi);
                res += T::Embedded::pack_bytes(vii); 
            } if ohv.sii.inner[i]{
                res += T::Embedded::pack_bytes(vi);
            }
        }
        match rec{
            TripleVector::MAL(r) =>{
                let flat_i:  Vec<GF2p64> = val_i.iter().map(|e| T::Embedded::from_u8(*e).embed()).collect();
                let flat_ii: Vec<GF2p64> = val_ii.iter().map(|e| T::Embedded::from_u8(*e).embed()).collect();
                let bi:      Vec<GF2p64> = ohv.si.inner.iter().map(|e| GF2p64::new(*e)).collect();
                let bii:     Vec<GF2p64> = ohv.sii.inner.iter().map(|e| GF2p64::new(*e)).collect();
                r.record_dot_in(
                    std::slice::from_ref(&bi), 
                    std::slice::from_ref(&bii), 
                    std::slice::from_ref(&flat_i), 
                    std::slice::from_ref(&flat_ii)
                );
            },
            TripleVector::SEMI(_) => {}
        }
        res
    }

    pub fn get_coordinates(&self, party: &mut MainParty, context: &mut BroadcastContext) -> MpcResult<Vec<T::Embedded>> {
        let row = self.row_ohv.random_offset;
        let col = self.col_ohv.random_offset;
        let lay = self.lay_ohv.random_offset;
        let v = vec![row, col, lay];
        open_rss_many::<T>(party, context, v)
    }

    pub fn compute_offset(&mut self, offsets: &[RssShare<T::Embedded>]) -> RssShareVec<T::Embedded> {
        debug_assert_eq!(offsets.len(), 3);
        [&self.row_ohv, &self.col_ohv, &self.lay_ohv]
            .into_iter()
            .zip(offsets)
            .map(|(ohv, share)| RssShare {
                si: share.si + ohv.random_offset.si,
                sii: share.sii + ohv.random_offset.sii,
            })
            .collect()
    }

    pub fn rotate(&mut self, offset: &[T::Embedded]){
        self.row_ohv.rotate(offset[0]);
        self.col_ohv.rotate(offset[1]);
        self.lay_ohv.rotate(offset[2]);
    }

}

impl<T:GFTTrait, const SIZE: usize, const SIZE_RED: usize> RndOhvOutput<T,SIZE,SIZE_RED>{
    pub fn new(ohv_output: (OhvVec<T,SIZE,SIZE_RED>,OhvVec<T,SIZE,SIZE_RED>), index: RssShare<T::Embedded>) -> Self { 
        Self{
            si: ohv_output.0,
            sii: ohv_output.1,
            random_offset: index,
            _marker: PhantomData,
        }
    }
    pub fn rotate(&mut self, offset: T::Embedded) {
        // println!("offset {:?}", offset);
        self.random_offset.si += offset;
        self.random_offset.sii += offset;
        self.si.rotate(offset);
        self.sii.rotate(offset);
    }
}

impl<T: GFTTrait, const SIZE: usize, const SIZE_RED: usize> OhvVec<T,SIZE,SIZE_RED>
where T::Embedded: Share, T::Wrapper: Share {
    pub fn new(val: [bool; SIZE]) -> Self{
        Self { inner: val, _marker: PhantomData }
    }

    pub fn rotate(&mut self, offset: T::Embedded){
        let mut tmp = [false; SIZE];
        for (i, el) in self.inner.iter().enumerate(){
            let index = i ^ offset.to_usize();
            tmp[index] = *el;
        }
        self.inner = tmp;
    }

    pub fn collapse_columns(
        &self, 
        lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE]
    ) -> [[T::Wrapper; SIZE_RED]; SIZE]
    {
        let mut res_matrix = [[T::Wrapper::default(); SIZE_RED]; SIZE];
        for (layer, res) in lut_table.iter().zip(res_matrix.iter_mut()){
            for (i, row) in layer.iter().enumerate(){
                if self.inner[i] {
                    res.iter_mut().zip(row.iter()).for_each(|(r, &val)| *r += T::Wrapper::new(val));
                }
            }
        }
        // println!("{:?}", res_matrix);
        res_matrix
    }

}


/// This function implements the LUT sampling benchmark.
///
/// The arguments are
/// - `connected` - the local party
/// - `simd` - number of parallel samples calls
/// - `n_worker_threads` - number of worker threads
#[instrument(name = "Run LUT benchmark", skip_all)]
pub fn lut_sampler_benchmark<
T: GFTTrait,
const SIZE: usize,
const SIZE_RED: usize,
>(
    connected: ConnectedParty, 
    simd: usize, 
    dist: &[IndexSampling],
    skew: usize,
    mal_sec: bool,
    n_worker_threads: Option<usize>,
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE]
) {
    let span_setup = span!(Level::INFO, "Setup").entered();
    let mut party: LUTSamplerParty<T,SIZE,SIZE_RED> = LUTSamplerParty::setup(connected, n_worker_threads, None, mal_sec, skew).unwrap();
    
    span_setup.exit();
    let setup_comm_stats = party.io().reset_comm_stats();
    
    // party.do_preprocessing(simd, dist).unwrap();
    let span = span!(Level::INFO, "Preprocessing").entered();
    party.sample_ohvs(simd).unwrap();
    let ohv_comm_stats = party.io().reset_comm_stats();
    party.sample_indices(simd, dist).unwrap();
    let index_sampling_comm_stats = party.io().reset_comm_stats();
    party.rotate_ohvs().unwrap();
    let rotation_comm_stats = party.io().reset_comm_stats();
    span.exit();

    println!("Sample LUT");
    let _output = party.sample_with_lut(lut_table, simd).unwrap();
    let online_comm_stats = party.io().reset_comm_stats();

    println!("Verifying mult triples products");
    let span_verify = span!(Level::INFO, "Verifying dot products").entered();
    let valid = party.verify_triples().unwrap();
    span_verify.exit();
    let verify_comm_stats = party.io().reset_comm_stats();
    if !valid{
        panic!("Error Triple Verification failed!");
    }

    let span_end = span!(Level::INFO, "Checking and Teardown").entered();
    println!("Checking the result with the cube coordinates");
    let samples = open_rss_many::<T>(party.inner.as_party_mut(), &mut party.broadcast_context, _output).unwrap();
    for i in 0..samples.len(){
        let coordinates = party.get_coordinates(i).unwrap();
        compare_result::<T,SIZE,SIZE_RED>(coordinates, samples[i], lut_table);
    }
    println!("Samples {:?}", samples.iter().map(|x| x.inner()).collect::<Vec<_>>());
    span_end.exit();

    party.inner.teardown().unwrap();
    println!("Finished benchmark");
    println!("Setup:");
    setup_comm_stats.print_comm_statistics(party.inner.party_index());
    println!("\nOne-hot Vectors:");
    ohv_comm_stats.print_comm_statistics(party.inner.party_index());
    println!("\nIndex Sampling:");
    index_sampling_comm_stats.print_comm_statistics(party.inner.party_index());
    println!("\nRotations:");
    rotation_comm_stats.print_comm_statistics(party.inner.party_index());
    println!("\nOnline Phase:");
    online_comm_stats.print_comm_statistics(party.inner.party_index());
    println!("\nVerify Triples:");
    verify_comm_stats.print_comm_statistics(party.inner.party_index());
    party.inner.print_statistics();
}
