use maestro::{
        chida::ChidaParty, rep3_core::{
        network::ConnectedParty,
        party::{CombinedCommStats, broadcast::{Broadcast, BroadcastContext}, error::MpcResult},
        share::RssShareVec
    }, share::Field, util::ArithmeticBlackBox
};
use crate::{
    mult_verification::{self, TripleVector},
    offline::{self, compute_ohv_vectors_cube, compute_ohv_vectors_matrix, extract_byte_from_matrix},
    ohv_container::{CubeOhv, MatrixOhv},
    online::{open_rss_many, sample_many_cube, sample_many_matrix},
    share::{gf_template::{GFTTrait, Share, ShareType}, gf2p64::GF2p64, helper_types::AllowedTypes},
    util::mul_triple_vec::{GF2p64DotEncoder, GF2p64Encoder, MulTripleVector, NoMulTripleRecording}
};
use tracing::instrument;

pub trait LutParty {
    fn rotate_ohvs(&mut self, offsets: &[RssShareVec<ShareType<Self::IndexType>>], network: &mut Network) -> MpcResult<()>;
    fn sample_ohvs(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<()>;

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

        let res = if d == 2{
            mult_verification::verify_multiplication_triples(
                &mut network.chida.as_party_mut(),
                &mut network.broadcast_context,
                &mut[
                    &mut GF2p64Encoder(&mut prep),
                    dot1,
                    ],
                false
            )?
        } else {
            let dot2 = &mut GF2p64DotEncoder::new(size[2], dot_products[1].get_mut_triple_vector());
            mult_verification::verify_multiplication_triples(
                &mut network.chida.as_party_mut(),
                &mut network.broadcast_context,
                &mut[
                    &mut GF2p64Encoder(&mut prep),
                    dot1,
                    dot2
                    ],
                false
            )?
        };
        // Commit all openings recorded so far (offset openings, verification
        // coin flips) by comparing the accumulated broadcast views.
        network.check_view()?;
        Ok(res)
    }

    fn sample_with_lut(&mut self, n_samples: usize, network: &mut Network) -> MpcResult<RssShareVec<<Self::T as GFTTrait>::Embedded>>;
    fn get_coordinates(&mut self, index: usize, network: &mut Network) -> MpcResult<Vec<ShareType<Self::IndexType>>>;
    fn open_many(&mut self, shares: &RssShareVec<<Self::T as GFTTrait>::Embedded>, network: &mut Network) -> MpcResult<Vec<<Self::T as GFTTrait>::Embedded>>{
        open_rss_many::<<Self::T as GFTTrait>::Embedded>(network.chida.as_party_mut(), &mut network.broadcast_context, shares)
    }
    fn compare_result(&self, indices: Vec<ShareType<Self::IndexType>>, value: <Self::T as GFTTrait>::Embedded);
    fn print_ohvs(&self);
    /// Bit width of the largest value stored in the LUT — the `bit_width`
    /// needed for the daBit-based B2A conversion of this party's outputs.
    /// One linear scan over the (public) table; zero-padding lanes cannot
    /// raise the maximum. At least 1.
    ///
    /// Debug/test accounting only: release paths should size the conversion
    /// from generation-time metadata (e.g. an exported `N_MAX` const via
    /// `dabit::bits_for_max_value`) instead of scanning online.
    fn max_value_bits(&self) -> usize;
    fn mal_sec(&self) -> bool;
    fn dot_check_vec(&mut self) -> &mut [TripleVector];
    fn prep_check_vec(&mut self) -> &mut MulTripleVector<GF2p64>;
    fn clear_self(&mut self);
    fn k(&self) -> &[usize];
    fn dim(&self) -> usize;
    fn sizes(&self) -> Vec<usize> {
        self.k().iter().map(|ki| 1 << ki).collect()
    }
    type T: GFTTrait;
    type IndexType: AllowedTypes;
}

/// Bit width of the largest embedded value among the given packed table
/// cells (see `LutParty::max_value_bits`).
fn max_value_bits_of_cells<T: GFTTrait>(
    cells: impl Iterator<Item = <<T as GFTTrait>::Wrapper as Share>::InnerType>,
) -> usize {
    let mut max = 0usize;
    for cell in cells {
        let wrapper = <T::Wrapper as Share>::new(cell);
        for i in 0..T::RATIO {
            max = max.max(T::get_element(&wrapper, i).to_usize());
        }
    }
    (usize::BITS as usize - max.leading_zeros() as usize).max(1)
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
    /// Compares the views of all `open_rss` openings recorded in the broadcast
    /// context so far with the neighbors and starts a fresh context. Errors if
    /// a party equivocated. `LutParty::verify_triples` calls this; after any
    /// opening that happens *later* (e.g. `open_many`, `b2a_many`), call it
    /// again before releasing the opened values.
    pub fn check_view(&mut self) -> MpcResult<()> {
        let context = std::mem::replace(&mut self.broadcast_context, BroadcastContext::new());
        self.chida.as_party_mut().compare_view(context)
    }
    pub fn teardown(&mut self) -> MpcResult<()> {
        self.chida.as_party_mut().teardown()
    }
}

pub struct LutPartyCube<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE3: usize, const SIZE3_RED: usize>{
    lut_table: &'a [[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2];SIZE1],
    mal_sec: bool,
    dot_check_vec: [TripleVector; 2],
    prep_check_vec: MulTripleVector<GF2p64>,
    prep_cubes: Vec<CubeOhv<T, <Self as LutParty>::IndexType, SIZE1, SIZE2, SIZE3>>,
    k: [usize; 3],
}

pub struct LutPartyMatrix<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize>{
    lut_table: &'a [[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE2_RED]; SIZE1],
    mal_sec: bool,
    dot_check_vec: [TripleVector;1],
    prep_check_vec: MulTripleVector<GF2p64>,
    prep_matrices: Vec<MatrixOhv<T, <Self as LutParty>::IndexType, SIZE1, SIZE2>>,
    k: [usize; 2],
}

impl<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE3: usize, const SIZE3_RED: usize> LutPartyCube<'a, T, SIZE1, SIZE2, SIZE3,SIZE3_RED>{
    pub fn setup(mal_sec: bool, k: &[usize], lut_table: &'a [[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2];SIZE1]) -> Self{
        debug_assert!(k.len() == 3, "not enough k parameters for cube setting");
        debug_assert!(1 << k[0] == SIZE1, "k must be such that 2^K == SIZE");
        debug_assert!(1 << k[1] == SIZE2, "k must be such that 2^K == SIZE");
        debug_assert!(1 << k[2]== SIZE3, "k must be such that 2^K == SIZE");
        debug_assert!(SIZE3.div_ceil(T::RATIO) == SIZE3_RED, "SIZE3_RED must be ceil(SIZE3 / RATIO)");
        Self {
            lut_table,
            mal_sec,
            dot_check_vec: [TripleVector::new(mal_sec), TripleVector::new(mal_sec)],
            prep_check_vec: MulTripleVector::<GF2p64>::new(),
            prep_cubes: Vec::new(),
            k: [k[0], k[1], k[2]],
        }
    }
}

impl<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize> LutPartyMatrix<'a, T, SIZE1, SIZE2,SIZE2_RED>{
    pub fn setup(mal_sec: bool, k: &[usize], lut_table: &'a [[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE2_RED]; SIZE1]) -> Self{
        debug_assert!(k.len() == 2 , "need two k values for matrix setting.");
        debug_assert!(1 << k[0] == SIZE1, "k must be such that 2^K == SIZE");
        debug_assert!(1 << k[1] == SIZE2, "k must be such that 2^K == SIZE");
        debug_assert!(SIZE2.div_ceil(T::RATIO) == SIZE2_RED, "SIZE2_RED must be ceil(SIZE2 / RATIO)");
        Self {
            lut_table,
            mal_sec,
            dot_check_vec: [TripleVector::new(mal_sec)],
            prep_check_vec: MulTripleVector::<GF2p64>::new(),
            prep_matrices: Vec::new(),
            k: [k[0], k[1]],
        }
    }
}

impl<'a, T: GFTTrait, const SIZE1: usize, const SIZE2: usize, const SIZE2_RED: usize> LutParty for LutPartyMatrix<'a, T, SIZE1, SIZE2, SIZE2_RED>{

    fn clear_self(&mut self) {
        self.prep_check_vec.clear();
        self.prep_matrices.clear();
    }

    #[instrument(name = "Rotate one-hot vectors", level = "trace", skip_all)]
    fn rotate_ohvs(&mut self, offsets: &[RssShareVec<ShareType<Self::IndexType>>], network: &mut Network) -> MpcResult<()> {
        debug_assert_eq!(self.prep_matrices.len()*self.dim(), offsets.len()*offsets[0].len());
        let mut offsets_public = Vec::with_capacity(offsets.len());
        for (offset, matrix) in
            offsets.iter()
            .zip(self.prep_matrices.iter_mut())
        {
            offsets_public.append(&mut matrix.compute_offset(offset));
        }

        let offsets_p = open_rss_many::<ShareType::<Self::IndexType>>(network.chida.as_party_mut(), &mut network.broadcast_context, &offsets_public)?;
        for (offset, matrix) in offsets_p
            .chunks(self.dim())
            .zip(self.prep_matrices.iter_mut()){
            matrix.rotate(offset);
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
            println!("Extract pos {:?}: expected: {:?}, got: {:?}", indices_usize, expected, value);
        }
    }

    fn max_value_bits(&self) -> usize {
        max_value_bits_of_cells::<T>(self.lut_table.iter().flatten().copied())
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

    fn k(&self) -> &[usize] {
        &self.k
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
> LutParty for LutPartyCube<'a, T, SIZE1, SIZE2, SIZE3, SIZE3_RED>{

    fn clear_self(&mut self) {
        self.prep_check_vec.clear();
        self.prep_cubes.clear();
    }

    #[instrument(name = "Rotate one-hot vectors", level = "trace", skip_all)]
    fn rotate_ohvs(&mut self, offsets: &[RssShareVec<ShareType<Self::IndexType>>], network: &mut Network) -> MpcResult<()>{
        debug_assert_eq!(self.prep_cubes.len()*self.dim(), offsets.len()*offsets[0].len());
        let mut offsets_public = Vec::with_capacity(offsets.len());
        for (offset, cube) in
            offsets.iter()
            .zip(self.prep_cubes.iter_mut())
        {
            offsets_public.append(&mut cube.compute_offset(offset));
        }

        let offsets_p = open_rss_many::<ShareType<Self::IndexType>>(network.chida.as_party_mut(), &mut network.broadcast_context, &offsets_public)?;
        for (offset, cube) in offsets_p
            .chunks(self.dim())
            .zip(self.prep_cubes.iter_mut()){
            cube.rotate(offset);
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

    fn max_value_bits(&self) -> usize {
        max_value_bits_of_cells::<T>(self.lut_table.iter().flatten().flatten().copied())
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

    fn k(&self) -> &[usize] {
        &self.k
    }

    type T = T;

    fn dim(&self) -> usize {
        3
    }

    type IndexType = u16;
}
