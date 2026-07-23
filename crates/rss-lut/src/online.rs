use std::ops::Add;

use tracing::{instrument, span, Level};

use maestro::rep3_core::{
        network::{NetSerializable, task::Direction}, party::{
            DigestExt, MainParty, Party, broadcast::{Broadcast, BroadcastContext}, error::MpcResult
        }, share::{RssShare, RssShareVec}
    };

use crate::{
    ohv_container::MatrixOhv, share::{gf_template::{GFTTrait, Share}, gf2p64::{GF2p64, GF2p64Subfield}, helper_types::AllowedTypes}, util::mul_triple_vec::DotProdRecorder
};

use crate::{ohv_container::CubeOhv, mult_verification::TripleVector};

pub fn open_rss_many<
    T: NetSerializable + Add<Output=T> + Clone + DigestExt
    >
    (party: &mut MainParty, context: &mut BroadcastContext, shares: &RssShareVec<T>) -> MpcResult<Vec<T>>{
    let (si_values, sii_values): (Vec<T>, Vec<T>) = shares
        .iter()
        .map(|share| (share.si.clone(), share.sii.clone()))
        .unzip();
    // println!("opening: {},{} si and sii vals", si_values.len(), sii_values.len());
    party.open_rss(context, &si_values, &sii_values)
}


pub fn sample_many_matrix<
T: GFTTrait,
IndexType: AllowedTypes,
const SIZE: usize, 
const SIZE2: usize,
const SIZE_RED: usize
>(
    party: &mut MainParty, 
    matrix_vector: &Vec<MatrixOhv<T, IndexType, SIZE, SIZE2>>, 
    lut_table: &[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE], 
    rec: &mut TripleVector, 
    amount: usize
) -> MpcResult<RssShareVec<T::Embedded>> 
{
    let alphas: Vec<T::Embedded> = party.generate_alpha::<T::Embedded>(amount).collect();
    let mut vector_i = Vec::with_capacity(amount);
    let mut vector_ii = vec![T::Embedded::default(); amount];
    let mut res = Vec::with_capacity(amount);
    for matrix in matrix_vector.iter(){
        let vectors = matrix.collapse_rows_local(lut_table);
        let res = matrix.collapse_cols(&vectors, rec);
        vector_i.push(res);
    }
    for i in 0..amount{
        vector_i[i] += alphas[i];
    }
    party.send_field::<T::Embedded>(Direction::Previous, vector_i.iter(), amount);
    party.receive_field_slice(Direction::Next, &mut vector_ii).rcv()?;
    let flat_i:  Vec<GF2p64> = vector_i.iter().map(|e| e.embed()).collect();
    let flat_ii: Vec<GF2p64> = vector_ii.iter().map(|e| e.embed()).collect();
    rec.record_dot_out(&flat_i, &flat_ii);

    for i in 0..amount{
        res.push(RssShare{
            si: vector_i[i],
            sii: vector_ii[i]
        });
    }
    Ok(res)
}



pub fn sample_many_cube<
T: GFTTrait,
IndexType: AllowedTypes,
const SIZE1: usize,
const SIZE2: usize, 
const SIZE3: usize,
const SIZE3_RED: usize
>(
    party: &mut MainParty, 
    cube_vector: &Vec<CubeOhv<T, IndexType, SIZE1, SIZE2, SIZE3>>, 
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2]], 
    rec: &mut [TripleVector; 2], 
    amount: usize
) -> MpcResult<RssShareVec<T::Embedded>> 
{

    let mut vectors_i: Vec<T::Wrapper> = Vec::with_capacity(SIZE3_RED*amount);
    let alphas: Vec<T::Wrapper> = party.generate_alpha::<T::Wrapper>(amount*SIZE3_RED).collect();
    let mut samples_i = Vec::with_capacity(amount);
    let mut samples_ii = vec![T::Embedded::default(); amount];
    let mut res = Vec::with_capacity(amount);
    // println!("Sample many {}, {}", amount, cube_vector.len());
    collapse_rows_and_columns(cube_vector, lut_table, &mut rec[0], &mut vectors_i);

    // Mask alpha to the real lanes so padding lanes stay zero in both shares.
    let wrap_bits = 8 * std::mem::size_of::<<T::Wrapper as Share>::InnerType>();
    let emb_bits = 8 * std::mem::size_of::<<T::Embedded as Share>::InnerType>();
    for i in 0..amount*SIZE3_RED{
        let w = i % SIZE3_RED;
        let bits = std::cmp::min(T::RATIO, SIZE3 - w*T::RATIO) * emb_bits;
        let mask = (!<<T::Wrapper as Share>::InnerType as AllowedTypes>::ZERO) >> (wrap_bits - bits);
        vectors_i[i] += alphas[i] * T::Wrapper::new(mask);
    }

    // Transfer only the real lanes of each sample (padding lanes are dropped / zero-filled).
    let stride = SIZE3_RED * T::RATIO;
    let small_i = T::unpack_slice(&vectors_i);
    let mut compact_i = Vec::with_capacity(amount*SIZE3);
    for s in 0..amount{
        compact_i.extend_from_slice(&small_i[s*stride .. s*stride + SIZE3]);
    }
    let mut compact_ii = vec![T::Embedded::default(); amount*SIZE3];
    party.send_field::<T::Embedded>(Direction::Previous, compact_i.iter(), amount*SIZE3);
    party.receive_field_slice(Direction::Next, &mut compact_ii).rcv()?;

    let mut padded_ii = vec![T::Embedded::default(); amount*stride];
    for s in 0..amount{
        padded_ii[s*stride .. s*stride + SIZE3].copy_from_slice(&compact_ii[s*SIZE3 .. (s+1)*SIZE3]);
    }
    let vectors_ii = T::pack_slice(&padded_ii);



    let flat_i:  Vec<GF2p64> = vectors_i.iter().map(|e| e.embed()).collect();
    let flat_ii: Vec<GF2p64> = vectors_ii.iter().map(|e| e.embed()).collect();
    rec[0].record_dot_out(&flat_i, &flat_ii);
    // println!("Collapse cols");
    let span = span!(Level::TRACE, "Collapse Layers").entered();
    for (i, (chunk_i, chunk_ii)) in vectors_i.chunks(SIZE3_RED).zip(vectors_ii.chunks(SIZE3_RED)).enumerate() {
        samples_i.push(cube_vector[i].collapse_layers(chunk_i, chunk_ii, &mut  rec[1]));
    }
    span.exit();
    // Send and receive amount bytes
    // println!("Send and receive");
    party.send_field::<T::Embedded>(Direction::Previous, samples_i.iter(), amount);
    party.receive_field_slice(Direction::Next, &mut samples_ii).rcv()?;
    let flat_i:  Vec<GF2p64> = samples_i.iter().map(|e| e.embed()).collect();
    let flat_ii: Vec<GF2p64> = samples_ii.iter().map(|e| e.embed()).collect();
    // print_many(&flat_i, &flat_ii, &flat_i, &flat_ii, &flat_ii, &flat_i);
    rec[1].record_dot_out(&flat_i, &flat_ii);
    // println!("Create RSS vector");
    for i in 0..amount{
        res.push(RssShare{
            si: samples_i[i],
            sii: samples_ii[i]
        });
    }
    Ok(res)
}

#[instrument(name = "Collapse Rows and Columns", level = "trace", skip_all)]
fn collapse_rows_and_columns<
T: GFTTrait,
IndexType: AllowedTypes,
const SIZE1: usize, 
const SIZE2: usize, 
const SIZE3: usize, 
const SIZE3_RED: usize,
>(
    cube_vector: &Vec<CubeOhv<T,IndexType,SIZE1,SIZE2,SIZE3>>, 
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE3_RED]; SIZE2]],
    rec: &mut TripleVector,
    res_vectors: &mut Vec<T::Wrapper>
) {
    for cube in cube_vector.iter(){
        let matrices = cube.collapse_columns_local(lut_table);
        let vectors = cube.collapse_rows(&matrices, rec);
        res_vectors.append(&mut vectors.to_vec());
    }
}