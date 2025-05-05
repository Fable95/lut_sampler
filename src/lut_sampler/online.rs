use tracing::{instrument, span, Level};

use maestro::rep3_core::{
        network::task::Direction, party::{
            broadcast::{Broadcast, BroadcastContext},
            error::MpcResult, MainParty, Party,
        }, share::{RssShare, RssShareVec}
    };

use crate::{
    share::{gf2p64::{GF2p64, GF2p64Subfield}, gf_template::{GFTTrait, Share}},
    util::mul_triple_vec::DotProdRecorder
};

use super::{CubeOhv, TripleVector};

pub fn open_rss_many<T: GFTTrait>
    (party: &mut MainParty, context: &mut BroadcastContext, shares: RssShareVec<T::Embedded>) -> MpcResult<Vec<T::Embedded>>{
    let (si_values, sii_values): (Vec<T::Embedded>, Vec<T::Embedded>) = shares
        .iter()
        .map(|share| (share.si, share.sii))
        .unzip();
    // println!("opening: {},{} si and sii vals", si_values.len(), sii_values.len());
    party.open_rss(context, &si_values, &sii_values)
}

#[instrument(name = "Collapse Rows and Columns", skip_all)]
fn collapse_rows_and_columns<
T: GFTTrait,
const SIZE: usize, 
const SIZE_RED: usize
>(
    cube_vector: &Vec<CubeOhv<T,SIZE,SIZE_RED>>, 
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE],
    rec: &mut TripleVector,
    res_vectors: &mut Vec<T::Wrapper>
) {
    for cube in cube_vector.iter(){
        let matrices = cube.collapse_columns_local(lut_table);
        let vectors = cube.collapse_rows(&matrices, rec);
        res_vectors.append(&mut vectors.to_vec());
    }
}

pub fn sample_many<T: GFTTrait,const SIZE: usize, const SIZE_RED: usize>(
    party: &mut MainParty, 
    cube_vector: &Vec<CubeOhv<T,SIZE,SIZE_RED>>, 
    lut_table: &[[[<<T as GFTTrait>::Wrapper as Share>::InnerType; SIZE_RED]; SIZE]; SIZE], 
    rec: &mut TripleVector, 
    amount: usize
) -> MpcResult<RssShareVec<T::Embedded>> 
{
    let mut vectors_i: Vec<T::Wrapper> = Vec::with_capacity(SIZE_RED*amount);
    let alphas: Vec<T::Wrapper> = party.generate_alpha::<T::Wrapper>(amount*SIZE_RED).collect();
    let mut vectors_ii = vec![T::Wrapper::default(); SIZE_RED*amount];
    let mut samples_i = Vec::with_capacity(amount);
    let mut samples_ii = vec![T::Embedded::default(); amount];
    let mut res = Vec::with_capacity(amount);
    // println!("Sample many {}, {}", amount, cube_vector.len());
    collapse_rows_and_columns(cube_vector, lut_table, rec, &mut vectors_i);

    // println!("Gen alphas");
    for i in 0..amount*SIZE_RED{
        vectors_i[i] += alphas[i];
    }
    // Send and receive 2 * SIZE_RED * amount * 16 byte = 32 * SIZE_RED * amount
    party.send_field::<T::Wrapper>(Direction::Previous, vectors_i.iter(), amount*SIZE_RED);
    party.receive_field_slice(Direction::Next, &mut vectors_ii).rcv()?;
    let flat_i:  Vec<GF2p64> = vectors_i.iter().flat_map(|e| T::new(*e).embed()).collect();
    let flat_ii: Vec<GF2p64> = vectors_ii.iter().flat_map(|e| T::new(*e).embed()).collect();
    rec.record_dot_out(&flat_i, &flat_ii);
    // println!("Collapse cols");
    let span = span!(Level::INFO, "Collapse Layers").entered();
    for (i, (chunk_i, chunk_ii)) in vectors_i.chunks(SIZE_RED).zip(vectors_ii.chunks(SIZE_RED)).enumerate() {
        samples_i.push(cube_vector[i].collapse_layers(chunk_i, chunk_ii, rec));
    }
    span.exit();
    // Send and receive amount bytes
    // println!("Send and receive");
    party.send_field::<T::Embedded>(Direction::Previous, samples_i.iter(), amount);
    party.receive_field_slice(Direction::Next, &mut samples_ii).rcv()?;
    let flat_i:  Vec<GF2p64> = samples_i.iter().map(|e| e.embed()).collect();
    let flat_ii: Vec<GF2p64> = samples_ii.iter().map(|e| e.embed()).collect();
    rec.record_dot_out(&flat_i, &flat_ii);
    // println!("Create RSS vector");
    for i in 0..amount{
        res.push(RssShare{
            si: samples_i[i],
            sii: samples_ii[i]
        });
    }
    Ok(res)
}