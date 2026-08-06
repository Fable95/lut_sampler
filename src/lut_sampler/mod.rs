// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
use clap::ValueEnum;
use maestro::rep3_core::{party::error::MpcResult, share::RssShareVec};
use rss_lut::dabit::{b2a_many, generate_dabits, random_zp_signs, zp_mul_rss, DaBitStore};
use rss_lut::mult_verification::verify_zp_triples;
use rss_lut::online::open_rss_many;
use rss_lut::share::gf_template::{GFTTrait, Share, ShareType};
use rss_lut::share::zp::{Zp, P};
use rss_lut::util::mul_triple_vec::{MulTripleVector, NoMulTripleRecording};
use tracing::{instrument, span, Level};

use crate::lut_sampler::index_bias::compute_biased_offsets;

pub mod index_bias;
pub mod tables;

pub use rss_lut::party::{LutParty, LutPartyCube, LutPartyMatrix, Network};

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

/// Samples the cheap (biased) index distribution and returns the secret-shared
/// offsets to be applied to the one-hot vectors.
#[instrument(name = "Run index sampling", level = "trace", skip_all)]
pub fn sample_indices<P: LutParty>(
    network: &mut Network,
    party: &mut P,
    n_samples: usize,
    skew: usize,
    l: usize,
) -> MpcResult<Vec<RssShareVec<ShareType<P::IndexType>>>> {
    let mal_sec = party.mal_sec();
    let k: Vec<usize> = party.k().to_vec();
    if mal_sec {
        compute_biased_offsets(network, party.prep_check_vec(), n_samples, mal_sec, skew, &k, l)
    } else {
        compute_biased_offsets(network, &mut NoMulTripleRecording, n_samples, mal_sec, skew, &k, l)
    }
}

#[instrument(name = "Do preprocessing", level = "trace", skip_all)]
pub fn do_preprocessing<P: LutParty>(
    network: &mut Network,
    party: &mut P,
    n_samples: usize,
    skew: usize,
    l: usize,
) -> MpcResult<()> {
    party.sample_ohvs(n_samples, network)?;
    let offsets = sample_indices(network, party, n_samples, skew, l)?;
    party.rotate_ohvs(&offsets, network)?;
    Ok(())
}

/// This function implements the LUT sampling benchmark.
///
/// The arguments are
/// - `simd` - number of parallel samples calls
/// - `net` - the local party network
/// - `party` - the LUT party
/// - `skew`, `l` - index-biasing parameters
#[instrument(name = "Run LUT benchmark", level = "trace", skip_all)]
pub fn lut_sampler_benchmark<
T: GFTTrait,
P: LutParty,
>(
    simd: usize,
    net: &mut Network,
    party: &mut P,
    skew: usize,
    l: usize,
    network: bool,
    debug: bool
) {
    let span_tot = span!(Level::TRACE, "Total runtime").entered();
    let span = span!(Level::TRACE, "OHV gen, index sampling, rotation.").entered();
    party.sample_ohvs(simd, net).unwrap();
    let ohv_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    let offsets = sample_indices(net, party, simd, skew, l).unwrap();
    let index_sampling_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    party.rotate_ohvs(&offsets, net).unwrap();
    let rotation_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    span.exit();

    let _output = party.sample_with_lut(simd, net).unwrap();
    let online_comm_stats = net.reset_comm_stats::<T::Wrapper>();

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
    }
}

/// LUT sampling benchmark with the final signed binary -> `Z_p` conversion:
/// the sampled one-sided (folded) magnitudes are turned into two-sided
/// arithmetic samples `sigma * x` with a fresh secret uniform sign per sample
/// (`sigma = 2b - 1` from a free random bit, applied with one `Zp`
/// multiplication after the conversion).
///
/// Mirrors [`lut_sampler_benchmark`]; the daBit/sign preprocessing and the
/// B2A conversion each live in their own timing span and network accounting.
///
/// Additional arguments over the original:
/// - `value_bits` - conversion bit width; pass
///   `rss_lut::dabit::bits_for_max_value(Table::N_MAX)` (const-evaluable).
///   `N_MAX` is asserted against the table data at export time, so no online
///   scan is needed; a `debug_assert` re-checks the table data in debug
///   builds (zero release cost).
#[instrument(name = "Run signed LUT benchmark", level = "trace", skip_all)]
pub fn lut_sampler_benchmark_signed<
T: GFTTrait,
P: LutParty,
>(
    simd: usize,
    net: &mut Network,
    party: &mut P,
    skew: usize,
    l: usize,
    value_bits: usize,
    network: bool,
    debug: bool
) {
    let mal_sec = party.mal_sec();
    let mut zp_triples = MulTripleVector::<Zp>::new();

    let span_tot = span!(Level::TRACE, "Total runtime").entered();
    let span = span!(Level::TRACE, "OHV gen, index sampling, rotation.").entered();
    party.sample_ohvs(simd, net).unwrap();
    let ohv_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    let offsets = sample_indices(net, party, simd, skew, l).unwrap();
    let index_sampling_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    party.rotate_ohvs(&offsets, net).unwrap();
    let rotation_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    span.exit();

    // The exported N_MAX (asserted at generation time) must agree with the
    // table data; re-check via a table scan in debug builds only.
    debug_assert_eq!(
        value_bits,
        party.max_value_bits(),
        "value_bits does not match the table data — N_MAX metadata is stale"
    );

    // daBit + random-sign preprocessing (input-independent, could run any
    // time before the conversion).
    let span_dabit = span!(Level::TRACE, "daBit preprocessing").entered();
    let (mut store, signs): (DaBitStore<<P::T as GFTTrait>::Embedded>, RssShareVec<Zp>) = if mal_sec {
        (
            generate_dabits(net.chida.as_party_mut(), &mut zp_triples, value_bits, simd).unwrap(),
            random_zp_signs::<<P::T as GFTTrait>::Embedded, _>(net.chida.as_party_mut(), &mut zp_triples, simd).unwrap(),
        )
    } else {
        (
            generate_dabits(net.chida.as_party_mut(), &mut NoMulTripleRecording, value_bits, simd).unwrap(),
            random_zp_signs::<<P::T as GFTTrait>::Embedded, _>(net.chida.as_party_mut(), &mut NoMulTripleRecording, simd).unwrap(),
        )
    };
    span_dabit.exit();
    let dabit_comm_stats = net.reset_comm_stats::<T::Wrapper>();

    let output = party.sample_with_lut(simd, net).unwrap();
    let online_comm_stats = net.reset_comm_stats::<T::Wrapper>();

    // B2A conversion (one opening round), then the random sign applied with
    // one multiplication (one more round): signed_output = sigma * x.
    let span_b2a = span!(Level::TRACE, "B2A conversion").entered();
    let converted = b2a_many(
        net.chida.as_party_mut(),
        &mut net.broadcast_context,
        &mut store,
        &output,
    ).unwrap();
    let signed_output = if mal_sec {
        zp_mul_rss(net.chida.as_party_mut(), &mut zp_triples, &converted, &signs).unwrap()
    } else {
        zp_mul_rss(net.chida.as_party_mut(), &mut NoMulTripleRecording, &converted, &signs).unwrap()
    };
    span_b2a.exit();
    let b2a_comm_stats = net.reset_comm_stats::<T::Wrapper>();

    let span_verify = span!(Level::TRACE, "Verifying dot products").entered();
    // GF(2^64) triples; in the malicious path this also commits all openings
    // recorded so far (offsets, the B2A opening) via check_view.
    let valid = party.verify_triples(net).unwrap();
    if !valid {
        panic!("Error Triple Verification failed!");
    }
    // Zp triples from the daBit generation, then commit the verification's
    // own coin-flip openings.
    if mal_sec {
        let valid_zp = verify_zp_triples(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &mut zp_triples,
            false,
        ).unwrap();
        if !valid_zp {
            panic!("Error Zp Triple Verification failed!");
        }
        net.check_view().unwrap();
    }
    span_verify.exit();
    let verify_comm_stats = net.reset_comm_stats::<T::Wrapper>();
    span_tot.exit();

    if debug {
        let span_end = span!(Level::TRACE, "Checking and Teardown").entered();
        println!("Checking the result with the cube coordinates");
        let samples = party.open_many(&output, net).unwrap();
        for i in 0..samples.len() {
            let coordinates = party.get_coordinates(i, net).unwrap();
            party.compare_result(coordinates, samples[i]);
        }
        let zp_samples = open_rss_many::<Zp>(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &signed_output,
        ).unwrap();
        net.check_view().unwrap();
        // Print as signed integers: representatives above p/2 are negative.
        let signed: Vec<i64> = zp_samples.iter().map(|z| {
            let v = z.value();
            if v > P / 2 { v as i64 - P as i64 } else { v as i64 }
        }).collect();
        // Consistency: each signed sample must be +-(binary magnitude).
        for (s, (bin, z)) in samples.iter().zip(&signed).enumerate() {
            let mag = bin.to_usize() as i64;
            if z.abs() != mag {
                println!("Sample {}: signed value {} does not match magnitude {}", s, z, mag);
            }
        }
        println!("Magnitudes {:?}", samples.iter().map(|x| x.inner()).collect::<Vec<_>>());
        println!("Signed samples {:?}", signed);
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
        println!("\ndaBit Preprocessing:");
        dabit_comm_stats.print_comm_statistics(net.party_index());
        println!("\nOnline Phase:");
        online_comm_stats.print_comm_statistics(net.party_index());
        println!("\nB2A Conversion:");
        b2a_comm_stats.print_comm_statistics(net.party_index());
        println!("\nVerify Triples (GF(2^64) + Zp):");
        verify_comm_stats.print_comm_statistics(net.party_index());
    }
}
