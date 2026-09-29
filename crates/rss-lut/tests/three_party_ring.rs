// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid

//! End-to-end tests for replicated sharing over the integer ring `Z_{2^K}`
//! (`share::z2k`) with the generic multiplication `online::mul_rss`, run over
//! the real network stack with three localhost parties.
//!
//! Inputs are genuine random sharings (`generate_random`), opened in the test
//! only to compute the expected plaintext results.

mod common;

use common::{agree3, localhost_connect};
use maestro::rep3_core::{
    network::ConnectedParty,
    party::Party,
    share::{HasZero, RssShare, RssShareVec},
};
use maestro::share::Field;

use rss_lut::dabit::zp_mul_rss;
use rss_lut::online::{mul_rss, open_rss_many};
use rss_lut::party::Network;
use rss_lut::share::z2k::{Z2k, Z2p16, Z2p32, Z2p64};
use rss_lut::share::zp::Zp;
use rss_lut::util::mul_triple_vec::NoMulTripleRecording;

/// Public constants exercising zero, one, the sign bit and wraparound.
fn edge_constants<const K: u32>() -> Vec<Z2k<K>> {
    let m = Z2k::<K>::MASK;
    [0, 1, 2, 3, m, m - 1, m >> 1, (m >> 1) + 1]
        .into_iter()
        .map(Z2k::new)
        .collect()
}

/// One party's program over `Z_{2^K}`: open random sharings `r`, multiply
/// `r * (r + c)` for public edge constants `c` and `r * s` for a second
/// random sharing `s`, then a depth-2 product `(r*s)*r`. Every product is
/// checked against plaintext arithmetic; returns the opened values.
fn ring_program<const K: u32>() -> impl FnOnce(ConnectedParty) -> Vec<u64> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let party = net.chida.as_party_mut();
        let mut rec = NoMulTripleRecording;
        let consts = edge_constants::<K>();
        let n = consts.len() + 64;

        let r: RssShareVec<Z2k<K>> = party.generate_random(n);
        let s: RssShareVec<Z2k<K>> = party.generate_random(n);
        // x_i = r_i + c_i for the first entries (public constant shifts).
        let x: RssShareVec<Z2k<K>> = r
            .iter()
            .enumerate()
            .map(|(i, ri)| match consts.get(i) {
                Some(c) => *ri + party.constant(*c),
                None => *ri,
            })
            .collect();

        let rx = mul_rss(party, &mut rec, &r, &x).unwrap();
        let rs = mul_rss(party, &mut rec, &r, &s).unwrap();
        let rsr = mul_rss(party, &mut rec, &rs, &r).unwrap();
        // Local linear operations on shares, including scalar multiplication.
        let lin: RssShareVec<Z2k<K>> = rs
            .iter()
            .zip(&r)
            .map(|(a, b)| *a * Z2k::new(3) - *b + party.constant(Z2k::ONE))
            .collect();

        let ctx = &mut net.broadcast_context;
        let party = net.chida.as_party_mut();
        let open = |party: &mut _, ctx: &mut _, v: &RssShareVec<Z2k<K>>| {
            open_rss_many::<Z2k<K>>(party, ctx, v).unwrap()
        };
        let (r_o, s_o) = (open(party, ctx, &r), open(party, ctx, &s));
        let (rx_o, rs_o) = (open(party, ctx, &rx), open(party, ctx, &rs));
        let (rsr_o, lin_o) = (open(party, ctx, &rsr), open(party, ctx, &lin));

        for i in 0..n {
            let c = consts.get(i).copied().unwrap_or(Z2k::ZERO);
            assert_eq!(rx_o[i], r_o[i] * (r_o[i] + c), "r*(r+c) at {}", i);
            assert_eq!(rs_o[i], r_o[i] * s_o[i], "r*s at {}", i);
            assert_eq!(rsr_o[i], r_o[i] * s_o[i] * r_o[i], "(r*s)*r at {}", i);
            assert_eq!(
                lin_o[i],
                rs_o[i] * Z2k::new(3) - r_o[i] + Z2k::ONE,
                "linear at {}",
                i
            );
            assert!(rx_o[i].value() <= Z2k::<K>::MASK);
        }
        // Randomness sanity: two independent random vectors differ somewhere.
        assert_ne!(r_o, s_o);

        net.teardown().unwrap();
        [r_o, rx_o, rs_o, rsr_o, lin_o]
            .concat()
            .into_iter()
            .map(|z| z.value())
            .collect()
    }
}

#[test]
fn ring_multiplication_z2p16() {
    agree3(localhost_connect(
        ring_program::<16>(),
        ring_program::<16>(),
        ring_program::<16>(),
    ));
}

#[test]
fn ring_multiplication_z2p32() {
    agree3(localhost_connect(
        ring_program::<32>(),
        ring_program::<32>(),
        ring_program::<32>(),
    ));
}

#[test]
fn ring_multiplication_z2p64() {
    agree3(localhost_connect(
        ring_program::<64>(),
        ring_program::<64>(),
        ring_program::<64>(),
    ));
}

#[test]
fn ring_multiplication_non_byte_width() {
    // K = 12 serializes to 2 bytes; high bits must never leak into values.
    agree3(localhost_connect(
        ring_program::<12>(),
        ring_program::<12>(),
        ring_program::<12>(),
    ));
}

/// `zp_mul_rss` is now a thin wrapper around `mul_rss`; both agree on Zp.
fn zp_wrapper_program() -> impl FnOnce(ConnectedParty) -> Vec<u64> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let party = net.chida.as_party_mut();
        let mut rec = NoMulTripleRecording;
        let a: RssShareVec<Zp> = party.generate_random(32);
        let b: RssShareVec<Zp> = party.generate_random(32);
        let generic = mul_rss(party, &mut rec, &a, &b).unwrap();
        let wrapper = zp_mul_rss(party, &mut rec, &a, &b).unwrap();
        let ctx = &mut net.broadcast_context;
        let party = net.chida.as_party_mut();
        let a_o = open_rss_many::<Zp>(party, ctx, &a).unwrap();
        let b_o = open_rss_many::<Zp>(party, ctx, &b).unwrap();
        let g_o = open_rss_many::<Zp>(party, ctx, &generic).unwrap();
        let w_o = open_rss_many::<Zp>(party, ctx, &wrapper).unwrap();
        for i in 0..a_o.len() {
            assert_eq!(g_o[i], a_o[i] * b_o[i]);
            assert_eq!(w_o[i], g_o[i]);
        }
        net.teardown().unwrap();
        g_o.into_iter().map(|z| z.value()).collect()
    }
}

#[test]
fn zp_wrapper_matches_generic_multiplication() {
    agree3(localhost_connect(
        zp_wrapper_program(),
        zp_wrapper_program(),
        zp_wrapper_program(),
    ));
}

// Compile-time: the aliases are the documented widths.
#[test]
fn aliases() {
    assert_eq!(<Z2p16 as Field>::NBITS, 16);
    assert_eq!(<Z2p32 as Field>::NBITS, 32);
    assert_eq!(<Z2p64 as Field>::NBITS, 64);
    let _unused: RssShare<Z2p16> = RssShare::from(Z2p16::ZERO, Z2p16::ZERO);
}
