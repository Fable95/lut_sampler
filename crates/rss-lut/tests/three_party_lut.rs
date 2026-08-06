// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid

//! End-to-end correctness test for the oblivious LUT primitive, run over the
//! real network stack with three localhost parties.
//!
//! The three-party localhost harness (`localhost_connect` / `create_certificates`)
//! is adapted from the maestro / tabularasta test utilities; everything else
//! exercises `rss-lut`'s public API only.

use std::{
    fs::File,
    io::BufReader,
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    str::FromStr,
    thread,
};

use maestro::rep3_core::{
    network::{Config, ConnectedParty, CreatedParty},
    share::{RssShare, RssShareVec},
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

use rss_lut::offline::{extract_byte_from_cube, extract_byte_from_matrix};
use rss_lut::party::{LutParty, LutPartyCube, LutPartyMatrix, Network};
use rss_lut::share::gf_template::{GFT, ShareType};

// --- LUT instances under test ---------------------------------------------

// Wrapper u64 packs RATIO = 4 embedded u16 values (one per last-dim position).
type GF = GFT<u64, u16, 4>;
const SAMPLES: usize = 8;

// Divisible last dim (SIZE_last == RATIO) and the new small case (SIZE_last < RATIO).
const CK: [usize; 3] = [2, 2, 2]; // cube, SIZE3 = 4 = RATIO
const CK_SMALL: [usize; 3] = [2, 2, 1]; // cube, SIZE3 = 2 < RATIO -> SIZE3_RED = 1
const MK: [usize; 2] = [2, 2]; // matrix, SIZE2 = 4 = RATIO
const MK_SMALL: [usize; 2] = [2, 1]; // matrix, SIZE2 = 2 < RATIO -> SIZE2_RED = 1

// Pack `s3` real u16 layer values (rest zero) into a u64 cell.
const fn cube_cell(r: usize, c: usize, s3: usize) -> u64 {
    let mut lay = 0;
    let mut packed = 0u64;
    while lay < s3 {
        packed |= ((r * 64 + c * 8 + lay + 1) as u64) << (lay * 16);
        lay += 1;
    }
    packed
}

const fn build_cube<const S1: usize, const S2: usize, const S3: usize>() -> [[[u64; 1]; S2]; S1] {
    let mut t = [[[0u64; 1]; S2]; S1];
    let mut r = 0;
    while r < S1 {
        let mut c = 0;
        while c < S2 {
            t[r][c][0] = cube_cell(r, c, S3);
            c += 1;
        }
        r += 1;
    }
    t
}

const fn build_matrix<const S1: usize, const S2: usize>() -> [[u64; 1]; S1] {
    let mut t = [[0u64; 1]; S1];
    let mut r = 0;
    while r < S1 {
        let mut col = 0;
        let mut packed = 0u64;
        while col < S2 {
            packed |= ((r * 8 + col + 1) as u64) << (col * 16);
            col += 1;
        }
        t[r][0] = packed;
        r += 1;
    }
    t
}

const CUBE: [[[u64; 1]; 4]; 4] = build_cube::<4, 4, 4>();       
const CUBE_SMALL: [[[u64; 1]; 4]; 4] = build_cube::<4, 4, 2>(); 
const MATRIX: [[u64; 1]; 4] = build_matrix::<4, 4>();               
const MATRIX_SMALL: [[u64; 1]; 4] = build_matrix::<4, 2>();         

/// One party's cube program: preprocess, look up, verify, and check every
/// opened sample against the public table at its (opened) coordinates. Returns
/// the opened sample values so the caller can cross-check party agreement.
fn cube_program<const S1: usize, const S2: usize, const S3: usize, const S3R: usize>(
    k: [usize; 3],
    table: &'static [[[u64; S3R]; S2]; S1],
    mal_sec: bool,
) -> impl FnOnce(ConnectedParty) -> Vec<u16> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let mut party = LutPartyCube::<GF, S1, S2, S3, S3R>::setup(mal_sec, &k, table);

        party.sample_ohvs(SAMPLES, &mut net).unwrap();
        // Zero offsets look up each OHV at its own random position.
        let zero = RssShare { si: ShareType(0u16), sii: ShareType(0u16) };
        let offsets: Vec<RssShareVec<ShareType<u16>>> = vec![vec![zero; 3]; SAMPLES];
        party.rotate_ohvs(&offsets, &mut net).unwrap();

        let out = party.sample_with_lut(SAMPLES, &mut net).unwrap();
        assert!(party.verify_triples(&mut net).unwrap(), "triple verification failed");

        let samples = party.open_many(&out, &mut net).unwrap();
        let mut vals = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            let coords = party.get_coordinates(i, &mut net).unwrap();
            let coords_usize: Vec<usize> = coords.iter().map(|c| c.0 as usize).collect();
            let expected = extract_byte_from_cube::<GF, S2, S3R>(&coords_usize, table);
            assert_eq!(
                expected.0, samples[i].0,
                "party {}: lookup at {:?} = {}, table says {}",
                net.party_index(), coords_usize, samples[i].0, expected.0
            );
            vals.push(samples[i].0);
        }
        net.teardown().unwrap();
        vals
    }
}

fn matrix_program<const S1: usize, const S2: usize, const S2R: usize>(
    k: [usize; 2],
    table: &'static [[u64; S2R]; S1],
    mal_sec: bool,
) -> impl FnOnce(ConnectedParty) -> Vec<u16> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let mut party = LutPartyMatrix::<GF, S1, S2, S2R>::setup(mal_sec, &k, table);

        party.sample_ohvs(SAMPLES, &mut net).unwrap();
        let zero = RssShare { si: ShareType(0u16), sii: ShareType(0u16) };
        let offsets: Vec<RssShareVec<ShareType<u16>>> = vec![vec![zero; 2]; SAMPLES];
        party.rotate_ohvs(&offsets, &mut net).unwrap();

        let out = party.sample_with_lut(SAMPLES, &mut net).unwrap();
        assert!(party.verify_triples(&mut net).unwrap(), "triple verification failed");

        let samples = party.open_many(&out, &mut net).unwrap();
        let mut vals = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            let coords = party.get_coordinates(i, &mut net).unwrap();
            let coords_usize: Vec<usize> = coords.iter().map(|c| c.0 as usize).collect();
            let expected = extract_byte_from_matrix::<GF, S1, S2R>(&coords_usize, table);
            assert_eq!(
                expected.0, samples[i].0,
                "party {}: lookup at {:?} = {}, table says {}",
                net.party_index(), coords_usize, samples[i].0, expected.0
            );
            vals.push(samples[i].0);
        }
        net.teardown().unwrap();
        vals
    }
}

fn agree3<T: PartialEq + std::fmt::Debug>(v: (T, T, T)) {
    assert_eq!(v.0, v.1, "party 1 and 2 disagree on opened samples");
    assert_eq!(v.0, v.2, "party 1 and 3 disagree on opened samples");
}

#[test]
fn cube_lookup_semi_honest() {
    agree3(localhost_connect(
        cube_program::<4, 4, 4, 1>(CK, &CUBE, false),
        cube_program::<4, 4, 4, 1>(CK, &CUBE, false),
        cube_program::<4, 4, 4, 1>(CK, &CUBE, false),
    ));
}

#[test]
fn cube_lookup_malicious() {
    agree3(localhost_connect(
        cube_program::<4, 4, 4, 1>(CK, &CUBE, true),
        cube_program::<4, 4, 4, 1>(CK, &CUBE, true),
        cube_program::<4, 4, 4, 1>(CK, &CUBE, true),
    ));
}

#[test]
fn cube_lookup_small_last_dim_semi_honest() {
    agree3(localhost_connect(
        cube_program::<4, 4, 2, 1>(CK_SMALL, &CUBE_SMALL, false),
        cube_program::<4, 4, 2, 1>(CK_SMALL, &CUBE_SMALL, false),
        cube_program::<4, 4, 2, 1>(CK_SMALL, &CUBE_SMALL, false),
    ));
}

#[test]
fn cube_lookup_small_last_dim_malicious() {
    agree3(localhost_connect(
        cube_program::<4, 4, 2, 1>(CK_SMALL, &CUBE_SMALL, true),
        cube_program::<4, 4, 2, 1>(CK_SMALL, &CUBE_SMALL, true),
        cube_program::<4, 4, 2, 1>(CK_SMALL, &CUBE_SMALL, true),
    ));
}

#[test]
fn matrix_lookup_semi_honest() {
    agree3(localhost_connect(
        matrix_program::<4, 4, 1>(MK, &MATRIX, false),
        matrix_program::<4, 4, 1>(MK, &MATRIX, false),
        matrix_program::<4, 4, 1>(MK, &MATRIX, false),
    ));
}

#[test]
fn matrix_lookup_malicious() {
    agree3(localhost_connect(
        matrix_program::<4, 4, 1>(MK, &MATRIX, true),
        matrix_program::<4, 4, 1>(MK, &MATRIX, true),
        matrix_program::<4, 4, 1>(MK, &MATRIX, true),
    ));
}

#[test]
fn matrix_lookup_small_last_dim_semi_honest() {
    agree3(localhost_connect(
        matrix_program::<4, 2, 1>(MK_SMALL, &MATRIX_SMALL, false),
        matrix_program::<4, 2, 1>(MK_SMALL, &MATRIX_SMALL, false),
        matrix_program::<4, 2, 1>(MK_SMALL, &MATRIX_SMALL, false),
    ));
}

#[test]
fn matrix_lookup_small_last_dim_malicious() {
    agree3(localhost_connect(
        matrix_program::<4, 2, 1>(MK_SMALL, &MATRIX_SMALL, true),
        matrix_program::<4, 2, 1>(MK_SMALL, &MATRIX_SMALL, true),
        matrix_program::<4, 2, 1>(MK_SMALL, &MATRIX_SMALL, true),
    ));
}

// --- three-party localhost harness ----------------------------------------

const TEST_KEY_DIR: &str = "keys";
type KeyPair = (PrivateKeyDer<'static>, CertificateDer<'static>);

fn create_certificates() -> (KeyPair, KeyPair, KeyPair) {
    fn key_path(filename: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push(TEST_KEY_DIR);
        p.push(filename);
        p
    }

    fn load_key(name: &str) -> PrivateKeyDer<'static> {
        let mut reader =
            BufReader::new(File::open(key_path(name)).unwrap_or_else(|_| panic!("Cannot open {}", name)));
        rustls_pemfile::private_key(&mut reader)
            .unwrap_or_else(|_| panic!("Cannot read private key in {}", name))
            .unwrap_or_else(|| panic!("No private key in {}", name))
    }

    fn load_cert(name: &str) -> CertificateDer<'static> {
        let mut reader =
            BufReader::new(File::open(key_path(name)).unwrap_or_else(|_| panic!("Cannot open {}", name)));
        let cert: Vec<_> = rustls_pemfile::certs(&mut reader)
            .map(|r| r.unwrap_or_else(|_| panic!("Cannot read certificate in {}", name)))
            .collect();
        assert_eq!(cert.len(), 1);
        cert[0].clone()
    }

    (
        (load_key("p1.key"), load_cert("p1.pem")),
        (load_key("p2.key"), load_cert("p2.pem")),
        (load_key("p3.key"), load_cert("p3.pem")),
    )
}

fn localhost_connect<
    T1: Send,
    F1: Send + FnOnce(ConnectedParty) -> T1,
    T2: Send,
    F2: Send + FnOnce(ConnectedParty) -> T2,
    T3: Send,
    F3: Send + FnOnce(ConnectedParty) -> T3,
>(
    f1: F1,
    f2: F2,
    f3: F3,
) -> (T1, T2, T3) {
    let addr: Vec<Ipv4Addr> = (0..3).map(|_| Ipv4Addr::from_str("127.0.0.1").unwrap()).collect();
    let party1 = CreatedParty::bind(0, IpAddr::V4(addr[0]), 0).unwrap();
    let party2 = CreatedParty::bind(1, IpAddr::V4(addr[1]), 0).unwrap();
    let party3 = CreatedParty::bind(2, IpAddr::V4(addr[2]), 0).unwrap();

    let port1 = party1.port().unwrap();
    let port2 = party2.port().unwrap();
    let port3 = party3.port().unwrap();

    let certs = create_certificates();
    let (sk1, pk1) = certs.0;
    let (sk2, pk2) = certs.1;
    let (sk3, pk3) = certs.2;

    let certificates = vec![pk1.clone(), pk2.clone(), pk3.clone()];
    let ports = vec![port1, port2, port3];

    let (p1_res, p2_res, p3_res) = thread::scope(|scope| {
        let party1 = {
            let config = Config::new(addr.clone(), ports.clone(), certificates.clone(), pk1, sk1);
            thread::Builder::new()
                .name("party1".to_string())
                .stack_size(1024 * 1024 * 32)
                .spawn_scoped(scope, move || {
                    let party1 = party1.connect(config, None).unwrap();
                    f1(party1)
                })
                .unwrap()
        };

        let party2 = {
            let addr: Vec<Ipv4Addr> = (0..3).map(|_| Ipv4Addr::from_str("127.0.0.1").unwrap()).collect();
            let config = Config::new(addr, ports.clone(), certificates.clone(), pk2, sk2);
            thread::Builder::new()
                .name("party2".to_string())
                .stack_size(1024 * 1024 * 32)
                .spawn_scoped(scope, move || {
                    let party2 = party2.connect(config, None).unwrap();
                    f2(party2)
                })
                .unwrap()
        };

        let party3 = {
            let addr: Vec<Ipv4Addr> = (0..3).map(|_| Ipv4Addr::from_str("127.0.0.1").unwrap()).collect();
            let config = Config::new(addr, ports, certificates, pk3, sk3);
            thread::Builder::new()
                .name("party3".to_string())
                .stack_size(1024 * 1024 * 32)
                .spawn_scoped(scope, move || {
                    let party3 = party3.connect(config, None).unwrap();
                    f3(party3)
                })
                .unwrap()
        };

        (party1.join(), party2.join(), party3.join())
    });

    (p1_res.unwrap(), p2_res.unwrap(), p3_res.unwrap())
}
