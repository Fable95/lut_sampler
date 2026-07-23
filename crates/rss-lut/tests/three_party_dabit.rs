//! End-to-end tests for the daBit-based binary -> Z_p conversion, run over
//! the real network stack with three localhost parties. All three real party
//! indices run, which matters because the injection step in
//! `generate_dabits` is party-index dependent.
//!
//! The three-party localhost harness (`localhost_connect` /
//! `create_certificates`) is the same as in `three_party_lut.rs`.

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
    party::Party,
    share::{RssShare, RssShareVec},
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

use rss_lut::dabit::{
    b2a_many, dabits_required, generate_dabits, random_zp_signs, zp_mul_rss, DaBitStore,
    ZP_TRIPLES_PER_DABIT, ZP_TRIPLES_PER_SIGN,
};
use rss_lut::mult_verification::verify_zp_triples;
use rss_lut::online::open_rss_many;
use rss_lut::party::{LutParty, LutPartyCube, Network};
use rss_lut::share::gf_template::{ShareType, GFT};
use rss_lut::share::zp::{Zp, P};
use rss_lut::util::mul_triple_vec::{MulTripleVector, NoMulTripleRecording};

// --- standalone daBit generation + conversion ------------------------------

const K: usize = 12; // bit width under test, < 16 to exercise the masking
const M: usize = 8; // daBit-store samples in the first generation batch

/// One party's program: generate daBits in two batches, cross-check the two
/// domains bit-for-bit (test-only openings — this burns the daBits' secrecy,
/// which is fine for a correctness test), convert known public constants,
/// verify, and open. Returns the opened Zp values for cross-party agreement.
fn dabit_program(mal_sec: bool) -> impl FnOnce(ConnectedParty) -> Vec<u64> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let mut zp_triples = MulTripleVector::<Zp>::new();
        let mut no_rec = NoMulTripleRecording;

        // Generation (offline; independent of any input).
        let mut store: DaBitStore<ShareType<u16>> = if mal_sec {
            generate_dabits(net.chida.as_party_mut(), &mut zp_triples, K, M).unwrap()
        } else {
            generate_dabits(net.chida.as_party_mut(), &mut no_rec, K, M).unwrap()
        };
        assert_eq!(store.available_samples(), M);
        if mal_sec {
            assert_eq!(zp_triples.len(), dabits_required(K, M) * ZP_TRIPLES_PER_DABIT);
        }

        // Chunked generation: a second batch merges into the same store.
        let more: DaBitStore<ShareType<u16>> = if mal_sec {
            generate_dabits(net.chida.as_party_mut(), &mut zp_triples, K, 2).unwrap()
        } else {
            generate_dabits(net.chida.as_party_mut(), &mut no_rec, K, 2).unwrap()
        };
        store.append(more);
        assert_eq!(store.available_samples(), M + 2);

        // Domain consistency (test-only): open both sides of every daBit and
        // compare bit-for-bit; every Zp share must open to 0 or 1, and to the
        // low K bits only (the masking of generate_random's output).
        let bin_open = open_rss_many::<ShareType<u16>>(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &store.masks().to_vec(),
        )
        .unwrap();
        let zp_open = open_rss_many::<Zp>(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &store.zp_bits().to_vec(),
        )
        .unwrap();
        for (s, mask) in bin_open.iter().enumerate() {
            assert_eq!(mask.0 >> K, 0, "mask has bits above bit_width");
            for j in 0..K {
                let b = ((mask.0 >> j) & 1) as u64;
                let z = zp_open[s * K + j].value();
                assert!(z <= 1, "Zp daBit is not a bit: {}", z);
                assert_eq!(b, z, "daBit domains disagree at sample {}, bit {}", s, j);
            }
        }

        // Conversion of known public constants (valid RSS sharings via
        // `party.constant`), consuming daBits from the store.
        let values: [u16; 4] = [0x000, 0x001, 0xABC, 0xFFF];
        let xs: RssShareVec<ShareType<u16>> = values
            .iter()
            .map(|v| net.chida.as_party_mut().constant(ShareType(*v)))
            .collect();
        let ys = b2a_many(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &mut store,
            &xs,
        )
        .unwrap();
        // Single-use: exactly xs.len() masks were consumed.
        assert_eq!(store.available_samples(), M + 2 - values.len());

        // Verify-then-open: Zp triple check + commit all recorded openings.
        if mal_sec {
            assert!(
                verify_zp_triples(
                    net.chida.as_party_mut(),
                    &mut net.broadcast_context,
                    &mut zp_triples,
                    false
                )
                .unwrap(),
                "Zp triple verification failed"
            );
            assert_eq!(zp_triples.len(), 0, "verification should clear the triples");
        }
        net.check_view().unwrap();

        let opened = open_rss_many::<Zp>(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &ys,
        )
        .unwrap();
        net.check_view().unwrap();
        for (v, o) in values.iter().zip(&opened) {
            assert_eq!(*v as u64, o.value(), "converted value mismatch");
        }
        net.teardown().unwrap();
        opened.iter().map(|z| z.value()).collect()
    }
}

#[test]
fn dabit_generation_and_conversion_semi_honest() {
    agree3(localhost_connect(
        dabit_program(false),
        dabit_program(false),
        dabit_program(false),
    ));
}

#[test]
fn dabit_generation_and_conversion_malicious() {
    agree3(localhost_connect(
        dabit_program(true),
        dabit_program(true),
        dabit_program(true),
    ));
}

// --- signed conversion (random sign for one-sided distributions) -----------

/// Enough signs that "both signs occur" is a sound assertion
/// (P[all 64 signs equal] = 2^-63).
const SIGNED_M: usize = 64;

/// One party's program: generate daBits and random signs, cross-check the
/// signs via test-only openings (sigma ∈ {+1,-1}, both signs occur), then
/// convert known constants with `b2a_many`, apply the signs with one
/// `zp_mul_rss`, and check the output is exactly sigma * x for the opened
/// sigmas. Returns the opened signed values.
fn signed_program(mal_sec: bool) -> impl FnOnce(ConnectedParty) -> Vec<u64> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let mut zp_triples = MulTripleVector::<Zp>::new();
        let mut no_rec = NoMulTripleRecording;

        let values: [u16; 4] = [0x000, 0x001, 0xABC, 0xFFF];
        let (mut store, signs): (DaBitStore<ShareType<u16>>, Vec<_>) = if mal_sec {
            (
                generate_dabits(net.chida.as_party_mut(), &mut zp_triples, K, values.len())
                    .unwrap(),
                random_zp_signs::<ShareType<u16>, _>(
                    net.chida.as_party_mut(),
                    &mut zp_triples,
                    SIGNED_M,
                )
                .unwrap(),
            )
        } else {
            (
                generate_dabits(net.chida.as_party_mut(), &mut no_rec, K, values.len()).unwrap(),
                random_zp_signs::<ShareType<u16>, _>(
                    net.chida.as_party_mut(),
                    &mut no_rec,
                    SIGNED_M,
                )
                .unwrap(),
            )
        };
        assert_eq!(signs.len(), SIGNED_M);
        if mal_sec {
            assert_eq!(
                zp_triples.len(),
                dabits_required(K, values.len()) * ZP_TRIPLES_PER_DABIT
                    + SIGNED_M * ZP_TRIPLES_PER_SIGN
            );
        }

        // Sign structure (test-only opening, burns the signs' secrecy):
        // sigma is ±1 and both signs occur (P[failure] = 2 * 2^-64 with
        // honest randomness).
        let sigma_open = open_rss_many::<Zp>(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &signs,
        )
        .unwrap();
        let minus_one = P - 1;
        for sig in &sigma_open {
            assert!(
                sig.value() == 1 || sig.value() == minus_one,
                "sigma is not ±1: {}",
                sig.value()
            );
        }
        assert!(sigma_open.iter().any(|s| s.value() == 1), "no positive sign drawn");
        assert!(sigma_open.iter().any(|s| s.value() == minus_one), "no negative sign drawn");

        // Convert known constants, then apply the first values.len() signs
        // with one multiplication; expected output is sigma_s * x_s.
        let xs: RssShareVec<ShareType<u16>> = values
            .iter()
            .map(|v| net.chida.as_party_mut().constant(ShareType(*v)))
            .collect();
        let converted = b2a_many(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &mut store,
            &xs,
        )
        .unwrap();
        assert_eq!(store.available_samples(), 0, "store should be exhausted");
        let ys = if mal_sec {
            zp_mul_rss(
                net.chida.as_party_mut(),
                &mut zp_triples,
                &converted,
                &signs[..values.len()],
            )
            .unwrap()
        } else {
            zp_mul_rss(
                net.chida.as_party_mut(),
                &mut no_rec,
                &converted,
                &signs[..values.len()],
            )
            .unwrap()
        };

        if mal_sec {
            assert!(
                verify_zp_triples(
                    net.chida.as_party_mut(),
                    &mut net.broadcast_context,
                    &mut zp_triples,
                    false
                )
                .unwrap(),
                "Zp triple verification failed"
            );
        }
        net.check_view().unwrap();

        let opened = open_rss_many::<Zp>(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &ys,
        )
        .unwrap();
        net.check_view().unwrap();
        for (s, (v, o)) in values.iter().zip(&opened).enumerate() {
            let expected = (sigma_open[s] * Zp::new(*v as u64)).value();
            assert_eq!(expected, o.value(), "signed conversion mismatch at sample {}", s);
        }
        net.teardown().unwrap();
        opened.iter().map(|z| z.value()).collect()
    }
}

#[test]
fn signed_dabit_conversion_semi_honest() {
    agree3(localhost_connect(
        signed_program(false),
        signed_program(false),
        signed_program(false),
    ));
}

#[test]
fn signed_dabit_conversion_malicious() {
    agree3(localhost_connect(
        signed_program(true),
        signed_program(true),
        signed_program(true),
    ));
}

// --- LUT pipeline -> b2a end-to-end -----------------------------------------

// Same cube instance as three_party_lut.rs: u64 wrapper packs RATIO = 4
// embedded u16 values; entries are r*64 + c*8 + lay + 1 <= 220 < 2^8.
type GF = GFT<u64, u16, 4>;
const SAMPLES: usize = 8;
const CK: [usize; 3] = [2, 2, 2];
const LUT_BITS: usize = 8;

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

const CUBE: [[[u64; 1]; 4]; 4] = build_cube::<4, 4, 4>();

/// No network needed: `max_value_bits` is a local scan over the public table.
#[test]
fn max_value_bits_matches_table() {
    // largest CUBE entry is 3*64 + 3*8 + 3 + 1 = 220 -> 8 bits
    let party = LutPartyCube::<GF, 4, 4, 4, 1>::setup(false, &CK, &CUBE);
    assert_eq!(party.max_value_bits(), 8);
    // all-zero table clamps to 1 (generate_*_dabits requires bit_width >= 1)
    const ZERO: [[[u64; 1]; 4]; 4] = [[[0u64; 1]; 4]; 4];
    let party = LutPartyCube::<GF, 4, 4, 4, 1>::setup(false, &CK, &ZERO);
    assert_eq!(party.max_value_bits(), 1);
}

/// Full pipeline: daBit preprocessing (sized via `dabits_required`), oblivious
/// LUT sampling, conversion of the binary outputs to Zp, both verifications,
/// then opening in both domains — the Zp value must equal the binary integer.
fn lut_b2a_program(mal_sec: bool) -> impl FnOnce(ConnectedParty) -> Vec<u64> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let mut party = LutPartyCube::<GF, 4, 4, 4, 1>::setup(mal_sec, &CK, &CUBE);
        let mut zp_triples = MulTripleVector::<Zp>::new();
        let mut no_rec = NoMulTripleRecording;

        // daBit preprocessing, exactly enough for SAMPLES conversions.
        assert_eq!(dabits_required(LUT_BITS, SAMPLES), LUT_BITS * SAMPLES);
        let mut store: DaBitStore<ShareType<u16>> = if mal_sec {
            generate_dabits(net.chida.as_party_mut(), &mut zp_triples, LUT_BITS, SAMPLES).unwrap()
        } else {
            generate_dabits(net.chida.as_party_mut(), &mut no_rec, LUT_BITS, SAMPLES).unwrap()
        };

        // Oblivious LUT sampling as in three_party_lut.rs (zero offsets).
        party.sample_ohvs(SAMPLES, &mut net).unwrap();
        let zero = RssShare { si: ShareType(0u16), sii: ShareType(0u16) };
        let offsets: Vec<RssShareVec<ShareType<u16>>> = vec![vec![zero; 3]; SAMPLES];
        party.rotate_ohvs(&offsets, &mut net).unwrap();
        let out = party.sample_with_lut(SAMPLES, &mut net).unwrap();

        // Convert the binary LUT outputs (each < 2^LUT_BITS) to Zp.
        let ys = b2a_many(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &mut store,
            &out,
        )
        .unwrap();
        assert_eq!(store.available_samples(), 0, "store should be exhausted");

        // Verify everything before opening the results:
        // GF(2^64) triples (includes check_view in the malicious path) ...
        assert!(party.verify_triples(&mut net).unwrap(), "triple verification failed");
        // ... the Zp triples from daBit generation ...
        if mal_sec {
            assert!(
                verify_zp_triples(
                    net.chida.as_party_mut(),
                    &mut net.broadcast_context,
                    &mut zp_triples,
                    false
                )
                .unwrap(),
                "Zp triple verification failed"
            );
        }
        // ... and the views of all remaining recorded openings.
        net.check_view().unwrap();

        // Open in both domains and compare.
        let bin = party.open_many(&out, &mut net).unwrap();
        let zp = open_rss_many::<Zp>(net.chida.as_party_mut(), &mut net.broadcast_context, &ys)
            .unwrap();
        net.check_view().unwrap();
        for (b, z) in bin.iter().zip(&zp) {
            assert_eq!(b.0 as u64, z.value(), "Zp output != binary LUT output");
        }
        net.teardown().unwrap();
        zp.iter().map(|z| z.value()).collect()
    }
}

#[test]
fn lut_output_b2a_semi_honest() {
    agree3(localhost_connect(
        lut_b2a_program(false),
        lut_b2a_program(false),
        lut_b2a_program(false),
    ));
}

#[test]
fn lut_output_b2a_malicious() {
    agree3(localhost_connect(
        lut_b2a_program(true),
        lut_b2a_program(true),
        lut_b2a_program(true),
    ));
}

/// Full signed pipeline: one-sided LUT magnitudes -> `b2a_many` -> sign
/// multiplication -> the opened Zp value must be `+b` or `-b` for the opened
/// binary magnitude `b`. (With only SAMPLES = 8 signs, sign *distribution* is
/// asserted in `signed_program`, not here.)
fn lut_signed_b2a_program(mal_sec: bool) -> impl FnOnce(ConnectedParty) -> Vec<u64> + Send {
    move |conn| {
        let mut net = Network::setup(conn).unwrap();
        let mut party = LutPartyCube::<GF, 4, 4, 4, 1>::setup(mal_sec, &CK, &CUBE);
        let mut zp_triples = MulTripleVector::<Zp>::new();
        let mut no_rec = NoMulTripleRecording;

        let (mut store, signs): (DaBitStore<ShareType<u16>>, Vec<_>) = if mal_sec {
            (
                generate_dabits(net.chida.as_party_mut(), &mut zp_triples, LUT_BITS, SAMPLES)
                    .unwrap(),
                random_zp_signs::<ShareType<u16>, _>(
                    net.chida.as_party_mut(),
                    &mut zp_triples,
                    SAMPLES,
                )
                .unwrap(),
            )
        } else {
            (
                generate_dabits(net.chida.as_party_mut(), &mut no_rec, LUT_BITS, SAMPLES)
                    .unwrap(),
                random_zp_signs::<ShareType<u16>, _>(
                    net.chida.as_party_mut(),
                    &mut no_rec,
                    SAMPLES,
                )
                .unwrap(),
            )
        };

        party.sample_ohvs(SAMPLES, &mut net).unwrap();
        let zero = RssShare { si: ShareType(0u16), sii: ShareType(0u16) };
        let offsets: Vec<RssShareVec<ShareType<u16>>> = vec![vec![zero; 3]; SAMPLES];
        party.rotate_ohvs(&offsets, &mut net).unwrap();
        let out = party.sample_with_lut(SAMPLES, &mut net).unwrap();

        let converted = b2a_many(
            net.chida.as_party_mut(),
            &mut net.broadcast_context,
            &mut store,
            &out,
        )
        .unwrap();
        assert_eq!(store.available_samples(), 0, "store should be exhausted");
        let ys = if mal_sec {
            zp_mul_rss(net.chida.as_party_mut(), &mut zp_triples, &converted, &signs).unwrap()
        } else {
            zp_mul_rss(net.chida.as_party_mut(), &mut no_rec, &converted, &signs).unwrap()
        };

        assert!(party.verify_triples(&mut net).unwrap(), "triple verification failed");
        if mal_sec {
            assert!(
                verify_zp_triples(
                    net.chida.as_party_mut(),
                    &mut net.broadcast_context,
                    &mut zp_triples,
                    false
                )
                .unwrap(),
                "Zp triple verification failed"
            );
        }
        net.check_view().unwrap();

        let bin = party.open_many(&out, &mut net).unwrap();
        let zp = open_rss_many::<Zp>(net.chida.as_party_mut(), &mut net.broadcast_context, &ys)
            .unwrap();
        net.check_view().unwrap();
        for (b, z) in bin.iter().zip(&zp) {
            let mag = b.0 as u64;
            let z = z.value();
            assert!(
                z == mag || (mag != 0 && z == P - mag),
                "signed output {} is neither +{} nor -{}",
                z,
                mag,
                mag
            );
        }
        net.teardown().unwrap();
        zp.iter().map(|z| z.value()).collect()
    }
}

#[test]
fn lut_output_signed_b2a_semi_honest() {
    agree3(localhost_connect(
        lut_signed_b2a_program(false),
        lut_signed_b2a_program(false),
        lut_signed_b2a_program(false),
    ));
}

#[test]
fn lut_output_signed_b2a_malicious() {
    agree3(localhost_connect(
        lut_signed_b2a_program(true),
        lut_signed_b2a_program(true),
        lut_signed_b2a_program(true),
    ));
}

// --- three-party localhost harness ----------------------------------------

fn agree3<T: PartialEq + std::fmt::Debug>(v: (T, T, T)) {
    assert_eq!(v.0, v.1, "party 1 and 2 disagree on opened values");
    assert_eq!(v.0, v.2, "party 1 and 3 disagree on opened values");
}

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
