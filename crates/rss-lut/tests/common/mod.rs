// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid

//! Three-party localhost harness shared by the integration tests: TLS
//! certificates from `keys/`, three real party indices on ephemeral ports,
//! each running its program in its own thread.
#![allow(dead_code)]

use std::{
    fs::File,
    io::BufReader,
    net::{IpAddr, Ipv4Addr},
    path::PathBuf,
    str::FromStr,
    thread,
};

use maestro::rep3_core::network::{Config, ConnectedParty, CreatedParty};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

pub fn agree3<T: PartialEq + std::fmt::Debug>(v: (T, T, T)) {
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
        let mut reader = BufReader::new(
            File::open(key_path(name)).unwrap_or_else(|_| panic!("Cannot open {}", name)),
        );
        rustls_pemfile::private_key(&mut reader)
            .unwrap_or_else(|_| panic!("Cannot read private key in {}", name))
            .unwrap_or_else(|| panic!("No private key in {}", name))
    }

    fn load_cert(name: &str) -> CertificateDer<'static> {
        let mut reader = BufReader::new(
            File::open(key_path(name)).unwrap_or_else(|_| panic!("Cannot open {}", name)),
        );
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

pub fn localhost_connect<
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
    let addr: Vec<Ipv4Addr> = (0..3)
        .map(|_| Ipv4Addr::from_str("127.0.0.1").unwrap())
        .collect();
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
            let addr: Vec<Ipv4Addr> = (0..3)
                .map(|_| Ipv4Addr::from_str("127.0.0.1").unwrap())
                .collect();
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
            let addr: Vec<Ipv4Addr> = (0..3)
                .map(|_| Ipv4Addr::from_str("127.0.0.1").unwrap())
                .collect();
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
