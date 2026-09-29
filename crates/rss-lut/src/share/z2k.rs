// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
//! Integer ring `Z_{2^K}` for `1 <= K <= 64`.
//!
//! Arithmetic is two's-complement wrapping, reduced to the low `K` bits; the
//! canonical representative is kept in `[0, 2^K)`. Elements serialize to
//! `ceil(K/8)` little-endian bytes, so the network cost of a sharing is the
//! byte-rounded ring width (choose `K` as a multiple of 8 for exact accounting).
//!
//! It implements the maestro element traits, so `RssShare<Z2k<K>>` supports
//! local linear operations, `generate_random`, `generate_alpha`, `constant`,
//! `open_rss` and the generic multiplication [`crate::online::mul_rss`].
//!
//! `Z_{2^K}` is a ring, not a field: most elements have no inverse, so it
//! deliberately implements neither `Invertible` nor `InnerProduct`. The
//! field-based malicious checks (`verify_multiplication_triples`,
//! `verify_dot_product_opt`) are not sound over it and must not be
//! instantiated with this type; use it in the semi-honest setting (pass
//! `NoMulTripleRecording`) unless a ring-specific check is added.

use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::{CryptoRng, Rng};
use sha2::Digest;

use maestro::rep3_core::network::NetSerializable;
use maestro::rep3_core::party::{DigestExt, RngExt};
use maestro::rep3_core::share::HasZero;
use maestro::share::{Field, HasTwo};

/// Element of `Z_{2^K}`; the canonical representative is kept in `[0, 2^K)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Z2k<const K: u32>(u64);

/// `Z_{2^8}`.
pub type Z2p8 = Z2k<8>;
/// `Z_{2^16}`.
pub type Z2p16 = Z2k<16>;
/// `Z_{2^32}`.
pub type Z2p32 = Z2k<32>;
/// `Z_{2^64}`.
pub type Z2p64 = Z2k<64>;

impl<const K: u32> Z2k<K> {
    /// Rejects `K = 0` and `K > 64` at compile time wherever the type is used.
    const VALID: () = assert!(K >= 1 && K <= 64, "Z2k requires 1 <= K <= 64");
    /// Mask of the low `K` bits.
    pub const MASK: u64 = if K >= 64 { u64::MAX } else { (1u64 << K) - 1 };
    /// Serialized size in bytes, `ceil(K/8)`.
    const BYTES: usize = K.div_ceil(8) as usize;

    /// Reduces `x` modulo `2^K`.
    #[inline]
    pub const fn new(x: u64) -> Self {
        #[allow(clippy::let_unit_value)]
        let _ = Self::VALID;
        Self(x & Self::MASK)
    }

    /// Canonical representative in `[0, 2^K)`.
    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }

    /// The representative interpreted in two's complement, in
    /// `[-2^(K-1), 2^(K-1))`.
    #[inline]
    pub const fn signed_value(self) -> i64 {
        let shift = 64 - K;
        ((self.0 << shift) as i64) >> shift
    }
}

impl<const K: u32> From<u64> for Z2k<K> {
    #[inline]
    fn from(x: u64) -> Self {
        Self::new(x)
    }
}

impl<const K: u32> HasZero for Z2k<K> {
    const ZERO: Self = Z2k(0);
}

impl<const K: u32> Add for Z2k<K> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0.wrapping_add(rhs.0))
    }
}

impl<const K: u32> AddAssign for Z2k<K> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const K: u32> Sub for Z2k<K> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.0.wrapping_sub(rhs.0))
    }
}

impl<const K: u32> SubAssign for Z2k<K> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const K: u32> Neg for Z2k<K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(self.0.wrapping_neg())
    }
}

impl<const K: u32> Mul for Z2k<K> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.0.wrapping_mul(rhs.0))
    }
}

impl<const K: u32> MulAssign for Z2k<K> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const K: u32> Field for Z2k<K> {
    const NBYTES: usize = Self::BYTES;
    const NBITS: usize = K as usize;
    const ONE: Self = Z2k(1);
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl<const K: u32> HasTwo for Z2k<K> {
    /// The integer 2 (zero for `K = 1`).
    const TWO: Self = Z2k(2 & Self::MASK);
}

impl<const K: u32> NetSerializable for Z2k<K> {
    fn serialized_size(n_elements: usize) -> usize {
        Self::BYTES * n_elements
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(Self::BYTES * len);
        for e in it {
            v.extend_from_slice(&e.borrow().0.to_le_bytes()[..Self::BYTES]);
        }
        v
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        Self::as_byte_vec(elements.iter(), elements.len())
    }

    fn from_byte_vec(v: Vec<u8>, len: usize) -> Vec<Self> {
        v.chunks_exact(Self::BYTES)
            .take(len)
            .map(Self::from_le_chunk)
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        for (c, d) in v.chunks_exact(Self::BYTES).zip(dest.iter_mut()) {
            *d = Self::from_le_chunk(c);
        }
    }
}

impl<const K: u32> Z2k<K> {
    /// Decodes one `ceil(K/8)`-byte little-endian chunk; bits above `K` are
    /// dropped, so malformed input still yields a canonical element.
    #[inline]
    fn from_le_chunk(c: &[u8]) -> Self {
        let mut b = [0u8; 8];
        b[..Self::BYTES].copy_from_slice(c);
        Self::new(u64::from_le_bytes(b))
    }
}

impl<const K: u32> RngExt for Z2k<K> {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        // Masking a uniform u64 is uniform over Z_{2^K}: no rejection needed.
        for slot in buf.iter_mut() {
            *slot = Self::new(rng.next_u64());
        }
    }
}

impl<const K: u32> DigestExt for Z2k<K> {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for m in message {
            digest.update(&m.0.to_le_bytes()[..Self::BYTES]);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use maestro::rep3_core::share::RssShare;
    use rand::{rngs::StdRng, SeedableRng};

    fn rand_elem<const K: u32>(rng: &mut StdRng) -> Z2k<K> {
        let mut b = [Z2k::<K>::ZERO];
        Z2k::<K>::fill(rng, &mut b);
        b[0]
    }

    /// Exact reference arithmetic modulo 2^K in u128.
    fn ref_mod<const K: u32>(x: u128) -> u64 {
        (x % (1u128 << K)) as u64
    }

    fn ring_ops_against_reference<const K: u32>(seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let m = 1u128 << K;
        for _ in 0..10_000 {
            let a = rand_elem::<K>(&mut rng);
            let b = rand_elem::<K>(&mut rng);
            let (x, y) = (a.0 as u128, b.0 as u128);
            assert!(x < m && y < m);
            assert_eq!((a + b).0, ref_mod::<K>(x + y));
            assert_eq!((a - b).0, ref_mod::<K>(x + m - y));
            assert_eq!((-a).0, ref_mod::<K>(m - x));
            assert_eq!((a * b).0, ref_mod::<K>(x * y));
            let mut c = a;
            c += b;
            c -= b;
            c *= Z2k::ONE;
            assert_eq!(c, a);
        }
    }

    #[test]
    fn ring_ops_all_standard_widths() {
        ring_ops_against_reference::<1>(1);
        ring_ops_against_reference::<7>(2);
        ring_ops_against_reference::<8>(3);
        ring_ops_against_reference::<13>(4);
        ring_ops_against_reference::<16>(5);
        ring_ops_against_reference::<32>(6);
        ring_ops_against_reference::<63>(7);
        ring_ops_against_reference::<64>(8);
    }

    #[test]
    fn exhaustive_small_ring() {
        // Z_{2^4}: every pair against integer arithmetic.
        for x in 0..16u64 {
            for y in 0..16u64 {
                let (a, b) = (Z2k::<4>::new(x), Z2k::<4>::new(y));
                assert_eq!((a + b).0, (x + y) % 16);
                assert_eq!((a - b).0, (x + 16 - y) % 16);
                assert_eq!((a * b).0, (x * y) % 16);
            }
        }
    }

    #[test]
    fn ring_axioms_random() {
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..5_000 {
            let a = rand_elem::<16>(&mut rng);
            let b = rand_elem::<16>(&mut rng);
            let c = rand_elem::<16>(&mut rng);
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
            assert_eq!(a + (-a), Z2p16::ZERO);
        }
    }

    #[test]
    fn constants_and_widths() {
        assert_eq!(Z2p16::ONE + Z2p16::ONE, Z2p16::TWO);
        assert_eq!(Z2k::<1>::TWO, Z2k::<1>::ZERO);
        assert_eq!(<Z2p16 as Field>::NBYTES, 2);
        assert_eq!(<Z2p16 as Field>::NBITS, 16);
        assert_eq!(<Z2k<12> as Field>::NBYTES, 2);
        assert_eq!(<Z2k<12> as Field>::NBITS, 12);
        assert_eq!(<Z2p64 as Field>::NBYTES, 8);
        assert_eq!(Z2p8::new(0x1ff).value(), 0xff);
        assert_eq!(Z2p64::new(u64::MAX).value(), u64::MAX);
        assert_eq!(Z2p16::new(0xffff).signed_value(), -1);
        assert_eq!(Z2p16::new(0x7fff).signed_value(), 0x7fff);
        assert_eq!(Z2p16::new(0x8000).signed_value(), -0x8000);
        assert_eq!(Z2p64::new(u64::MAX).signed_value(), -1);
        assert!(Z2p32::ZERO.is_zero());
    }

    fn serialization_roundtrip<const K: u32>(seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let vals: Vec<Z2k<K>> = (0..257).map(|_| rand_elem::<K>(&mut rng)).collect();
        let bytes = Z2k::<K>::as_byte_vec(vals.iter(), vals.len());
        assert_eq!(bytes.len(), Z2k::<K>::serialized_size(vals.len()));
        assert_eq!(bytes, Z2k::<K>::as_byte_vec_slice(&vals));
        assert_eq!(vals, Z2k::<K>::from_byte_vec(bytes.clone(), vals.len()));
        let mut dest = vec![Z2k::<K>::ZERO; vals.len()];
        Z2k::<K>::from_byte_slice(bytes, &mut dest);
        assert_eq!(vals, dest);
    }

    #[test]
    fn serialization_roundtrips() {
        serialization_roundtrip::<5>(10);
        serialization_roundtrip::<8>(11);
        serialization_roundtrip::<16>(12);
        serialization_roundtrip::<24>(13);
        serialization_roundtrip::<32>(14);
        serialization_roundtrip::<64>(15);
    }

    #[test]
    fn deserialization_is_canonical() {
        // A 12-bit element occupies 2 bytes; stray high bits are dropped.
        let back = Z2k::<12>::from_byte_vec(vec![0xff, 0xff], 1);
        assert_eq!(back[0].value(), 0xfff);
    }

    #[test]
    fn rng_in_range_and_covers_ring() {
        let mut rng = StdRng::seed_from_u64(16);
        let mut buf = vec![Z2k::<3>::ZERO; 10_000];
        Z2k::<3>::fill(&mut rng, &mut buf);
        assert!(buf.iter().all(|z| z.0 < 8));
        for v in 0..8 {
            assert!(buf.iter().any(|z| z.0 == v), "value {} never drawn", v);
        }
    }

    // Three-party replicated sharing (party j holds (s_j, s_{j+1})): the local
    // product terms s_j t_j + s_j t_{j+1} + s_{j+1} t_j of the three parties
    // sum to the product -- the invariant behind `online::mul_rss`.
    #[test]
    fn replicated_local_products_sum_to_product() {
        let mut rng = StdRng::seed_from_u64(17);
        for _ in 0..1_000 {
            let x = rand_elem::<16>(&mut rng);
            let y = rand_elem::<16>(&mut rng);
            let share = |v: Z2p16, rng: &mut StdRng| {
                let s0 = rand_elem::<16>(rng);
                let s1 = rand_elem::<16>(rng);
                let s = [s0, s1, v - s0 - s1];
                [0, 1, 2].map(|j| RssShare::from(s[j], s[(j + 1) % 3]))
            };
            let (xs, ys) = (share(x, &mut rng), share(y, &mut rng));
            let mut sum = Z2p16::ZERO;
            for j in 0..3 {
                sum += xs[j].si * ys[j].si + xs[j].si * ys[j].sii + xs[j].sii * ys[j].si;
            }
            assert_eq!(sum, x * y);
        }
    }

    #[test]
    fn satisfies_rss_element_bounds() {
        fn assert_bounds<F: Field + DigestExt + HasTwo + Send + Sync>() {}
        assert_bounds::<Z2p16>();
        assert_bounds::<Z2k<21>>();
    }
}
