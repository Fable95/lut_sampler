//! Prime field `Z_p` for the Mersenne prime `p = 2^61 - 1`.
//!
//! Standalone: it implements the maestro field traits so the generic malicious
//! verification (`verify_dot_product_opt`, `verify_multiplication_triples`, ...)
//! instantiates over `Z_p` by type substitution. Not wired into the LUT pipeline.

use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

use rand::{CryptoRng, Rng};
use sha2::Digest;

use maestro::rep3_core::network::NetSerializable;
use maestro::rep3_core::party::{DigestExt, RngExt};
use maestro::rep3_core::share::{HasZero, RssShare};
use maestro::share::{Field, HasTwo, InnerProduct, Invertible};

/// The Mersenne prime `2^61 - 1`.
pub const P: u64 = (1u64 << 61) - 1;
const MASK61: u64 = (1u64 << 61) - 1;

/// Element of `Z_p`; the canonical representative is kept in `[0, P)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Zp(u64);

impl Zp {
    #[inline]
    pub fn new(x: u64) -> Self {
        Self(reduce_u64(x))
    }
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Reduce `x < 2^64` modulo `P` (uses `2^61 ≡ 1 mod P`).
#[inline]
fn reduce_u64(x: u64) -> u64 {
    let mut r = (x & MASK61) + (x >> 61);
    if r >= P {
        r -= P;
    }
    r
}

/// Reduce a product `x < 2^122` modulo `P`.
#[inline]
fn reduce_u128(x: u128) -> u64 {
    let lo = (x as u64) & MASK61;
    let mid = (x >> 61) as u64;
    let mut r = lo + mid;
    r = (r & MASK61) + (r >> 61);
    if r >= P {
        r -= P;
    }
    r
}

/// `x^(2^n)` by repeated squaring.
#[inline]
fn sqn(mut x: Zp, n: u32) -> Zp {
    for _ in 0..n {
        x = x * x;
    }
    x
}

impl HasZero for Zp {
    const ZERO: Self = Zp(0);
}

impl Add for Zp {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let s = self.0 + rhs.0;
        Zp(if s >= P { s - P } else { s })
    }
}

impl AddAssign for Zp {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Zp {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Zp(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 + P - rhs.0
        })
    }
}

impl Neg for Zp {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Zp(if self.0 == 0 { 0 } else { P - self.0 })
    }
}

impl Mul for Zp {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Zp(reduce_u128((self.0 as u128) * (rhs.0 as u128)))
    }
}

impl Field for Zp {
    const NBYTES: usize = 8;
    const ONE: Self = Zp(1);
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl HasTwo for Zp {
    const TWO: Self = Zp(2);
}

impl Invertible for Zp {
    fn inverse(self) -> Self {
        // Fermat: a^(p-2) with p-2 = 2^61 - 3 = 4*(2^59 - 1) + 1, via an
        // addition chain on run lengths (59 = 32+16+8+2+1): 60 squarings and
        // 10 multiplications instead of ~119 for square-and-multiply.
        // Constant-time; zero maps to zero.
        let t2 = sqn(self, 1) * self; // a^(2^2 - 1)
        let t4 = sqn(t2, 2) * t2;
        let t8 = sqn(t4, 4) * t4;
        let t16 = sqn(t8, 8) * t8;
        let t32 = sqn(t16, 16) * t16;
        let t48 = sqn(t32, 16) * t16;
        let t56 = sqn(t48, 8) * t8;
        let t58 = sqn(t56, 2) * t2;
        let t59 = sqn(t58, 1) * self; // a^(2^59 - 1)
        sqn(t59, 2) * self
    }
}

impl InnerProduct for Zp {
    fn inner_product(a: &[Self], b: &[Self]) -> Self {
        a.iter().zip(b).fold(Self::ZERO, |s, (x, y)| s + *x * *y)
    }

    fn weak_inner_product(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.iter().zip(b).fold(Self::ZERO, |sum, (x, y)| sum + weak_prod(x, y))
    }

    fn weak_inner_product2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.iter()
            .zip(b)
            .step_by(2)
            .fold(Self::ZERO, |sum, (x, y)| sum + weak_prod(x, y))
    }

    fn weak_inner_product3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.chunks(2).zip(b.chunks(2)).fold(Self::ZERO, |sum, (x, y)| {
            // f_k(2) for f_k(t) = x0 + (x1 - x0)*t; matches `compute_poly`.
            let x = x[0] + (x[1] - x[0]) * Self::TWO;
            let y = y[0] + (y[1] - y[0]) * Self::TWO;
            sum + weak_prod(&x, &y)
        })
    }
}

/// Local product term of two replicated shares; summing over the three parties
/// yields the true product `x * y` (valid in any field).
#[inline]
fn weak_prod(x: &RssShare<Zp>, y: &RssShare<Zp>) -> Zp {
    x.si * y.si + x.si * y.sii + x.sii * y.si
}

impl NetSerializable for Zp {
    fn serialized_size(n_elements: usize) -> usize {
        8 * n_elements
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 * len);
        for e in it {
            v.extend_from_slice(&e.borrow().0.to_le_bytes());
        }
        v
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 * elements.len());
        for e in elements {
            v.extend_from_slice(&e.0.to_le_bytes());
        }
        v
    }

    fn from_byte_vec(v: Vec<u8>, len: usize) -> Vec<Self> {
        v.chunks_exact(8)
            .take(len)
            .map(|c| Zp::new(u64::from_le_bytes(c.try_into().unwrap())))
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        for (c, d) in v.chunks_exact(8).zip(dest.iter_mut()) {
            *d = Zp::new(u64::from_le_bytes(c.try_into().unwrap()));
        }
    }
}

impl RngExt for Zp {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        for slot in buf.iter_mut() {
            // Rejection sampling over [0, P) to stay uniform.
            let v = loop {
                let cand = rng.next_u64() & MASK61;
                if cand < P {
                    break cand;
                }
            };
            *slot = Zp(v);
        }
    }
}

impl DigestExt for Zp {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for m in message {
            digest.update(m.0.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn rand_zp(rng: &mut StdRng) -> Zp {
        let mut b = [Zp::ZERO];
        Zp::fill(rng, &mut b);
        b[0]
    }

    fn ref_mul(a: u64, b: u64) -> u64 {
        (((a as u128) * (b as u128)) % (P as u128)) as u64
    }

    #[test]
    fn add_sub_mul_against_reference() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..10_000 {
            let a = rand_zp(&mut rng);
            let b = rand_zp(&mut rng);
            assert_eq!((a + b).0, (a.0 + b.0) % P);
            assert_eq!((a - b).0, ((a.0 + P) - b.0) % P);
            assert_eq!((a * b).0, ref_mul(a.0, b.0));
        }
    }

    #[test]
    fn additive_and_multiplicative_identities() {
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..10_000 {
            let a = rand_zp(&mut rng);
            assert_eq!(a + Zp::ZERO, a);
            assert_eq!(a * Zp::ONE, a);
            assert_eq!(a + (-a), Zp::ZERO);
            assert_eq!(a - a, Zp::ZERO);
        }
    }

    #[test]
    fn field_axioms_random() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..5_000 {
            let a = rand_zp(&mut rng);
            let b = rand_zp(&mut rng);
            let c = rand_zp(&mut rng);
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }

    #[test]
    fn multiplicative_inverse() {
        let mut rng = StdRng::seed_from_u64(4);
        assert_eq!(Zp::ZERO.inverse(), Zp::ZERO);
        for _ in 0..5_000 {
            let a = rand_zp(&mut rng);
            if a.is_zero() {
                continue;
            }
            assert_eq!(a * a.inverse(), Zp::ONE);
        }
    }

    // Statically guarantees Zp meets the exact bound set of the generic
    // verification (`verify_dot_product_opt`), i.e. it drops in by type substitution.
    #[test]
    fn satisfies_verification_bounds() {
        fn assert_bounds<F: Field + DigestExt + HasTwo + Invertible + InnerProduct + Send + Sync>() {}
        assert_bounds::<Zp>();
    }

    #[test]
    fn constants() {
        assert_eq!(Zp::ONE + Zp::ONE, Zp::TWO);
        assert_ne!(Zp::TWO, Zp::ZERO);
        assert_ne!(Zp::TWO, Zp::ONE);
        assert!(Zp::ZERO.is_zero());
        assert_eq!(Zp::NBYTES, 8);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut rng = StdRng::seed_from_u64(5);
        let vals: Vec<Zp> = (0..257).map(|_| rand_zp(&mut rng)).collect();
        let bytes = Zp::as_byte_vec(vals.iter(), vals.len());
        assert_eq!(bytes.len(), Zp::serialized_size(vals.len()));
        let back = Zp::from_byte_vec(bytes, vals.len());
        assert_eq!(vals, back);
    }

    #[test]
    fn rng_in_range() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut buf = vec![Zp::ZERO; 100_000];
        Zp::fill(&mut rng, &mut buf);
        assert!(buf.iter().all(|z| z.0 < P));
    }

    // Splits each plaintext into a 3-party replicated sharing (party j holds
    // (v_j, v_{j+1})) and checks that the three local weak inner products sum to
    // the true inner product -- the invariant the verification relies on.
    #[test]
    fn weak_inner_product_sums_to_inner_product() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 64;
        let x: Vec<Zp> = (0..n).map(|_| rand_zp(&mut rng)).collect();
        let y: Vec<Zp> = (0..n).map(|_| rand_zp(&mut rng)).collect();

        let split = |v: &[Zp], rng: &mut StdRng| -> [Vec<RssShare<Zp>>; 3] {
            let mut p = [Vec::new(), Vec::new(), Vec::new()];
            for val in v {
                let s0 = rand_zp(rng);
                let s1 = rand_zp(rng);
                let s2 = *val - s0 - s1;
                let sh = [s0, s1, s2];
                for j in 0..3 {
                    p[j].push(RssShare::from(sh[j], sh[(j + 1) % 3]));
                }
            }
            p
        };

        let xs = split(&x, &mut rng);
        let ys = split(&y, &mut rng);

        let mut sum = Zp::ZERO;
        for j in 0..3 {
            sum += Zp::weak_inner_product(&xs[j], &ys[j]);
        }
        assert_eq!(sum, Zp::inner_product(&x, &y));
    }
}
