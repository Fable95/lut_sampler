//! This module implements the 64-bit finite field `GF(2^64)`.
//!
//! The field modulus is `X^64 + X^4 + X^3 + X + 1`.
//! Field elements are represented by the [GF2p64] data type.
//!
//! On supported hardware the feature `clmul` provides faster field multiplication using CLMUL CPU instructions.
use std::{
    borrow::Borrow,
    fmt::{Debug, Formatter},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub},
};

use itertools::Itertools;
use rand::{CryptoRng, Rng};
use maestro::rep3_core::{network::NetSerializable, party::{DigestExt, RngExt}, share::{HasZero, RssShare}};
use sha2::Digest;

use maestro::share::{
    bs_bool16::BsBool16, gf4::GF4, gf8::GF8, Field, HasTwo, InnerProduct, Invertible
};

/// An element of `GF(2^64) := GF(2)[X] / X^64 + X^4 + X^3 + X + 1`
///
/// An element is represented as an 64-bit vector in the form of a [u64].
#[derive(Copy, Clone, Default, PartialEq)]
pub struct GF2p64(u64);

impl GF2p64 {
    /// Returns a new field element from a bit-vector.
    pub fn new<T>(x: T) -> Self
    where
        u64: From<T>,
    {
        GF2p64(u64::from(x))
    }

    /// A 64-bit vector representing the irreducible polynomial `X^64 + X^4 + X^3 + X + 1`.
    //const MODULUS: u64 = 0x80_00_00_00_00_00_00_1B;

    /// MODULUS - X^64 = X^4+X^3+X+1
    const MOD_MINUS_XN: u64 = 0x00_00_00_00_00_00_00_1B;

    /// Multiply an element by `X`` (i.e. by `0x0000000000000002`)
    /// This is essentially a left shift of the bit-vector mod MODULUS
    fn mul_by_x(&mut self) {
        let high_bit = self.0 >> (Self::NBITS - 1);
        self.0 <<= 1;
        if high_bit != 0 {
            self.0 ^= Self::MOD_MINUS_XN;
        }
    }

    /// Slow multiplication.
    ///
    /// Uses addition as a fallback in case the `cmul` feature is not supported.
    #[deprecated(note = "Do not use directly, only exposed for benchmarking purposes.")]
    pub fn mul_using_add(mut self, other: &Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..Self::NBITS {
            if other.0 & (1 << i) != 0 {
                result.0 ^= self.0
            }
            self.mul_by_x()
        }
        result
    }

    /// Carry propagation for CLMUL (cf. <https://github.com/gendx/horcrux/blob/main/horcrux/src/gf2n.rs>)
    /// Requires the `clmul` feature.
    #[cfg(any(
        test,
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    fn propagate_carries(mut word: u64, carry: u64) -> Self {
        let mut c = carry;
        while c != 0 {
            word ^= c ^ (c << 4) ^ (c << 3) ^ (c << 1);
            c = (c >> (Self::NBITS - 4)) ^ (c >> (Self::NBITS - 3)) ^ (c >> (Self::NBITS - 1))
        }
        Self(word)
    }


    /// Carry propagation for CLMUL (cf. <https://github.com/gendx/horcrux/blob/main/horcrux/src/gf2n.rs>)
    /// Requires the `clmul` feature.
    #[cfg(all(
        feature = "clmul",
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    fn propagate_carry_fast(word: u64, carry: u64) -> Self {
        use std::arch::aarch64::vmull_p64;
        unsafe{
            let c1 = vmull_p64(carry,GF2p64::MOD_MINUS_XN);
            let carry = (carry >> (Self::NBITS - 4)) ^ (carry >> (Self::NBITS - 3)) ^ (carry >> (Self::NBITS - 1));
            let c2 = vmull_p64(carry,GF2p64::MOD_MINUS_XN);
            Self(word ^ c1 as u64 ^ c2 as u64)
        }
    }

    /// Carry propagation for CLMUL (cf. <https://github.com/gendx/horcrux/blob/main/horcrux/src/gf2n.rs>)
    /// Requires the `clmul` feature.
    #[cfg(all(
        feature = "clmul",
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "pclmulqdq"
    ))]
    fn propagate_carry_fast(word: u64, carry: u64) -> Self {
        use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x, _mm_storeu_si128, _mm_xor_si128};
        unsafe{
            let carry2 = (carry >> (Self::NBITS - 4)) ^ (carry >> (Self::NBITS - 3)) ^ (carry >> (Self::NBITS - 1));
            let poly = _mm_set_epi64x(0, GF2p64::MOD_MINUS_XN as i64);
            let carry = _mm_set_epi64x(carry2 as i64, carry as i64);
            let w = _mm_xor_si128(_mm_clmulepi64_si128(carry, poly, 0),_mm_clmulepi64_si128(carry, poly, 1));
            let mut res: [u64; 2] = [0u64, 0u64];
            _mm_storeu_si128(&mut res as *mut _ as *mut __m128i, w);
            Self(word ^ res[0])
        }
    }

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    pub fn mul_clmul_u64_fast(&self, other: &Self) -> Self {
        let (word, carry) = Self::clmul_u64(self, other);
        Self::propagate_carry_fast(word, carry)
    }

    /// Carryless multiplication for x86 architecture, requires the `clmul` feature.
    #[cfg(all(
        feature = "clmul",
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "pclmulqdq"
    ))]
    fn clmul_u64(&self, other: &Self) -> (u64, u64) {
        use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x, _mm_storeu_si128};

        let x: __m128i = unsafe { _mm_set_epi64x(0, self.0 as i64) };
        let y: __m128i = unsafe { _mm_set_epi64x(0, other.0 as i64) };
        let clmul: __m128i = unsafe { _mm_clmulepi64_si128(x, y, 0) };
        let mut cc: [u64; 2] = [0u64, 0u64];
        unsafe { _mm_storeu_si128(&mut cc as *mut _ as *mut __m128i, clmul) };

        let word = cc[0];
        let carry = cc[1];
        (word, carry)
    }

    /// Carryless multiplication for ARM architecture, requires the `clmul` feature.
    #[cfg(all(
        feature = "clmul",
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    fn clmul_u64(&self, other: &Self) -> (u64, u64) {
        use std::arch::aarch64::vmull_p64;
        let clmul: u128 = unsafe { vmull_p64(self.0, other.0) };
        let word = clmul as u64;
        let carry = (clmul >> 64) as u64;
        (word, carry)
    }

    /// Multiplication using CLMUL, requires the `clmul` feature.
    ///
    /// Implemented along the lines of <https://github.com/gendx/horcrux/blob/main/horcrux/src/gf2n.rs>.
    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    pub fn mul_clmul_u64(&self, other: &Self) -> Self {
        let (word, carry) = Self::clmul_u64(self, other);
        Self::propagate_carries(word, carry)
    }

    /// Inner product using multiplication
    ///
    /// This acts as a fallback in case the `clmul` feature is not supported.
    #[deprecated(note = "Do not use directly, only exposed for benchmarking purposes.")]
    pub fn fallback_inner_product(a: &[Self], b: &[Self]) -> Self {
        a.iter().zip(b).fold(Self::ZERO, |s, (a, b)| s + *a * *b)
    }

    /// Inner product using CLMUL with delayed carry propagation.
    ///
    /// Requires the `clmul` feature.
    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    pub fn clmul_inner_product(a: &[Self], b: &[Self]) -> Self {
        let (word, carry) = a.iter().zip(b).fold((0u64, 0u64), |(wrd, car), (a, b)| {
            let (w, c) = Self::clmul_u64(a, b);
            (wrd ^ w, car ^ c)
        });
        Self::propagate_carries(word, carry)
    }

    #[cfg(all(
        feature = "clmul",
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    pub fn fast_clmul_inner_product(a: &[Self], b: &[Self]) -> Self {
        use std::arch::aarch64::{vaddq_p128, vmull_p64};
        let wrdcar =  a.iter().zip(b).fold(0, |wrdcar, (a, b)| {
            unsafe { 
                let res = vmull_p64(a.0, b.0);
                vaddq_p128(wrdcar,res)
            }
        });
        let word = wrdcar as u64;
        let carry = (wrdcar >> 64) as u64;        
        Self::propagate_carry_fast(word, carry)
    }

    #[cfg(all(
        feature = "clmul",
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "pclmulqdq"
    ))]
    pub fn fast_clmul_inner_product(a: &[Self], b: &[Self]) -> Self {
        use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x, _mm_storeu_si128, _mm_setzero_si128, _mm_xor_si128};
        let wrdcar: __m128i = unsafe { a.iter().zip(b).fold(_mm_setzero_si128(), |wrdcar, (a, b)| {
                let x = _mm_set_epi64x(0, a.0 as i64);
                let y = _mm_set_epi64x(0, b.0 as i64);
                let res = _mm_clmulepi64_si128(x, y, 0);
                _mm_xor_si128(wrdcar, res)
        })};
        let mut cc: [u64; 2] = [0u64, 0u64];
        unsafe { _mm_storeu_si128(&mut cc as *mut _ as *mut __m128i, wrdcar) };
        let word = cc[0];
        let carry = cc[1];  
        Self::propagate_carries(word, carry)
    }

    /// Weak inner product using basic multiplication.
    ///
    /// This acts as a fallback in case the `clmul` feature is not supported.
    /// See [Self::clmul_weak_inner_product] for more details.
    #[deprecated(note = "Do not use directly, only exposed for benchmarking purposes.")]
    pub fn fallback_weak_ip(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.iter().zip(b).fold(Self::ZERO, |sum, (x, y)| {
            sum + x.si * y.si + (x.si + x.sii) * (y.si + y.sii)
        })
    }

    /// Weak inner product2 using basic multiplication
    /// This acts as a fallback in case the `clmul` feature is not supported.
    /// See [Self::clmul_weak_inner_product2] for more details.
    #[deprecated(note = "Do not use directly, only exposed for benchmarking purposes.")]
    pub fn fallback_weak_ip2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.iter().zip(b).step_by(2).fold(Self::ZERO, |sum, (x, y)| {
            sum + x.si * y.si + (x.si + x.sii) * (y.si + y.sii)
        })
    }

    /// Weak inner product3 using basic multiplication
    /// This acts as a fallback in case the `clmul` feature is not supported.
    /// See [Self::clmul_weak_inner_product3] for more details.
    #[deprecated(note = "Do not use directly, only exposed for benchmarking purposes.")]
    pub fn fallback_weak_ip3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        a.chunks(2).zip(b.chunks(2)).fold(Self::ZERO, |sum, (x, y)| {
            let x = x[0] + (x[0]+x[1]) * Self::TWO;
            let y = y[0] + (y[0]+y[1]) * Self::TWO;
            sum + x.si * y.si + (x.si + x.sii) * (y.si + y.sii)
        })
    }

    /// Weak inner product using CLMUL with delayed carry propagation.
    ///
    /// This function computes the inner product of two RSS-shared vectors.
    /// The result is sum-shared.
    /// Requires the `clmul` feature.
    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    pub fn clmul_weak_inner_product(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        let (word, carry) = a.iter().zip(b).fold((0u64, 0u64), |(wrd, car), (a, b)| {
            let (w1, c1) = Self::clmul_u64(&a.si, &b.si);
            let (w2, c2) = Self::clmul_u64(&(a.si + a.sii), &(b.si + b.sii));
            (wrd ^ w1 ^ w2, car ^ c1 ^ c2)
        });
        Self::propagate_carries(word, carry)
    }

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    pub fn clmul_weak_inner_product2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        let (word, carry) = a.iter().zip(b).step_by(2).fold((0u64, 0u64), |(wrd, car), (a, b)| {
            let (w1, c1) = Self::clmul_u64(&a.si, &b.si);
            let (w2, c2) = Self::clmul_u64(&(a.si + a.sii), &(b.si + b.sii));
            (wrd ^ w1 ^ w2, car ^ c1 ^ c2)
        });
        Self::propagate_carries(word, carry)
    }

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    pub fn clmul_weak_inner_product3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        let (word, carry) = a.chunks(2).zip(b.chunks(2)).fold((0u64, 0u64), |(wrd, car), (a, b)| {
            let a = a[0] + (a[0]+a[1])*Self::TWO;
            let b = b[0] + (b[0]+b[1])*Self::TWO;
            let (w1, c1) = Self::clmul_u64(&a.si, &b.si);
            let (w2, c2) = Self::clmul_u64(&(a.si + a.sii), &(b.si + b.sii));
            (wrd ^ w1 ^ w2, car ^ c1 ^ c2)
        });
        Self::propagate_carries(word, carry)
    }

    #[inline]
    pub fn extend_from_bit(vec: &mut Vec<Self>, bit: &BsBool16) {
        let mut x = bit.as_u16();
        for _ in 0..16 {
            vec.push(Self::new(x & 1));
            x >>= 1;
        }
    }

    #[inline]
    pub fn extend_from_bitstring(vec: &mut Vec<Self>, bitstring: &[BsBool16], simd_len: usize) {
        debug_assert!(bitstring.len() <= Self::NBITS * simd_len);
        let mut transposed = vec![Self::ZERO; 16*simd_len];
        bitstring.chunks_exact(simd_len).enumerate().for_each(|(bitpos, simd_bit)| {
            simd_bit.iter().enumerate().for_each(|(j,bit)| {
                let mut bit = bit.as_u16() as u64;
                for i in 0..16 {
                    transposed[16*j+i].0 |= (bit & 1) << bitpos;
                    bit >>= 1;
                }
            })
        });
        vec.extend_from_slice(&transposed);
    }
}

impl Field for GF2p64 {
    const NBYTES: usize = 8;

    const ONE: Self = Self(1u64);

    fn is_zero(&self) -> bool {
        self.0 == 0u64
    }
}

impl HasZero for GF2p64 {
    const ZERO: Self = Self(0u64);
}

impl NetSerializable for GF2p64 {
    fn as_byte_vec(it: impl IntoIterator<Item = impl Borrow<Self>>, _len: usize) -> Vec<u8> {
        // Using big-endian encoding
        it.into_iter()
            .flat_map(|gf| gf.borrow().0.to_be_bytes())
            .collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        // Using big-endian encoding
        let mut res = vec![0u8; Self::serialized_size(elements.len())];
        res.chunks_exact_mut(8).zip_eq(elements).for_each(|(chunk, el)| {
            chunk.copy_from_slice(&el.0.to_be_bytes());
        });
        res
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        debug_assert!(v.len() % Self::NBYTES == 0);
        v.into_iter()
            .chunks(Self::NBYTES)
            .into_iter()
            .map(|c| {
                let x = u64::from_be_bytes(
                    c.collect::<Vec<u8>>()
                        .try_into()
                        .expect("chunk with incorrect length"),
                );
                Self(x)
            })
            .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        debug_assert_eq!(v.len(), Self::NBYTES * dest.len());
        dest.iter_mut()
            .zip(v.chunks(Self::NBYTES))
            .for_each(|(dst, c)| {
                dst.0 = u64::from_be_bytes(c.try_into().expect("chunk with incorrect length"));
            })
    }

    fn serialized_size(n_elements: usize) -> usize {
        n_elements * Self::NBYTES
    }
}

impl Invertible for GF2p64 {
    /// Multiplicative inverse
    fn inverse(self) -> Self {
        if self == Self::ZERO {
            return self;
        }
        let mut p = self;
        // Compute x^(2^n - 2)
        let mut result = Self::ONE;
        for _ in 1..Self::NBITS {
            p = p * p;
            result *= p;
        }
        result
    }
}

impl HasTwo for GF2p64 {
    const TWO: Self = Self(2u64);
}

impl Add for GF2p64 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl AddAssign for GF2p64 {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Sub for GF2p64 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

/// Additive Inverse
impl Neg for GF2p64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}

impl Mul for GF2p64 {
    type Output = Self;

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul_clmul_u64(&self, &rhs)
    }

    #[cfg(not(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    )))]
    fn mul(self, rhs: Self) -> Self::Output {
        //Fall back
        self.mul_using_add(&rhs)
    }
}

impl MulAssign for GF2p64 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

#[cfg(not(any(
    all(
        feature = "clmul",
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "pclmulqdq"
    ),
    all(
        feature = "clmul",
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    )
)))]
impl InnerProduct for GF2p64 {
    fn inner_product(a: &[Self], b: &[Self]) -> Self {
        Self::fallback_inner_product(a, b)
    }

    fn weak_inner_product(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        Self::fallback_weak_ip(a, b)
    }

    fn weak_inner_product2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        Self::fallback_weak_ip2(a, b)
    }

    fn weak_inner_product3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        Self::fallback_weak_ip3(a, b)
    }
}

#[cfg(any(
    all(
        feature = "clmul",
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "pclmulqdq"
    ),
    all(
        feature = "clmul",
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    )
))]
impl InnerProduct for GF2p64 {
    fn inner_product(a: &[Self], b: &[Self]) -> Self {
        Self::clmul_inner_product(a, b)
    }

    fn weak_inner_product(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        Self::clmul_weak_inner_product(a, b)
    }

    fn weak_inner_product2(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        Self::clmul_weak_inner_product2(a, b)
    }

    fn weak_inner_product3(a: &[RssShare<Self>], b: &[RssShare<Self>]) -> Self {
        Self::clmul_weak_inner_product3(a, b)
    }
}

pub struct GF2p64InnerProd {
    #[cfg(not(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    )))]
    el: GF2p64,
    #[cfg(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    ))]
    word: u64,
    #[cfg(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    ))]
    carry: u64,
}

impl GF2p64InnerProd {
    #[cfg(not(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    )))]
    pub fn new() -> Self {
        Self { el: GF2p64::ZERO }
    }

    #[cfg(not(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    )))]
    pub fn add_prod(&mut self, a: &GF2p64, b: &GF2p64) {
        self.el += *a * *b;
    }

    #[cfg(not(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    )))]
    pub fn sum(self) -> GF2p64 {
        self.el
    }

    #[cfg(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    ))]
    pub fn new() -> Self {
        Self { word: 0u64, carry: 0u64 }
    }

    #[cfg(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    ))]
    pub fn add_prod(&mut self, a: &GF2p64, b: &GF2p64) {
        let (w, c) = GF2p64::clmul_u64(a, b);
        self.word ^= w;
        self.carry ^= c;
    }

    #[cfg(any(
        all(feature = "clmul",target_arch = "x86_64", target_feature = "sse2", target_feature = "pclmulqdq"),
        all(feature = "clmul", target_arch = "aarch64", target_feature = "neon", target_feature = "aes")
    ))]
    pub fn sum(self) -> GF2p64 {
        GF2p64::propagate_carries(self.word, self.carry)
    }
}

impl Debug for GF2p64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "GF2p64(0x{:016x})", self.0)
    }
}

impl RngExt for GF2p64 {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let mut v = vec![0u8; buf.len() * GF2p64::NBYTES];
        rng.fill_bytes(&mut v);
        Self::from_byte_slice(v, buf)
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0; n * GF2p64::NBYTES];
        rng.fill_bytes(&mut r);
        GF2p64::from_byte_vec(r, n)
    }
}

impl DigestExt for GF2p64 {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        for x in message {
            digest.update(x.0.to_be_bytes());
        }
    }
}

/// Field that can be embedded into `GF(2^64)`.
pub trait GF2p64Subfield: Field {
    /// Embeds a field element into `GF(2^64)`.
    fn embed(self) -> GF2p64;
}

#[rustfmt::skip]
pub const GF8_EB_TABLE: [u64; 256] = [
    0x0000000000000000, 0x0000000000000001, 0x033ce8beddc8a656, 0x033ce8beddc8a657, 
    0x512620375ed2a108, 0x512620375ed2a109, 0x521ac889831a075e, 0x521ac889831a075f, 
    0x0c9e636090aafc01, 0x0c9e636090aafc00, 0x0fa28bde4d625a57, 0x0fa28bde4d625a56, 
    0x5db84357ce785d09, 0x5db84357ce785d08, 0x5e84abe913b0fb5f, 0x5e84abe913b0fb5e, 
    0xba4f3cd82801769c, 0xba4f3cd82801769d, 0xb973d466f5c9d0ca, 0xb973d466f5c9d0cb, 
    0xeb691cef76d3d794, 0xeb691cef76d3d795, 0xe855f451ab1b71c2, 0xe855f451ab1b71c3, 
    0xb6d15fb8b8ab8a9d, 0xb6d15fb8b8ab8a9c, 0xb5edb70665632ccb, 0xb5edb70665632cca, 
    0xe7f77f8fe6792b95, 0xe7f77f8fe6792b94, 0xe4cb97313bb18dc3, 0xe4cb97313bb18dc2, 
    0xba26e7904adb4a47, 0xba26e7904adb4a46, 0xb91a0f2e9713ec11, 0xb91a0f2e9713ec10, 
    0xeb00c7a71409eb4f, 0xeb00c7a71409eb4e, 0xe83c2f19c9c14d19, 0xe83c2f19c9c14d18, 
    0xb6b884f0da71b646, 0xb6b884f0da71b647, 0xb5846c4e07b91010, 0xb5846c4e07b91011, 
    0xe79ea4c784a3174e, 0xe79ea4c784a3174f, 0xe4a24c79596bb118, 0xe4a24c79596bb119, 
    0x0069db4862da3cdb, 0x0069db4862da3cda, 0x035533f6bf129a8d, 0x035533f6bf129a8c, 
    0x514ffb7f3c089dd3, 0x514ffb7f3c089dd2, 0x527313c1e1c03b85, 0x527313c1e1c03b84, 
    0x0cf7b828f270c0da, 0x0cf7b828f270c0db, 0x0fcb50962fb8668c, 0x0fcb50962fb8668d, 
    0x5dd1981faca261d2, 0x5dd1981faca261d3, 0x5eed70a1716ac784, 0x5eed70a1716ac785, 
    0x467698598926dc01, 0x467698598926dc00, 0x454a70e754ee7a57, 0x454a70e754ee7a56, 
    0x1750b86ed7f47d09, 0x1750b86ed7f47d08, 0x146c50d00a3cdb5f, 0x146c50d00a3cdb5e,
    0x4ae8fb39198c2000, 0x4ae8fb39198c2001, 0x49d41387c4448656, 0x49d41387c4448657, 
    0x1bcedb0e475e8108, 0x1bcedb0e475e8109, 0x18f233b09a96275e, 0x18f233b09a96275f, 
    0xfc39a481a127aa9d, 0xfc39a481a127aa9c, 0xff054c3f7cef0ccb, 0xff054c3f7cef0cca, 
    0xad1f84b6fff50b95, 0xad1f84b6fff50b94, 0xae236c08223dadc3, 0xae236c08223dadc2, 
    0xf0a7c7e1318d569c, 0xf0a7c7e1318d569d, 0xf39b2f5fec45f0ca, 0xf39b2f5fec45f0cb, 
    0xa181e7d66f5ff794, 0xa181e7d66f5ff795, 0xa2bd0f68b29751c2, 0xa2bd0f68b29751c3, 
    0xfc507fc9c3fd9646, 0xfc507fc9c3fd9647, 0xff6c97771e353010, 0xff6c97771e353011, 
    0xad765ffe9d2f374e, 0xad765ffe9d2f374f, 0xae4ab74040e79118, 0xae4ab74040e79119, 
    0xf0ce1ca953576a47, 0xf0ce1ca953576a46, 0xf3f2f4178e9fcc11, 0xf3f2f4178e9fcc10, 
    0xa1e83c9e0d85cb4f, 0xa1e83c9e0d85cb4e, 0xa2d4d420d04d6d19, 0xa2d4d420d04d6d18, 
    0x461f4311ebfce0da, 0x461f4311ebfce0db, 0x4523abaf3634468c, 0x4523abaf3634468d, 
    0x17396326b52e41d2, 0x17396326b52e41d3, 0x14058b9868e6e784, 0x14058b9868e6e785, 
    0x4a8120717b561cdb, 0x4a8120717b561cda, 0x49bdc8cfa69eba8d, 0x49bdc8cfa69eba8c, 
    0x1ba700462584bdd3, 0x1ba700462584bdd2, 0x189be8f8f84c1b85, 0x189be8f8f84c1b84, 
    0x4418ae808b28bdd0, 0x4418ae808b28bdd1, 0x4724463e56e01b86, 0x4724463e56e01b87, 
    0x153e8eb7d5fa1cd8, 0x153e8eb7d5fa1cd9, 0x160266090832ba8e, 0x160266090832ba8f, 
    0x4886cde01b8241d1, 0x4886cde01b8241d0, 0x4bba255ec64ae787, 0x4bba255ec64ae786, 
    0x19a0edd74550e0d9, 0x19a0edd74550e0d8, 0x1a9c05699898468f, 0x1a9c05699898468e, 
    0xfe579258a329cb4c, 0xfe579258a329cb4d, 0xfd6b7ae67ee16d1a, 0xfd6b7ae67ee16d1b, 
    0xaf71b26ffdfb6a44, 0xaf71b26ffdfb6a45, 0xac4d5ad12033cc12, 0xac4d5ad12033cc13, 
    0xf2c9f1383383374d, 0xf2c9f1383383374c, 0xf1f51986ee4b911b, 0xf1f51986ee4b911a, 
    0xa3efd10f6d519645, 0xa3efd10f6d519644, 0xa0d339b1b0993013, 0xa0d339b1b0993012, 
    0xfe3e4910c1f3f797, 0xfe3e4910c1f3f796, 0xfd02a1ae1c3b51c1, 0xfd02a1ae1c3b51c0, 
    0xaf1869279f21569f, 0xaf1869279f21569e, 0xac24819942e9f0c9, 0xac24819942e9f0c8, 
    0xf2a02a7051590b96, 0xf2a02a7051590b97, 0xf19cc2ce8c91adc0, 0xf19cc2ce8c91adc1, 
    0xa3860a470f8baa9e, 0xa3860a470f8baa9f, 0xa0bae2f9d2430cc8, 0xa0bae2f9d2430cc9, 
    0x447175c8e9f2810b, 0x447175c8e9f2810a, 0x474d9d76343a275d, 0x474d9d76343a275c, 
    0x155755ffb7202003, 0x155755ffb7202002, 0x166bbd416ae88655, 0x166bbd416ae88654, 
    0x48ef16a879587d0a, 0x48ef16a879587d0b, 0x4bd3fe16a490db5c, 0x4bd3fe16a490db5d, 
    0x19c9369f278adc02, 0x19c9369f278adc03, 0x1af5de21fa427a54, 0x1af5de21fa427a55, 
    0x026e36d9020e61d1, 0x026e36d9020e61d0, 0x0152de67dfc6c787, 0x0152de67dfc6c786, 
    0x534816ee5cdcc0d9, 0x534816ee5cdcc0d8, 0x5074fe508114668f, 0x5074fe508114668e, 
    0x0ef055b992a49dd0, 0x0ef055b992a49dd1, 0x0dccbd074f6c3b86, 0x0dccbd074f6c3b87, 
    0x5fd6758ecc763cd8, 0x5fd6758ecc763cd9, 0x5cea9d3011be9a8e, 0x5cea9d3011be9a8f, 
    0xb8210a012a0f174d, 0xb8210a012a0f174c, 0xbb1de2bff7c7b11b, 0xbb1de2bff7c7b11a, 
    0xe9072a3674ddb645, 0xe9072a3674ddb644, 0xea3bc288a9151013, 0xea3bc288a9151012, 
    0xb4bf6961baa5eb4c, 0xb4bf6961baa5eb4d, 0xb78381df676d4d1a, 0xb78381df676d4d1b, 
    0xe5994956e4774a44, 0xe5994956e4774a45, 0xe6a5a1e839bfec12, 0xe6a5a1e839bfec13, 
    0xb848d14948d52b96, 0xb848d14948d52b97, 0xbb7439f7951d8dc0, 0xbb7439f7951d8dc1, 
    0xe96ef17e16078a9e, 0xe96ef17e16078a9f, 0xea5219c0cbcf2cc8, 0xea5219c0cbcf2cc9, 
    0xb4d6b229d87fd797, 0xb4d6b229d87fd796, 0xb7ea5a9705b771c1, 0xb7ea5a9705b771c0, 
    0xe5f0921e86ad769f, 0xe5f0921e86ad769e, 0xe6cc7aa05b65d0c9, 0xe6cc7aa05b65d0c8, 
    0x0207ed9160d45d0a, 0x0207ed9160d45d0b, 0x013b052fbd1cfb5c, 0x013b052fbd1cfb5d, 
    0x5321cda63e06fc02, 0x5321cda63e06fc03, 0x501d2518e3ce5a54, 0x501d2518e3ce5a55, 
    0x0e998ef1f07ea10b, 0x0e998ef1f07ea10a, 0x0da5664f2db6075d, 0x0da5664f2db6075c, 
    0x5fbfaec6aeac0003, 0x5fbfaec6aeac0002, 0x5c8346787364a655, 0x5c8346787364a654, 
];

impl GF2p64Subfield for GF8 {
    fn embed(self) -> GF2p64 {
        GF2p64(GF8_EB_TABLE[self.0 as usize])
    }
}

#[rustfmt::skip]
const GF4_EB_TABLE: [u64; 16] = [
    0x0000000000000000, 0x0000000000000001, 0xa181e7d66f5ff794, 0xa181e7d66f5ff795,
    0xb848d14948d52b96, 0xb848d14948d52b97, 0x19c9369f278adc02, 0x19c9369f278adc03,
    0xfc39a481a127aa9d, 0xfc39a481a127aa9c, 0x5db84357ce785d09, 0x5db84357ce785d08,
    0x447175c8e9f2810b, 0x447175c8e9f2810a, 0xe5f0921e86ad769f, 0xe5f0921e86ad769e,
];

impl GF2p64Subfield for GF4 {
    fn embed(self) -> GF2p64 {
        GF2p64(GF4_EB_TABLE[self.as_u8() as usize])
    }
}

#[rustfmt::skip]
const GF4_EXT_EB_TABLE0: [u64; 16] = [0x0000000000000000, 0x0000000000000001, 0xb848d14948d52b97, 0xb848d14948d52b96, 0xa181e7d66f5ff794, 0xa181e7d66f5ff795, 0x19c9369f278adc03, 0x19c9369f278adc02, 0x5db84357ce785d09, 0x5db84357ce785d08, 0xe5f0921e86ad769e, 0xe5f0921e86ad769f, 0xfc39a481a127aa9d, 0xfc39a481a127aa9c, 0x447175c8e9f2810a, 0x447175c8e9f2810b];
#[rustfmt::skip]
const GF4_EXT_EB_TABLE1: [u64; 16] = [0x0000000000000000, 0x14f1968d182dd50f, 0xe783b218b4f7828e, 0xf3722495acda5781, 0x425086d25fa48e9a, 0x56a1105f47895b95, 0xa5d334caeb530c14, 0xb122a247f37ed91b, 0x5176233df09ddbe4, 0x4587b5b0e8b00eeb, 0xb6f59125446a596a, 0xa20407a85c478c65, 0x1326a5efaf39557e, 0x07d73362b7148071, 0xf4a517f71bced7f0, 0xe054817a03e302ff];
#[rustfmt::skip]
const GF4_EXT_EB_TABLE2: [u64; 16] = [0x0000000000000000, 0x1bf7034c8bcbc73e, 0xbacfec5a0a44ed0d, 0xa138ef16818f2a33, 0x58d19564f7685dd5, 0x432696287ca39aeb, 0xe21e793efd2cb0d8, 0xf9e97a7276e777e6, 0xa6cb12e9ca5a79b5, 0xbd3c11a54191be8b, 0x1c04feb3c01e94b8, 0x07f3fdff4bd55386, 0xfe1a878d3d322460, 0xe5ed84c1b6f9e35e, 0x44d56bd73776c96d, 0x5f22689bbcbd0e53];
// const GF4_EXT_EB_TABLE3: [u64; 16] = [0x0000000000000000, 0xa5d51fb65fe6c30e, 0xb9ddfd59eb0f5923, 0x1c08e2efb4e99a2d, 0xfaeb4e7aafdb3005, 0x5f3e51ccf03df30b, 0x4336b32344d46926, 0xe6e3ac951b32aa28, 0xf54ee43fdf842e10, 0x509bfb898062ed1e, 0x4c931966348b7733, 0xe94606d06b6db43d, 0x0fa5aa45705f1e15, 0xaa70b5f32fb9dd1b, 0xb678571c9b504736, 0x13ad48aac4b68438];

/// Embeds a0 + a1 * alpha from the degree 4 extension GF(2^4) \[alpha\] / 2*alpha^4 + 2*alpha^2 + 4*alpha + 8
/// into [GF2p64].
pub fn embed_gf4p4_deg2(a0: GF4, a1: GF4) -> GF2p64 {
    // the embedding is done in two steps (pre-computed in lookup tables)
    // (1) compute the isomorphism from GF(2^4)^4 to GF(2^16)
    // (2) embed GF(2^16) into a subgroup in GF(2^64) of the same size
    //
    // for (1), let x be the generator for GF(2^4), alpha be the generator for GF(2^4)^4 and y be the generator for GF(2^16), then
    // the isomorphism psi maps x -> 1 + y + y^3 + y^4 and alpha -> 1 + y + y^2.
    //
    // the tables Ti are computed as follows: Ti[x0 + x1 * x + x2 * x^2 + x3 * x^3] = (x0 + x1 * psi(x) + x2 * psi(x)^2 + x3 * psi(x)^3) * psi(alpha)**i
    // thus the isomorphism (a0 + a1 * alpha + a2 * alpha^2 + a3 * alpha^3) -> GF(2^16) can be computed using the tables as 
    // T0[a0] + T1[a1] + T2[a2] + T3[a3]
    // we only need at most deg3, so we omit T3
    //
    // for (2), the embedding is via u^56 + u^55 + u^54 + u^53 + u^51 + u^50 + u^48 + u^47 + u^46 + u^44 + u^43 + u^39 + u^37 + u^36 + u^34 + u^30 + u^29 + u^28 + u^23
    // + u^22 + u^21 + u^20 + u^19 + u^18 + u^17 + u^16 + u^13 + u^12 + u^10 + u^9 + u^7 + u^6 + u^5 + u^4 + u^3 + u + 1
    // where u is the generator of GF(2^64)
    // each table entry is embedded already
    GF2p64(GF4_EXT_EB_TABLE0[a0.as_u8() as usize] ^ GF4_EXT_EB_TABLE1[a1.as_u8() as usize])
}

/// Embeds a0 + a1 * alpha + a2 * alpha^2 from the degree 4 extension GF(2^4) \[alpha\] / 2* alpha^4 + 2*alpha^2 + 4*alpha + 8
/// into [GF2p64].
pub fn embed_gf4p4_deg3(a0: GF4, a1: GF4, a2: GF4) -> GF2p64 {
    GF2p64(GF4_EXT_EB_TABLE0[a0.as_u8() as usize] ^ GF4_EXT_EB_TABLE1[a1.as_u8() as usize] ^ GF4_EXT_EB_TABLE2[a2.as_u8() as usize])
}

/// Embeds b0 + b1 * u + b2 * u^2 into [GF2p64]. This is bitsliced by 16.
pub fn embed_gf2_deg012(dst: &mut[GF2p64], b0: BsBool16, b1: BsBool16, b2: BsBool16) {
    debug_assert_eq!(dst.len(), 16);
    for i in 0..16 {
        let b = ((b0.as_u16() >> i) & 0b1) | ((b1.as_u16() >> i) & 0b1) << 1 | ((b2.as_u16() >> i) & 0b1) << 2;
        dst[i] = GF2p64(b as u64);
    }
}

/// Embeds b0 + b3 * u^3 + b6 * u^6 into [GF2p64]. This is bitsliced by 16.
pub fn embed_gf2_deg036(dst: &mut [GF2p64], b0: BsBool16, b3: BsBool16, b6: BsBool16) {
    debug_assert_eq!(dst.len(), 16);
    for i in 0..16 {
        let b = ((b0.as_u16() >> i) & 0b1) | ((b3.as_u16() >> i) & 0b1) << 3 | ((b6.as_u16() >> i) & 0b1) << 6;
        dst[i] = GF2p64(b as u64);
    }
}

/// Embeds sum_{i=0}^8 bi * u^i into [GF2p64]. This is bitsliced by 16.
pub fn embed_gf2_deg8(dst: &mut [GF2p64], b0: BsBool16, b1: BsBool16, b2: BsBool16, b3: BsBool16, b4: BsBool16, b5: BsBool16, b6: BsBool16, b7: BsBool16, b8: BsBool16) {
    debug_assert_eq!(dst.len(), 16);
    for i in 0..16 {
        let b = ((b0.as_u16() >> i) & 0b1) | ((b1.as_u16() >> i) & 0b1) << 1 | ((b2.as_u16() >> i) & 0b1) << 2
        | ((b3.as_u16() >> i) & 0b1) << 3
        | ((b4.as_u16() >> i) & 0b1) << 4
        | ((b5.as_u16() >> i) & 0b1) << 5
        | ((b6.as_u16() >> i) & 0b1) << 6
        | ((b7.as_u16() >> i) & 0b1) << 7
        | ((b8.as_u16() >> i) & 0b1) << 8;
        dst[i] = GF2p64(b as u64);
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use rand::thread_rng;
    use maestro::rep3_core::{network::NetSerializable, party::RngExt, share::HasZero};

    use maestro::share::{
        bs_bool16::BsBool16, gf2p64::GF2p64Subfield, gf4::GF4, gf8::GF8, Field, InnerProduct, Invertible
    };

    

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    use crate::share::test::random_secret_shared_vector;

    use super::{GF2p64, GF2p64InnerProd};

    fn get_test_values() -> Vec<GF2p64> {
        vec![
            GF2p64(0),
            GF2p64(1),
            GF2p64(0xffffffffffffffff),
            GF2p64(0xfffffffffeffffff),
        ]
    }

    fn get_non_zero_test_values() -> Vec<GF2p64> {
        vec![
            GF2p64(1),
            GF2p64(0xffffffffffffffff),
            GF2p64(0xfffffffffeffffff),
        ]
    }

    #[test]
    fn test_mul() {
        let test_elements = get_test_values();
        for &x in &test_elements {
            for &y in &test_elements {
                //println!("{0:?} * {1:?} = {2:?} = {3:?}",x,y,x*y,y*x);
                assert_eq!(x * y, y * x)
            }
        }
        let zero = GF2p64::ZERO;
        for &x in &test_elements {
            assert_eq!(x * zero, zero)
        }
        for &x in &test_elements[1..] {
            assert_eq!(x * GF2p64::ONE, x)
        }
    }

    #[test]
    fn test_inverse() {
        let test_values = get_non_zero_test_values();
        for x in test_values {
            let inv_x = x.inverse();
            assert_eq!(x * inv_x, GF2p64::ONE);
            assert_eq!(inv_x * x, GF2p64::ONE)
        }
    }

    #[test]
    fn test_serialization() {
        let v = vec![
            GF2p64(0x1),
            GF2p64(0xffffffffffffffff),
            GF2p64(0x3),
            GF2p64(0x123456781234578),
        ];
        let as_bytes = GF2p64::as_byte_vec(v.iter(), 4);
        let v_new = GF2p64::from_byte_vec(as_bytes, 4);
        assert_eq!(v_new, v);
    }

    #[test]
    fn test_gf8_embedding() {
        for i in 0..255 {
            let x = GF8(i);
            for j in 0..255 {
                let y = GF8(j);
                assert_eq!((x + y).embed(), x.embed() + y.embed());
                assert_eq!((x * y).embed(), x.embed() * y.embed())
            }
        }
    }

    #[test]
    fn test_gf4_embedding() {
        for i in 0..16 {
            let x = GF4::new_unchecked(i);
            for j in 0..16 {
                let y = GF4::new_unchecked(j);
                assert_eq!((x + y).embed(), x.embed() + y.embed());
                assert_eq!((x * y).embed(), x.embed() * y.embed())
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn test_fall_back_inner_product() {
        let a = vec![GF2p64(0x1), GF2p64(0x1), GF2p64(0x4), GF2p64(0x8)];
        let b = vec![GF2p64(0x1), GF2p64(0x2), GF2p64(0x1), GF2p64(0x1)];
        assert_eq!(GF2p64::fallback_inner_product(&a, &b), GF2p64(0xf))
    }

    #[test]
    #[allow(deprecated)]
    fn test_inner_product() {
        let mut rng = thread_rng();
        let a: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        let b: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        assert_eq!(
            GF2p64::fallback_inner_product(&a, &b),
            GF2p64::inner_product(&a, &b)
        )
    }

    #[test]
    #[cfg(all(
        feature = "clmul",
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    fn fast_test_inner_product_aarch64() {
        let mut rng = thread_rng();
        let a: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        let b: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        assert_eq!(
            GF2p64::inner_product(&a, &b),
            GF2p64::fast_clmul_inner_product(&a, &b)
        )
    }

    #[test]
    #[cfg(all(
        feature = "clmul",
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "pclmulqdq"
    ))]
    fn fast_test_inner_product_x86_64() {
        let mut rng = thread_rng();
        let a: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        let b: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        assert_eq!(
            GF2p64::inner_product(&a, &b),
            GF2p64::fast_clmul_inner_product(&a, &b)
        )
    }

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    #[test]
    fn test_fast_mul() {
        let mut rng = thread_rng();
        let a: Vec<GF2p64> = GF2p64::generate(&mut rng, 128);
        let b: Vec<GF2p64> = GF2p64::generate(&mut rng, 128);
        a.into_iter().zip(b).for_each(|(a, b)| {
            assert_eq!(a*b, GF2p64::mul_clmul_u64_fast(&a,&b))
        });
    }

    #[cfg(any(
        all(
            feature = "clmul",
            target_arch = "x86_64",
            target_feature = "sse2",
            target_feature = "pclmulqdq"
        ),
        all(
            feature = "clmul",
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        )
    ))]
    #[test]
    #[allow(deprecated)]
    fn test_weak_ip_consistency() {
        let (_, a, b, _) = random_secret_shared_vector(32);
        assert_eq!(
            GF2p64::fallback_weak_ip(&a, &b),
            GF2p64::clmul_weak_inner_product(&a, &b)
        )
    }
    
    #[test]
    fn test_inner_gf2p64_inner_prod_correctness() {
        let mut rng = thread_rng();
        let a: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);
        let b: Vec<GF2p64> = GF2p64::generate(&mut rng, 29);

        let expected = GF2p64::inner_product(&a, &b);
        let mut actual = GF2p64InnerProd::new();
        for (a,b) in a.iter().zip(b.iter()) {
            actual.add_prod(a, b);
        }
        let actual = actual.sum();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_extend_from_bit() {
        let inputs = (0..16).map(|i| BsBool16::new(i as u16)).collect_vec();
        let mut v = Vec::new();
        for (i,inp) in inputs.into_iter().enumerate() {
            GF2p64::extend_from_bit(&mut v, &inp);
            assert_eq!(v.len(), 16*(i+1));
            let bits = inp.as_u16();
            for j in 0..16 {
                assert_eq!(v[v.len()-16+j], GF2p64::new((bits >> j) & 0x1));
            }
        }
    }

    #[test]
    fn test_extend_from_bitstring() {
        // simple case

        fn test(string: &[BsBool16], expected: GF2p64) {
            let mut v = Vec::new();
            GF2p64::extend_from_bitstring(&mut v, string, 1);
            assert_eq!(v.len(), 16);
            assert_eq!(v[0], expected);
            for i in 1..16 {
                assert_eq!(v[i], GF2p64::ZERO);
            }
        }

        test(&[BsBool16::new(0b1)], GF2p64::new(1u64));
        test(&[BsBool16::new(1), BsBool16::new(0), BsBool16::new(1), BsBool16::new(0), BsBool16::new(1), BsBool16::new(1), BsBool16::new(0), BsBool16::new(1)], GF2p64::new(0b10110101u64));
    }
}
