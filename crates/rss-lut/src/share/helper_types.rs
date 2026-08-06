// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid

use std::{borrow::Borrow, fmt::Debug, ops::{BitAnd, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr}};

use rand::{CryptoRng, Rng};

use crate::share::{gf_template::ShareType, gf2p64::GF8_EB_TABLE};



#[derive(Copy, Clone, PartialEq, Debug)]
pub struct BitShare(pub u8);


/// Trait to restrict `T` to only specific types
pub trait AllowedTypes: 
    Copy
    + Clone
    + Default 
    + Debug
    + BitOrAssign 
    + BitOr<Output = Self>
    + BitAnd<Output = Self>
    + BitXor<Output = Self>
    + BitXorAssign
    + Not<Output = Self>
    + Shl<usize, Output = Self> 
    + Shr<usize, Output = Self>
    + Borrow<Self>
    + PartialEq
    + std::fmt::Binary
    {
        const ZERO: Self;
        const ONE: Self;
        fn to_u8(self) -> u8;
        fn to_u64(self) -> u64;
        fn to_usize(self) -> usize;
        fn from_usize(x: usize) -> Self;
        fn from(x: u64) -> Self;
        fn embed(self) -> u64{
            self.to_u64()
        }
        fn pack_bytes(x: &[u8]) -> ShareType<Self>{
            debug_assert!(x.len() <= std::mem::size_of::<Self>(), "There are more bytes than the size of the type {} {}", x.len(), std::mem::size_of::<Self>());
            let mut result = Self::default();
            for (i, &byte) in x.iter().enumerate() {
                    result |= Self::from(byte as u64) << (i * 8);
            }
            ShareType(result)
        }
        fn serialized_size(n_elements: usize) -> usize {
            n_elements * std::mem::size_of::<Self>()
        }

        fn as_byte_vec(it: impl IntoIterator<Item = impl std::borrow::Borrow<ShareType<Self>>>, _len: usize) -> Vec<u8> {
            it.into_iter()
                .flat_map(|gf| 
                (0..std::mem::size_of::<Self>())
                .map(move |i| (gf.borrow().0 >> (i * 8)).to_u8()))
                .collect()
        }

        fn as_byte_vec_slice(elements: &[ShareType<Self>]) -> Vec<u8> {
            elements.iter()
                .flat_map(|gf| 
                (0..std::mem::size_of::<Self>())
                .map(move |i| (gf.borrow().0 >> (i * 8)).to_u8()))
                .collect()
        }

        fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<ShareType<Self>> {
            v.chunks(std::mem::size_of::<Self>())
            .map(Self::pack_bytes)
            .collect()
        }

        fn from_byte_slice(v: Vec<u8>, dest: &mut [ShareType<Self>]) {
            v.chunks(std::mem::size_of::<Self>())
                .zip(dest.iter_mut())
                .for_each(|(chunk, dst)| 
                    *dst = Self::pack_bytes(chunk)
                );
        }

        fn fill_self<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [ShareType<Self>]) {
            let len = buf.len() * std::mem::size_of::<Self>();
            let mut v = vec![0u8; len];
            rng.fill_bytes(&mut v);
            v.chunks(std::mem::size_of::<Self>())
                .zip(buf)
                .for_each(|(val, x)| *x = Self::pack_bytes(val));
        }
    }

impl AllowedTypes for BitShare{
    const ZERO: Self = BitShare(0);

    const ONE: Self = BitShare(1);

    fn to_u8(self) -> u8 {
        self.0
    }

    fn to_usize(self) -> usize {
        self.0 as usize
    }

    fn from_usize(x: usize) -> Self {
        Self((x % 2) as u8)
    }

    fn serialized_size(n_elements: usize) -> usize {
        (n_elements + 7) / 8
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl std::borrow::Borrow<ShareType<Self>>>, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::serialized_size(len));
        let mut current_byte = 0u8;
        let mut bit_index = 0;

        for bit in it {
            current_byte |= (bit.borrow().0.0 & 1) << bit_index;
            bit_index += 1;

            if bit_index == 8 {
                bytes.push(current_byte);
                current_byte = 0;
                bit_index = 0;
            }
        }

        if bit_index > 0 {
            bytes.push(current_byte);
        }

        bytes
    }

    fn as_byte_vec_slice(elements: &[ShareType<Self>]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::serialized_size(elements.len()));
        let mut current_byte = 0u8;
        let mut bit_index = 0;

        for bit in elements {
            current_byte |= (bit.0.0 & 1) << bit_index;
            bit_index += 1;

            if bit_index == 8 {
                bytes.push(current_byte);
                current_byte = 0;
                bit_index = 0;
            }
        }

        if bit_index > 0 {
            bytes.push(current_byte);
        }

        bytes
    }

    fn from_byte_vec(v: Vec<u8>, len: usize) -> Vec<ShareType<Self>> {
        let mut bits = Vec::with_capacity(len);
        for byte in v {
            for i in 0..8 {
                if bits.len() < len {
                    bits.push(ShareType(BitShare((byte >> i) & 1)));
                } else {
                    break;
                }
            }
        }
        bits
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [ShareType<Self>]) {
        let mut bit_index = 0;
        for byte in v {
            for i in 0..8 {
                if bit_index < dest.len() {
                    dest[bit_index] = ShareType(BitShare((byte >> i) & 1));
                    bit_index += 1;
                } else {
                    break;
                }
            }
        }
    }
    fn pack_bytes(x: &[u8]) -> ShareType<Self>{
        debug_assert!(x.len() == 1, "There are more bytes than the size of the type {} 1", x.len());
        debug_assert!(x[0] <= 1, "The provided byte is not a bit {}", x[0]);
        ShareType(BitShare(x[0] & 1))
    }
        
    fn fill_self<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [ShareType<Self>]) {
        let len = buf.len() * std::mem::size_of::<Self>();
        let mut v = vec![0u8; len];
        rng.fill_bytes(&mut v);
        v.iter()
            .zip(buf)
            .for_each(|(val, x)| *x = ShareType(BitShare(*val & 1)));
    }

    fn from(x: u64) -> Self {
        Self((x % 2) as u8)
    }
    
    fn to_u64(self) -> u64 {
        self.0 as u64
    }
}

impl std::fmt::Binary for BitShare{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl BitOrAssign for BitShare{
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl Default for BitShare{
    fn default() -> Self {
        Self(0)
    }
}

impl Not for BitShare{
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(1 - self.0)
    }
}

impl BitOr for BitShare{
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitAnd for BitShare{
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitXor for BitShare{
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for BitShare{
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl From<BitShare> for u64{
    fn from(x: BitShare) -> Self {
        x.0 as u64
    }
}

impl Shl<usize> for BitShare{
    type Output = Self;
    fn shl(self, _rhs: usize) -> Self::Output {
        Self(0)
    }
}

impl Shr<usize> for BitShare{
    type Output = Self;
    fn shr(self, _rhs: usize) -> Self::Output {
        Self(0)
    }
}


impl AllowedTypes for u8   {
    const ZERO: Self = 0;
    const ONE:  Self = 0xFF;
    fn to_u8(self) -> u8 {
        self
    }
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(x: usize) -> Self{
        x as Self
    }
    fn from(x: u64) -> Self {
        x as u8
    }
    
    fn to_u64(self) -> u64 {
        self as u64
    }
    fn embed(self) -> u64 {
        GF8_EB_TABLE[self as usize]
    }
}
impl AllowedTypes for u16  {
    const ZERO: Self = 0;
    const ONE:  Self = 0xFFFF;
    fn to_u8(self) -> u8 {
        (self % 256) as u8
    }
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(x: usize) -> Self{
        x as Self
    }
    fn from(x: u64) -> Self {
        x as u16
    }
    fn to_u64(self) -> u64 {
        self as u64
    }
}
impl AllowedTypes for u32  {
    const ZERO: Self = 0;
    const ONE:  Self = 0xFFFFFFFF;
    fn to_u8(self) -> u8 {
        (self % 256) as u8
    }
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(x: usize) -> Self{
        x as Self
    }

    fn from(x: u64) -> Self {
        x as u32
    }
    fn to_u64(self) -> u64 {
        self as u64
    }
}
impl AllowedTypes for u64  {
    const ZERO: Self = 0;
    const ONE:  Self = 0xFFFFFFFFFFFFFFFF;
    fn to_u8(self) -> u8 {
        (self % 256) as u8
    }
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(x: usize) -> Self{
        x as Self
    }
    fn from(x: u64) -> Self {
        x
    }
    fn to_u64(self) -> u64 {
        self as u64
    }
}
impl AllowedTypes for u128 {
    const ZERO: Self = 0;
    const ONE:  Self = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
    fn to_u8(self) -> u8 {
        (self % 256) as u8
    }
    fn to_usize(self) -> usize {
        self as usize
    }
    fn from_usize(x: usize) -> Self{
        x as Self
    }
    fn from(x: u64) -> Self {
        x as u128
    }
    fn to_u64(self) -> u64 {
        self as u64
    }
}