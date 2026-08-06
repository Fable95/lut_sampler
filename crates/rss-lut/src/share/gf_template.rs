// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid
// Portions adapted from MAESTRO (https://github.com/KULeuven-COSIC/maestro),
// Copyright © 2024 COSIC-KU Leuven and Concordium AG, licensed under the MIT
// License. See THIRD-PARTY-NOTICES for the full notice.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};
use std::marker::PhantomData;


use rand::{CryptoRng, Rng};
use sha2::Digest;

use maestro::rep3_core::{network::NetSerializable, party::{DigestExt, RngExt}, share::HasZero};
use maestro::share::Field;
use crate::share::helper_types::{AllowedTypes};

use super::gf2p64::{GF2p64, GF2p64Subfield};




pub trait Share:
    Default
    + HasZero
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Field
    + NetSerializable
    + RngExt
    + DigestExt
    + Debug
    + Sized
    + GF2p64Subfield
{
    type InnerType: AllowedTypes;   
    fn pack_bytes(x: &[u8]) -> Self;
    fn new(x: Self::InnerType) -> Self;
    fn from_u8(x: u8) -> Self;
    fn from_usize(x: usize) -> Self;
    fn inner(&self) -> Self::InnerType;
    fn inner_mut(&mut self) -> &mut Self::InnerType;
    fn to_u8(&self) -> u8;
    fn to_usize(&self) -> usize;
    fn bit_embed_gf2p64(&self) -> Vec<GF2p64>;

}

pub trait GFTTrait:
    Default
    + Debug
    + Clone
    + Copy
    + Sized
{
    type Wrapper: Share;
    type Embedded: Share;
    const RATIO: usize;
    fn new(x: Self::Wrapper) -> Self;
    fn get_element(element: &Self::Wrapper, index: usize) -> Self::Embedded;
    fn pack(elements: &[Self::Embedded]) -> Self::Wrapper;
    fn pack_slice(elements: &[Self::Embedded]) -> Vec<Self::Wrapper>;
    fn unpack_slice(elements: &[Self::Wrapper]) -> Vec<Self::Embedded>;
    fn unpack(element: &Self::Wrapper) -> Vec<Self::Embedded>;
    // fn embed(element: &Self::Wrapper) -> Vec<GF2p64>{
    //     (0..Self::RATIO)
    //         .map(|i| Self::get_element(element, i).embed())
    //         .collect()
    // }
}



#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub struct ShareType<T: AllowedTypes>(pub T);

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct GFT<T: AllowedTypes, E: AllowedTypes, const RATIO: usize>
    where T: From<E>
{
    pub inner: ShareType<T>,
    _marker: PhantomData<E>,
}

impl<T: AllowedTypes, E: AllowedTypes, const RATIO: usize> 
Default for GFT<T,E,RATIO>
where T: From<E>{
    fn default() -> Self {
        Self{
            inner: ShareType::<T>(T::ZERO),
            _marker: PhantomData
        }
    }
}


impl<T: AllowedTypes, E: AllowedTypes, const RATIO: usize> GFT<T,E,RATIO>
where T: From<E>{
    const CHECK: () = assert!(std::mem::size_of::<E>() < std::mem::size_of::<T>(), 
    "The wrapper type must be larger than the embedded type");
}

impl<T: AllowedTypes, E: AllowedTypes, const R: usize> GFTTrait 
for GFT<T,E,R>
where T: From<E> 
{
    const RATIO: usize = R;
    type Embedded = ShareType<E>;
    type Wrapper =  ShareType<T>;

    /// Generates a new packed GF8 element from a vector of bytes
    fn new(x: Self::Wrapper) -> Self {
        Self { inner: x, _marker: PhantomData }
    }

    fn unpack(element: &Self::Wrapper) -> Vec<Self::Embedded> {
        (0..R)
            .map(|i| Self::get_element(element, i))
            .collect()
    }

    fn unpack_slice(elements: &[Self::Wrapper]) -> Vec<Self::Embedded> {
        elements.iter().flat_map(|e| Self::unpack(e)).collect()
    }

    fn get_element(element: &Self::Wrapper, index: usize) -> Self::Embedded {
        debug_assert!(index < R);
        
        <Self::Embedded as Share>::new(E::from((element.0 >> (index * 8 * std::mem::size_of::<E>())).to_usize() as u64))
    }
    
    fn pack(elements: &[Self::Embedded]) -> Self::Wrapper {
        debug_assert!(elements.len() == R);
        let mut result = T::ZERO;
        for (i, e) in elements.iter().enumerate() {
            result |= <T as AllowedTypes>::from(e.to_usize() as u64) << (i * 8 * std::mem::size_of::<E>());
        }
        ShareType(result)
    }

    fn pack_slice(elements: &[Self::Embedded]) -> Vec<Self::Wrapper> {
        elements.chunks(R).map(|chunk| Self::pack(chunk)).collect()
    }
}

impl<T: AllowedTypes> Share for ShareType<T>{
    fn pack_bytes(x: &[u8]) -> Self{
        Self::InnerType::pack_bytes(x)
    }
    
    type InnerType = T;
    
    fn new(x: Self::InnerType) -> Self {
        Self(x)
    }
    
    fn inner(&self) -> Self::InnerType {
        self.0
    }

    fn inner_mut(&mut self) -> &mut Self::InnerType {
        &mut self.0
    }

    fn to_u8(&self) -> u8 {
        self.0.to_u8()
    }
    
    fn to_usize(&self) -> usize {
        self.0.to_usize()
    }
    
    
    fn from_u8(x: u8) -> Self {
        Self::new(Self::InnerType::from(x as u64))
    }
    fn from_usize(x: usize) -> Self {
        Self::new(Self::InnerType::from_usize(x))
    }
    
    fn bit_embed_gf2p64(&self) -> Vec<GF2p64> {
        let len = <Self as Field>::NBITS;
        debug_assert!(len <= 64);
        let mut bits = self.0.to_usize() as u64;
    
        (0..len)
            .map(|_| {
                let bit = bits & 1;
                bits >>= 1;
                GF2p64::new(bit)
            })
            .collect()
    }
    
}

impl<T: AllowedTypes> GF2p64Subfield for ShareType<T>{
    fn embed(self) -> super::gf2p64::GF2p64 {
        // GF2p64::new(GF8_EB_TABLE[(self.0.to_u8()) as usize])
        GF2p64::new(self.0.embed())
    }
}

impl<T: AllowedTypes> HasZero for ShareType<T>{
    const ZERO: Self = Self(T::ZERO);
}

impl<T: AllowedTypes> Add for ShareType<T>{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<T: AllowedTypes> AddAssign for ShareType<T>{
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl<T: AllowedTypes> Sub for ShareType<T>{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}


impl<T: AllowedTypes> Neg for ShareType<T>{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self
    }
}

impl<T: AllowedTypes> Mul for ShareType<T> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<T: AllowedTypes> Field for ShareType<T> {
    const NBYTES: usize = std::mem::size_of::<T>();
    const NBITS: usize = 8 * Self::NBYTES;

    const ONE: Self = Self(T::ONE); // all bits set to 1
    
    fn is_zero(&self) -> bool {
        self.0 == T::ZERO
    }
}

impl<T: AllowedTypes> NetSerializable  for ShareType<T> {
    fn serialized_size(n_elements: usize) -> usize {
        T::serialized_size(n_elements)
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl std::borrow::Borrow<Self>>, len: usize) -> Vec<u8> {
        T::as_byte_vec(it, len)
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        T::as_byte_vec_slice(elements)
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        T::from_byte_vec(v, _len)
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        T::from_byte_slice(v, dest)
    }
}

impl<T: AllowedTypes> RngExt  for ShareType<T> {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        T::fill_self(rng, buf);
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![Self::default(); n];
        Self::fill(rng, &mut r);
        r
    }
}

impl<T: AllowedTypes> DigestExt  for ShareType<T> {
    fn update<D: Digest>(digest: &mut D, message: &[Self]) {
        let vec = Self::as_byte_vec(message, message.len());
        for x in vec.chunks(std::mem::size_of::<T>()){
            digest.update(x);
        }
    }
}


#[cfg(test)]
mod test{
    

    use rand::{rngs::ThreadRng, thread_rng};

    
    use maestro::rep3_core::{network::NetSerializable, party::RngExt, share::HasZero};
    use crate::share::{gf_template::Share, helper_types::BitShare};

    use super::{GFTTrait, GFT};

    type GF128_8  = GFT<u128,u8,16>;
    type GF128_16 = GFT<u128,u16,8>;
    type GF128_32 = GFT<u128,u32,4>;

    type GF64_8  =  GFT<u64,u8,8>;
    type GF64_16 =  GFT<u64,u16,4>;
    type GF64_32 =  GFT<u64,u32,2>;


    fn check_slice<T: Share>(items: &Vec<T>, slice: &mut [T]){
        <T as NetSerializable>::from_byte_slice(
            <T as NetSerializable>::as_byte_vec(items, items.len()), slice);
        assert_eq!(&slice, &items);
    }

    fn check_packing<T: GFTTrait, const N: usize>(rng: &mut ThreadRng){
        println!("Checking packing for type {}", std::any::type_name::<T>());
        let l: Vec<T::Embedded> = <T::Embedded as RngExt>::generate(rng, N);
        let old_len = size_of::<T::Wrapper>();
        let vec1 = <T::Embedded as NetSerializable>::as_byte_vec(&l, l.len());
        let len_2 = vec1.len() / old_len;
        let vec2 = <T::Wrapper as NetSerializable>::from_byte_vec(vec1.clone(), len_2);
        
        let mut vec3  = <T::Wrapper as NetSerializable>::as_byte_vec(&vec2, vec2.len());
        vec3.truncate(vec1.len());
        assert_eq!(vec1, vec3);

        let vec4 = <T::Embedded as NetSerializable>::from_byte_vec(vec3, N);
        assert_eq!(&vec4, &l);
    }

    fn check_compose<T: GFTTrait, const N: usize>(rng: &mut ThreadRng, _debug: bool){
        
        let l: Vec<T::Wrapper> = <T::Wrapper as RngExt>::generate(rng, N);
        let mut slice = [<T::Wrapper as HasZero>::ZERO; N];
        assert_eq!(l, <T::Wrapper as NetSerializable>::from_byte_vec(<T::Wrapper as NetSerializable>::as_byte_vec(&l, l.len()),l.len()));
        check_slice::<T::Wrapper>(&l, &mut slice);
        check_packing::<T,N>(rng);
    }

    #[test]
    fn test_serialization() {
        let mut rng = thread_rng();
        check_compose::<GF128_8 , 500>  (&mut rng, false);
        check_compose::<GF128_16, 500>  (&mut rng, false);
        check_compose::<GF128_32, 500>  (&mut rng, false);
        check_compose::<GF64_8  , 500>  (&mut rng, false);
        check_compose::<GF64_16 , 500>  (&mut rng, false);
        check_compose::<GF64_32 , 500>  (&mut rng, false);
        check_compose::<GF128_8 , 45>   (&mut rng, false);
        check_compose::<GF128_16, 45>   (&mut rng, false);
        check_compose::<GF128_32, 45>   (&mut rng, false);
        check_compose::<GF64_8  , 45>   (&mut rng, false);
        check_compose::<GF64_16 , 45>   (&mut rng, false);
        check_compose::<GF64_32 , 45>   (&mut rng, false);


    }

    type GFBin = GFT<u64, BitShare, 8>;
    
    #[test]
    fn test_bin_serialization() {
        let mut rng = thread_rng();
        check_compose::<GFBin , 500> (&mut rng, true);
        check_compose::<GFBin , 45>  (&mut rng, true);
    }
}