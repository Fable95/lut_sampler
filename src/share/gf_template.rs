use std::fmt::Debug;
use std::ops::{Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, Neg, Shl, Shr, Sub};
use std::marker::PhantomData;


use rand::{CryptoRng, Rng};
use sha2::Digest;

use maestro::rep3_core::{network::NetSerializable, party::{DigestExt, RngExt}, share::HasZero};
use maestro::share::Field;

use super::gf2p64::{GF2p64, GF2p64Subfield, GF8_EB_TABLE};

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
    // + Not<Output = Self>
    + Shl<usize, Output = Self> 
    + Shr<usize, Output = Self>
    + PartialEq
    + From<u8>
    {
        const ZERO: Self;
        const ONE: Self;
        fn to_u8(self) -> u8;
        fn to_usize(self) -> usize;
        fn from_usize(x: usize) -> Self;
        fn pop_count(&self) -> Self;
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
    
    fn pop_count(&self) -> Self {
        self.count_ones() as u8
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
    fn pop_count(&self) -> Self {
        self.count_ones() as u16
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
    fn pop_count(&self) -> Self {
        self.count_ones() as u32
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
    fn pop_count(&self) -> Self {
        self.count_ones() as u64
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
    fn pop_count(&self) -> Self {
        self.count_ones() as u128
    }
}


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
    fn count_ones(&self) -> Self;
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
    fn get_element(&self, index: usize) -> Self::Embedded;
    fn inner(&self) -> Self::Wrapper;
    fn pack(x: &[Self::Embedded]) -> Self;
    fn unpack(&self) -> Vec<Self::Embedded>;
    fn embed(&self) -> Vec<GF2p64>;
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
where T: From<E> {
    const RATIO: usize = R;
    type Embedded = ShareType<E>;
    type Wrapper =  ShareType<T>;

    /// Generates a new packed GF8 element from a vector of bytes
    fn new(x: Self::Wrapper) -> Self {
        Self { inner: x, _marker: PhantomData }
    }

    fn unpack(&self) -> Vec<Self::Embedded> {
        (0..R)
            .map(|i| self.get_element(i))
            .collect()
    }

    fn get_element(&self, index: usize) -> Self::Embedded {
        debug_assert!(index < R);
        
        <Self::Embedded as Share>::new(E::from((self.inner.0 >> (index * std::mem::size_of::<E>())).to_u8()))
    }

    fn pack(x: &[Self::Embedded]) -> Self {
        let mut result = T::default();
        for (i, &el) in x.iter().enumerate() {
            if i < std::mem::size_of::<T>()/R {
                result |= T::from(el.0) << (i * std::mem::size_of::<E>());
            }
        }
        GFT::<T,E,R>{
            inner: <Self::Wrapper as Share>::new(result),
            _marker: PhantomData
        }
    }
    
    fn inner(&self) -> Self::Wrapper {
        self.inner
    }
    
    fn embed(&self) -> Vec<GF2p64> {
        (0..R)
            .map(|i| self.get_element(i).embed())
            .collect()
    }
    
}

impl<T: AllowedTypes> Share for ShareType<T>{
    fn pack_bytes(x: &[u8]) -> Self{
        let mut result = T::default();
        for (i, &byte) in x.iter().enumerate() {
            if i < 16 {
                result |= T::from(byte) << (i * 8);
            }
        }
        Self(result)
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
    
    fn count_ones(&self) -> Self {
        println!("count_ones {:?}", self.0.pop_count());
        Self(self.0.pop_count())
    }
    
    fn from_u8(x: u8) -> Self {
        Self::new(Self::InnerType::from(x))
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
        GF2p64::new(GF8_EB_TABLE[(self.0.to_u8()) as usize])
        // GF2p64::new(self.0.to_u8() as u64)
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
        n_elements * std::mem::size_of::<T>()
    }

    fn as_byte_vec(it: impl IntoIterator<Item = impl std::borrow::Borrow<Self>>, _len: usize) -> Vec<u8> {
        it.into_iter()
            .flat_map(|gf| 
            (0..std::mem::size_of::<T>())
            .map(move |i| (gf.borrow().0 >> (i * 8)).to_u8()))
            .collect()
    }

    fn as_byte_vec_slice(elements: &[Self]) -> Vec<u8> {
        elements.iter()
            .flat_map(|gf| 
            (0..std::mem::size_of::<T>())
            .map(move |i| (gf.0 >> (i * 8)).to_u8()))
            .collect()
    }

    fn from_byte_vec(v: Vec<u8>, _len: usize) -> Vec<Self> {
        v.chunks(std::mem::size_of::<T>())
        .map(Self::pack_bytes)
        .collect()
    }

    fn from_byte_slice(v: Vec<u8>, dest: &mut [Self]) {
        v.chunks(std::mem::size_of::<T>())
            .zip(dest.iter_mut())
            .for_each(|(chunk, dst)| 
                *dst = Self::pack_bytes(chunk)
            );
    }
}

impl<T: AllowedTypes> RngExt  for ShareType<T> {
    fn fill<R: Rng + CryptoRng>(rng: &mut R, buf: &mut [Self]) {
        let len = buf.len() * std::mem::size_of::<T>();
        let mut v = vec![0u8; len];
        rng.fill_bytes(&mut v);
        v.chunks(std::mem::size_of::<T>())
            .zip(buf)
            .for_each(|(val, x)| x.0 = Self::pack_bytes(val).0);
    }

    fn generate<R: Rng + CryptoRng>(rng: &mut R, n: usize) -> Vec<Self> {
        let mut r = vec![0u8; std::mem::size_of::<T>() * n];
        rng.fill_bytes(&mut r);
        r.chunks(std::mem::size_of::<T>())
            .map(|x| Self::pack_bytes(x))
            .collect()
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
    use super::{GFTTrait, GFT};

    type GF128_8  = GFT<u128,u8,16>;
    type GF128_16 = GFT<u128,u16,8>;
    type GF128_32 = GFT<u128,u32,4>;

    type GF64_8  =  GFT<u64,u8,8>;
    type GF64_16 =  GFT<u64,u16,4>;
    type GF64_32 =  GFT<u64,u32,2>;

    #[test]
    fn test_serialization() {
        let mut rng = thread_rng();
        
        fn check_slice<T: GFTTrait>(items: &Vec<T::Wrapper>, slice: &mut [T::Wrapper]){
            <T::Wrapper as NetSerializable>::from_byte_slice(
                <T::Wrapper as NetSerializable>::as_byte_vec(items, items.len()), slice);
            assert_eq!(&slice, &items);
        }

        fn comp<T: GFTTrait, const N: usize>(rng: &mut ThreadRng){
            
            let l: Vec<T::Wrapper> = <T::Wrapper as RngExt>::generate(rng, N);
            let mut slice = [<T::Wrapper as HasZero>::ZERO; N];
            assert_eq!(l, <T::Wrapper as NetSerializable>::from_byte_vec(<T::Wrapper as NetSerializable>::as_byte_vec(&l, l.len()),l.len()));
            check_slice::<T>(&l, &mut slice);
        }

        comp::<GF128_8 , 500> (&mut rng);
        comp::<GF128_16, 500>(&mut rng);
        comp::<GF128_32, 500>(&mut rng);
        comp::<GF64_8  , 500>  (&mut rng);
        comp::<GF64_16 , 500> (&mut rng);
        comp::<GF64_32 , 500> (&mut rng);
        comp::<GF128_8 , 45>  (&mut rng);
        comp::<GF128_16, 45> (&mut rng);
        comp::<GF128_32, 45> (&mut rng);
        comp::<GF64_8  , 45>   (&mut rng);
        comp::<GF64_16 , 45>  (&mut rng);
        comp::<GF64_32 , 45>  (&mut rng);

    }
}