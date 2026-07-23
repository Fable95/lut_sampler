use std::slice;

use itertools::Itertools;
use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};
use maestro::rep3_core::{
    network::task::Direction,
    party::{broadcast::{Broadcast, BroadcastContext}, error::MpcResult, DigestExt, MainParty, Party}, share::{HasZero, RssShare, RssShareVec},
};
use maestro::share::{Field, HasTwo, InnerProduct, Invertible};

use crate::{
    share::gf2p64::{GF2p64, GF2p64InnerProd},
    share::zp::Zp,
    util::mul_triple_vec::{DotProdRecorder, DotProdVector, MulTripleEncoder, MulTripleVector, NoMulTripleRecording}
};

#[derive(Clone)]
pub enum TripleVector{
    SEMI(NoMulTripleRecording),
    MAL(DotProdVector<GF2p64>)
}

impl TripleVector{
    pub fn new(mal: bool) -> Self{
        if mal {
            return Self::MAL(DotProdVector::new());
        } else {
            return Self::SEMI(NoMulTripleRecording);
        }
    }

    pub fn get_mut_triple_vector(&mut self) -> &mut DotProdVector<GF2p64>{
        match self {
            TripleVector::SEMI(_) => panic!("Malicious setting expected"),
            TripleVector::MAL(vec) => vec,  
        }
    }

    pub fn get_triple_vector(&self) -> &DotProdVector<GF2p64>{
        match self {
            TripleVector::SEMI(_) => panic!("Malicious setting expected"),
            TripleVector::MAL(vec) => vec,  
        }
    }

    pub fn get_len(&self) -> usize {
        match self {
            TripleVector::SEMI(_) => panic!("Malicious setting expected"),
            TripleVector::MAL(vec) => vec.len(),  
        }
    }

    pub fn append(&mut self, mut other: TripleVector) {
        let o = other.get_mut_triple_vector();
        let i = self.get_mut_triple_vector();
        i.append(o);
    }

}


impl DotProdRecorder<GF2p64> for TripleVector{
    fn reserve_for_more_dotprods(&mut self, n: usize) {
        match self{
            TripleVector::SEMI(no_mul_triple_recording) => 
                DotProdRecorder::<GF2p64>::reserve_for_more_dotprods(no_mul_triple_recording, n),
            TripleVector::MAL(mul_triple_vector) => 
                mul_triple_vector.reserve_for_more_dotprods(n),
        }
    }

    fn record_dot_prod(&mut self, a_i: &[Vec<GF2p64>], a_ii: &[Vec<GF2p64>], b_i: &[Vec<GF2p64>], b_ii: &[Vec<GF2p64>], c_i: &Vec<GF2p64>, c_ii: &Vec<GF2p64>) {
        match self{
            TripleVector::SEMI(no_mul_triple_recording) => 
                no_mul_triple_recording.record_dot_prod(a_i, a_ii, b_i, b_ii, c_i, c_ii),
            TripleVector::MAL(mul_triple_vector) => 
                mul_triple_vector.record_dot_prod(a_i, a_ii, b_i, b_ii, c_i, c_ii),
        }
    }
    
    fn record_dot_in(&mut self, a_i: &[Vec<GF2p64>], a_ii: &[Vec<GF2p64>], b_i: &[Vec<GF2p64>], b_ii: &[Vec<GF2p64>]) {
        match self{
            TripleVector::SEMI(no_rec) => DotProdRecorder::<GF2p64>::record_dot_in(no_rec, a_i, a_ii, b_i, b_ii),
            TripleVector::MAL(rec) => rec.record_dot_in(a_i, a_ii, b_i, b_ii),
        }
    }
    
    fn record_dot_out(&mut self, c_i: &Vec<GF2p64>, c_ii: &Vec<GF2p64>) {
        match self{
            TripleVector::SEMI(no_rec) => DotProdRecorder::<GF2p64>::record_dot_out(no_rec, c_i, c_ii),
            TripleVector::MAL(rec) => rec.record_dot_out(c_i, c_ii),
        }
    }
}


pub fn verify_multiplication_triples(party: &mut MainParty, context: &mut BroadcastContext, triples: &mut [&mut (dyn MulTripleEncoder + Send + Sync)], dont_clear: bool) -> MpcResult<bool> {
    let lengths: usize = triples.iter().map(|enc| enc.len_triples_out()).sum();
    if lengths == 0 {
        return Ok(true);
    }
    let n = lengths.checked_next_power_of_two().expect("n too large");

    let r: GF2p64 = coin_flip(party, context)?;

    let mut x_vec = vec![RssShare::from(GF2p64::ZERO, GF2p64::ZERO); n];
    let mut y_vec = vec![RssShare::from(GF2p64::ZERO, GF2p64::ZERO); n];
    let mut zi = GF2p64InnerProd::new();
    let mut zii = GF2p64InnerProd::new();
    let mut weight = GF2p64::ONE;

    let mut i = 0;
    triples.iter_mut().for_each(|enc| {
        let len = enc.len_triples_out();
        // encode
        (*enc).add_triples(&mut x_vec[i..(i+len)], &mut y_vec[i..(i+len)], &mut zi, &mut zii, &mut weight, r);
        if !dont_clear {
            enc.clear();
        }
        i += len;
    });
    let z = RssShare::from(zi.sum(), zii.sum());
    // println!("add_triples_time={}s", add_triples_time.elapsed().as_secs_f64());
    verify_dot_product_opt(party, context, x_vec, y_vec, z)
}

#[rustfmt::skip]
pub fn verify_multiplication_triples_mt(party: &mut MainParty, context: &mut BroadcastContext, triples: &mut [&mut (dyn MulTripleEncoder + Send + Sync)], dont_clear: bool) -> MpcResult<bool>
{
    let length: usize = triples.iter().map(|enc| enc.len_triples_out()).sum();
    let n = length.checked_next_power_of_two().expect("n too large");
    if n < (1 << 14) {
        // don't use multi-threading for such small task
        return verify_multiplication_triples(party, context, triples, dont_clear);
    }

    let n_threads = party.num_worker_threads();
    let chunk_sizes = triples.iter().map(|enc| {
        let len = enc.len_triples_in();
        if len < 4096 {
            None
        }else{
            Some(party.chunk_size_for_task(len))
        }
    }).collect_vec();
    
    let r: Vec<GF2p64> = coin_flip_n(party, context, triples.len()*n_threads)?;

    let mut x_vec = vec![RssShare::from(GF2p64::ZERO, GF2p64::ZERO); n];
    let mut y_vec = vec![RssShare::from(GF2p64::ZERO, GF2p64::ZERO); n];

    let indices = triples.iter().map(|enc| enc.len_triples_out());
    let x_vec_chunks = split_at_indices_mut(&mut x_vec[..length], indices.clone());
    let y_vec_chunks = split_at_indices_mut(&mut y_vec[..length], indices);
    
    let z_vec = party.run_in_threadpool(|| {
        let vec: Vec<_> = triples.par_iter_mut()
            .zip_eq(x_vec_chunks.into_par_iter())
            .zip_eq(y_vec_chunks.into_par_iter())
            .zip_eq(chunk_sizes.into_par_iter())
            .zip_eq(r.par_chunks_exact(n_threads))
            .map(|((((enc, x_vec), y_vec), chunk_size), rand)| {
                match chunk_size {
                    None => {
                        // do all in a single thread
                        let mut zi = GF2p64InnerProd::new();
                        let mut zii = GF2p64InnerProd::new();
                        let mut weight = GF2p64::ONE;
                        enc.add_triples(x_vec, y_vec, &mut zi, &mut zii, &mut weight, rand[0]);
                        if !dont_clear { enc.clear() }
                        RssShare::from(zi.sum(), zii.sum())
                    },
                    Some(chunk_size) => {
                        // chunk with multiple threads
                        let mut z = RssShare::from(GF2p64::ZERO, GF2p64::ZERO);
                        enc.add_triples_par(x_vec, y_vec, &mut z, GF2p64::ONE, rand, chunk_size);
                        if !dont_clear { enc.clear() }
                        z
                    }
                }
            }).collect();
        Ok(vec)
    })?;
    // sum all z values
    let z = z_vec.into_iter().fold(RssShare::from(GF2p64::ZERO, GF2p64::ZERO), |acc, x| acc + x);

    // println!("Add triples: {}", add_triples_time.elapsed().as_secs_f64());
    verify_dot_product_opt(party, context, x_vec, y_vec, z)
}

fn split_at_indices_mut<T, I>(mut slice: &mut[T], indices: I) -> Vec<&mut[T]>
where I: IntoIterator<Item=usize>
{
    let it = indices.into_iter();
    let mut chunks = Vec::with_capacity(it.size_hint().0);
    for index in it {
        let (chunk, rest) = slice.split_at_mut(index);
        slice = rest;
        chunks.push(chunk);
    }
    chunks
}

/// Batched verification of recorded `Zp` multiplication triples
/// (random-linear-combination + Chida-style dot-product compression, the same
/// recursion as the GF(2^64) check — `verify_dot_product_opt` is field-generic
/// and `Zp` satisfies its exact bound set, see the `Zp` unit tests).
///
/// Independent of the GF(2^64) check: run any time between recording (e.g. by
/// `dabit::zp_mul_rss`) and releasing values derived from the multiplications.
/// Soundness error is O(log n / p) ≈ 2^-55 for realistic batch sizes.
/// Clears `triples` unless `dont_clear` is set.
pub fn verify_zp_triples(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    triples: &mut MulTripleVector<Zp>,
    dont_clear: bool,
) -> MpcResult<bool> {
    let len = triples.len();
    if len == 0 {
        return Ok(true);
    }
    let n = len.checked_next_power_of_two().expect("n too large");
    let r: Zp = coin_flip(party, context)?;

    // Random linear combination z = sum_i r^i * c_i with x_i scaled by r^i;
    // padding with zero triples (0 * 0 = 0 is a valid triple).
    let mut x_vec = vec![RssShare::from(Zp::ZERO, Zp::ZERO); n];
    let mut y_vec = vec![RssShare::from(Zp::ZERO, Zp::ZERO); n];
    let mut z = RssShare::from(Zp::ZERO, Zp::ZERO);
    let mut weight = Zp::ONE;
    let (ai, aii, bi, bii, ci, cii) = triples.as_mut_slices();
    for i in 0..len {
        x_vec[i] = RssShare::from(ai[i], aii[i]) * weight;
        y_vec[i] = RssShare::from(bi[i], bii[i]);
        z = z + RssShare::from(ci[i], cii[i]) * weight;
        weight = weight * r;
    }
    if !dont_clear {
        triples.clear();
    }
    verify_dot_product_opt(party, context, x_vec, y_vec, z)
}

/// Protocol to verify the component-wise multiplication triples
///
/// This protocol assumes that the input vectors are of length 2^n for some n.
fn verify_dot_product<F: Field + DigestExt + HasTwo + Invertible>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    x_vec: Vec<RssShare<F>>,
    y_vec: Vec<RssShare<F>>,
    z: RssShare<F>,
) -> MpcResult<bool>
where
    F: InnerProduct,
{
    let n = x_vec.len();
    debug_assert_eq!(n, y_vec.len());
    debug_assert!(n & (n - 1) == 0 && n != 0);
    if n == 1 {
        return check_triple(party, context, x_vec[0], y_vec[0], z);
    }
    // let inner_prod_time = Instant::now();
    // Compute dot products
    let f1: RssShareVec<F> = x_vec.iter().skip(1).step_by(2).copied().collect();
    let g1: RssShareVec<F> = y_vec.iter().skip(1).step_by(2).copied().collect();
    let f2: Vec<_> = x_vec
        .chunks(2)
        .map(|c| c[0] + (c[1] - c[0]) * F::TWO)
        .collect();
    let g2: Vec<_> = y_vec
        .chunks(2)
        .map(|c| c[0] + (c[1] - c[0]) * F::TWO)
        .collect();
    // let inner_prod_time = inner_prod_time.elapsed();
    // let weak_inner_prod_time = Instant::now();
    let mut hs = [F::ZERO; 2];
    hs[0] = F::weak_inner_product(&f1, &g1);
    hs[1] = F::weak_inner_product(&f2, &g2);
    // let weak_inner_prod_time = weak_inner_prod_time.elapsed();
    // let ss_rss_time = Instant::now();
    let h = ss_to_rss_shares(party, &hs)?;
    // let ss_rss_time = ss_rss_time.elapsed();
    let h1 = &h[0];
    let h2 = &h[1];
    let h0 = z - *h1;
    // let coin_flip_time = Instant::now();
    // Coin flip
    let r = coin_flip(party, context)?;
    // For large F this is very unlikely
    debug_assert!(r != F::ZERO && r != F::ONE);
    // let coin_flip_time = coin_flip_time.elapsed();

    // let poly_time = Instant::now();
    // Compute polynomials
    let fr: Vec<_> = x_vec.chunks(2).map(|c| c[0] + (c[1] - c[0]) * r).collect();
    let gr: Vec<_> = y_vec.chunks(2).map(|c| c[0] + (c[1] - c[0]) * r).collect();
    // let poly_time = poly_time.elapsed();
    let hr = lagrange_deg2(&h0, h1, h2, r);
    // println!("[vfy-dp] n={}, inner_prod_time={}s, weak_inner_prod_time={}s, ss_rss_time={}s, coin_flip_time={}s, poly_time={}s", n, inner_prod_time.as_secs_f32(), weak_inner_prod_time.as_secs_f32(), ss_rss_time.as_secs_f32(), coin_flip_time.as_secs_f32(), poly_time.as_secs_f32());
    verify_dot_product(party, context, fr, gr, hr)
}

/// Evaluates the pairing polynomials `f_k(t) = x_{2k} + (x_{2k+1} - x_{2k})*t`
/// (so `f_k(0) = x_{2k}`, `f_k(1) = x_{2k+1}` in any characteristic) at `t = r`.
#[inline]
fn compute_poly<F: Field>(x: &mut [RssShare<F>], r: F) {
    let mut i = 0;
    for k in 0..x.len()/2 {
        x[k] = x[i] + (x[i+1] - x[i])*r;
        i += 2;
    }
}

#[inline]
fn compute_poly_dst<F: Field>(dst: &mut [RssShare<F>], x: &[RssShare<F>], r: F) {
    debug_assert_eq!(2*dst.len(), x.len());
    let mut i = 0;
    for k in 0..dst.len() {
        dst[k] = x[i] + (x[i+1] - x[i])*r;
        i += 2;
    }
}

fn verify_dot_product_opt<F: Field + DigestExt + HasTwo + Invertible + Send + Sync>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    mut x_vec: Vec<RssShare<F>>,
    mut y_vec: Vec<RssShare<F>>,
    z: RssShare<F>,
) -> MpcResult<bool>
where
    F: InnerProduct,
{
    let n = x_vec.len();
    // println!("n = {}", n);
    debug_assert_eq!(n, y_vec.len());
    debug_assert!(n & (n - 1) == 0 && n != 0);
    if n == 1 {
        return check_triple(party, context, x_vec[0], y_vec[0], z);
    }
    let multi_threading = party.has_multi_threading() && n >= (1 << 13);
    let mut chunk_size = if x_vec.len() % party.num_worker_threads() == 0 {
        x_vec.len() / party.num_worker_threads()
    }else{
        x_vec.len() / party.num_worker_threads() +1
    };
    // make sure chunk size is even
    if chunk_size % 2 != 0 { chunk_size += 1 }

    // let inner_prod_time = Instant::now();
    let mut hs = [F::ZERO; 2];
    if !multi_threading {
        hs[0] = F::weak_inner_product2(&x_vec[1..], &y_vec[1..]);
        hs[1] = F::weak_inner_product3(&x_vec, &y_vec);
    }else{
        let mut h0 = F::ZERO;
        let mut h1 = F::ZERO;
        party.run_in_threadpool_scoped(|scope| {
            scope.spawn(|_| { 
                h0 = x_vec[1..]
                    .par_chunks(chunk_size)
                    .zip_eq(y_vec[1..].par_chunks(chunk_size))
                    .map(|(x,y)| F::weak_inner_product2(x, y))
                    .reduce(|| F::ZERO, |sum, v| sum + v);
            });
            scope.spawn(|_| {
                h1 = x_vec.par_chunks(chunk_size)
                    .zip_eq(y_vec.par_chunks(chunk_size))
                    .map(|(x,y)| F::weak_inner_product3(x, y))
                    .reduce(|| F::ZERO, |sum, v| sum + v);
            });
        });
        hs[0] = h0;
        hs[1] = h1;
    }
    
    // let inner_prod_time = inner_prod_time.elapsed();
    // let ss_rss_time = Instant::now();
    let h = ss_to_rss_shares(party, &hs)?;
    // let ss_rss_time = ss_rss_time.elapsed();
    let h1 = &h[0];
    let h2 = &h[1];
    let h0 = z - *h1;
    // let coin_flip_time = Instant::now();
    // Coin flip
    let r = coin_flip(party, context)?;
    // For large F this is very unlikely
    debug_assert!(r != F::ZERO && r != F::ONE);
    // let coin_flip_time = coin_flip_time.elapsed();

    // let poly_time = Instant::now();
    // Compute polynomials
    let (fr, gr) = if !multi_threading {
        compute_poly(&mut x_vec, r);
        x_vec.truncate(x_vec.len()/2);
        let fr = x_vec;
        compute_poly(&mut y_vec, r);
        y_vec.truncate(y_vec.len()/2);
        let gr = y_vec;
        (fr, gr)
    }else{
        let mut fr = vec![RssShare::from(F::ZERO, F::ZERO); x_vec.len()/2];
        let mut gr = vec![RssShare::from(F::ZERO, F::ZERO); x_vec.len()/2];
        party.run_in_threadpool_scoped(|scope| {
            scope.spawn(|_| {
                fr.par_chunks_mut(chunk_size/2)
                .zip_eq(x_vec.par_chunks(chunk_size))
                .for_each(|(dst, x)| {
                    compute_poly_dst(dst, x, r);
                });
            });

            scope.spawn(|_| {
                gr.par_chunks_mut(chunk_size/2)
                .zip_eq(y_vec.par_chunks(chunk_size))
                .for_each(|(dst, y)| {
                    compute_poly_dst(dst, y, r);
                });
            });
        });
        (fr, gr)
    };
    
    // let poly_time = poly_time.elapsed();
    let hr = lagrange_deg2(&h0, h1, h2, r);
    // println!("[vfy-dp-opt] n={}, inner_prod_time={}s, ss_rss_time={}s, coin_flip_time={}s, poly_time={}s", n, inner_prod_time.as_secs_f32(), ss_rss_time.as_secs_f32(), coin_flip_time.as_secs_f32(), poly_time.as_secs_f32());
    verify_dot_product_opt(party, context, fr, gr, hr)
}

/// Protocol 1 CheckTriple
fn check_triple<F: Field + DigestExt>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    x: RssShare<F>,
    y: RssShare<F>,
    z: RssShare<F>,
) -> MpcResult<bool>
where
    F: InnerProduct,
{
    // Generate RSS sharing of random value
    let x_prime = party.generate_random(1)[0];
    let z_prime = weak_mult(party, &x_prime, &y)?;
    let t = coin_flip(party, context)?;
    let rho = reconstruct(party, context, x + x_prime * t)?;
    reconstruct(party, context, z + z_prime * t - y * rho).map(|x| x.is_zero())
}

/// Shared lagrange evaluation of the polynomial h at position x for given (shared) points h(0), h(1), h(2)
#[inline]
fn lagrange_deg2<F: Field + HasTwo + Invertible>(
    h0: &RssShare<F>,
    h1: &RssShare<F>,
    h2: &RssShare<F>,
    x: F,
) -> RssShare<F> {
    // Lagrange weights for interpolation points {0, 1, TWO}, char-agnostic.
    // w0^-1 = (0-1)*(0-2) = 2
    let w0 = F::TWO.inverse();
    // w1^-1 = (1-0)*(1-2) = 1 - TWO
    let w1 = (F::ONE - F::TWO).inverse();
    // w2^-1 = (2-0)*(2-1) = TWO*(TWO-1)
    let w2 = (F::TWO * (F::TWO - F::ONE)).inverse();
    let l0 = w0 * (x - F::ONE) * (x - F::TWO);
    let l1 = w1 * x * (x - F::TWO);
    let l2 = w2 * x * (x - F::ONE);
    // Lagrange interpolation
    (*h0) * l0 + (*h1) * l1 + (*h2) * l2
}

fn reconstruct<F: Field + DigestExt>(party: &mut MainParty, context: &mut BroadcastContext, rho: RssShare<F>) -> MpcResult<F> {
    party
        .open_rss(
            context,
            slice::from_ref(&rho.si),
            slice::from_ref(&rho.sii),
        )
        .map(|v| v[0])
}

/// Coin flip protocol returns a random value in F
///
/// Generates a sharing of a random value that is then reconstructed globally.
fn coin_flip<F: Field + DigestExt>(party: &mut MainParty, context: &mut BroadcastContext) -> MpcResult<F> {
    let r: RssShare<F> = party.generate_random(1)[0];
    reconstruct(party, context, r)
}

/// Coin flip protocol returns a n random values in F
///
/// Generates a sharing of a n random values that is then reconstructed globally.
fn coin_flip_n<F: Field + DigestExt>(party: &mut MainParty, context: &mut BroadcastContext, n: usize) -> MpcResult<Vec<F>> {
    let (r_i, r_ii): (Vec<_>, Vec<_>) = party.generate_random::<F>(n).into_iter().map(|rss| (rss.si, rss.sii)).unzip();
    party.open_rss(context, &r_i, &r_ii)
}

/// Computes the components wise multiplication of replicated shared x and y.
fn weak_mult<F: Field + Copy + Sized>(
    party: &mut MainParty,
    x: &RssShare<F>,
    y: &RssShare<F>,
) -> MpcResult<RssShare<F>>
where
    F: InnerProduct,
{
    // Compute a sum sharing of x*y
    let zs = F::weak_inner_product(&[*x], &[*y]);
    single_ss_to_rss_shares(party, zs)
}

/// Converts a vector of sum sharings into a replicated sharing
#[inline]
fn ss_to_rss_shares<F: Field + Copy + Sized>(
    party: &mut MainParty,
    sum_shares: &[F],
) -> MpcResult<RssShareVec<F>> {
    let n = sum_shares.len();
    let alphas = party.generate_alpha(n);
    let s_i: Vec<F> = sum_shares.iter().zip(alphas).map(|(s, a)| *s + a).collect();
    let mut s_ii = vec![F::ZERO; n];
    party.send_field_slice(Direction::Previous, &s_i);
    party.receive_field_slice(Direction::Next, &mut s_ii)
        .rcv()?;
    party.wait_for_completion();
    let res: RssShareVec<F> = s_ii
        .iter()
        .zip(s_i)
        .map(|(sii, si)| RssShare::from(si, *sii))
        .collect();
    Ok(res)
}

/// Converts a sum sharing into a replicated sharing
#[inline]
fn single_ss_to_rss_shares<F: Field + Copy + Sized>(
    party: &mut MainParty,
    sum_share: F,
) -> MpcResult<RssShare<F>> {
    // Convert zs to RSS sharing
    let s_i = [sum_share + party.generate_alpha(1).next().unwrap()];
    let mut s_ii = [F::ZERO; 1];
    party.send_field_slice(Direction::Previous, &s_i);
    party.receive_field_slice(Direction::Next, &mut s_ii)
        .rcv()?;
    party.io().wait_for_completion();
    Ok(RssShare::from(s_i[0], s_ii[0]))
}

#[cfg(test)]
mod test {
    use maestro::rep3_core::share::{HasZero, RssShare};
    use maestro::share::{Field, InnerProduct};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use crate::share::zp::Zp;
    use super::{compute_poly, lagrange_deg2};

    // `lagrange_deg2` must interpolate the quadratic through {0,1,TWO} in any
    // field; here we check it over the prime field Z_p (odd characteristic),
    // which the char-2 weights would have gotten wrong.
    #[test]
    fn lagrange_deg2_interpolates_over_zp() {
        let z = |v: u64| Zp::new(v);
        let rss = |v: Zp| RssShare::from(v, Zp::new(0));
        let (c0, c1, c2) = (z(7), z(123456), z(999983));
        let q = |t: Zp| c0 + c1 * t + c2 * t * t;
        let h0 = rss(q(z(0)));
        let h1 = rss(q(z(1)));
        let h2 = rss(q(z(2)));
        for r in [z(3), z(12345), z((1u64 << 60) + 5)] {
            let got = lagrange_deg2(&h0, &h1, &h2, r);
            assert_eq!(got.si, q(r), "interpolation mismatch at r={}", r.value());
        }
    }

    // Completeness of the compression algebra over an odd-characteristic field:
    // runs the honest recursion of `verify_dot_product_opt` on plain values
    // (sii = 0, so the weak inner products equal the true ones) and checks that
    // the carried claim stays true down to the length-1 base case. This is the
    // check that fails if the pairing polynomial and the Lagrange nodes {0,1,2}
    // are inconsistent.
    #[test]
    fn compression_recursion_is_complete_over_zp() {
        let mut rng = StdRng::seed_from_u64(8);
        let n = 64;
        let rand_vec = |rng: &mut StdRng| -> Vec<RssShare<Zp>> {
            (0..n).map(|_| RssShare::from(Zp::new(rng.gen()), Zp::ZERO)).collect()
        };
        let mut x_vec = rand_vec(&mut rng);
        let mut y_vec = rand_vec(&mut rng);
        let true_ip = |x: &[RssShare<Zp>], y: &[RssShare<Zp>]| {
            Zp::inner_product(
                &x.iter().map(|s| s.si).collect::<Vec<_>>(),
                &y.iter().map(|s| s.si).collect::<Vec<_>>(),
            )
        };
        let mut z = RssShare::from(true_ip(&x_vec, &y_vec), Zp::ZERO);

        while x_vec.len() > 1 {
            let h1 = RssShare::from(Zp::weak_inner_product2(&x_vec[1..], &y_vec[1..]), Zp::ZERO);
            let h2 = RssShare::from(Zp::weak_inner_product3(&x_vec, &y_vec), Zp::ZERO);
            let h0 = z - h1;
            let r = Zp::new(rng.gen());
            compute_poly(&mut x_vec, r);
            x_vec.truncate(x_vec.len() / 2);
            compute_poly(&mut y_vec, r);
            y_vec.truncate(y_vec.len() / 2);
            z = lagrange_deg2(&h0, &h1, &h2, r);
            // invariant: the carried claim <x, y> = z still holds
            assert_eq!(true_ip(&x_vec, &y_vec), z.si, "claim broken at length {}", x_vec.len());
        }
        assert_eq!(x_vec[0].si * y_vec[0].si, z.si);
    }
}


