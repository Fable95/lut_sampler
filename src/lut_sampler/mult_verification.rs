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
    util::mul_triple_vec::MulTripleEncoder
};


/// Protocol `8` to verify the multiplication triples at the end of the protocol.
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
        .map(|c| c[0] + (c[0] + c[1]) * F::TWO)
        .collect();
    let g2: Vec<_> = y_vec
        .chunks(2)
        .map(|c| c[0] + (c[0] + c[1]) * F::TWO)
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
    let fr: Vec<_> = x_vec.chunks(2).map(|c| c[0] + (c[0] + c[1]) * r).collect();
    let gr: Vec<_> = y_vec.chunks(2).map(|c| c[0] + (c[0] + c[1]) * r).collect();
    // let poly_time = poly_time.elapsed();
    let hr = lagrange_deg2(&h0, h1, h2, r);
    // println!("[vfy-dp] n={}, inner_prod_time={}s, weak_inner_prod_time={}s, ss_rss_time={}s, coin_flip_time={}s, poly_time={}s", n, inner_prod_time.as_secs_f32(), weak_inner_prod_time.as_secs_f32(), ss_rss_time.as_secs_f32(), coin_flip_time.as_secs_f32(), poly_time.as_secs_f32());
    verify_dot_product(party, context, fr, gr, hr)
}

#[inline]
fn compute_poly<F: Field>(x: &mut [RssShare<F>], r: F) {
    let mut i = 0;
    for k in 0..x.len()/2 {
        x[k] = x[i] + (x[i] + x[i+1])*r;
        i += 2;
    }
}

#[inline]
fn compute_poly_dst<F: Field>(dst: &mut [RssShare<F>], x: &[RssShare<F>], r: F) {
    debug_assert_eq!(2*dst.len(), x.len());
    let mut i = 0;
    for k in 0..dst.len() {
        dst[k] = x[i] + (x[i] + x[i+1])*r;
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
    // Lagrange weights
    // w0^-1 = (1-0)*(2-0) = 1*2 = 2
    let w0 = F::TWO.inverse();
    // w1^-1 = (0-1)*(2-1) = 1*(2-1) = (2-1) = 2+1
    let w1 = (F::TWO + F::ONE).inverse();
    // w2^-1 = (0-2)*(1-2) = 2*(1+2) = 2 * (2+1)
    let w2 = w0 * w1;
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


