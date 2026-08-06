// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Fabian Schmid

//! daBit-based binary -> `Z_p` share conversion (B2A).
//!
//! A daBit is a pair `([r]^B, [r]^Zp)` of sharings of the *same* random bit
//! `r ∈ {0,1}` in the binary and the arithmetic domain. With `bit_width`
//! daBits, a binary-shared value `x < 2^bit_width` converts to a `Zp` sharing
//! with one `bit_width`-bit opening plus local arithmetic.
//!
//! The three phases are independently schedulable:
//! - **Generation** ([`generate_dabits`], offline, 2 rounds per batch) fills a
//!   caller-owned [`DaBitStore`] and records its `Zp` multiplication triples
//!   into a caller-owned recorder (`MulTripleVector<Zp>` in the malicious
//!   setting, `NoMulTripleRecording` in the semi-honest setting).
//! - **Use** ([`b2a_many`], online, 1 round) consumes daBits from the store.
//! - **Verification** (`mult_verification::verify_zp_triples`) checks the
//!   recorded triples in one batch, any time between recording and releasing
//!   values derived from the conversion. The opening inside [`b2a_many`] is
//!   additionally committed by `Network::check_view` (compare-views).
//!
//! For two-sided outputs from one-sided magnitudes, [`random_zp_signs`]
//! produces uniform `±1` sharings; apply one with a single [`zp_mul_rss`]
//! multiplication.

use maestro::rep3_core::{
    network::task::Direction,
    party::{broadcast::BroadcastContext, error::MpcResult, MainParty, Party},
    share::{HasZero, RssShare, RssShareVec},
};
use maestro::share::{Field, HasTwo};

use crate::{
    online::open_rss_many,
    share::{gf_template::Share, zp::Zp},
    util::mul_triple_vec::MulTripleRecorder,
};

/// Exact number of daBits needed to convert `n_samples` binary-shared values
/// of `bit_width` meaningful bits each: one daBit per bit, no slack
/// (RSS-native generation has no cut-and-choose waste).
pub const fn dabits_required(bit_width: usize, n_samples: usize) -> usize {
    bit_width * n_samples
}

/// `Zp` multiplication triples recorded per generated daBit (for sizing the
/// verification batch).
pub const ZP_TRIPLES_PER_DABIT: usize = 2;

/// `Zp` multiplication triples recorded per sign by [`random_zp_signs`]
/// (one daBit's worth for the underlying bit; applying the sign later via
/// [`zp_mul_rss`] records one more).
pub const ZP_TRIPLES_PER_SIGN: usize = ZP_TRIPLES_PER_DABIT;

/// Conversion bit width for a table whose largest value is `n_max`
/// (`bits(n_max)`, at least 1). Const-evaluable so callers can size the
/// conversion from an exported `N_MAX` table constant at compile time,
/// without any runtime scan of the table data.
pub const fn bits_for_max_value(n_max: u16) -> usize {
    let bits = (u16::BITS - n_max.leading_zeros()) as usize;
    if bits == 0 {
        1
    } else {
        bits
    }
}

/// Largest supported `bit_width`. The recombination computes the value mod
/// `p = 2^61 - 1`, so the conversion is exact iff the value is below `p`:
/// automatic for `bit_width <= 60`; at 61 bits the caller must additionally
/// ensure the value is at most `p - 1` (the all-ones value `2^61 - 1 = p`
/// wraps to 0).
pub const MAX_BIT_WIDTH: usize = 61;

/// Preprocessed daBits, grouped as one `bit_width`-bit mask per future
/// conversion. Single-use: [`b2a_many`] drains from the front and masks are
/// never handed out twice. Deliberately not `Clone`.
///
/// The `Zp` verification data lives *outside* this store (in the caller's
/// `MulTripleVector<Zp>`), so the store can be consumed while its generation
/// triples still await verification.
pub struct DaBitStore<E: Share> {
    bit_width: usize,
    /// One packed binary mask `[r]^B` per sample; low `bit_width` bits used.
    bin: Vec<RssShare<E>>,
    /// `bit_width` `Zp` bit-shares per mask, LSB-first, same order as `bin`.
    zp: Vec<RssShare<Zp>>,
}

impl<E: Share> DaBitStore<E> {
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Number of conversions this store can still serve.
    pub fn available_samples(&self) -> usize {
        self.bin.len()
    }

    /// Merge another batch of the same `bit_width` (generation in chunks).
    pub fn append(&mut self, mut other: DaBitStore<E>) {
        assert_eq!(
            self.bit_width, other.bit_width,
            "cannot merge daBit stores of different bit widths"
        );
        self.bin.append(&mut other.bin);
        self.zp.append(&mut other.zp);
    }

    /// Read access to the packed binary masks (testing/debugging; opening
    /// these burns the daBits' secrecy).
    pub fn masks(&self) -> &[RssShare<E>] {
        &self.bin
    }

    /// Read access to the `Zp` bit shares (testing/debugging; LSB-first,
    /// `bit_width` entries per mask).
    pub fn zp_bits(&self) -> &[RssShare<Zp>] {
        &self.zp
    }
}

/// Batched RSS multiplication over `Zp`: `c = a * b` component-wise, one `Zp`
/// element sent per party. Records the native `Zp` triples `(a, b, c)` into
/// `rec` for later verification with `verify_zp_triples` (pass
/// `NoMulTripleRecording` in the semi-honest setting).
pub fn zp_mul_rss<R: MulTripleRecorder<Zp>>(
    party: &mut MainParty,
    rec: &mut R,
    a: &[RssShare<Zp>],
    b: &[RssShare<Zp>],
) -> MpcResult<Vec<RssShare<Zp>>> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    // Local sum-share of the product, rerandomized with a fresh zero-sharing:
    //   c_p = a_p*b_p + a_p*b_{p+1} + a_{p+1}*b_p + alpha_p
    let alphas = party.generate_alpha::<Zp>(len);
    let ci: Vec<Zp> = alphas
        .zip(a.iter().zip(b))
        .map(|(alpha, (x, y))| x.si * y.si + x.si * y.sii + x.sii * y.si + alpha)
        .collect();
    let mut cii = vec![Zp::ZERO; len];
    party.send_field::<Zp>(Direction::Previous, ci.iter(), len);
    party.receive_field_slice(Direction::Next, &mut cii).rcv()?;
    // Record only after cii is received: the recorded c must be the full RSS share.
    let ai: Vec<Zp> = a.iter().map(|x| x.si).collect();
    let aii: Vec<Zp> = a.iter().map(|x| x.sii).collect();
    let bi: Vec<Zp> = b.iter().map(|x| x.si).collect();
    let bii: Vec<Zp> = b.iter().map(|x| x.sii).collect();
    rec.record_mul_triple(&ai, &aii, &bi, &bii, &ci, &cii);
    Ok(ci
        .into_iter()
        .zip(cii)
        .map(|(si, sii)| RssShare::from(si, sii))
        .collect())
}

/// Generates `dabits_required(bit_width, n_samples)` daBits, grouped as
/// `n_samples` packed `bit_width`-bit masks. Input-independent; two
/// communication rounds per call regardless of batch size; 2 `Zp` elements
/// sent per party per daBit.
pub fn generate_dabits<E: Share, R: MulTripleRecorder<Zp>>(
    party: &mut MainParty,
    rec: &mut R,
    bit_width: usize,
    n_samples: usize,
) -> MpcResult<DaBitStore<E>> {
    assert!(
        bit_width >= 1 && bit_width <= MAX_BIT_WIDTH,
        "bit_width must be in 1..={}",
        MAX_BIT_WIDTH
    );
    assert!(
        bit_width <= <E as Field>::NBITS,
        "bit_width {} exceeds the binary share type ({} bits)",
        bit_width,
        <E as Field>::NBITS
    );
    let bin = random_masked_bits(party, bit_width, n_samples);
    let zp = packed_bits_to_zp(party, rec, &[(&bin, bit_width)])?;
    Ok(DaBitStore { bit_width, bin, zp })
}

/// Generates `n` sharings of uniform random signs `sigma ∈ {+1, -1}` over
/// `Zp`: one free random binary bit `b` per sign, arithmetized like a daBit
/// (two batched multiplication rounds, [`ZP_TRIPLES_PER_SIGN`] recorded
/// triples per sign), then `sigma = 2b - 1` locally. Apply to a converted
/// value with a single [`zp_mul_rss`] multiplication.
///
/// Correctness caveat for samplers: `sigma * 0 = 0`, so a folded (one-sided)
/// distribution must encode the mass at zero NOT halved.
pub fn random_zp_signs<E: Share, R: MulTripleRecorder<Zp>>(
    party: &mut MainParty,
    rec: &mut R,
    n: usize,
) -> MpcResult<Vec<RssShare<Zp>>> {
    let bits = random_masked_bits::<E>(party, 1, n);
    let bits_zp = packed_bits_to_zp(party, rec, &[(&bits, 1)])?;
    let one = party.constant(Zp::ONE);
    Ok(bits_zp.iter().map(|b| *b + *b - one).collect())
}

/// RSS sharings of `n` uniform random `bit_count`-bit strings, zero
/// communication (`generate_random` + masking). Mask BOTH si and sii (Mul on
/// the share type is AND): each piece is masked identically at both of its
/// holders, so the sharing stays consistent.
fn random_masked_bits<E: Share>(
    party: &mut MainParty,
    bit_count: usize,
    n: usize,
) -> Vec<RssShare<E>> {
    let mask = E::from_usize((1usize << bit_count) - 1);
    let mut bits = party.generate_random::<E>(n);
    for r in bits.iter_mut() {
        r.si = r.si * mask;
        r.sii = r.sii * mask;
    }
    bits
}

/// Converts packed random binary bit sharings into per-bit `Zp` sharings of
/// the same bits. `bit_groups` pairs slices of packed shares with their
/// bits-per-share count; the result is in group order, share order, LSB-first
/// — one `RssShare<Zp>` per bit. All groups share the same two batched
/// multiplication rounds.
fn packed_bits_to_zp<E: Share, R: MulTripleRecorder<Zp>>(
    party: &mut MainParty,
    rec: &mut R,
    bit_groups: &[(&[RssShare<E>], usize)],
) -> MpcResult<Vec<RssShare<Zp>>> {
    let total_bits: usize = bit_groups
        .iter()
        .map(|(shares, bits_per_share)| shares.len() * bits_per_share)
        .sum();

    // Each additive piece s_q of a bit is known to exactly two parties and
    // injects into Zp for free as the trivial sharing that is s_q in slot q
    // and zero elsewhere. The vectors [s_0],[s_1],[s_2] are GLOBAL sharings
    // (party p holds piece s_p as si and piece s_{p+1} as sii), so this
    // filling is deliberately party-index dependent.
    let mut s0 = Vec::with_capacity(total_bits);
    let mut s1 = Vec::with_capacity(total_bits);
    let mut s2 = Vec::with_capacity(total_bits);
    for (shares, bits_per_share) in bit_groups {
        for share in shares.iter() {
            let si_bits = share.si.to_usize();
            let sii_bits = share.sii.to_usize();
            for j in 0..*bits_per_share {
                let si_bit = Zp::new(((si_bits >> j) & 1) as u64);
                let sii_bit = Zp::new(((sii_bits >> j) & 1) as u64);
                let inject = |q: usize| {
                    RssShare::from(
                        if q == party.i { si_bit } else { Zp::ZERO },
                        if q == (party.i + 1) % 3 { sii_bit } else { Zp::ZERO },
                    )
                };
                s0.push(inject(0));
                s1.push(inject(1));
                s2.push(inject(2));
            }
        }
    }

    // Arithmetize r = s0 ⊕ s1 ⊕ s2 via a ⊕ b = a + b - 2ab, one recorded
    // multiplication round per XOR.
    let prod01 = zp_mul_rss(party, rec, &s0, &s1)?;
    let t: Vec<RssShare<Zp>> = s0
        .iter()
        .zip(&s1)
        .zip(&prod01)
        .map(|((a, b), ab)| *a + *b - *ab * Zp::TWO)
        .collect();

    let prod = zp_mul_rss(party, rec, &t, &s2)?;
    Ok(t.iter()
        .zip(&s2)
        .zip(&prod)
        .map(|((a, b), ab)| *a + *b - *ab * Zp::TWO)
        .collect())
}

/// Converts binary-shared values into `Zp` sharings, consuming `xs.len()`
/// masks from the store (single-use). One communication round.
///
/// Caller obligations:
/// - every `x` has meaningful value `< 2^bit_width` (true for LUT outputs of
///   `bit_width`-bit tables); higher bits would not be masked and would leak.
///   At `bit_width = 61` the value must also stay below `p` (see
///   [`MAX_BIT_WIDTH`]);
/// - malicious setting: before releasing the results, run
///   `verify_zp_triples` on the generation recorder AND commit the opening
///   below via `Network::check_view` (compare-views).
pub fn b2a_many<E: Share>(
    party: &mut MainParty,
    context: &mut BroadcastContext,
    store: &mut DaBitStore<E>,
    xs: &[RssShare<E>],
) -> MpcResult<RssShareVec<Zp>> {
    let bit_width = store.bit_width;
    let n_samples = xs.len();
    assert!(
        store.available_samples() >= n_samples,
        "not enough daBits preprocessed: need {}, have {}",
        n_samples,
        store.available_samples()
    );
    let bin: Vec<RssShare<E>> = store.bin.drain(..n_samples).collect();
    let zp: Vec<RssShare<Zp>> = store.zp.drain(..n_samples * bit_width).collect();

    // (1) mask locally: c = x XOR r (Add on the binary share type is XOR)
    let masked: RssShareVec<E> = xs
        .iter()
        .zip(&bin)
        .map(|(x, r)| RssShare::from(x.si + r.si, x.sii + r.sii))
        .collect();
    // (2) open c; the exchanged views are committed by a later check_view
    let c_pub: Vec<E> = open_rss_many::<E>(party, context, &masked)?;
    // (3) local recombination
    let one = party.constant(Zp::ONE);
    let out = c_pub
        .iter()
        .enumerate()
        .map(|(sample, c)| {
            let mask_bits = &zp[sample * bit_width..(sample + 1) * bit_width];
            recombine_sample(c.to_usize(), mask_bits, one)
        })
        .collect();
    Ok(out)
}

/// Local recombination of one sample from the public `c = x ⊕ r` and the
/// mask's `Zp` bit shares (LSB-first): `x = sum_j 2^j * (c_j + (1 - 2 c_j) *
/// r_j)`, i.e. bit_j is `r_j` if `c_j == 0`, else `1 - r_j`. `one` must be
/// the party's RSS sharing of the public constant 1.
fn recombine_sample(c_bits: usize, mask_bits: &[RssShare<Zp>], one: RssShare<Zp>) -> RssShare<Zp> {
    let mut acc = RssShare::from(Zp::ZERO, Zp::ZERO);
    let mut pow = Zp::ONE; // = 2^j
    for (j, r_j) in mask_bits.iter().enumerate() {
        let x_j = if (c_bits >> j) & 1 == 1 { one - *r_j } else { *r_j };
        acc = acc + x_j * pow;
        pow = pow * Zp::TWO;
    }
    acc
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::share::zp::P;
    use maestro::rep3_core::party::RngExt;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn rand_zp(rng: &mut StdRng) -> Zp {
        let mut b = [Zp::ZERO];
        Zp::fill(rng, &mut b);
        b[0]
    }

    /// Random 3-party RSS sharing of `v` in the replicated layout
    /// (party p holds pieces (s_p, s_{p+1})).
    fn share3(v: Zp, rng: &mut StdRng) -> [RssShare<Zp>; 3] {
        let pieces = {
            let s0 = rand_zp(rng);
            let s1 = rand_zp(rng);
            [s0, s1, v - s0 - s1]
        };
        [0, 1, 2].map(|p| RssShare::from(pieces[p], pieces[(p + 1) % 3]))
    }

    /// Reconstructs the shared value and asserts the replication invariant
    /// (party p's sii equals party p+1's si).
    fn open3(shares: &[RssShare<Zp>]) -> Zp {
        assert_eq!(shares.len(), 3);
        for p in 0..3 {
            assert_eq!(shares[p].sii, shares[(p + 1) % 3].si, "replication broken");
        }
        shares.iter().fold(Zp::ZERO, |acc, s| acc + s.si)
    }

    /// The three parties' sharings of a public constant, matching
    /// `party.constant` (value in additive slot 0).
    fn constant3(v: Zp) -> [RssShare<Zp>; 3] {
        [
            RssShare::from(v, Zp::ZERO),
            RssShare::from(Zp::ZERO, Zp::ZERO),
            RssShare::from(Zp::ZERO, v),
        ]
    }

    #[test]
    fn dabits_required_is_exact() {
        assert_eq!(dabits_required(12, 100), 1200);
        assert_eq!(dabits_required(1, 1), 1);
        assert_eq!(dabits_required(8, 0), 0);
        // 2 Zp triples per daBit for sizing the verification batch
        assert_eq!(dabits_required(12, 100) * ZP_TRIPLES_PER_DABIT, 2400);
    }

    #[test]
    fn bits_for_max_value_is_exact() {
        assert_eq!(bits_for_max_value(0), 1); // degenerate all-zero table
        assert_eq!(bits_for_max_value(1), 1);
        assert_eq!(bits_for_max_value(15), 4);
        assert_eq!(bits_for_max_value(255), 8);
        assert_eq!(bits_for_max_value(511), 9);
        assert_eq!(bits_for_max_value(16383), 14);
        assert_eq!(bits_for_max_value(u16::MAX), 16);
    }

    #[test]
    fn max_bit_width_covers_zp() {
        // p - 1 = 2^61 - 2 needs 61 bits; the only excluded 61-bit value is
        // the all-ones string 2^61 - 1 = p itself.
        assert_eq!((1u64 << MAX_BIT_WIDTH) - 1, P);
    }

    /// The conversion's local recombination: for x, r < 2^bit_width and
    /// public c = x ^ r, the three parties' `recombine_sample` outputs form a
    /// valid RSS sharing of x — at every supported width.
    #[test]
    fn recombination_reconstructs_the_masked_value() {
        let mut rng = StdRng::seed_from_u64(42);
        let one = constant3(Zp::ONE);
        for bit_width in [1usize, 12, 60, MAX_BIT_WIDTH] {
            let width_mask = (1u64 << bit_width) - 1;
            // largest exactly-convertible value at this width
            let max_exact = if bit_width == MAX_BIT_WIDTH { P - 1 } else { width_mask };
            for trial in 0..20u64 {
                let x = match trial {
                    0 => 0,
                    1 => max_exact,
                    _ => (rng.gen::<u64>() & width_mask).min(max_exact),
                };
                let r = rng.gen::<u64>() & width_mask;
                let c = x ^ r;
                let mask_bit_shares: Vec<[RssShare<Zp>; 3]> = (0..bit_width)
                    .map(|j| share3(Zp::new((r >> j) & 1), &mut rng))
                    .collect();
                let out: Vec<RssShare<Zp>> = (0..3)
                    .map(|p| {
                        let mask_bits: Vec<RssShare<Zp>> =
                            mask_bit_shares.iter().map(|bit| bit[p]).collect();
                        recombine_sample(c as usize, &mask_bits, one[p])
                    })
                    .collect();
                assert_eq!(open3(&out), Zp::new(x), "x = {}, width = {}", x, bit_width);
            }
        }
    }

    /// The MAX_BIT_WIDTH caveat: at width 61, p - 1 converts exactly and the
    /// excluded all-ones value 2^61 - 1 = p wraps to 0.
    #[test]
    fn recombination_at_width_61_wraps_only_at_p() {
        let mut rng = StdRng::seed_from_u64(43);
        let one = constant3(Zp::ONE);
        for (x, expected) in [(P - 1, Zp::new(P - 1)), (P, Zp::ZERO)] {
            // r = 0, so c = x and the recombination sums x's bits directly
            let mask_bit_shares: Vec<[RssShare<Zp>; 3]> = (0..MAX_BIT_WIDTH)
                .map(|_| share3(Zp::ZERO, &mut rng))
                .collect();
            let out: Vec<RssShare<Zp>> = (0..3)
                .map(|p| {
                    let mask_bits: Vec<RssShare<Zp>> =
                        mask_bit_shares.iter().map(|bit| bit[p]).collect();
                    recombine_sample(x as usize, &mask_bits, one[p])
                })
                .collect();
            assert_eq!(open3(&out), expected, "x = {}", x);
        }
    }

    /// The sign formula of `random_zp_signs`: sigma = 2b - 1 computed on
    /// shares opens to +1 for b = 1 and to p - 1 (= -1) for b = 0.
    #[test]
    fn sign_from_bit_is_plus_minus_one() {
        let mut rng = StdRng::seed_from_u64(44);
        let one = constant3(Zp::ONE);
        for bit in [0u64, 1] {
            let b = share3(Zp::new(bit), &mut rng);
            let sigma: Vec<RssShare<Zp>> = (0..3).map(|p| b[p] + b[p] - one[p]).collect();
            let expected = if bit == 1 { Zp::ONE } else { Zp::ZERO - Zp::ONE };
            assert_eq!(open3(&sigma), expected, "bit = {}", bit);
        }
    }

    /// Sign application: the local product formula of `zp_mul_rss` (the alpha
    /// rerandomization sums to zero and is omitted) yields sigma * x, i.e.
    /// +x, -x, and — the folded-distribution caveat — 0 for x = 0.
    #[test]
    fn sign_application_flips_the_value() {
        let mut rng = StdRng::seed_from_u64(45);
        let one = constant3(Zp::ONE);
        for bit in [0u64, 1] {
            for x_val in [0u64, 1, 12345, P - 1] {
                let b = share3(Zp::new(bit), &mut rng);
                let sigma: Vec<RssShare<Zp>> = (0..3).map(|p| b[p] + b[p] - one[p]).collect();
                let x = share3(Zp::new(x_val), &mut rng);
                let product = (0..3).fold(Zp::ZERO, |acc, p| {
                    acc + sigma[p].si * x[p].si + sigma[p].si * x[p].sii + sigma[p].sii * x[p].si
                });
                let expected =
                    if bit == 1 { Zp::new(x_val) } else { Zp::ZERO - Zp::new(x_val) };
                assert_eq!(product, expected, "bit = {}, x = {}", bit, x_val);
            }
        }
    }
}
