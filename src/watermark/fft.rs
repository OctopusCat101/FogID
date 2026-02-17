//! FFT spectral-band watermark engine for robust audio watermarking.
//!
//! Inspired by audiowmark's approach: instead of embedding in absolute DCT
//! coefficients (fragile to lossy codecs), we modulate relative magnitudes
//! of frequency bands. Extraction compares "up" vs "down" band magnitudes —
//! the *ratio* survives lossy compression even when absolute values change.
//!
//! Architecture:
//! - Non-overlapping frames with Hann window for FFT analysis
//! - Delta-based embedding: compute spectral modification, IFFT the delta,
//!   and add directly to the original signal. Avoids COLA reconstruction issues.
//! - Extraction: windowed FFT → compare up/down band magnitudes.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const FRAME_SIZE: usize = 1024;

/// Number of paired bands per data bit per frame.
/// Each pair consists of two adjacent bins; one is "up", one is "down".
/// Adjacent bins have similar baseline energy, so the modulation stands out.
const PAIRS_PER_FRAME: usize = 30;

/// Frequency band range (bin indices). At 44100 Hz with frame_size=1024:
/// bin_freq = bin_index * 44100 / 1024 ≈ bin_index * 43.07 Hz
/// min_band=20 → ~861 Hz, max_band=100 → ~4307 Hz
const MIN_BAND: usize = 20;
const MAX_BAND: usize = 100;

/// Modulation strength (multiplicative delta for magnitude).
const WATER_DELTA: f64 = 0.08;

/// Hop size = frame_size (non-overlapping frames).
pub const HOP_SIZE: usize = FRAME_SIZE;


// ---------------------------------------------------------------------------
// Band selection (key-seeded pseudo-random)
// ---------------------------------------------------------------------------

/// For a given bit index and seed, select paired "up" and "down" frequency bands.
///
/// Pairs adjacent bins (e.g., 20&21, 22&23, ...) so baseline magnitudes match.
/// Randomly assigns which bin in each pair is "up" and which is "down".
fn select_bands(bit_index: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let bit_seed = seed.wrapping_add(bit_index as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut rng = ChaCha8Rng::seed_from_u64(bit_seed);

    // Create pairs from adjacent bins: (20,21), (22,23), ...
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut b = MIN_BAND;
    while b + 1 <= MAX_BAND {
        pairs.push((b, b + 1));
        b += 2;
    }
    pairs.shuffle(&mut rng);

    let mut up_bands = Vec::with_capacity(PAIRS_PER_FRAME);
    let mut down_bands = Vec::with_capacity(PAIRS_PER_FRAME);

    for &(a, b) in pairs.iter().take(PAIRS_PER_FRAME) {
        // Randomly assign which bin is up and which is down
        // Use a simple deterministic choice based on the pair seed
        let pair_seed = bit_seed.wrapping_add(a as u64).wrapping_mul(0x517CC1B727220A95);
        if pair_seed % 2 == 0 {
            up_bands.push(a);
            down_bands.push(b);
        } else {
            up_bands.push(b);
            down_bands.push(a);
        }
    }

    (up_bands, down_bands)
}

// ---------------------------------------------------------------------------
// FFT Engine
// ---------------------------------------------------------------------------

pub struct FftEngine {
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
}

impl FftEngine {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(FRAME_SIZE);
        let fft_inverse = planner.plan_fft_inverse(FRAME_SIZE);

        Self {
            fft_forward,
            fft_inverse,
        }
    }

    /// Embed one bit into a frame by direct spectral modification.
    ///
    /// FFT → modulate up/down bands → IFFT.
    /// Returns the modified frame (replaces original).
    pub fn embed_frame(
        &self,
        samples: &[f32],
        bit: u8,
        bit_index: usize,
        seed: u64,
    ) -> Vec<f32> {
        debug_assert_eq!(samples.len(), FRAME_SIZE);

        let bit_val = if bit >= 128 { 1 } else { 0 };
        let (up_bands, down_bands) = select_bands(bit_index, seed);

        // FFT (no windowing — spectral leakage is acceptable because
        // we compare up vs down band sums, and leakage affects both equally)
        let mut spectrum: Vec<Complex<f64>> = samples
            .iter()
            .map(|&s| Complex::new(s as f64, 0.0))
            .collect();
        self.fft_forward.process(&mut spectrum);

        // Modulate: bit=1 → boost up, attenuate down; bit=0 → vice versa
        let (boost_bands, attenuate_bands) = if bit_val == 1 {
            (&up_bands, &down_bands)
        } else {
            (&down_bands, &up_bands)
        };

        for &band in boost_bands {
            if band < spectrum.len() {
                let mag = spectrum[band].norm();
                let phase = spectrum[band].arg();
                let new_mag = mag * (1.0 + WATER_DELTA);
                spectrum[band] = Complex::from_polar(new_mag, phase);
                // Maintain conjugate symmetry for real-valued output
                let mirror = FRAME_SIZE - band;
                if mirror < spectrum.len() && mirror != band {
                    spectrum[mirror] = spectrum[band].conj();
                }
            }
        }

        for &band in attenuate_bands {
            if band < spectrum.len() {
                let mag = spectrum[band].norm();
                let phase = spectrum[band].arg();
                let new_mag = mag * (1.0 - WATER_DELTA);
                spectrum[band] = Complex::from_polar(new_mag, phase);
                let mirror = FRAME_SIZE - band;
                if mirror < spectrum.len() && mirror != band {
                    spectrum[mirror] = spectrum[band].conj();
                }
            }
        }

        // IFFT + normalize
        self.fft_inverse.process(&mut spectrum);
        let norm = FRAME_SIZE as f64;
        spectrum
            .iter()
            .map(|c| (c.re / norm) as f32)
            .collect()
    }

    /// Extract raw soft decision (signed) for one bit from a frame.
    ///
    /// Uses per-pair normalized comparison to cancel baseline magnitude bias.
    /// For each (up, down) pair: contribution = (up_mag - down_mag) / (up_mag + down_mag).
    /// Positive sum = bit 1, negative sum = bit 0.
    pub fn extract_frame_soft(
        &self,
        samples: &[f32],
        bit_index: usize,
        seed: u64,
    ) -> f64 {
        debug_assert_eq!(samples.len(), FRAME_SIZE);

        let (up_bands, down_bands) = select_bands(bit_index, seed);

        let mut spectrum: Vec<Complex<f64>> = samples
            .iter()
            .map(|&s| Complex::new(s as f64, 0.0))
            .collect();
        self.fft_forward.process(&mut spectrum);

        // Per-pair normalized comparison
        let mut soft = 0.0f64;
        for (&ub, &db) in up_bands.iter().zip(down_bands.iter()) {
            if ub < spectrum.len() && db < spectrum.len() {
                let up_mag = spectrum[ub].norm();
                let down_mag = spectrum[db].norm();
                let total = up_mag + down_mag;
                if total > 1e-12 {
                    soft += (up_mag - down_mag) / total;
                }
            }
        }

        soft
    }
}

impl Default for FftEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Public embedding / extraction API
// ---------------------------------------------------------------------------

/// Embed watermark bits into an audio buffer.
///
/// Cyclic embedding: each frame carries one bit, cycling through all bits.
/// Longer audio = more frames per bit = stronger watermark.
///
/// `bits` is the shuffled watermark bit array (0 or 255 values).
/// Returns the watermarked audio buffer (same length as input).
pub fn embed_overlap_add(
    samples: &[f32],
    bits: &[u8],
    seed: u64,
) -> Vec<f32> {
    let engine = FftEngine::new();
    let total_bits = bits.len();
    if total_bits == 0 {
        return samples.to_vec();
    }

    let mut output = samples.to_vec();
    let mut frame_idx = 0usize;
    let mut pos = 0usize;

    while pos + FRAME_SIZE <= samples.len() {
        // Cyclic: frame_idx maps to bit_idx via modulo
        let bit_idx = frame_idx % total_bits;

        let modified = engine.embed_frame(
            &samples[pos..pos + FRAME_SIZE],
            bits[bit_idx],
            bit_idx,
            seed,
        );

        output[pos..pos + FRAME_SIZE].copy_from_slice(&modified);

        frame_idx += 1;
        pos += HOP_SIZE;
    }

    output
}

/// Extract watermark bits from an audio buffer.
///
/// Cyclic extraction: each frame votes for one bit position.
/// Soft decisions are accumulated across multiple cycles.
/// Returns soft decisions for each bit position.
pub fn extract_soft_decisions(
    samples: &[f32],
    bit_count: usize,
    seed: u64,
) -> Vec<f64> {
    let engine = FftEngine::new();
    let mut soft_sums = vec![0.0f64; bit_count];
    if bit_count == 0 {
        return soft_sums;
    }

    let mut frame_idx = 0usize;
    let mut pos = 0usize;

    while pos + FRAME_SIZE <= samples.len() {
        let bit_idx = frame_idx % bit_count;

        let soft = engine.extract_frame_soft(
            &samples[pos..pos + FRAME_SIZE],
            bit_idx,
            seed,
        );
        soft_sums[bit_idx] += soft;

        frame_idx += 1;
        pos += HOP_SIZE;
    }

    soft_sums
}

/// Convert soft decisions to hard bits (0 or 255 for text.rs compatibility).
pub fn soft_to_hard_bits(soft: &[f64]) -> Vec<u8> {
    soft.iter()
        .map(|&v| if v > 0.0 { 255u8 } else { 0u8 })
        .collect()
}

// ---------------------------------------------------------------------------
// Alignment-search helpers (precomputed spectra)
// ---------------------------------------------------------------------------

/// Precompute FFT spectra for all non-overlapping frames.
pub fn precompute_frame_spectra(samples: &[f32]) -> Vec<Vec<Complex<f64>>> {
    let engine = FftEngine::new();
    let mut spectra = Vec::new();
    let mut pos = 0;
    while pos + FRAME_SIZE <= samples.len() {
        let mut spectrum: Vec<Complex<f64>> = samples[pos..pos + FRAME_SIZE]
            .iter()
            .map(|&s| Complex::new(s as f64, 0.0))
            .collect();
        engine.fft_forward.process(&mut spectrum);
        spectra.push(spectrum);
        pos += HOP_SIZE;
    }
    spectra
}

/// Precompute (up_bands, down_bands) for every bit index 0..bit_count.
pub fn precompute_bands(bit_count: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    (0..bit_count)
        .map(|bit_idx| select_bands(bit_idx, seed))
        .collect()
}

/// Accumulate soft decisions from precomputed spectra with a cyclic frame offset.
///
/// Frame `i` is treated as carrying bit `(i + frame_offset) % bands.len()`.
pub fn soft_decisions_from_spectra(
    spectra: &[Vec<Complex<f64>>],
    bands: &[(Vec<usize>, Vec<usize>)],
    frame_offset: usize,
) -> Vec<f64> {
    let bit_count = bands.len();
    let mut soft_sums = vec![0.0f64; bit_count];
    if bit_count == 0 {
        return soft_sums;
    }

    for (fi, spectrum) in spectra.iter().enumerate() {
        let bit_idx = (fi + frame_offset) % bit_count;
        let (ref up_bands, ref down_bands) = bands[bit_idx];

        let mut soft = 0.0f64;
        for (&ub, &db) in up_bands.iter().zip(down_bands.iter()) {
            if ub < spectrum.len() && db < spectrum.len() {
                let up_mag = spectrum[ub].norm();
                let down_mag = spectrum[db].norm();
                let total = up_mag + down_mag;
                if total > 1e-12 {
                    soft += (up_mag - down_mag) / total;
                }
            }
        }

        soft_sums[bit_idx] += soft;
    }

    soft_sums
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_full_pipeline_roundtrip() {
        let text = "Hello";
        let bits = crate::watermark::text::text_to_bits(text).unwrap();

        // Ensure multiple votes per watermark bit for robust hard decisions.
        // frames = floor((len - FRAME_SIZE) / HOP_SIZE) + 1
        let votes_per_bit = 4usize;
        let required_frames = bits.len() * votes_per_bit;
        let required_samples = FRAME_SIZE + (required_frames.saturating_sub(1)) * HOP_SIZE;
        let num_samples = required_samples.max(44100 * 6);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let audio: Vec<f32> = (0..num_samples)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();
        let seed = 12345u64;

        use crate::watermark::core;
        let shuffled = core::prepare_bits(&bits, seed);

        // Embed
        let watermarked = embed_overlap_add(&audio, &shuffled, seed);
        assert_eq!(watermarked.len(), audio.len());

        // Extract
        let soft = extract_soft_decisions(&watermarked, bits.len(), seed);
        let unshuffled = core::unshuffle_f64(&soft, seed);
        let candidate = soft_to_hard_bits(&unshuffled);

        assert!(
            crate::watermark::text::check_watermark_magic(&candidate),
            "Magic bytes not detected in extracted watermark"
        );

        let recovered = crate::watermark::text::bits_to_text(&candidate).unwrap();
        assert_eq!(recovered, text);
    }

    #[test]
    fn test_single_frame_embed_extract() {
        // Broadband frame with energy in the embedding band range
        let freqs = [900.0f32, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0, 4200.0];
        let frame: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| {
                let t = i as f32 / 44100.0;
                let mut v = 0.0f32;
                for &f in &freqs {
                    v += (2.0 * std::f32::consts::PI * f * t).sin();
                }
                v / freqs.len() as f32 * 0.5
            })
            .collect();

        let engine = FftEngine::new();
        let seed = 42u64;

        // Test bit=1
        let modified1 = engine.embed_frame(&frame, 255, 0, seed);
        let soft1 = engine.extract_frame_soft(&modified1, 0, seed);
        eprintln!("bit=1 soft decision: {soft1}");
        assert!(soft1 > 0.0, "Expected positive soft for bit=1, got {soft1}");

        // Test bit=0
        let modified0 = engine.embed_frame(&frame, 0, 0, seed);
        let soft0 = engine.extract_frame_soft(&modified0, 0, seed);
        eprintln!("bit=0 soft decision: {soft0}");
        assert!(soft0 < 0.0, "Expected negative soft for bit=0, got {soft0}");
    }

    #[test]
    fn test_band_selection_deterministic() {
        let (up1, down1) = select_bands(5, 99999);
        let (up2, down2) = select_bands(5, 99999);
        assert_eq!(up1, up2);
        assert_eq!(down1, down2);

        let (up3, _) = select_bands(6, 99999);
        assert_ne!(up1, up3);
    }

    #[test]
    fn test_different_seeds_different_bands() {
        let (up1, _) = select_bands(0, 111);
        let (up2, _) = select_bands(0, 222);
        assert_ne!(up1, up2);
    }

    #[test]
    fn test_trimmed_pipeline_roundtrip() {
        let text = "Hello";
        let bits = crate::watermark::text::text_to_bits(text).unwrap();

        // Need enough votes per bit to survive losing some frames to trimming.
        // Random noise gives weaker per-frame SNR than real audio, so use
        // generous redundancy.
        let votes_per_bit = 10usize;
        let required_frames = bits.len() * votes_per_bit;
        let required_samples = FRAME_SIZE + (required_frames.saturating_sub(1)) * HOP_SIZE;
        let num_samples = required_samples;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let audio: Vec<f32> = (0..num_samples)
            .map(|_| rng.random_range(-0.3f32..0.3f32))
            .collect();
        let seed = 12345u64;

        use crate::watermark::core;
        let shuffled = core::prepare_bits(&bits, seed);
        let watermarked = embed_overlap_add(&audio, &shuffled, seed);

        // Verify non-trimmed precomputed path matches classic path
        let bands = precompute_bands(bits.len(), seed);
        let spectra_full = precompute_frame_spectra(&watermarked);
        let soft_precomp = soft_decisions_from_spectra(&spectra_full, &bands, 0);
        let soft_classic = extract_soft_decisions(&watermarked, bits.len(), seed);
        for i in 0..bits.len() {
            assert!(
                (soft_precomp[i] - soft_classic[i]).abs() < 1e-10,
                "precomputed vs classic mismatch at bit {i}"
            );
        }

        // Simulate trimming: remove 37 frames from the start
        let trim_frames = 37usize;
        let trim_samples = trim_frames * FRAME_SIZE;
        let trimmed = &watermarked[trim_samples..];

        // Frame-offset-aware extraction
        let spectra = precompute_frame_spectra(trimmed);

        let mut found = false;
        for frame_offset in 0..bits.len() {
            let soft = soft_decisions_from_spectra(&spectra, &bands, frame_offset);
            let unshuffled = core::unshuffle_f64(&soft, seed);
            let candidate = soft_to_hard_bits(&unshuffled);
            if crate::watermark::text::check_watermark_magic(&candidate) {
                if let Ok(recovered) = crate::watermark::text::bits_to_text(&candidate) {
                    eprintln!("found at frame_offset={frame_offset} (expected {trim_frames})");
                    assert_eq!(recovered, text);
                    found = true;
                    break;
                }
            }
        }

        assert!(found, "Failed to extract watermark from trimmed audio");
    }
}
