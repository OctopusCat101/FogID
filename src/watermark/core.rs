use image::RgbImage;
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use super::dct::CosTable;
use super::dwt::{dwt2_multilevel, idwt2_multilevel};

/// U/V chrominance channels use a fraction of mod1 to avoid visible color artifacts.
/// YUV→RGB conversion amplifies U/V changes (U→B: 1.772×, V→R: 1.402×), so we
/// embed more gently in chrominance. Text encoding already provides high redundancy
/// (57× for short strings on 512×512), so reduced UV strength doesn't hurt extraction.
const UV_MOD_RATIO: f64 = 0.4;
/// Video path: fixed 8x8 block size for block-average (DC-like) QIM embedding.
const VIDEO_BLOCK_SIZE: usize = 8;
/// Video path: fixed QIM quantization step tuned for H.264 robustness.
/// With 8×8 blocks on 1080p, single-frame redundancy is ~63×; multi-frame
/// accumulation adds another order of magnitude. mod=8 keeps per-pixel
/// change ≤ ±6 (typically ±2), invisible in motion while still extractable.
const VIDEO_MOD: f64 = 8.0;
const VIDEO_BLOCK_PIXELS: usize = VIDEO_BLOCK_SIZE * VIDEO_BLOCK_SIZE;
/// Reference block variance for adaptive QIM strength (std-dev ~= 20).
const VIDEO_REF_VAR: f64 = 400.0;
const VIDEO_MOD_MIN_RATIO: f64 = 0.5;
const VIDEO_MOD_MAX_RATIO: f64 = 2.0;
const VIDEO_CONF_EPS: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct WatermarkParams {
    pub seed_wm: u64,
    pub seed_dct: u64,
    pub mod1: f64,
    pub mod2: Option<f64>,
    pub block_size: usize,
    pub dwt_deep: usize,
}

impl Default for WatermarkParams {
    fn default() -> Self {
        Self {
            seed_wm: 4399,
            seed_dct: 2333,
            mod1: 24.0,
            mod2: None,
            block_size: 4,
            dwt_deep: 1,
        }
    }
}

impl WatermarkParams {
    /// Derive parameters from a password string using SHA-256.
    pub fn from_password(password: &str) -> Self {
        let hash = Sha256::digest(password.as_bytes());
        let seed_wm = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        let seed_dct = u64::from_le_bytes(hash[8..16].try_into().unwrap());
        Self {
            seed_wm,
            seed_dct,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Color space conversion (BT.601)
// ---------------------------------------------------------------------------

fn rgb_to_yuv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let u = -0.169 * r - 0.331 * g + 0.500 * b + 128.0;
    let v = 0.500 * r - 0.419 * g - 0.081 * b + 128.0;
    (y, u, v)
}

fn yuv_to_rgb(y: f64, u: f64, v: f64) -> (f64, f64, f64) {
    let u2 = u - 128.0;
    let v2 = v - 128.0;
    let r = y + 1.402 * v2;
    let g = y - 0.344 * u2 - 0.714 * v2;
    let b = y + 1.772 * u2;
    (r, g, b)
}

// ---------------------------------------------------------------------------
// Image ↔ channel planes
// ---------------------------------------------------------------------------

/// Split an RGB image into three f64 planes (Y, U, V), padded so dimensions
/// are divisible by `2^dwt_deep * block_size`.
fn image_to_yuv_planes(
    img: &RgbImage,
    dwt_deep: usize,
    block_size: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, u32, u32) {
    let (w, h) = img.dimensions();
    let factor = (1 << dwt_deep) * block_size;
    let pad_h = ((h as usize + factor - 1) / factor) * factor;
    let pad_w = ((w as usize + factor - 1) / factor) * factor;

    let mut y_plane = vec![0.0f64; pad_h * pad_w];
    let mut u_plane = vec![0.0f64; pad_h * pad_w];
    let mut v_plane = vec![0.0f64; pad_h * pad_w];

    for row in 0..h as usize {
        for col in 0..w as usize {
            let px = img.get_pixel(col as u32, row as u32);
            let (y, u, v) = rgb_to_yuv(px[0] as f64, px[1] as f64, px[2] as f64);
            let idx = row * pad_w + col;
            y_plane[idx] = y;
            u_plane[idx] = u;
            v_plane[idx] = v;
        }
    }

    (y_plane, u_plane, v_plane, pad_h, pad_w, w, h)
}

/// Reassemble YUV planes into an RGB image, clipping to [0, 255].
fn yuv_planes_to_image(
    y_plane: &[f64],
    u_plane: &[f64],
    v_plane: &[f64],
    _pad_rows: usize,
    pad_cols: usize,
    orig_w: u32,
    orig_h: u32,
) -> RgbImage {
    let mut img = RgbImage::new(orig_w, orig_h);
    for row in 0..orig_h as usize {
        for col in 0..orig_w as usize {
            let idx = row * pad_cols + col;
            let (r, g, b) = yuv_to_rgb(y_plane[idx], u_plane[idx], v_plane[idx]);
            img.put_pixel(
                col as u32,
                row as u32,
                image::Rgb([
                    r.round().clamp(0.0, 255.0) as u8,
                    g.round().clamp(0.0, 255.0) as u8,
                    b.round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    img
}

// ---------------------------------------------------------------------------
// Watermark I/O helpers
// ---------------------------------------------------------------------------

/// Shuffle a bit array (each element 0 or 255) with a seeded RNG.
pub fn prepare_bits(bits: &[u8], seed_wm: u64) -> Vec<u8> {
    let mut flat = bits.to_vec();
    let mut rng = ChaCha8Rng::seed_from_u64(seed_wm);
    flat.shuffle(&mut rng);
    flat
}

/// Inverse-shuffle to recover original bit order.
pub fn unshuffle_bits(data: &[f64], seed_wm: u64) -> Vec<u8> {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed_wm);
    indices.shuffle(&mut rng);

    let mut result = vec![0u8; n];
    for (shuffled_pos, &orig_pos) in indices.iter().enumerate() {
        let val = data[shuffled_pos].round().clamp(0.0, 255.0) as u8;
        result[orig_pos] = val;
    }
    result
}

/// Inverse-shuffle f64 payload without quantization.
pub fn unshuffle_f64(data: &[f64], seed_wm: u64) -> Vec<f64> {
    let n = data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed_wm);
    indices.shuffle(&mut rng);

    let mut result = vec![0.0f64; n];
    for (shuffled_pos, &orig_pos) in indices.iter().enumerate() {
        result[orig_pos] = data[shuffled_pos];
    }
    result
}


// ---------------------------------------------------------------------------
// Block-level operations
// ---------------------------------------------------------------------------

/// Embed one watermark bit into a single block using DCT → shuffle → SVD → QIM.
fn block_embed(
    block: &[f64],
    block_size: usize,
    cos_table: &CosTable,
    dct_shuffle_index: &[usize],
    wm_bit: u8,
    mod1: f64,
    mod2: Option<f64>,
) -> Vec<f64> {
    let dct_coeffs = cos_table.dct2(block);

    let mut shuffled = vec![0.0f64; block_size * block_size];
    for (i, &idx) in dct_shuffle_index.iter().enumerate() {
        shuffled[i] = dct_coeffs[idx];
    }

    let mat = DMatrix::from_row_slice(block_size, block_size, &shuffled);
    let svd = mat.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let mut s = svd.singular_values;

    // QIM: quantize s[0] to encode the watermark bit
    if wm_bit >= 128 {
        s[0] = s[0] - s[0] % mod1 + 0.75 * mod1;
    } else {
        s[0] = s[0] - s[0] % mod1 + 0.25 * mod1;
    }

    // Optional: QIM on s[1] for extra robustness
    if let Some(m2) = mod2 {
        if s.len() > 1 {
            if wm_bit >= 128 {
                s[1] = s[1] - s[1] % m2 + 0.75 * m2;
            } else {
                s[1] = s[1] - s[1] % m2 + 0.25 * m2;
            }
        }
    }

    // Reconstruct matrix and convert back to spatial domain
    let sigma = DMatrix::from_diagonal(&s);
    let reconstructed = &u * sigma * &v_t;

    // nalgebra stores column-major; convert back to row-major
    let mut recon_row_major = vec![0.0f64; block_size * block_size];
    for r in 0..block_size {
        for c in 0..block_size {
            recon_row_major[r * block_size + c] = reconstructed[(r, c)];
        }
    }

    let mut unshuffled = vec![0.0f64; block_size * block_size];
    for (i, &idx) in dct_shuffle_index.iter().enumerate() {
        unshuffled[idx] = recon_row_major[i];
    }

    // IDCT
    cos_table.idct2(&unshuffled)
}

/// Extract one watermark bit from a single block using DCT → shuffle → SVD → QIM.
fn block_extract(
    block: &[f64],
    block_size: usize,
    cos_table: &CosTable,
    dct_shuffle_index: &[usize],
    mod1: f64,
    mod2: Option<f64>,
) -> f64 {
    let dct_coeffs = cos_table.dct2(block);

    let mut shuffled = vec![0.0f64; block_size * block_size];
    for (i, &idx) in dct_shuffle_index.iter().enumerate() {
        shuffled[i] = dct_coeffs[idx];
    }

    let mat = DMatrix::from_row_slice(block_size, block_size, &shuffled);
    let svd = mat.svd(false, false);
    let s = svd.singular_values;

    // QIM: decode bit from s[0] quantization residue
    let wm1 = if s[0] % mod1 > mod1 / 2.0 { 255.0 } else { 0.0 };

    // Optional: blend s[1] decision for extra robustness
    if let Some(m2) = mod2 {
        if s.len() > 1 {
            let wm2 = if s[1] % m2 > m2 / 2.0 { 255.0 } else { 0.0 };
            return (wm1 * 3.0 + wm2) / 4.0;
        }
    }

    wm1
}

// ---------------------------------------------------------------------------
// Position-independent block helpers
// ---------------------------------------------------------------------------

/// Derive a deterministic seed for a block at a given spatial position.
/// Uses SplitMix64-style mixing for fast, well-distributed hashing.
/// The same (seed_dct, row, col) ALWAYS produces the same result,
/// regardless of image dimensions or total block count.
fn block_shuffle_seed(seed_dct: u64, block_row: usize, block_col: usize) -> u64 {
    let mut h = seed_dct;
    h = h.wrapping_add((block_row as u64).wrapping_mul(0x9E3779B97F4A7C15));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h = h.wrapping_add((block_col as u64).wrapping_mul(0x517CC1B727220A95));
    h ^= h >> 30;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    h
}

/// Generate DCT shuffle index for a specific block at (row, col).
/// Position-independent: does not depend on total block count.
fn generate_block_dct_shuffle(
    seed_dct: u64,
    block_row: usize,
    block_col: usize,
    block_size: usize,
) -> Vec<usize> {
    let seed = block_shuffle_seed(seed_dct, block_row, block_col);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = block_size * block_size;
    let mut index: Vec<usize> = (0..n).collect();
    index.shuffle(&mut rng);
    index
}

/// Map block position to watermark bit index (position-independent).
/// Uses a large prime multiplier to ensure uniform distribution across wm_size.
#[inline]
fn block_to_wm_index(block_row: usize, block_col: usize, wm_size: usize) -> usize {
    ((block_row as u64)
        .wrapping_mul(100003)
        .wrapping_add(block_col as u64)
        % wm_size as u64) as usize
}

// ---------------------------------------------------------------------------
// Process all blocks in a channel's LL subband (parallel with rayon)
// ---------------------------------------------------------------------------

fn embed_channel(
    ll: &[f64],
    ll_rows: usize,
    ll_cols: usize,
    block_size: usize,
    wm_flat: &[u8],
    wm_size: usize,
    cos_table: &CosTable,
    seed_dct: u64,
    mod1: f64,
    mod2: Option<f64>,
) -> Vec<f64> {
    let blocks_r = ll_rows / block_size;
    let blocks_c = ll_cols / block_size;
    let total_blocks = blocks_r * blocks_c;

    // Parallel: compute each block independently
    let new_blocks: Vec<(usize, usize, Vec<f64>)> = (0..total_blocks)
        .into_par_iter()
        .map(|bi| {
            let br = bi / blocks_c;
            let bc = bi % blocks_c;
            let wm_idx = block_to_wm_index(br, bc, wm_size);

            // Extract block from LL
            let mut block_buf = vec![0.0f64; block_size * block_size];
            for r in 0..block_size {
                for c in 0..block_size {
                    block_buf[r * block_size + c] =
                        ll[(br * block_size + r) * ll_cols + (bc * block_size + c)];
                }
            }

            let shuffle = generate_block_dct_shuffle(seed_dct, br, bc, block_size);
            let new_block = block_embed(
                &block_buf,
                block_size,
                cos_table,
                &shuffle,
                wm_flat[wm_idx],
                mod1,
                mod2,
            );

            (br, bc, new_block)
        })
        .collect();

    // Write all blocks back
    let mut result = ll.to_vec();
    for (br, bc, new_block) in new_blocks {
        for r in 0..block_size {
            for c in 0..block_size {
                result[(br * block_size + r) * ll_cols + (bc * block_size + c)] =
                    new_block[r * block_size + c];
            }
        }
    }

    result
}

fn extract_channel(
    ll: &[f64],
    ll_rows: usize,
    ll_cols: usize,
    block_size: usize,
    wm_size: usize,
    cos_table: &CosTable,
    seed_dct: u64,
    mod1: f64,
    mod2: Option<f64>,
) -> Vec<f64> {
    let blocks_r = ll_rows / block_size;
    let blocks_c = ll_cols / block_size;
    let total_blocks = blocks_r * blocks_c;

    // Parallel: extract value from each block
    let block_results: Vec<(usize, f64)> = (0..total_blocks)
        .into_par_iter()
        .map(|bi| {
            let br = bi / blocks_c;
            let bc = bi % blocks_c;
            let wm_idx = block_to_wm_index(br, bc, wm_size);

            let mut block_buf = vec![0.0f64; block_size * block_size];
            for r in 0..block_size {
                for c in 0..block_size {
                    block_buf[r * block_size + c] =
                        ll[(br * block_size + r) * ll_cols + (bc * block_size + c)];
                }
            }

            let shuffle = generate_block_dct_shuffle(seed_dct, br, bc, block_size);
            let val = block_extract(&block_buf, block_size, cos_table, &shuffle, mod1, mod2);
            (wm_idx, val)
        })
        .collect();

    // Aggregate results sequentially (cumulative average)
    let mut wm_extracted = vec![0.0f64; wm_size];
    let mut wm_count = vec![0u32; wm_size];
    for (wm_idx, val) in block_results {
        let count = wm_count[wm_idx];
        if count == 0 {
            wm_extracted[wm_idx] = val;
        } else {
            wm_extracted[wm_idx] =
                (wm_extracted[wm_idx] * count as f64 + val) / (count as f64 + 1.0);
        }
        wm_count[wm_idx] += 1;
    }

    wm_extracted
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Embed watermark bits into a carrier RGB image.
///
/// `wm_bits` is the bit array from `text_to_bits` (each element 0 or 255).
/// Returns the watermarked RGB image.
/// `on_progress` receives values in `[0.0, 1.0]`.
pub fn embed(
    carrier_rgb: &RgbImage,
    wm_bits: &[u8],
    params: &WatermarkParams,
    on_progress: impl Fn(f32),
) -> Result<RgbImage, String> {
    let wm_flat = prepare_bits(wm_bits, params.seed_wm);
    let wm_size = wm_flat.len();

    // Convert carrier to YUV planes with padding
    let (y_plane, u_plane, v_plane, pad_h, pad_w, orig_w, orig_h) =
        image_to_yuv_planes(carrier_rgb, params.dwt_deep, params.block_size);

    // Check capacity
    let ll_rows = pad_h / (1 << params.dwt_deep);
    let ll_cols = pad_w / (1 << params.dwt_deep);
    let capacity = (ll_rows / params.block_size) * (ll_cols / params.block_size);
    if capacity < wm_size {
        return Err(format!(
            "水印太大：需要 {} 个块，但载体只有 {} 个块的容量。\n请减小水印尺寸或增大载体图片。",
            wm_size, capacity
        ));
    }

    // Multi-level DWT for each channel
    on_progress(0.05);
    let (y_ll, yr, yc, y_coeffs) = dwt2_multilevel(&y_plane, pad_h, pad_w, params.dwt_deep);
    let (u_ll, _ur, _uc, u_coeffs) = dwt2_multilevel(&u_plane, pad_h, pad_w, params.dwt_deep);
    let (v_ll, _vr, _vc, v_coeffs) = dwt2_multilevel(&v_plane, pad_h, pad_w, params.dwt_deep);
    on_progress(0.10);

    // Pre-compute shared resources
    let cos_table = CosTable::new(params.block_size);

    // U/V channels use reduced mod to minimize visible color artifacts
    let uv_mod1 = params.mod1 * UV_MOD_RATIO;
    let uv_mod2 = params.mod2.map(|m| m * UV_MOD_RATIO);

    // Embed watermark in each channel's LL subband
    let y_ll_new = embed_channel(
        &y_ll,
        yr,
        yc,
        params.block_size,
        &wm_flat,
        wm_size,
        &cos_table,
        params.seed_dct,
        params.mod1,
        params.mod2,
    );
    on_progress(0.35);

    let u_ll_new = embed_channel(
        &u_ll,
        yr,
        yc,
        params.block_size,
        &wm_flat,
        wm_size,
        &cos_table,
        params.seed_dct,
        uv_mod1,
        uv_mod2,
    );
    on_progress(0.60);

    let v_ll_new = embed_channel(
        &v_ll,
        yr,
        yc,
        params.block_size,
        &wm_flat,
        wm_size,
        &cos_table,
        params.seed_dct,
        uv_mod1,
        uv_mod2,
    );
    on_progress(0.85);

    // Inverse DWT → YUV → RGB
    let y_reconstructed = idwt2_multilevel(&y_ll_new, &y_coeffs);
    let u_reconstructed = idwt2_multilevel(&u_ll_new, &u_coeffs);
    let v_reconstructed = idwt2_multilevel(&v_ll_new, &v_coeffs);
    on_progress(0.95);

    Ok(yuv_planes_to_image(
        &y_reconstructed,
        &u_reconstructed,
        &v_reconstructed,
        pad_h,
        pad_w,
        orig_w,
        orig_h,
    ))
}

/// Extract watermark bits from a watermarked RGB image.
///
/// `wm_bit_count` is the expected number of bits (from `watermark_bit_count`).
/// Returns the extracted bit array (each element 0 or 255).
/// `on_progress` receives values in `[0.0, 1.0]`.
pub fn extract(
    img_rgb: &RgbImage,
    wm_bit_count: usize,
    params: &WatermarkParams,
    on_progress: impl Fn(f32),
) -> Result<Vec<u8>, String> {
    let wm_size = wm_bit_count;

    // Convert to YUV planes with padding
    let (y_plane, u_plane, v_plane, pad_h, pad_w, _orig_w, _orig_h) =
        image_to_yuv_planes(img_rgb, params.dwt_deep, params.block_size);

    // Multi-level DWT on each YUV channel
    on_progress(0.05);
    let (y_ll, yr, yc, _y_coeffs) = dwt2_multilevel(&y_plane, pad_h, pad_w, params.dwt_deep);
    let (u_ll, _ur, _uc, _u_coeffs) = dwt2_multilevel(&u_plane, pad_h, pad_w, params.dwt_deep);
    let (v_ll, _vr, _vc, _v_coeffs) = dwt2_multilevel(&v_plane, pad_h, pad_w, params.dwt_deep);
    on_progress(0.10);

    let cos_table = CosTable::new(params.block_size);

    // U/V channels use reduced mod (must match embedding)
    let uv_mod1 = params.mod1 * UV_MOD_RATIO;
    let uv_mod2 = params.mod2.map(|m| m * UV_MOD_RATIO);

    // Extract from each channel
    let wm_y = extract_channel(
        &y_ll,
        yr,
        yc,
        params.block_size,
        wm_size,
        &cos_table,
        params.seed_dct,
        params.mod1,
        params.mod2,
    );
    on_progress(0.40);

    let wm_u = extract_channel(
        &u_ll,
        yr,
        yc,
        params.block_size,
        wm_size,
        &cos_table,
        params.seed_dct,
        uv_mod1,
        uv_mod2,
    );
    on_progress(0.65);

    let wm_v = extract_channel(
        &v_ll,
        yr,
        yc,
        params.block_size,
        wm_size,
        &cos_table,
        params.seed_dct,
        uv_mod1,
        uv_mod2,
    );
    on_progress(0.90);

    // Weighted three-channel voting: Y gets higher weight (full mod = more reliable)
    let wm_avg: Vec<f64> = wm_y
        .iter()
        .zip(wm_u.iter())
        .zip(wm_v.iter())
        .map(|((&y, &u), &v)| ((y * 3.0 + u + v) / 5.0).round())
        .collect();

    // Inverse shuffle to recover original bit order
    Ok(unshuffle_bits(&wm_avg, params.seed_wm))
}

#[inline]
fn video_block_stats(frame_rgb: &RgbImage, block_row: usize, block_col: usize) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for r in 0..VIDEO_BLOCK_SIZE {
        for c in 0..VIDEO_BLOCK_SIZE {
            let x = (block_col * VIDEO_BLOCK_SIZE + c) as u32;
            let y = (block_row * VIDEO_BLOCK_SIZE + r) as u32;
            let px = frame_rgb.get_pixel(x, y);
            let lum = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
            sum += lum;
            sum_sq += lum * lum;
        }
    }
    let n = VIDEO_BLOCK_PIXELS as f64;
    let avg = sum / n;
    let var = (sum_sq / n - avg * avg).max(0.0);
    (avg, var)
}

#[inline]
fn adaptive_video_mod(variance: f64) -> f64 {
    let ratio = (variance / VIDEO_REF_VAR).clamp(VIDEO_MOD_MIN_RATIO, VIDEO_MOD_MAX_RATIO);
    (VIDEO_MOD * ratio).max(VIDEO_CONF_EPS)
}

#[inline]
fn qim_embed_delta(avg: f64, mod_v: f64, bit: u8) -> f64 {
    let base = avg - avg % mod_v;
    let target = if bit >= 128 {
        base + 0.75 * mod_v
    } else {
        base + 0.25 * mod_v
    };
    target - avg
}

#[inline]
fn qim_extract_bit_and_conf(avg: f64, mod_v: f64) -> (f64, f64) {
    let rem = avg % mod_v;
    let half = mod_v * 0.5;
    let bit = if rem > half { 255.0 } else { 0.0 };
    let conf = ((rem - half).abs() / half).max(VIDEO_CONF_EPS);
    (bit, conf)
}

/// Fast in-place video-frame embedding using 8x8 block-average QIM on luminance.
///
/// This path is designed for compressed video workflows (e.g. H.264 + yuv420p):
/// - Y-only signal (implemented by equal RGB offset => luminance shift)
/// - no DWT / DCT / SVD
/// - deterministic bit mapping with password-derived shuffle (`seed_wm`)
/// - adaptive QIM strength by block variance
pub fn embed_video_frame_in_place(
    frame_rgb: &mut RgbImage,
    wm_bits: &[u8],
    params: &WatermarkParams,
) -> Result<(), String> {
    if wm_bits.is_empty() {
        return Err("水印数据为空".into());
    }

    let wm_flat = prepare_bits(wm_bits, params.seed_wm);
    let wm_size = wm_flat.len();
    let (w, h) = frame_rgb.dimensions();
    let blocks_c = w as usize / VIDEO_BLOCK_SIZE;
    let blocks_r = h as usize / VIDEO_BLOCK_SIZE;
    let capacity = blocks_r * blocks_c;
    if capacity == 0 {
        return Err("视频帧尺寸过小，无法嵌入水印".into());
    }

    // Parallel phase: compute block deltas from the unmodified frame.
    let deltas: Vec<f64> = {
        let frame_ref: &RgbImage = &*frame_rgb;
        (0..capacity)
            .into_par_iter()
            .map(|bi| {
                let br = bi / blocks_c;
                let bc = bi % blocks_c;
                let wm_idx = block_to_wm_index(br, bc, wm_size);
                let bit = wm_flat[wm_idx];
                let (avg, var) = video_block_stats(frame_ref, br, bc);
                let mod_v = adaptive_video_mod(var);
                qim_embed_delta(avg, mod_v, bit)
            })
            .collect()
    };

    // Write phase: apply deltas in-place.
    for (bi, delta) in deltas.into_iter().enumerate() {
        let br = bi / blocks_c;
        let bc = bi % blocks_c;
        for r in 0..VIDEO_BLOCK_SIZE {
            for c in 0..VIDEO_BLOCK_SIZE {
                let x = (bc * VIDEO_BLOCK_SIZE + c) as u32;
                let y = (br * VIDEO_BLOCK_SIZE + r) as u32;
                let p = frame_rgb.get_pixel_mut(x, y);
                for ch in 0..3 {
                    let v = p[ch] as f64 + delta;
                    p[ch] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(())
}

/// Fast video-frame embedding wrapper that preserves return-by-value API.
#[allow(dead_code)]
pub fn embed_video_frame(
    frame_rgb: &RgbImage,
    wm_bits: &[u8],
    params: &WatermarkParams,
) -> Result<RgbImage, String> {
    let mut out = frame_rgb.clone();
    embed_video_frame_in_place(&mut out, wm_bits, params)?;
    Ok(out)
}

/// Fast video-frame extraction using 8x8 block-average QIM on luminance
/// with confidence output for weighted multi-frame accumulation.
pub fn extract_video_frame_with_confidence(
    frame_rgb: &RgbImage,
    wm_bit_count: usize,
    params: &WatermarkParams,
) -> Result<(Vec<u8>, Vec<f64>), String> {
    if wm_bit_count == 0 {
        return Err("水印位数无效".into());
    }

    let (w, h) = frame_rgb.dimensions();
    let blocks_c = w as usize / VIDEO_BLOCK_SIZE;
    let blocks_r = h as usize / VIDEO_BLOCK_SIZE;
    if blocks_r == 0 || blocks_c == 0 {
        return Err("视频帧尺寸过小，无法提取水印".into());
    }
    let capacity = blocks_r * blocks_c;

    // Parallel phase: per-block bit/confidence estimation.
    let block_results: Vec<(usize, f64, f64)> = (0..capacity)
        .into_par_iter()
        .map(|bi| {
            let br = bi / blocks_c;
            let bc = bi % blocks_c;
            let wm_idx = block_to_wm_index(br, bc, wm_bit_count);
            let (avg, var) = video_block_stats(frame_rgb, br, bc);
            let mod_v = adaptive_video_mod(var);
            let (bit_val, conf) = qim_extract_bit_and_conf(avg, mod_v);
            (wm_idx, bit_val, conf)
        })
        .collect();

    // Sequential aggregation: confidence-weighted average per watermark bit.
    let mut wm_weighted_sum = vec![0.0f64; wm_bit_count];
    let mut wm_conf_sum = vec![0.0f64; wm_bit_count];
    let mut wm_count = vec![0u32; wm_bit_count];
    for (wm_idx, bit_val, conf) in block_results {
        wm_weighted_sum[wm_idx] += bit_val * conf;
        wm_conf_sum[wm_idx] += conf;
        wm_count[wm_idx] += 1;
    }

    let mut wm_values = vec![0.0f64; wm_bit_count];
    let mut wm_conf_avg = vec![0.0f64; wm_bit_count];
    for i in 0..wm_bit_count {
        if wm_conf_sum[i] > 0.0 {
            wm_values[i] = wm_weighted_sum[i] / wm_conf_sum[i];
        }
        if wm_count[i] > 0 {
            wm_conf_avg[i] = wm_conf_sum[i] / wm_count[i] as f64;
        }
    }

    Ok((
        unshuffle_bits(&wm_values, params.seed_wm),
        unshuffle_f64(&wm_conf_avg, params.seed_wm),
    ))
}

/// Fast video-frame extraction compatibility API (without confidence output).
#[allow(dead_code)]
pub fn extract_video_frame(
    frame_rgb: &RgbImage,
    wm_bit_count: usize,
    params: &WatermarkParams,
) -> Result<Vec<u8>, String> {
    let (bits, _conf) = extract_video_frame_with_confidence(frame_rgb, wm_bit_count, params)?;
    Ok(bits)
}

// ---------------------------------------------------------------------------
// Robust extraction with alignment search
// ---------------------------------------------------------------------------

/// Pad an image on the left/top by (dx, dy) pixels using edge replication.
fn pad_image_top_left(img: &RgbImage, dx: u32, dy: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let new_w = w + dx;
    let new_h = h + dy;
    let mut padded = RgbImage::new(new_w, new_h);

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = if x >= dx { (x - dx).min(w - 1) } else { 0 };
            let src_y = if y >= dy { (y - dy).min(h - 1) } else { 0 };
            padded.put_pixel(x, y, *img.get_pixel(src_x, src_y));
        }
    }

    padded
}

/// Extract watermark with brute-force grid alignment search.
///
/// Uses the fixed watermark size (`FIXED_WM_BITS`) so no prior knowledge of
/// the watermark content is needed. When an image is cropped by a few pixels,
/// the DWT block grid shifts; this function tries all possible padding offsets
/// to find the correct alignment.
///
/// Returns the decoded watermark text on success.
pub fn extract_robust(
    img_rgb: &RgbImage,
    params: &WatermarkParams,
    on_progress: impl Fn(f32),
) -> Result<String, String> {
    let wm_bit_count = super::text::FIXED_WM_BITS;

    // First try exact dimensions (most common case: no crop)
    if let Ok(bits) = extract(img_rgb, wm_bit_count, params, |_| {}) {
        if let Ok(text) = super::text::bits_to_text(&bits) {
            on_progress(1.0);
            return Ok(text);
        }
    }

    // Brute-force: try all grid alignments
    let grid_period = (1usize << params.dwt_deep) * params.block_size;
    let total_attempts = grid_period * grid_period;

    for attempt in 1..total_attempts {
        let dy = (attempt / grid_period) as u32;
        let dx = (attempt % grid_period) as u32;

        let padded = pad_image_top_left(img_rgb, dx, dy);

        if let Ok(bits) = extract(&padded, wm_bit_count, params, |_| {}) {
            if super::text::check_watermark_magic(&bits) {
                if let Ok(text) = super::text::bits_to_text(&bits) {
                    on_progress(1.0);
                    return Ok(text);
                }
            }
        }

        on_progress(attempt as f32 / total_attempts as f32);
    }

    Err("未检测到水印（已尝试多种对齐方式）".into())
}

