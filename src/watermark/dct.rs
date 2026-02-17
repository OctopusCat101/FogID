/// 2D DCT-II and IDCT (DCT-III) for small square blocks.
///
/// Uses the orthonormal formulation matching OpenCV's `cv2.dct`.
/// Cosine values are precomputed in a lookup table for speed.
use std::f64::consts::PI;

/// Precomputed cosine table for a given block size N.
///
/// `table[k * n + i] = cos(PI * (2*i + 1) * k / (2*N))`
/// `alpha[k]` = orthonormal scaling factor.
pub struct CosTable {
    n: usize,
    table: Vec<f64>, // n * n
    alpha: Vec<f64>, // n
}

impl CosTable {
    pub fn new(n: usize) -> Self {
        let nf = n as f64;
        let mut table = vec![0.0; n * n];
        let mut alpha = vec![0.0; n];

        for k in 0..n {
            alpha[k] = if k == 0 {
                (1.0 / nf).sqrt()
            } else {
                (2.0 / nf).sqrt()
            };
            for i in 0..n {
                table[k * n + i] = (PI * (2.0 * i as f64 + 1.0) * k as f64 / (2.0 * nf)).cos();
            }
        }

        Self { n, table, alpha }
    }

    /// 1D DCT-II (orthonormal) using precomputed table.
    #[inline]
    pub fn dct1d(&self, input: &[f64], output: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            let mut sum = 0.0;
            let row = k * n;
            for i in 0..n {
                sum += input[i] * self.table[row + i];
            }
            output[k] = self.alpha[k] * sum;
        }
    }

    /// 1D IDCT (DCT-III, orthonormal) using precomputed table.
    #[inline]
    pub fn idct1d(&self, input: &[f64], output: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += self.alpha[k] * input[k] * self.table[k * n + i];
            }
            output[i] = sum;
        }
    }

    /// 2D DCT-II of an n×n block (row-major).
    pub fn dct2(&self, block: &[f64]) -> Vec<f64> {
        let n = self.n;
        debug_assert_eq!(block.len(), n * n);

        // Row-wise DCT
        let mut temp = vec![0.0f64; n * n];
        let mut row_buf = vec![0.0f64; n];
        for r in 0..n {
            self.dct1d(&block[r * n..(r + 1) * n], &mut row_buf);
            temp[r * n..(r + 1) * n].copy_from_slice(&row_buf);
        }

        // Column-wise DCT
        let mut result = vec![0.0f64; n * n];
        let mut col_in = vec![0.0f64; n];
        let mut col_out = vec![0.0f64; n];
        for c in 0..n {
            for r in 0..n {
                col_in[r] = temp[r * n + c];
            }
            self.dct1d(&col_in, &mut col_out);
            for r in 0..n {
                result[r * n + c] = col_out[r];
            }
        }
        result
    }

    /// 2D IDCT of an n×n block (row-major).
    pub fn idct2(&self, block: &[f64]) -> Vec<f64> {
        let n = self.n;
        debug_assert_eq!(block.len(), n * n);

        // Column-wise IDCT
        let mut temp = vec![0.0f64; n * n];
        let mut col_in = vec![0.0f64; n];
        let mut col_out = vec![0.0f64; n];
        for c in 0..n {
            for r in 0..n {
                col_in[r] = block[r * n + c];
            }
            self.idct1d(&col_in, &mut col_out);
            for r in 0..n {
                temp[r * n + c] = col_out[r];
            }
        }

        // Row-wise IDCT
        let mut result = vec![0.0f64; n * n];
        let mut row_buf = vec![0.0f64; n];
        for r in 0..n {
            self.idct1d(&temp[r * n..(r + 1) * n], &mut row_buf);
            result[r * n..(r + 1) * n].copy_from_slice(&row_buf);
        }
        result
    }
}

/// Partial cosine table: only the first `k` rows out of `n`.
///
/// For audio watermarking we only need DCT coefficients 0..k (typically k=9),
/// so the table is k×n instead of n×n.  For n=4096, k=9 this is 36 864
/// elements instead of 16 777 216 — a ~460× reduction.
#[allow(dead_code)]
pub struct PartialCosTable {
    n: usize,
    k: usize,
    table: Vec<f64>, // k * n
    alpha: Vec<f64>, // k
}
#[allow(dead_code)]

impl PartialCosTable {
    pub fn new(n: usize, k: usize) -> Self {
        assert!(k <= n);
        let nf = n as f64;
        let mut table = vec![0.0; k * n];
        let mut alpha = vec![0.0; k];

        for row in 0..k {
            alpha[row] = if row == 0 {
                (1.0 / nf).sqrt()
            } else {
                (2.0 / nf).sqrt()
            };
            for i in 0..n {
                table[row * n + i] =
                    (PI * (2.0 * i as f64 + 1.0) * row as f64 / (2.0 * nf)).cos();
            }
        }

        Self { n, k, table, alpha }
    }

    /// Compute only the first `k` DCT-II coefficients.  O(k·n).
    #[inline]
    pub fn forward(&self, input: &[f64], output: &mut [f64]) {
        debug_assert!(input.len() >= self.n);
        debug_assert!(output.len() >= self.k);
        let n = self.n;
        for row in 0..self.k {
            let mut sum = 0.0;
            let base = row * n;
            for i in 0..n {
                sum += input[i] * self.table[base + i];
            }
            output[row] = self.alpha[row] * sum;
        }
    }

    /// Compute the time-domain delta caused by modifying a few DCT coefficients.
    ///
    /// For each `(index, delta)` pair, accumulates
    /// `alpha[index] * delta * cos_table[index][i]` into `output[i]`.
    ///
    /// `output` must be zeroed by the caller.  O(modified_count · n).
    #[inline]
    pub fn inverse_delta(&self, deltas: &[(usize, f64)], output: &mut [f64]) {
        debug_assert!(output.len() >= self.n);
        let n = self.n;
        for &(idx, delta) in deltas {
            debug_assert!(idx < self.k);
            let a = self.alpha[idx];
            let base = idx * n;
            let ad = a * delta;
            for i in 0..n {
                output[i] += ad * self.table[base + i];
            }
        }
    }
}
