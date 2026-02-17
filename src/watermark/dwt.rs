/// 2D Haar Discrete Wavelet Transform
///
/// Data is stored in row-major order as Vec<f64> of length rows*cols.

const SQRT2: f64 = std::f64::consts::SQRT_2;

/// Result of a single-level 2D DWT decomposition.
pub struct Dwt2Result {
    /// Low-low (approximation) — size (rows/2, cols/2)
    pub ll: Vec<f64>,
    /// Low-high (horizontal detail)
    pub lh: Vec<f64>,
    /// High-low (vertical detail)
    pub hl: Vec<f64>,
    /// High-high (diagonal detail)
    pub hh: Vec<f64>,
    pub half_rows: usize,
    pub half_cols: usize,
}

/// High-frequency coefficients stored for IDWT reconstruction.
pub struct DwtCoeffs {
    pub lh: Vec<f64>,
    pub hl: Vec<f64>,
    pub hh: Vec<f64>,
    pub half_rows: usize,
    pub half_cols: usize,
}

/// Perform one level of 2D Haar DWT.
///
/// Input: `data` of size `rows × cols` (both must be even).
/// Returns the four sub-bands, each of size `(rows/2) × (cols/2)`.
pub fn dwt2_haar(data: &[f64], rows: usize, cols: usize) -> Dwt2Result {
    assert!(rows % 2 == 0 && cols % 2 == 0, "dimensions must be even");
    assert_eq!(data.len(), rows * cols);

    let hr = rows / 2;
    let hc = cols / 2;

    // Step 1: row-wise 1D Haar → temp[rows][cols] where left half = low, right half = high
    let mut temp = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..hc {
            let a = data[r * cols + 2 * c];
            let b = data[r * cols + 2 * c + 1];
            temp[r * cols + c] = (a + b) / SQRT2;
            temp[r * cols + hc + c] = (a - b) / SQRT2;
        }
    }

    // Step 2: column-wise 1D Haar on each half
    let mut ll = vec![0.0f64; hr * hc];
    let mut lh = vec![0.0f64; hr * hc];
    let mut hl = vec![0.0f64; hr * hc];
    let mut hh = vec![0.0f64; hr * hc];

    for c in 0..hc {
        for r in 0..hr {
            let a = temp[2 * r * cols + c];
            let b = temp[(2 * r + 1) * cols + c];
            ll[r * hc + c] = (a + b) / SQRT2;
            hl[r * hc + c] = (a - b) / SQRT2;
        }
    }
    for c in 0..hc {
        for r in 0..hr {
            let a = temp[2 * r * cols + hc + c];
            let b = temp[(2 * r + 1) * cols + hc + c];
            lh[r * hc + c] = (a + b) / SQRT2;
            hh[r * hc + c] = (a - b) / SQRT2;
        }
    }

    Dwt2Result {
        ll,
        lh,
        hl,
        hh,
        half_rows: hr,
        half_cols: hc,
    }
}

/// Perform one level of inverse 2D Haar DWT.
///
/// Reconstructs data of size `(2*half_rows) × (2*half_cols)` from the four sub-bands.
pub fn idwt2_haar(
    ll: &[f64],
    lh: &[f64],
    hl: &[f64],
    hh: &[f64],
    half_rows: usize,
    half_cols: usize,
) -> Vec<f64> {
    let rows = half_rows * 2;
    let cols = half_cols * 2;
    let hc = half_cols;

    // Step 1: inverse column-wise Haar
    let mut temp = vec![0.0f64; rows * cols];
    // Left half (from ll + hl)
    for c in 0..hc {
        for r in 0..half_rows {
            let lo = ll[r * hc + c];
            let hi = hl[r * hc + c];
            temp[2 * r * cols + c] = (lo + hi) / SQRT2;
            temp[(2 * r + 1) * cols + c] = (lo - hi) / SQRT2;
        }
    }
    // Right half (from lh + hh)
    for c in 0..hc {
        for r in 0..half_rows {
            let lo = lh[r * hc + c];
            let hi = hh[r * hc + c];
            temp[2 * r * cols + hc + c] = (lo + hi) / SQRT2;
            temp[(2 * r + 1) * cols + hc + c] = (lo - hi) / SQRT2;
        }
    }

    // Step 2: inverse row-wise Haar
    let mut out = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..hc {
            let lo = temp[r * cols + c];
            let hi = temp[r * cols + hc + c];
            out[r * cols + 2 * c] = (lo + hi) / SQRT2;
            out[r * cols + 2 * c + 1] = (lo - hi) / SQRT2;
        }
    }
    out
}

/// Multi-level 2D Haar DWT.
///
/// Returns the final LL subband and a stack of high-frequency coefficients
/// (from finest to coarsest level).
pub fn dwt2_multilevel(
    data: &[f64],
    rows: usize,
    cols: usize,
    levels: usize,
) -> (Vec<f64>, usize, usize, Vec<DwtCoeffs>) {
    let mut ll = data.to_vec();
    let mut r = rows;
    let mut c = cols;
    let mut coeffs_stack = Vec::with_capacity(levels);

    for _ in 0..levels {
        let res = dwt2_haar(&ll, r, c);
        coeffs_stack.push(DwtCoeffs {
            lh: res.lh,
            hl: res.hl,
            hh: res.hh,
            half_rows: res.half_rows,
            half_cols: res.half_cols,
        });
        ll = res.ll;
        r = res.half_rows;
        c = res.half_cols;
    }

    (ll, r, c, coeffs_stack)
}

/// Multi-level inverse 2D Haar DWT.
///
/// `coeffs_stack` must be in the same order as returned by `dwt2_multilevel`
/// (finest to coarsest). We iterate in reverse (coarsest to finest) for reconstruction.
pub fn idwt2_multilevel(ll: &[f64], coeffs_stack: &[DwtCoeffs]) -> Vec<f64> {
    let mut current = ll.to_vec();

    for coeff in coeffs_stack.iter().rev() {
        current = idwt2_haar(
            &current,
            &coeff.lh,
            &coeff.hl,
            &coeff.hh,
            coeff.half_rows,
            coeff.half_cols,
        );
    }

    current
}
