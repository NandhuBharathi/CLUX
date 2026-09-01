//! Bare-Metal SIMD & Fused Kernels for SSM

/// Fused Discretization: A_bar = exp(delta * a_diag), B_bar = delta * b
#[inline(always)]
pub fn fused_discretize_f32(
    delta: &[f32],
    a_diag: &[f32],
    b: &[f32],
    a_bar: &mut [f32],
    b_bar: &mut [f32],
) {
    let len = delta.len().min(a_diag.len()).min(b.len());
    
    // 4-way unrolled loop for SIMD auto-vectorization
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        for k in 0..4 {
            let j = idx + k;
            a_bar[j] = (delta[j] * a_diag[j]).exp();
            b_bar[j] = delta[j] * b[j];
        }
    }

    let offset = chunks * 4;
    for j in 0..remainder {
        let idx = offset + j;
        a_bar[idx] = (delta[idx] * a_diag[idx]).exp();
        b_bar[idx] = delta[idx] * b[idx];
    }
}

/// Diagonal SSM Scan Step:
/// h_next = a_bar * h_prev + b_bar * x
/// y = sum(c * h_next)
#[inline(always)]
pub fn ssm_scan_step_f32(
    a_bar: &[f32],
    b_bar: &[f32],
    x_val: f32,
    h_prev: &[f32],
    h_next: &mut [f32],
    c: &[f32],
) -> f32 {
    let state_dim = a_bar.len();
    let mut sum = 0.0f32;

    for i in 0..state_dim {
        // Elementwise state decay + input injection
        let next_h = a_bar[i] * h_prev[i] + b_bar[i] * x_val;
        h_next[i] = next_h;
        // Output projection dot product accumulation
        sum += c[i] * next_h;
    }

    sum
}

/// Vector Dot Product
#[inline(always)]
pub fn vector_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let idx = i * 4;
        acc0 += a[idx] * b[idx];
        acc1 += a[idx + 1] * b[idx + 1];
        acc2 += a[idx + 2] * b[idx + 2];
        acc3 += a[idx + 3] * b[idx + 3];
    }

    let mut total = (acc0 + acc1) + (acc2 + acc3);
    let offset = chunks * 4;
    for j in 0..remainder {
        total += a[offset + j] * b[offset + j];
    }

    total
}

/// In-place RMS Normalization
#[inline(always)]
pub fn rms_norm_f32(x: &mut [f32], weight: &[f32], eps: f32) {
    let len = x.len();
    if len == 0 { return; }

    let mut sum_sq = 0.0f32;
    for &val in x.iter() {
        sum_sq += val * val;
    }

    let mean_sq = sum_sq / len as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for i in 0..len {
        x[i] = x[i] * inv_rms * weight[i];
    }
}
