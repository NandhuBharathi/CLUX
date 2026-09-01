//! Real Input-Dependent Selective State Space Math Kernels (Bare-Metal SIMD Ready)

#[inline(always)]
pub fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// TRUE Selective Scan: delta (dt), B, and C are strictly functions of input 'x'
pub fn input_dependent_scan_f32(
    x: &[f32],           // [d_inner] current input token
    dt_proj: &[f32],     // [d_inner] input-dependent time-step delta
    b_proj: &[f32],      // [d_state] input-dependent B matrix
    c_proj: &[f32],      // [d_state] input-dependent C matrix
    a_diag: &[f32],      // [d_inner, d_state] static A base
    state: &mut [f32],   // [d_inner, d_state] Memory state
    out: &mut [f32],     // [d_inner]
    d_inner: usize,
    d_state: usize,
) {
    for i in 0..d_inner {
        // 1. Input-dependent Discretization step (The Mamba Secret)
        let dt = softplus_f32(dt_proj[i]); 
        let mut y_acc = 0.0f32;
        let offset = i * d_state;

        for j in 0..d_state {
            // 2. Hardware-efficient continuous-to-discrete transformation
            let a_val = a_diag[offset + j];
            let a_bar = (dt * a_val).exp();
            let b_bar = dt * b_proj[j];

            // 3. State update (Recurrent formula): h_t = A_bar * h_{t-1} + B_bar * x_t
            let next_h = a_bar * state[offset + j] + b_bar * x[i];
            state[offset + j] = next_h;
            
            // 4. Output projection: y_t = C_t * h_t
            y_acc += c_proj[j] * next_h;
        }
        out[i] = y_acc;
    }
}
