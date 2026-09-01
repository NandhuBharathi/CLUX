//! CLUX JIT & Bare-Metal SIMD Execution Core

pub mod arch;
pub mod kernels;

pub use arch::{CpuFeatures, TargetArch};
pub use kernels::{fused_discretize_f32, ssm_scan_step_f32, vector_dot_f32, rms_norm_f32};

pub struct JitEngine {
    pub target: TargetArch,
}

impl JitEngine {
    pub fn new() -> Self {
        Self {
            target: CpuFeatures::detect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_detection() {
        let engine = JitEngine::new();
        println!("Detected CPU Execution Target: {:?}", engine.target);
    }

    #[test]
    fn test_fused_discretize_and_scan_step() {
        let state_dim = 16;
        let delta = vec![0.1f32; state_dim];
        let a_diag = vec![-1.0f32; state_dim];
        let b = vec![1.0f32; state_dim];
        let c = vec![0.5f32; state_dim];

        let mut a_bar = vec![0.0f32; state_dim];
        let mut b_bar = vec![0.0f32; state_dim];

        // 1. Discretize
        fused_discretize_f32(&delta, &a_diag, &b, &mut a_bar, &mut b_bar);

        // a_bar should be exp(0.1 * -1.0) = exp(-0.1) approx 0.904837
        assert!((a_bar[0] - 0.904837).abs() < 1e-4);
        // b_bar should be 0.1 * 1.0 = 0.1
        assert!((b_bar[0] - 0.1).abs() < 1e-5);

        // 2. Scan Step
        let h_prev = vec![0.0f32; state_dim];
        let mut h_next = vec![0.0f32; state_dim];
        let x_val = 2.0f32;

        let y_out = ssm_scan_step_f32(&a_bar, &b_bar, x_val, &h_prev, &mut h_next, &c);

        // h_next = 0.904837 * 0 + 0.1 * 2.0 = 0.2
        assert!((h_next[0] - 0.2).abs() < 1e-4);
        // y_out = 16 * (0.5 * 0.2) = 16 * 0.1 = 1.6
        assert!((y_out - 1.6).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![2.0f32, 2.0f32, 2.0f32, 2.0f32];
        let weight = vec![1.0f32; 4];
        rms_norm_f32(&mut x, &weight, 1e-5);
        // RMS of [2, 2, 2, 2] is 2, normalized = 1.0
        for &val in &x {
            assert!((val - 1.0).abs() < 1e-4);
        }
    }
}
