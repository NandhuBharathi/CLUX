use clux_jit::{causal_conv1d_f32, rms_norm_f32, selective_scan_f32, silu_f32, vector_dot_f32};

pub struct SsmLayer {
    pub d_model: usize, pub d_inner: usize, pub d_state: usize,
    pub norm_weight: Vec<f32>, pub in_proj_x: Vec<f32>, pub in_proj_z: Vec<f32>,
    pub conv_weight: Vec<f32>, pub delta_bias: Vec<f32>, pub a_diag: Vec<f32>,
    pub b_weight: Vec<f32>, pub c_weight: Vec<f32>, pub out_proj: Vec<f32>,
    pub conv_state: Vec<f32>, pub ssm_state: Vec<f32>,
}

impl SsmLayer {
    pub fn new(dm: usize, di: usize, ds: usize, idx: usize) -> Self {
        let init = |r: usize, c: usize, off: usize| -> Vec<f32> {
            (0..r * c).map(|i| (((i * 19 + off * 31) % 97) as f32 / 97.0 - 0.5) * 0.05).collect()
        };
        Self {
            d_model: dm, d_inner: di, d_state: ds,
            norm_weight: vec![1.0; dm],
            in_proj_x: init(di, dm, idx + 1), in_proj_z: init(di, dm, idx + 2),
            conv_weight: init(di, 4, idx + 3), delta_bias: vec![0.05; di],
            a_diag: vec![-0.1; di * ds], b_weight: vec![0.1; ds], c_weight: vec![0.1; ds],
            out_proj: init(dm, di, idx + 4), conv_state: vec![0.0; di * 3], ssm_state: vec![0.0; di * ds],
        }
    }

    pub fn reset_state(&mut self) { self.conv_state.fill(0.0); self.ssm_state.fill(0.0); }

    pub fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        let (dm, di, ds) = (self.d_model, self.d_inner, self.d_state);
        let mut norm_x = input.to_vec();
        rms_norm_f32(&mut norm_x, &self.norm_weight, 1e-5);

        let mut x_b = vec![0.0; di]; let mut z_b = vec![0.0; di];
        for i in 0..di {
            x_b[i] = vector_dot_f32(&norm_x, &self.in_proj_x[i * dm..(i + 1) * dm]);
            z_b[i] = vector_dot_f32(&norm_x, &self.in_proj_z[i * dm..(i + 1) * dm]);
        }

        let mut conv_out = vec![0.0; di];
        causal_conv1d_f32(&x_b, &self.conv_weight, &mut self.conv_state, &mut conv_out, di);
        for v in &mut conv_out { *v = silu_f32(*v); }

        let mut ssm_y = vec![0.0; di];
        selective_scan_f32(&conv_out, &self.delta_bias, &self.a_diag, &self.b_weight, &self.c_weight, &mut self.ssm_state, &mut ssm_y, di, ds);

        let mut fused = vec![0.0; di];
        for i in 0..di { fused[i] = ssm_y[i] * silu_f32(z_b[i]); }

        for m in 0..dm {
            output[m] = input[m] + vector_dot_f32(&fused, &self.out_proj[m * di..(m + 1) * di]);
        }
    }
}
