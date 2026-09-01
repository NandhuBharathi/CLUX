pub fn silu_f32(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
pub fn softplus_f32(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
pub fn vector_dot_f32(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() }

pub fn rms_norm_f32(x: &mut [f32], weight: &[f32], eps: f32) {
    let len = x.len();
    if len == 0 { return; }
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / ((sum_sq / len as f32) + eps).sqrt();
    for i in 0..len { x[i] = x[i] * inv * weight[i]; }
}

pub fn causal_conv1d_f32(input: &[f32], w: &[f32], s: &mut [f32], out: &mut [f32], ch: usize) {
    for c in 0..ch {
        let cw = &w[c * 4..(c + 1) * 4];
        let cs = &mut s[c * 3..(c + 1) * 3];
        let v = input[c];
        out[c] = cs[0] * cw[0] + cs[1] * cw[1] + cs[2] * cw[2] + v * cw[3];
        cs[0] = cs[1]; cs[1] = cs[2]; cs[2] = v;
    }
}

pub fn selective_scan_f32(x: &[f32], dt: &[f32], a: &[f32], b: &[f32], c: &[f32], h: &mut [f32], out: &mut [f32], di: usize, ds: usize) {
    for i in 0..di {
        let delta = softplus_f32(dt[i]);
        let mut y = 0.0f32;
        let off = i * ds;
        for j in 0..ds {
            let a_bar = (delta * a[off + j]).exp();
            let b_bar = delta * b[j];
            let next_h = a_bar * h[off + j] + b_bar * x[i];
            h[off + j] = next_h;
            y += c[j] * next_h;
        }
        out[i] = y;
    }
}

pub fn adamw_step_f32(param: &mut f32, grad: f32, m: &mut f32, v: &mut f32, lr: f32, b1: f32, b2: f32, wd: f32, eps: f32) {
    *param -= lr * wd * *param;
    *m = b1 * *m + (1.0 - b1) * grad;
    *v = b2 * *v + (1.0 - b2) * (grad * grad);
    *param -= lr * (*m / (1.0 - b1)) / ((*v / (1.0 - b2)).sqrt() + eps);
}
