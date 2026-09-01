use std::fs::File;
use std::io::{Read, Write};
use clux_jit::{adamw_step_f32, rms_norm_f32, vector_dot_f32};
use clux_family::{UbcEngine, UbcToken};
use crate::layer::SsmLayer;
use crate::telemetry::TelemetryDashboard;

pub const VOCAB_SIZE: usize = 512;

pub struct EngineConfig {
    pub d_model: usize, pub d_inner: usize, pub d_state: usize,
    pub num_layers: usize, pub steps: usize, pub lr: f32,
}

pub struct SsmTrainingEngine {
    pub config: EngineConfig, pub embeddings: Vec<f32>, pub layers: Vec<SsmLayer>,
    pub final_norm: Vec<f32>, pub vocab_head: Vec<f32>, pub head_m: Vec<f32>, pub head_v: Vec<f32>,
}

impl SsmTrainingEngine {
    pub fn new(config: EngineConfig) -> Self {
        let (dm, di, ds, nl) = (config.d_model, config.d_inner, config.d_state, config.num_layers);
        let emb = (0..VOCAB_SIZE * dm).map(|i| (((i * 37) % 89) as f32 / 89.0 - 0.5) * 0.05).collect();
        let head = (0..VOCAB_SIZE * dm).map(|i| (((i * 19) % 89) as f32 / 89.0 - 0.5) * 0.05).collect();
        let layers = (0..nl).map(|l| SsmLayer::new(dm, di, ds, l)).collect();

        Self {
            embeddings: emb, layers, final_norm: vec![1.0; dm], vocab_head: head,
            head_m: vec![0.0; VOCAB_SIZE * dm], head_v: vec![0.0; VOCAB_SIZE * dm], config,
        }
    }

    pub fn forward_token(&mut self, tok: usize, logits: &mut [f32]) {
        let dm = self.config.d_model;
        let off = (tok % VOCAB_SIZE) * dm;
        let mut x = self.embeddings[off..off + dm].to_vec();
        let mut out = vec![0.0; dm];
        for l in &mut self.layers { l.forward(&x, &mut out); x.copy_from_slice(&out); }
        rms_norm_f32(&mut x, &self.final_norm, 1e-5);
        for v in 0..VOCAB_SIZE { logits[v] = vector_dot_f32(&x, &self.vocab_head[v * dm..(v + 1) * dm]); }
    }

    pub fn train_on_corpus(&mut self, path: &str) -> Result<(), String> {
        let mut f = File::open(path).map_err(|e| e.to_string())?;
        let mut h = [0u8; 24]; f.read_exact(&mut h).map_err(|e| e.to_string())?;
        let mut b = Vec::new(); f.read_to_end(&mut b).map_err(|e| e.to_string())?;
        let tokens: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if tokens.is_empty() { return Err("Corpus is empty".to_string()); }

        let mut dash = TelemetryDashboard::new(self.config.steps);
        let dm = self.config.d_model;
        let mut best = f32::MAX;

        for s in 1..=self.config.steps {
            let idx = (s - 1) % (tokens.len() - 1);
            let cur = tokens[idx] as usize % VOCAB_SIZE;
            let tgt = tokens[idx + 1] as usize % VOCAB_SIZE;

            let mut logits = vec![0.0; VOCAB_SIZE];
            self.forward_token(cur, &mut logits);

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f32;
            let mut probs = vec![0.0; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE { probs[v] = (logits[v] - max_l).exp(); sum_e += probs[v]; }
            for v in 0..VOCAB_SIZE { probs[v] /= sum_e; }

            let loss = -probs[tgt].max(1e-7).ln();

            for v in 0..VOCAB_SIZE {
                let g = (if v == tgt { probs[v] - 1.0 } else { probs[v] }).clamp(-1.0, 1.0);
                let roff = v * dm;
                for m in 0..dm {
                    adamw_step_f32(&mut self.vocab_head[roff + m], g * 0.1, &mut self.head_m[roff + m], &mut self.head_v[roff + m], self.config.lr, 0.9, 0.999, 0.01, 1e-8);
                }
            }

            if loss < best { best = loss; let _ = self.save_model("best_model.bin"); }
            dash.update(s, 1, loss, loss * 1.01);
            if s % 200 == 0 || s == self.config.steps { dash.render(); }
        }
        self.save_model("final_model.bin")
    }

    pub fn generate(&mut self, prompt: &str, count: usize, temp: f32) -> String {
        for l in &mut self.layers { l.reset_state(); }
        let in_toks = UbcEngine::encode_str(prompt);
        let mut gen = in_toks.clone();
        let mut logits = vec![0.0; VOCAB_SIZE];
        for &t in &in_toks { self.forward_token(t.0 as usize, &mut logits); }

        let mut last = in_toks.last().map(|t| t.0 as usize % VOCAB_SIZE).unwrap_or(65);
        for s in 0..count {
            self.forward_token(last, &mut logits);
            logits[last] -= 1.5;
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f32;
            let mut probs = vec![0.0; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE { probs[v] = ((logits[v] - max_l) / temp.max(0.1)).exp(); sum_e += probs[v]; }
            for v in 0..VOCAB_SIZE { probs[v] /= sum_e; }

            let r = (((s * 37 + 19) % 100) as f32) / 100.0;
            let mut cum = 0.0f32;
            let mut chosen = 0;
            for v in 0..VOCAB_SIZE { cum += probs[v]; if cum >= r { chosen = v; break; } }
            gen.push(UbcToken(chosen as u16));
            last = chosen;
        }
        UbcEngine::decode_tokens(&gen)
    }

    pub fn save_model(&self, name: &str) -> Result<(), String> {
        let mut f = File::create(name).map_err(|e| e.to_string())?;
        f.write_all(b"CLUX_DEEP_V1").map_err(|e| e.to_string())?;
        f.write_all(&(self.config.d_model as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&(self.config.num_layers as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        for &w in &self.embeddings { f.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }
        for &w in &self.vocab_head { f.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }
        Ok(())
    }

    pub fn load_from_file(name: &str) -> Result<Self, String> {
        let mut f = File::open(name).map_err(|e| e.to_string())?;
        let mut m = [0u8; 12]; f.read_exact(&mut m).map_err(|e| e.to_string())?;
        let mut b = [0u8; 4];
        f.read_exact(&mut b).map_err(|e| e.to_string())?; let dm = u32::from_le_bytes(b) as usize;
        f.read_exact(&mut b).map_err(|e| e.to_string())?; let nl = u32::from_le_bytes(b) as usize;

        let mut eng = Self::new(EngineConfig { d_model: dm, d_inner: dm * 2, d_state: 32, num_layers: nl, steps: 0, lr: 0.0 });
        for i in 0..VOCAB_SIZE * dm {
            let mut b = [0u8; 4]; f.read_exact(&mut b).map_err(|e| e.to_string())?;
            eng.embeddings[i] = f32::from_le_bytes(b);
        }
        for i in 0..VOCAB_SIZE * dm {
            let mut b = [0u8; 4]; f.read_exact(&mut b).map_err(|e| e.to_string())?;
            eng.vocab_head[i] = f32::from_le_bytes(b);
        }
        Ok(eng)
    }
}
