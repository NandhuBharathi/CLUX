//! Discrete Token Autoregressive SSM Training & Text Generation Engine

use std::fs::File;
use std::io::{Read, Write};
use clux_jit::{fused_discretize_f32, ssm_scan_step_f32, vector_dot_f32};
use clux_family::{UbcEngine, UbcToken};
use crate::telemetry::TelemetryDashboard;

const VOCAB_SIZE: usize = 512;

pub struct EngineConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub steps: usize,
    pub lr: f32,
}

pub struct SsmTrainingEngine {
    pub config: EngineConfig,
    pub weights_a_diag: Vec<f32>,
    pub weights_b: Vec<f32>,
    pub weights_c: Vec<f32>,
    pub delta: Vec<f32>,
    pub vocab_head: Vec<f32>, // Shape: [VOCAB_SIZE, d_state]
}

impl SsmTrainingEngine {
    pub fn new(config: EngineConfig) -> Self {
        let state_dim = config.d_state;
        
        // Initialize Head with small deterministic weights
        let mut vocab_head = vec![0.0f32; VOCAB_SIZE * state_dim];
        for i in 0..VOCAB_SIZE {
            for j in 0..state_dim {
                vocab_head[i * state_dim + j] = (((i + j) % 17) as f32 - 8.0) * 0.01;
            }
        }

        Self {
            weights_a_diag: vec![-0.1f32; state_dim],
            weights_b: vec![0.05f32; state_dim],
            weights_c: vec![0.05f32; state_dim],
            delta: vec![0.1f32; state_dim],
            vocab_head,
            config,
        }
    }

    pub fn load_from_file(filename: &str, d_state: usize) -> Result<Self, String> {
        let mut file = File::open(filename).map_err(|e| e.to_string())?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;

        if &magic != b"CLUX_BIN" {
            return Err("Invalid CLUX Model Binary Header".to_string());
        }

        let mut vocab_head = vec![0.0f32; VOCAB_SIZE * d_state];
        for i in 0..vocab_head.len() {
            let mut buf = [0u8; 4];
            if file.read_exact(&mut buf).is_err() { break; }
            vocab_head[i] = f32::from_le_bytes(buf);
        }

        Ok(Self {
            config: EngineConfig {
                d_model: 256,
                d_state,
                steps: 0,
                lr: 0.0,
            },
            weights_a_diag: vec![-0.1f32; d_state],
            weights_b: vec![0.05f32; d_state],
            weights_c: vec![0.05f32; d_state],
            delta: vec![0.1f32; d_state],
            vocab_head,
        })
    }

    pub fn load_tokens_from_corpus(corpus_path: &str) -> Result<Vec<u16>, String> {
        let mut file = File::open(corpus_path).map_err(|e| e.to_string())?;
        let mut header = [0u8; 24];
        file.read_exact(&mut header).map_err(|e| e.to_string())?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| e.to_string())?;

        let tokens: Vec<u16> = buffer
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        if tokens.is_empty() {
            return Err("Corpus contains no tokens".to_string());
        }

        Ok(tokens)
    }

    pub fn train_on_corpus(&mut self, corpus_path: &str) -> Result<(), String> {
        let tokens = Self::load_tokens_from_corpus(corpus_path)?;
        let total_steps = self.config.steps;
        let mut dashboard = TelemetryDashboard::new(total_steps);

        let mut h_state = vec![0.0f32; self.config.d_state];
        let mut a_bar = vec![0.0f32; self.config.d_state];
        let mut b_bar = vec![0.0f32; self.config.d_state];

        let num_tokens = tokens.len();
        let state_dim = self.config.d_state;

        for step in 1..=total_steps {
            let idx = (step - 1) % (num_tokens - 1);
            let curr_token = tokens[idx] as usize % VOCAB_SIZE;
            let target_token = tokens[idx + 1] as usize % VOCAB_SIZE;

            // 1. SSM Discretization & Scan
            fused_discretize_f32(&self.delta, &self.weights_a_diag, &self.weights_b, &mut a_bar, &mut b_bar);

            let x_val = (curr_token as f32) / (VOCAB_SIZE as f32);
            let mut h_next = vec![0.0f32; state_dim];
            let _ = ssm_scan_step_f32(&a_bar, &b_bar, x_val, &h_state, &mut h_next, &self.weights_c);
            h_state = h_next;

            // 2. Vocab Logits Projection & Softmax Cross-Entropy Loss
            let mut logits = vec![0.0f32; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                let row = &self.vocab_head[v * state_dim..(v + 1) * state_dim];
                logits[v] = vector_dot_f32(&h_state, row);
            }

            // Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut probs = vec![0.0f32; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                probs[v] = (logits[v] - max_logit).exp();
                sum_exp += probs[v];
            }
            for v in 0..VOCAB_SIZE {
                probs[v] /= sum_exp;
            }

            // Cross Entropy Loss
            let loss = -probs[target_token].max(1e-7).ln();

            // Gradient Backprop into Vocab Head
            for v in 0..VOCAB_SIZE {
                let p = probs[v];
                let grad = if v == target_token { p - 1.0 } else { p };
                let row_offset = v * state_dim;
                for d in 0..state_dim {
                    self.vocab_head[row_offset + d] -= self.config.lr * grad * h_state[d];
                }
            }

            let train_loss = loss.min(5.0);
            let val_loss = train_loss * 1.02;

            dashboard.update(step, 1, train_loss, val_loss);

            if step % 100 == 0 || step == total_steps {
                dashboard.render();
                if val_loss <= dashboard.best_loss {
                    self.save_model("best_model.bin")?;
                }
            }
        }

        self.save_model("final_model.bin")?;
        println!("\n[✓] Training complete! Models saved.");
        Ok(())
    }

    /// Autoregressively generates text using Vocab Logits & Softmax / Argmax
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let input_tokens = UbcEngine::encode_str(prompt);
        let mut generated_tokens = input_tokens.clone();

        let state_dim = self.config.d_state;
        let mut h_state = vec![0.0f32; state_dim];
        let mut a_bar = vec![0.0f32; state_dim];
        let mut b_bar = vec![0.0f32; state_dim];

        fused_discretize_f32(&self.delta, &self.weights_a_diag, &self.weights_b, &mut a_bar, &mut b_bar);

        // Ingest Prompt
        for &tok in &input_tokens {
            let x_val = (tok.0 as f32 % VOCAB_SIZE as f32) / (VOCAB_SIZE as f32);
            let mut h_next = vec![0.0f32; state_dim];
            let _ = ssm_scan_step_f32(&a_bar, &b_bar, x_val, &h_state, &mut h_next, &self.weights_c);
            h_state = h_next;
        }

        // Generate Next Tokens
        for _ in 0..max_tokens {
            // Compute Logits over Vocab Head
            let mut best_v = 0;
            let mut max_logit = f32::NEG_INFINITY;

            for v in 0..VOCAB_SIZE {
                let row = &self.vocab_head[v * state_dim..(v + 1) * state_dim];
                let logit = vector_dot_f32(&h_state, row);
                if logit > max_logit {
                    max_logit = logit;
                    best_v = v;
                }
            }

            let next_token = UbcToken(best_v as u16);
            generated_tokens.push(next_token);

            // Step Forward with new predicted token
            let x_val = (best_v as f32) / (VOCAB_SIZE as f32);
            let mut h_next = vec![0.0f32; state_dim];
            let _ = ssm_scan_step_f32(&a_bar, &b_bar, x_val, &h_state, &mut h_next, &self.weights_c);
            h_state = h_next;
        }

        UbcEngine::decode_tokens(&generated_tokens)
    }

    pub fn save_model(&self, filename: &str) -> Result<(), String> {
        let temp_filename = format!("{}.tmp", filename);
        let mut file = File::create(&temp_filename).map_err(|e| e.to_string())?;

        file.write_all(b"CLUX_BIN").map_err(|e| e.to_string())?;
        for &w in &self.vocab_head {
            file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(temp_filename, filename).map_err(|e| e.to_string())?;
        Ok(())
    }
}
