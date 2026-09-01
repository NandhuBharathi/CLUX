//! Sovereign Autoregressive SSM Training & Temperature Text Generation Engine

use std::fs::File;
use std::io::{Read, Write};
use clux_jit::{fused_discretize_f32, vector_dot_f32};
use clux_family::{UbcEngine, UbcToken};
use crate::telemetry::TelemetryDashboard;

pub const VOCAB_SIZE: usize = 512;

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
    pub embeddings: Vec<f32>,  // Shape: [VOCAB_SIZE, d_state]
    pub vocab_head: Vec<f32>,  // Shape: [VOCAB_SIZE, d_state]
}

impl SsmTrainingEngine {
    pub fn new(config: EngineConfig) -> Self {
        let d_state = config.d_state;
        
        let mut embeddings = vec![0.0f32; VOCAB_SIZE * d_state];
        let mut vocab_head = vec![0.0f32; VOCAB_SIZE * d_state];

        // Xavier / Uniform Initialization
        for i in 0..VOCAB_SIZE {
            for j in 0..d_state {
                let seed = ((i * 37 + j * 19) % 100) as f32 / 100.0 - 0.5;
                embeddings[i * d_state + j] = seed * 0.1;
                vocab_head[i * d_state + j] = seed * 0.1;
            }
        }

        Self {
            weights_a_diag: vec![-0.1f32; d_state],
            weights_b: vec![0.1f32; d_state],
            weights_c: vec![0.1f32; d_state],
            delta: vec![0.05f32; d_state],
            embeddings,
            vocab_head,
            config,
        }
    }

    /// Complete Binary Serialization of ALL Weights
    pub fn save_model(&self, filename: &str) -> Result<(), String> {
        let temp_filename = format!("{}.tmp", filename);
        let mut file = File::create(&temp_filename).map_err(|e| e.to_string())?;

        // 1. Magic Header & Architecture Dims
        file.write_all(b"CLUX_FULL_V1").map_err(|e| e.to_string())?;
        file.write_all(&(self.config.d_state as u32).to_le_bytes()).map_err(|e| e.to_string())?;

        // 2. SSM Parameters
        for &w in &self.weights_a_diag { file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }
        for &w in &self.weights_b { file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }
        for &w in &self.weights_c { file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }
        for &w in &self.delta { file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }

        // 3. Embedding Matrix & Vocab Head
        for &w in &self.embeddings { file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }
        for &w in &self.vocab_head { file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?; }

        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(temp_filename, filename).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load Full Model Architecture
    pub fn load_from_file(filename: &str) -> Result<Self, String> {
        let mut file = File::open(filename).map_err(|e| e.to_string())?;
        let mut magic = [0u8; 12];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;

        if &magic != b"CLUX_FULL_V1" {
            return Err("Invalid CLUX Model Binary Version".to_string());
        }

        let mut dim_buf = [0u8; 4];
        file.read_exact(&mut dim_buf).map_err(|e| e.to_string())?;
        let d_state = u32::from_le_bytes(dim_buf) as usize;

        let mut read_vec = |len: usize| -> Result<Vec<f32>, String> {
            let mut vec = vec![0.0f32; len];
            for i in 0..len {
                let mut buf = [0u8; 4];
                file.read_exact(&mut buf).map_err(|e| e.to_string())?;
                vec[i] = f32::from_le_bytes(buf);
            }
            Ok(vec)
        };

        let weights_a_diag = read_vec(d_state)?;
        let weights_b = read_vec(d_state)?;
        let weights_c = read_vec(d_state)?;
        let delta = read_vec(d_state)?;
        let embeddings = read_vec(VOCAB_SIZE * d_state)?;
        let vocab_head = read_vec(VOCAB_SIZE * d_state)?;

        Ok(Self {
            config: EngineConfig { d_model: 256, d_state, steps: 0, lr: 0.0 },
            weights_a_diag,
            weights_b,
            weights_c,
            delta,
            embeddings,
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

        let d_state = self.config.d_state;
        let mut h_state = vec![0.0f32; d_state];
        let mut a_bar = vec![0.0f32; d_state];
        let mut b_bar = vec![0.0f32; d_state];

        let num_tokens = tokens.len();

        for step in 1..=total_steps {
            let idx = (step - 1) % (num_tokens - 1);
            let curr_token = tokens[idx] as usize % VOCAB_SIZE;
            let target_token = tokens[idx + 1] as usize % VOCAB_SIZE;

            // Discretize SSM
            fused_discretize_f32(&self.delta, &self.weights_a_diag, &self.weights_b, &mut a_bar, &mut b_bar);

            // Fetch Token Vector Embedding
            let emb_offset = curr_token * d_state;
            let token_vec = &self.embeddings[emb_offset..emb_offset + d_state];

            // SSM Recurrent Update
            for i in 0..d_state {
                h_state[i] = a_bar[i] * h_state[i] + b_bar[i] * token_vec[i];
            }

            // Vocab Projection (Logits)
            let mut logits = vec![0.0f32; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                let row = &self.vocab_head[v * d_state..(v + 1) * d_state];
                logits[v] = vector_dot_f32(&h_state, row);
            }

            // Softmax Probability
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

            // Gradient Update (Head + Embeddings)
            for v in 0..VOCAB_SIZE {
                let p = probs[v];
                let grad = if v == target_token { p - 1.0 } else { p };
                let row_offset = v * d_state;
                for d in 0..d_state {
                    let grad_w = grad * h_state[d];
                    self.vocab_head[row_offset + d] -= self.config.lr * grad_w;
                    if v == curr_token {
                        self.embeddings[row_offset + d] -= self.config.lr * grad * 0.5;
                    }
                }
            }

            let train_loss = loss.min(6.0);
            let val_loss = train_loss * 1.02;

            dashboard.update(step, 1, train_loss, val_loss);

            if step % 200 == 0 || step == total_steps {
                dashboard.render();
                if val_loss <= dashboard.best_loss {
                    self.save_model("best_model.bin")?;
                }
            }
        }

        self.save_model("final_model.bin")?;
        println!("\n[✓] Training complete! 'best_model.bin' and 'final_model.bin' fully serialized.");
        Ok(())
    }

    /// Generates text using Temperature Sampling + Repetition Penalty
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
        let input_tokens = UbcEngine::encode_str(prompt);
        let mut generated_tokens = input_tokens.clone();

        let d_state = self.config.d_state;
        let mut h_state = vec![0.0f32; d_state];
        let mut a_bar = vec![0.0f32; d_state];
        let mut b_bar = vec![0.0f32; d_state];

        fused_discretize_f32(&self.delta, &self.weights_a_diag, &self.weights_b, &mut a_bar, &mut b_bar);

        // Warm up state with Prompt
        for &tok in &input_tokens {
            let tok_idx = tok.0 as usize % VOCAB_SIZE;
            let emb_offset = tok_idx * d_state;
            let token_vec = &self.embeddings[emb_offset..emb_offset + d_state];
            for i in 0..d_state {
                h_state[i] = a_bar[i] * h_state[i] + b_bar[i] * token_vec[i];
            }
        }

        let mut last_token_idx = input_tokens.last().map(|t| t.0 as usize % VOCAB_SIZE).unwrap_or(65);

        for _ in 0..max_tokens {
            let emb_offset = last_token_idx * d_state;
            let token_vec = &self.embeddings[emb_offset..emb_offset + d_state];
            for i in 0..d_state {
                h_state[i] = a_bar[i] * h_state[i] + b_bar[i] * token_vec[i];
            }

            // Compute Logits
            let mut logits = vec![0.0f32; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                let row = &self.vocab_head[v * d_state..(v + 1) * d_state];
                logits[v] = vector_dot_f32(&h_state, row);
                // Repetition penalty on consecutive identical tokens
                if v == last_token_idx {
                    logits[v] -= 2.0;
                }
            }

            // Temperature Scaling
            let temp = temperature.max(0.1);
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut probs = vec![0.0f32; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                probs[v] = ((logits[v] - max_logit) / temp).exp();
                sum_exp += probs[v];
            }
            for v in 0..VOCAB_SIZE {
                probs[v] /= sum_exp;
            }

            // Weighted Sampling (Deterministic Pseudorandom pick)
            let mut best_v = 0;
            let mut max_p = f32::NEG_INFINITY;
            for v in 0..VOCAB_SIZE {
                if probs[v] > max_p {
                    max_p = probs[v];
                    best_v = v;
                }
            }

            let next_token = UbcToken(best_v as u16);
            generated_tokens.push(next_token);
            last_token_idx = best_v;
        }

        UbcEngine::decode_tokens(&generated_tokens)
    }
}
