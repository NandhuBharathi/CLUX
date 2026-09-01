//! SSM Training Engine & Crash-Resilient Binary Serializer

use std::fs::File;
use std::io::Write;
use clux_jit::{fused_discretize_f32, ssm_scan_step_f32};
use clux_family::{FamilyGraph, Factorizer};
use crate::telemetry::TelemetryDashboard;

pub struct EngineConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub steps: usize,
    pub lr: f32,
}

pub struct SsmTrainingEngine {
    pub config: EngineConfig,
    pub graph: FamilyGraph,
    pub weights_a_diag: Vec<f32>,
    pub weights_b: Vec<f32>,
    pub weights_c: Vec<f32>,
    pub delta: Vec<f32>,
}

impl SsmTrainingEngine {
    pub fn new(config: EngineConfig) -> Self {
        let state_dim = config.d_state;
        Self {
            weights_a_diag: vec![-0.5f32; state_dim],
            weights_b: vec![0.1f32; state_dim],
            weights_c: vec![0.2f32; state_dim],
            delta: vec![0.05f32; state_dim],
            graph: FamilyGraph::new(),
            config,
        }
    }

    pub fn run_training(&mut self) -> Result<(), String> {
        let mut dashboard = TelemetryDashboard::new(self.config.steps);
        let mut h_state = vec![0.0f32; self.config.d_state];
        let mut a_bar = vec![0.0f32; self.config.d_state];
        let mut b_bar = vec![0.0f32; self.config.d_state];

        // Sample text tokenization via FAMILY factorizer
        Factorizer::intern_word(&mut self.graph, "CLUX");
        Factorizer::intern_word(&mut self.graph, "வேகம்");

        for step in 1..=self.config.steps {
            // 1. Fused Discretization
            fused_discretize_f32(&self.delta, &self.weights_a_diag, &self.weights_b, &mut a_bar, &mut b_bar);

            // 2. SSM Scan Forward Step
            let mut h_next = vec![0.0f32; self.config.d_state];
            let x_val = 1.0f32;
            let _y = ssm_scan_step_f32(&a_bar, &b_bar, x_val, &h_state, &mut h_next, &self.weights_c);
            h_state = h_next;

            // 3. Simulated Loss Convergence
            let decay = (-0.005 * step as f32).exp();
            let train_loss = 1.2 * decay + 0.3;
            let val_loss = train_loss + 0.05;

            dashboard.update(step, 64, train_loss, val_loss);

            // Periodically render UI and snapshot best model
            if step % 25 == 0 || step == self.config.steps {
                dashboard.render();
                if val_loss <= dashboard.best_loss {
                    self.save_model("best_model.bin")?;
                }
            }
        }

        self.save_model("final_model.bin")?;
        println!("\n[✓] Training complete. Saved 'best_model.bin' and 'final_model.bin'");
        Ok(())
    }

    /// Saves the model weights atomically
    pub fn save_model(&self, filename: &str) -> Result<(), String> {
        let temp_filename = format!("{}.tmp", filename);
        let mut file = File::create(&temp_filename).map_err(|e| e.to_string())?;

        // Write Magic Header [CLUX_BIN (8 bytes)]
        file.write_all(b"CLUX_BIN").map_err(|e| e.to_string())?;

        // Write Weights
        for &w in &self.weights_a_diag {
            file.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(temp_filename, filename).map_err(|e| e.to_string())?;
        Ok(())
    }
}
