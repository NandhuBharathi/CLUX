//! Zero-Overhead Terminal Telemetry & Live Training Metrics

use std::time::Instant;

pub struct TelemetryDashboard {
    start_time: Instant,
    pub total_tokens: usize,
    pub current_step: usize,
    pub total_steps: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub best_loss: f32,
}

impl TelemetryDashboard {
    pub fn new(total_steps: usize) -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            total_tokens: 0,
            current_step: 0,
            total_steps,
            train_loss: 4.0,
            val_loss: 4.0,
            best_loss: f32::MAX,
        }
    }

    pub fn update(&mut self, step: usize, tokens_processed: usize, train_loss: f32, val_loss: f32) {
        self.current_step = step;
        self.total_tokens += tokens_processed;
        self.train_loss = train_loss;
        self.val_loss = val_loss;
        if val_loss < self.best_loss {
            self.best_loss = val_loss;
        }
    }

    pub fn render(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let tok_per_sec = if elapsed > 0.0 { self.total_tokens as f32 / elapsed } else { 0.0 };
        
        let progress_pct = (self.current_step as f32 / self.total_steps as f32) * 100.0;
        let remaining_steps = self.total_steps.saturating_sub(self.current_step);
        let eta_secs = if self.current_step > 0 {
            (elapsed / self.current_step as f32) * remaining_steps as f32
        } else {
            0.0
        };

        let perplexity = (self.train_loss.min(10.0)).exp();

        // ANSI Terminal Refresh
        print!("\x1B[2J\x1B[1;1H");
        println!("================================================================================");
        println!("  CLUX ENGINE :: SOVEREIGN SSM TRAINING RUNNER [Native Bare-Metal]");
        println!("================================================================================");
        println!("  Progress    : [{:>5.1}%] Step {} / {}", progress_pct, self.current_step, self.total_steps);
        println!("  Elapsed     : {:02}m {:02}s  |  Dynamic ETA : {:02}m {:02}s", 
            (elapsed as u32) / 60, (elapsed as u32) % 60,
            (eta_secs as u32) / 60, (eta_secs as u32) % 60
        );
        println!("--------------------------------------------------------------------------------");
        println!("  Train Loss  : {:.4}        |  Val Loss     : {:.4}", self.train_loss, self.val_loss);
        println!("  Perplexity  : {:.3}         |  Best Val Loss: {:.4}", perplexity, self.best_loss);
        println!("  Throughput  : {:>10.1} tok/s |  Total Tokens  : {}", tok_per_sec, self.total_tokens);
        println!("================================================================================");
    }
}
