//! CLUX Sovereign AI Binary CLI

pub mod telemetry;
pub mod engine;

use std::env;
use clux_jit::JitEngine;
use engine::{EngineConfig, SsmTrainingEngine};

fn main() {
    let args: Vec<String> = env::args().collect();
    let jit = JitEngine::new();

    if args.len() > 1 && args[1] == "--info" {
        println!("==================================================");
        println!("  CLUX AI Engine [Sovereign Bare-Metal]");
        println!("  Target SIMD Architecture : {:?}", jit.target);
        println!("  Workspace Crates Loaded : 5 Crates Active");
        println!("==================================================");
        return;
    }

    println!("[*] Initializing CLUX SSM Training Simulation...");
    let config = EngineConfig {
        d_model: 256,
        d_state: 32,
        steps: 100,
        lr: 0.0003,
    };

    let mut engine = SsmTrainingEngine::new(config);
    if let Err(e) = engine.run_training() {
        eprintln!("[!] Engine Error: {}", e);
    }
}
