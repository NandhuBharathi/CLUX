//! CLUX Sovereign AI Binary CLI

pub mod telemetry;
pub mod engine;

use std::env;
use std::fs::File;
use std::io::Seek;
use clux_jit::JitEngine;
use clux_family::CorpusCompiler;
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

    // CLI Subcommand: prepare <input_path_or_url> [output_path]
    if args.len() > 2 && args[1] == "prepare" {
        let input_path = &args[2];
        let output_path = if args.len() > 3 { &args[3] } else { "corpus.bin" };

        println!("[*] Ingesting dataset from: {}", input_path);
        
        let input_file = match File::open(input_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[!] Failed to open input file: {}", e);
                return;
            }
        };

        let mut out_file = match File::create(output_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[!] Failed to create output binary: {}", e);
                return;
            }
        };

        match CorpusCompiler::compile_from_reader(input_file, &mut out_file) {
            Ok(total_tokens) => {
                // Update total token count in the header
                let mut out_file = File::options().write(true).open(output_path).unwrap();
                out_file.seek(std::io::SeekFrom::Start(16)).unwrap();
                std::io::Write::write_all(&mut out_file, &total_tokens.to_le_bytes()).unwrap();

                println!("[✓] Dataset successfully compiled to '{}'", output_path);
                println!("    Total UBC Tokens Processed: {}", total_tokens);
            }
            Err(e) => eprintln!("[!] Compilation error: {}", e),
        }
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
