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

    // CLI Subcommand: prepare <input_text> [output_bin]
    if args.len() > 2 && args[1] == "prepare" {
        let input_path = &args[2];
        let output_path = if args.len() > 3 { &args[3] } else { "corpus.bin" };

        println!("[*] Ingesting dataset from: {}", input_path);
        let input_file = File::open(input_path).expect("Failed to open input file");
        let mut out_file = File::create(output_path).expect("Failed to create output binary");

        let total_tokens = CorpusCompiler::compile_from_reader(input_file, &mut out_file).expect("Compilation error");
        let mut update_file = File::options().write(true).open(output_path).unwrap();
        update_file.seek(std::io::SeekFrom::Start(16)).unwrap();
        std::io::Write::write_all(&mut update_file, &total_tokens.to_le_bytes()).unwrap();

        println!("[✓] Dataset compiled to '{}' ({} tokens)", output_path, total_tokens);
        return;
    }

    // CLI Subcommand: train <corpus_bin>
    if args.len() > 2 && args[1] == "train" {
        let corpus_path = &args[2];
        let config = EngineConfig {
            d_model: 256,
            d_state: 32,
            steps: 300,
            lr: 0.001,
        };

        let mut engine = SsmTrainingEngine::new(config);
        if let Err(e) = engine.train_on_corpus(corpus_path) {
            eprintln!("[!] Training Error: {}", e);
        }
        return;
    }

    // CLI Subcommand: generate [model.bin] <prompt> [num_tokens]
    if args.len() > 2 && args[1] == "generate" {
        let model_path = if args.len() > 3 && args[2].ends_with(".bin") {
            &args[2]
        } else {
            "best_model.bin"
        };

        let prompt = if args[2].ends_with(".bin") {
            args.get(3).map(|s| s.as_str()).unwrap_or("வணக்கம்")
        } else {
            &args[2]
        };

        let max_tokens = args.last().and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);

        println!("[*] Loading model: {}", model_path);
        match SsmTrainingEngine::load_from_file(model_path, 32) {
            Ok(engine) => {
                println!("[*] Generating text for prompt: \"{}\"", prompt);
                let result = engine.generate(prompt, max_tokens);
                println!("--------------------------------------------------");
                println!("Output: {}", result);
                println!("--------------------------------------------------");
            }
            Err(e) => eprintln!("[!] Load Error: {}", e),
        }
        return;
    }

    println!("CLUX Engine CLI Usage:");
    println!("  clux --info");
    println!("  clux prepare <input_text.txt> [corpus.bin]");
    println!("  clux train <corpus.bin>");
    println!("  clux generate [model.bin] <prompt> [tokens_count]");
}
