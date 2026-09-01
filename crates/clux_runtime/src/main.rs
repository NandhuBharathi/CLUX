pub mod telemetry;
pub mod layer;
pub mod engine;

use std::env;
use std::fs::File;
use std::io::Seek;
use std::path::Path;
use clux_jit::JitEngine;
use clux_family::CorpusCompiler;
use engine::{EngineConfig, SsmTrainingEngine};

fn main() {
    let args: Vec<String> = env::args().collect();
    let jit = JitEngine::new();

    if args.len() > 1 && args[1] == "--info" {
        println!("==================================================");
        println!("  CLUX AI Engine [Deep Sovereign Bare-Metal]");
        println!("  Architecture : 4-Layer SSM | Top-K Softmax | AdamW");
        println!("==================================================");
        return;
    }

    if args.len() > 2 && args[1] == "prepare" {
        let input_path = &args[2];
        let output_path = if args.len() > 3 { &args[3] } else { "corpus.bin" };
        let input_file = File::open(input_path).expect("Failed to open input file");
        let mut out_file = File::create(output_path).expect("Failed to create output binary");
        let total_tokens = CorpusCompiler::compile_from_reader(input_file, &mut out_file).expect("Compilation error");
        let mut update_file = File::options().write(true).open(output_path).unwrap();
        update_file.seek(std::io::SeekFrom::Start(16)).unwrap();
        std::io::Write::write_all(&mut update_file, &total_tokens.to_le_bytes()).unwrap();
        println!("[✓] Dataset compiled to '{}' ({} tokens)", output_path, total_tokens);
        return;
    }

    if args.len() > 2 && args[1] == "train" {
        let corpus_path = &args[2];
        let steps = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(4000);
        let config = EngineConfig { d_model: 128, d_inner: 256, d_state: 32, num_layers: 4, steps, lr: 0.003 };
        let mut engine = SsmTrainingEngine::new(config);
        if let Err(e) = engine.train_on_corpus(corpus_path) { eprintln!("[!] Training Error: {}", e); }
        return;
    }

    if args.len() > 2 && args[1] == "generate" {
        let model_path = if Path::new("best_model.bin").exists() { "best_model.bin" } else { "final_model.bin" };
        let prompt = &args[2];
        let max_tokens = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(40);
        let temp = args.get(4).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.4);
        
        // FIXED: Fetch top_k from args or default to 5
        let top_k = args.get(5).and_then(|s| s.parse::<usize>().ok()).unwrap_or(5);

        println!("[*] Loading Model: {}", model_path);
        match SsmTrainingEngine::load_from_file(model_path) {
            Ok(mut engine) => {
                println!("[*] Prompt: \"{}\"", prompt);
                // FIXED: Passing all 4 arguments to engine.generate()
                let result = engine.generate(prompt, max_tokens, temp, top_k);
                println!("--------------------------------------------------");
                println!("Generated Output:\n{}", result);
                println!("--------------------------------------------------");
            }
            Err(e) => eprintln!("[!] Load Error: {}", e),
        }
        return;
    }
}
