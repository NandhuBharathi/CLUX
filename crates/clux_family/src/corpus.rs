//! Streaming Binary Corpus Compiler & Memory Reader

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use crate::ubc::UbcEngine;

pub struct CorpusCompiler;

impl CorpusCompiler {
    const MAGIC: &'static [u8; 12] = b"CLUX_CORPUS\0";
    const VERSION: u32 = 1;

    /// Compiles a raw text stream / file directly into a binary token stream
    pub fn compile_from_reader<R: Read, W: Write>(reader: R, writer: &mut W) -> Result<u64, String> {
        let mut buf_reader = BufReader::new(reader);
        let mut buf_writer = BufWriter::new(writer);

        // 1. Write Initial Header (Placeholder for token count)
        buf_writer.write_all(Self::MAGIC).map_err(|e| e.to_string())?;
        buf_writer.write_all(&Self::VERSION.to_le_bytes()).map_err(|e| e.to_string())?;
        buf_writer.write_all(&0u64.to_le_bytes()).map_err(|e| e.to_string())?;

        let mut total_tokens: u64 = 0;
        let mut line = String::new();

        // 2. Line-by-line streaming without loading entire dataset into RAM
        while buf_reader.read_line(&mut line).map_err(|e| e.to_string())? > 0 {
            let tokens = UbcEngine::encode_str(&line);
            for tok in tokens {
                buf_writer.write_all(&tok.0.to_le_bytes()).map_err(|e| e.to_string())?;
                total_tokens += 1;
            }
            line.clear();
        }

        buf_writer.flush().map_err(|e| e.to_string())?;
        Ok(total_tokens)
    }

    /// Reads metadata from a compiled corpus.bin
    pub fn inspect_corpus(path: &str) -> Result<(u32, u64), String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut magic = [0u8; 12];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;

        if &magic != Self::MAGIC {
            return Err("Invalid CLUX Corpus binary signature".to_string());
        }

        let mut ver_bytes = [0u8; 4];
        file.read_exact(&mut ver_bytes).map_err(|e| e.to_string())?;
        let version = u32::from_le_bytes(ver_bytes);

        let mut count_bytes = [0u8; 8];
        file.read_exact(&mut count_bytes).map_err(|e| e.to_string())?;
        let token_count = u64::from_le_bytes(count_bytes);

        Ok((version, token_count))
    }
}
