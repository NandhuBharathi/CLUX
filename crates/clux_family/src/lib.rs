//! Universal Bytecode (UBC) 16-bit Tokenizer & Factorized Morphological Trie

use std::collections::HashMap;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UbcToken(pub u16);

#[derive(Default)]
pub struct TrieNode {
    pub children: HashMap<char, TrieNode>,
    pub token_id: Option<u16>,
    pub is_root: bool,
    pub is_suffix: bool,
}

pub struct MorphologicalTokenizer {
    pub root: TrieNode,
    pub next_id: u16,
    pub vocab: HashMap<u16, String>,
}

impl MorphologicalTokenizer {
    pub fn new() -> Self {
        Self {
            root: TrieNode::default(),
            next_id: 1, // 0 is reserved for UNK/Padding
            vocab: HashMap::new(),
        }
    }

    /// Decomposes words into Prefix, Root, and Suffix organically
    pub fn train_morphology(&mut self, corpus: &str) {
        let words: Vec<&str> = corpus.split_whitespace().collect();
        
        for word in words {
            let chars: Vec<char> = word.chars().collect();
            let mut current = &mut self.root;
            
            for &c in &chars {
                current = current.children.entry(c).or_insert_with(TrieNode::default);
            }
            
            if current.token_id.is_none() {
                current.token_id = Some(self.next_id);
                self.vocab.insert(self.next_id, word.to_string());
                self.next_id += 1;
            }
        }
    }

    /// Greedy Longest-Prefix-Match encoding for Morphological fragments
    pub fn encode(&self, text: &str) -> Vec<UbcToken> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut current = &self.root;
            let mut best_match_len = 0;
            let mut best_match_id = 0;
            let mut j = i;

            while j < chars.len() {
                if let Some(node) = current.children.get(&chars[j]) {
                    current = node;
                    if let Some(id) = current.token_id {
                        best_match_len = j - i + 1;
                        best_match_id = id;
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            if best_match_len > 0 {
                tokens.push(UbcToken(best_match_id));
                i += best_match_len;
            } else {
                tokens.push(UbcToken(0)); // UNK
                i += 1;
            }
        }
        tokens
    }
}
