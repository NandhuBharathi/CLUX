//! Morphological Factorization & Radix Sub-string Extraction

use crate::{FamilyGraph, FamilyNode, NodeType, CompositeToken, NodeId};

pub struct Factorizer;

impl Factorizer {
    /// Common Tamil / Multilingual Suffixes
    const SUFFIXES: &'static [&'static str] = &[
        "கள்", "gal", "இல்", "il", "ஆல்", "aal", "உக்கு", "ukku", 
        "இன்", "in", "உடைய", "udaiya", "ing", "ed", "tion", "ly"
    ];

    /// Common Prefixes
    const PREFIXES: &'static [&'static str] = &[
        "அ", "இ", "எ", "un", "re", "in", "dis", "pre"
    ];

    /// Decomposes a word into (Prefix, Root, Suffix)
    pub fn decompose<'a>(word: &'a str) -> (&'a str, &'a str, &'a str) {
        if word.chars().count() <= 3 {
            return ("", word, "");
        }

        let mut prefix = "";
        let mut remainder = word;
        let mut suffix = "";

        // Check Prefix
        for p in Self::PREFIXES {
            if remainder.starts_with(p) && remainder.len() > p.len() + 2 {
                prefix = p;
                remainder = &remainder[p.len()..];
                break;
            }
        }

        // Check Suffix
        for s in Self::SUFFIXES {
            if remainder.ends_with(s) && remainder.len() > s.len() + 2 {
                suffix = s;
                remainder = &remainder[..remainder.len() - s.len()];
                break;
            }
        }

        (prefix, remainder, suffix)
    }

    /// Interns a factorized token into the FamilyGraph
    pub fn intern_word(graph: &mut FamilyGraph, word: &str) -> CompositeToken {
        let (prefix, root, suffix) = Self::decompose(word);

        let prefix_id = if !prefix.is_empty() {
            Some(graph.insert_node(NodeType::Prefix, prefix))
        } else {
            None
        };

        let root_id = graph.insert_node(NodeType::Root, root);

        let suffix_id = if !suffix.is_empty() {
            Some(graph.insert_node(NodeType::Suffix, suffix))
        } else {
            None
        };

        CompositeToken {
            prefix_id,
            root_id,
            suffix_id,
        }
    }
}
