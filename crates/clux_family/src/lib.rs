//! CLUX Family Graph & Factorized Morphological Tokenizer
//! 
//! Memory Layout:
//! 1. Atomic Grapheme Cluster (16-bit UBC Encoding)
//! 2. Factorized Graph: Prefix + Root + Suffix + Byte Pool
//! 3. Zero-Copy Lookup Tables

pub type NodeId = u32;

/// Universal AI Bytecode (UBC) - 16-bit Atomic Grapheme
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UbcToken(pub u16);

/// FAMILY Graph Node Types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Prefix   = 0x01,
    Root     = 0x02,
    Suffix   = 0x03,
    BytePool = 0x04,
}

/// 64-bit Packed Node inside the Morphological Graph
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct FamilyNode {
    pub node_id: NodeId,
    pub node_type: NodeType,
    pub flags: u8,
    pub length: u16,
}

/// Factorized Composite Token Definition
#[derive(Debug, Clone)]
pub struct CompositeToken {
    pub prefix_id: Option<NodeId>,
    pub root_id: NodeId,
    pub suffix_id: Option<NodeId>,
}

/// Core FAMILY Graph Structure
pub struct FamilyGraph {
    pub nodes: Vec<FamilyNode>,
    pub string_pool: Vec<u8>,
}

impl FamilyGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(65536),
            string_pool: Vec::with_capacity(1024 * 1024),
        }
    }

    #[inline(always)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}
