//! CLUX Family Graph & Factorized Morphological Tokenizer

pub mod ubc;
pub mod decomposer;

pub use ubc::{UbcEngine, UbcToken};
pub use decomposer::Factorizer;

pub type NodeId = u32;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Prefix   = 0x01,
    Root     = 0x02,
    Suffix   = 0x03,
    BytePool = 0x04,
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy)]
pub struct FamilyNode {
    pub node_id: NodeId,
    pub node_type: NodeType,
    pub flags: u8,
    pub length: u16,
    pub pool_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeToken {
    pub prefix_id: Option<NodeId>,
    pub root_id: NodeId,
    pub suffix_id: Option<NodeId>,
}

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

    pub fn insert_node(&mut self, node_type: NodeType, text: &str) -> NodeId {
        let node_id = self.nodes.len() as NodeId;
        let pool_offset = self.string_pool.len() as u32;
        let bytes = text.as_bytes();

        self.string_pool.extend_from_slice(bytes);
        self.nodes.push(FamilyNode {
            node_id,
            node_type,
            flags: 0,
            length: bytes.len() as u16,
            pool_offset,
        });

        node_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tamil_factorization() {
        let mut graph = FamilyGraph::new();
        let token = Factorizer::intern_word(&mut graph, "மரங்கள்"); // மரம் + கள்
        
        assert_eq!(graph.node_count(), 2); // Root(மரம்) + Suffix(கள்)
        assert!(token.prefix_id.is_none());
        assert!(token.suffix_id.is_some());
    }

    #[test]
    fn test_ubc_encoding() {
        let tokens = UbcEngine::encode_str("தமிழ்");
        assert_eq!(tokens.len(), 5); // 5 codepoints mapped to atomic 16-bit IDs
    }
}
