//! Universal AI Bytecode (UBC) Engine
//! Encodes and Decodes UTF-8 / Graphemes into 16-bit IDs.

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UbcToken(pub u16);

pub struct UbcEngine;

impl UbcEngine {
    #[inline(always)]
    pub fn encode_char(c: char) -> UbcToken {
        let val = c as u32;
        if val < 128 {
            UbcToken(val as u16)
        } else if (0x0B80..=0x0BFF).contains(&val) {
            UbcToken(0x0100 + (val - 0x0B80) as u16)
        } else {
            let folded = ((val >> 8) ^ (val & 0xFF)) as u16;
            UbcToken(0x0200 | (folded & 0x01FF))
        }
    }

    #[inline(always)]
    pub fn decode_token(tok: UbcToken) -> Option<char> {
        let val = tok.0 as u32;
        if val < 128 {
            char::from_u32(val)
        } else if (0x0100..=0x017F).contains(&val) {
            char::from_u32(0x0B80 + (val - 0x0100))
        } else {
            // Default printable fallback
            Some(' ')
        }
    }

    pub fn encode_str(text: &str) -> Vec<UbcToken> {
        text.chars().map(Self::encode_char).collect()
    }

    pub fn decode_tokens(tokens: &[UbcToken]) -> String {
        tokens.iter().filter_map(|&t| Self::decode_token(t)).collect()
    }
}
