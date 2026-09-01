//! Universal AI Bytecode (UBC) Engine
//! Encodes UTF-8 codepoints / atomic graphemes into compact 16-bit IDs.

use crate::UbcToken;

pub struct UbcEngine;

impl UbcEngine {
    /// ASCII (0..127) directly maps to 0x0000..0x007F
    /// Tamil unicode block (0x0B80..0x0BFF) maps to 0x0100..0x017F
    #[inline(always)]
    pub fn encode_char(c: char) -> UbcToken {
        let val = c as u32;
        if val < 128 {
            UbcToken(val as u16)
        } else if (0x0B80..=0x0BFF).contains(&val) {
            // Tamil Unicode Block offset mapping
            UbcToken(0x0100 + (val - 0x0B80) as u16)
        } else {
            // Fallback for general multilingual hash/range
            let folded = ((val >> 8) ^ (val & 0xFF)) as u16;
            UbcToken(0x0200 | (folded & 0x01FF))
        }
    }

    pub fn encode_str(text: &str) -> Vec<UbcToken> {
        text.chars().map(Self::encode_char).collect()
    }
}
