#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UbcToken(pub u16);

pub struct UbcEngine;

pub const TAMIL_ALPHABET: &str = concat!(
    " \n.,!?:;'",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "அஆஇஈஉஊஎஏஐஒஓஔஃ",
    "கஙசஞடணதநபமயரலவழளறனஜஷஸஹ",
    "்ாிீுூெேைொோௌ",
    "0123456789-"
);

impl UbcEngine {
    #[inline(always)]
    pub fn encode_char(c: char) -> UbcToken {
        if let Some(pos) = TAMIL_ALPHABET.chars().position(|x| x == c) {
            UbcToken(pos as u16)
        } else {
            UbcToken(0)
        }
    }

    #[inline(always)]
    pub fn decode_token(tok: UbcToken) -> Option<char> {
        TAMIL_ALPHABET.chars().nth(tok.0 as usize)
    }

    pub fn encode_str(text: &str) -> Vec<UbcToken> {
        text.chars().map(Self::encode_char).collect()
    }

    pub fn decode_tokens(tokens: &[UbcToken]) -> String {
        tokens.iter().filter_map(|&t| Self::decode_token(t)).collect()
    }

    pub fn vocab_size() -> usize {
        TAMIL_ALPHABET.chars().count()
    }
}
