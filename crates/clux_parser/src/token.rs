//! Lexical Tokens for CLUX DSL

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    KwModel,
    KwTraining,
    Ident(String),
    NumberInt(usize),
    NumberFloat(f32),
    StringLit(String),
    Colon,
    Comma,
    BraceOpen,
    BraceClose,
    Eof,
}
