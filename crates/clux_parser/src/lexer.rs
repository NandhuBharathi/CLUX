//! High-speed Byte Lexer

use crate::token::Token;

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            src: source.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b' ' | b'\t' | b'\r' | b'\n' => {
                    self.pos += 1;
                }
                b'#' => {
                    // Line comment
                    while let Some(c) = self.advance() {
                        if c == b'\n' { break; }
                    }
                }
                _ => break,
            }
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let Some(b) = self.peek() else {
            return Token::Eof;
        };

        match b {
            b'{' => { self.pos += 1; Token::BraceOpen }
            b'}' => { self.pos += 1; Token::BraceClose }
            b':' => { self.pos += 1; Token::Colon }
            b',' => { self.pos += 1; Token::Comma }
            b'"' => self.lex_string(),
            b'0'..=b'9' => self.lex_number(),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_ident_or_keyword(),
            _ => {
                self.pos += 1;
                self.next_token()
            }
        }
    }

    fn lex_string(&mut self) -> Token {
        self.advance(); // consume opening quote
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'"' {
                let s = String::from_utf8_lossy(&self.src[start..self.pos]).to_string();
                self.advance(); // consume closing quote
                return Token::StringLit(s);
            }
            self.pos += 1;
        }
        Token::Eof
    }

    fn lex_number(&mut self) -> Token {
        let start = self.pos;
        let mut has_dot = false;

        while let Some(b) = self.peek() {
            if b == b'.' {
                has_dot = true;
                self.pos += 1;
            } else if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }

        let slice = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
        if has_dot {
            Token::NumberFloat(slice.parse().unwrap_or(0.0))
        } else {
            Token::NumberInt(slice.parse().unwrap_or(0))
        }
    }

    fn lex_ident_or_keyword(&mut self) -> Token {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let ident = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        match ident {
            "model" => Token::KwModel,
            "training" => Token::KwTraining,
            _ => Token::Ident(ident.to_string()),
        }
    }
}
