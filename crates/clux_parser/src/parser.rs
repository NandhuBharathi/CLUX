//! Recursive Descent Parser for CLUX Specifications

use crate::ast::{CluxSpec, ModelConfig, Precision, TrainConfig};
use crate::lexer::Lexer;
use crate::token::Token;

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    curr: Token,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let curr = lexer.next_token();
        Self { lexer, curr }
    }

    fn bump(&mut self) -> Token {
        let prev = std::mem::replace(&mut self.curr, self.lexer.next_token());
        prev
    }

    pub fn parse_spec(&mut self) -> Result<CluxSpec, String> {
        let mut model_config = None;
        let mut train_config = None;

        while self.curr != Token::Eof {
            match self.curr {
                Token::KwModel => {
                    model_config = Some(self.parse_model_block()?);
                }
                Token::KwTraining => {
                    train_config = Some(self.parse_train_block()?);
                }
                _ => return Err(format!("Unexpected token: {:?}", self.curr)),
            }
        }

        let model = model_config.ok_or_else(|| "Missing 'model' definition block".to_string())?;

        Ok(CluxSpec {
            model,
            train: train_config,
        })
    }

    fn parse_model_block(&mut self) -> Result<ModelConfig, String> {
        self.bump(); // consume 'model'
        let name = match self.bump() {
            Token::Ident(n) => n,
            other => return Err(format!("Expected model identifier, got {:?}", other)),
        };

        if self.bump() != Token::BraceOpen {
            return Err("Expected '{' after model name".to_string());
        }

        let mut d_model = 512;
        let mut d_state = 64;
        let mut num_layers = 6;
        let mut precision = Precision::FP16;

        while self.curr != Token::BraceClose && self.curr != Token::Eof {
            let key = match self.bump() {
                Token::Ident(k) => k,
                other => return Err(format!("Expected field key, got {:?}", other)),
            };

            if self.bump() != Token::Colon {
                return Err("Expected ':' after field key".to_string());
            }

            match key.as_str() {
                "dim" => {
                    if let Token::NumberInt(val) = self.bump() { d_model = val; }
                }
                "state_dim" => {
                    if let Token::NumberInt(val) = self.bump() { d_state = val; }
                }
                "layers" => {
                    if let Token::NumberInt(val) = self.bump() { num_layers = val; }
                }
                "precision" => {
                    if let Token::StringLit(val) = self.bump() {
                        precision = match val.to_lowercase().as_str() {
                            "fp32" => Precision::FP32,
                            "fp16" => Precision::FP16,
                            "bf16" => Precision::BF16,
                            "int8" => Precision::INT8,
                            _ => return Err(format!("Unsupported precision: {}", val)),
                        };
                    }
                }
                _ => return Err(format!("Unknown model parameter: {}", key)),
            }

            if self.curr == Token::Comma {
                self.bump();
            }
        }

        if self.bump() != Token::BraceClose {
            return Err("Expected '}' at end of model block".to_string());
        }

        Ok(ModelConfig {
            name,
            d_model,
            d_state,
            num_layers,
            precision,
        })
    }

    fn parse_train_block(&mut self) -> Result<TrainConfig, String> {
        self.bump(); // consume 'training'
        if self.bump() != Token::BraceOpen {
            return Err("Expected '{' after training keyword".to_string());
        }

        let mut lr = 0.001;
        let mut batch_size = 32;
        let mut epochs = 10;
        let mut checkpoint_interval = 500;

        while self.curr != Token::BraceClose && self.curr != Token::Eof {
            let key = match self.bump() {
                Token::Ident(k) => k,
                other => return Err(format!("Expected field key in training, got {:?}", other)),
            };

            if self.bump() != Token::Colon {
                return Err("Expected ':' after training field key".to_string());
            }

            match key.as_str() {
                "lr" => {
                    match self.bump() {
                        Token::NumberFloat(v) => lr = v,
                        Token::NumberInt(v) => lr = v as f32,
                        other => return Err(format!("Invalid lr value: {:?}", other)),
                    }
                }
                "batch_size" => {
                    if let Token::NumberInt(val) = self.bump() { batch_size = val; }
                }
                "epochs" => {
                    if let Token::NumberInt(val) = self.bump() { epochs = val; }
                }
                "checkpoint_interval" => {
                    if let Token::NumberInt(val) = self.bump() { checkpoint_interval = val; }
                }
                _ => return Err(format!("Unknown training parameter: {}", key)),
            }

            if self.curr == Token::Comma {
                self.bump();
            }
        }

        if self.bump() != Token::BraceClose {
            return Err("Expected '}' at end of training block".to_string());
        }

        Ok(TrainConfig {
            lr,
            batch_size,
            epochs,
            checkpoint_interval,
        })
    }
}
