//! CLUX Architecture Configuration DSL Parser

pub mod ast;
pub mod token;
pub mod lexer;
pub mod parser;

pub use ast::{CluxSpec, ModelConfig, Precision, TrainConfig};
pub use parser::Parser;

pub fn parse(source: &str) -> Result<CluxSpec, String> {
    let mut parser = Parser::new(source);
    parser.parse_spec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsl_parsing() {
        let dsl = r#"
        model CLUX_Tamil_SSM {
            dim: 256,
            state_dim: 32,
            layers: 4,
            precision: "int8",
        }

        training {
            lr: 0.0003,
            batch_size: 64,
            epochs: 5,
            checkpoint_interval: 250,
        }
        "#;

        let spec = parse(dsl).expect("Parsing must succeed");
        assert_eq!(spec.model.name, "CLUX_Tamil_SSM");
        assert_eq!(spec.model.d_model, 256);
        assert_eq!(spec.model.precision, Precision::INT8);
        assert!(spec.train.is_some());
        assert_eq!(spec.train.unwrap().epochs, 5);
    }
}
