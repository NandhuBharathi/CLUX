//! Abstract Syntax Tree (AST) for CLUX Architecture DSL

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::FP16
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub name: String,
    pub d_model: usize,
    pub d_state: usize,
    pub num_layers: usize,
    pub precision: Precision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub lr: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub checkpoint_interval: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CluxSpec {
    pub model: ModelConfig,
    pub train: Option<TrainConfig>,
}
