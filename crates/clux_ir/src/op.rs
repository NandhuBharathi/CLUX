//! IR Operational Nodes for SSM & Tensor Math

use crate::tensor::TensorId;

#[derive(Debug, Clone, PartialEq)]
pub enum IrOp {
    /// Fused SSM Discretization: A_bar = exp(Delta * A_diag), B_bar = Delta * B
    SsmDiscretize {
        delta: TensorId,
        a_diag: TensorId,
        b: TensorId,
        out_a_bar: TensorId,
        out_b_bar: TensorId,
    },
    /// SSM State Update: h_t = A_bar * h_{t-1} + B_bar * x_t; y_t = C * h_t
    SsmScanStep {
        a_bar: TensorId,
        b_bar: TensorId,
        x: TensorId,
        h_prev: TensorId,
        h_next: TensorId,
        c: TensorId,
        out_y: TensorId,
    },
    /// Matrix Multiplication: Y = X * W
    MatMul {
        input: TensorId,
        weight: TensorId,
        out: TensorId,
    },
    /// RMS Normalization (Pre-layer norm)
    RmsNorm {
        input: TensorId,
        weight: TensorId,
        out: TensorId,
        eps: f32,
    },
    /// Elementwise Addition: out = a + b
    ElemAdd {
        a: TensorId,
        b: TensorId,
        out: TensorId,
    },
}
