//! CLUX Intermediate Representation (IR) & Static Arena Planner

pub mod tensor;
pub mod op;
pub mod graph;

pub use tensor::{DataType, TensorDesc, TensorId};
pub use op::IrOp;
pub use graph::IrGraph;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_ir_graph_construction() {
        let mut graph = IrGraph::new();

        // 1. Allocate Tensors: d_model = 256, d_state = 32
        let x = graph.allocate_tensor(vec![1, 256], DataType::FP16);
        let delta = graph.allocate_tensor(vec![1, 256], DataType::FP16);
        let a_diag = graph.allocate_tensor(vec![256, 32], DataType::FP16);
        let b = graph.allocate_tensor(vec![1, 32], DataType::FP16);
        let c = graph.allocate_tensor(vec![1, 32], DataType::FP16);

        let a_bar = graph.allocate_tensor(vec![256, 32], DataType::FP16);
        let b_bar = graph.allocate_tensor(vec![256, 32], DataType::FP16);
        let h_prev = graph.allocate_tensor(vec![256, 32], DataType::FP16);
        let h_next = graph.allocate_tensor(vec![256, 32], DataType::FP16);
        let out_y = graph.allocate_tensor(vec![1, 256], DataType::FP16);

        // 2. Push SSM Discretization Op
        graph.push_op(IrOp::SsmDiscretize {
            delta,
            a_diag,
            b,
            out_a_bar: a_bar,
            out_b_bar: b_bar,
        });

        // 3. Push SSM Scan Step Op
        graph.push_op(IrOp::SsmScanStep {
            a_bar,
            b_bar,
            x,
            h_prev,
            h_next,
            c,
            out_y,
        });

        assert_eq!(graph.op_count(), 2);
        assert_eq!(graph.tensors.len(), 10);
        // Ensure total arena size is 64-byte aligned and non-zero
        assert!(graph.total_arena_size > 0);
        assert_eq!(graph.total_arena_size % 64, 0);
    }
}
