//! IR Compute Graph & Static Memory Planning

use crate::tensor::{DataType, TensorDesc, TensorId};
use crate::op::IrOp;

pub struct IrGraph {
    pub tensors: Vec<TensorDesc>,
    pub ops: Vec<IrOp>,
    pub total_arena_size: usize,
}

impl IrGraph {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            ops: Vec::new(),
            total_arena_size: 0,
        }
    }

    /// Allocates a new tensor with automatic 64-byte alignment inside the static arena
    pub fn allocate_tensor(&mut self, shape: Vec<usize>, dtype: DataType) -> TensorId {
        let id = self.tensors.len() as TensorId;
        let offset = self.total_arena_size;
        let desc = TensorDesc::new(id, shape, dtype, offset);
        
        self.total_arena_size += desc.total_bytes;
        self.tensors.push(desc);
        id
    }

    pub fn push_op(&mut self, op: IrOp) {
        self.ops.push(op);
    }

    #[inline(always)]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }
}
