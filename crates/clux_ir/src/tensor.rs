//! Tensor Descriptors & Byte Memory Alignment

pub type TensorId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    FP32,
    FP16,
    BF16,
    INT8,
}

impl DataType {
    #[inline(always)]
    pub fn byte_size(&self) -> usize {
        match self {
            DataType::FP32 => 4,
            DataType::FP16 | DataType::BF16 => 2,
            DataType::INT8 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub byte_offset: usize, // Offset in static memory arena
    pub total_bytes: usize,
}

impl TensorDesc {
    pub fn new(id: TensorId, shape: Vec<usize>, dtype: DataType, byte_offset: usize) -> Self {
        let elem_count: usize = shape.iter().product();
        let raw_bytes = elem_count * dtype.byte_size();
        // 64-Byte Cache Line Alignment
        let total_bytes = (raw_bytes + 63) & !63;

        Self {
            id,
            shape,
            dtype,
            byte_offset,
            total_bytes,
        }
    }
}
