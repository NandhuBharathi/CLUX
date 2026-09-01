pub mod arch;
pub mod kernels;
pub use arch::{CpuFeatures, TargetArch};
pub use kernels::*;

pub struct JitEngine { pub target: TargetArch }
impl JitEngine { pub fn new() -> Self { Self { target: CpuFeatures::detect() } } }
