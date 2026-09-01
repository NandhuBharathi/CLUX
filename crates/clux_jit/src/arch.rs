//! Dynamic CPU Feature Detection & SIMD Target Routing

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Avx512,
    Avx2,
    Neon,
    GenericSimd,
}

pub struct CpuFeatures;

impl CpuFeatures {
    pub fn detect() -> TargetArch {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return TargetArch::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return TargetArch::Avx2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return TargetArch::Neon;
        }

        TargetArch::GenericSimd
    }
}
