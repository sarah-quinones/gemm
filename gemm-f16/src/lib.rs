#![cfg_attr(feature = "nightly", feature(stdsimd), feature(avx512_target_feature))]

pub mod gemm;
pub use half::f16;
