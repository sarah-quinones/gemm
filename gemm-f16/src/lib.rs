#![cfg_attr(feature = "nightly", feature(stdsimd), feature(avx512_target_feature))]

pub mod gemm;
pub mod microkernel;
pub use half::f16;

#[macro_use]
#[allow(unused_imports)]
extern crate gemm_common;
