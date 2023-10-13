#![cfg_attr(feature = "nightly", feature(stdsimd), feature(avx512_target_feature))]
#![cfg_attr(not(feature = "std"), no_std)]

pub mod gemm;
mod microkernel;

#[macro_use]
extern crate gemm_common;
