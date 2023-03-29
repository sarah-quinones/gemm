pub mod f64 {
    type T = f64;
    gemm_common::gemm_cplx_def!(f64, 1);
}
