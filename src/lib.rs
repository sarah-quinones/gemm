#![cfg_attr(feature = "nightly", feature(stdsimd), feature(avx512_target_feature))]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(rust_2018_idioms)]
#![allow(unused_macros)]

#[cfg(not(feature = "std"))]
macro_rules! feature_detected {
    ($tt: tt) => {
        cfg!(feature = $tt)
    };
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
macro_rules! feature_detected {
    ($tt: tt) => {
        ::std::arch::is_x86_feature_detected!($tt)
    };
}
#[cfg(all(feature = "std", target_arch = "aarch64"))]
macro_rules! feature_detected {
    ($tt: tt) => {
        ::std::arch::is_aarch64_feature_detected!($tt)
    };
}

mod cache;
mod gemm;
mod gemv;
mod gevv;
mod microkernel;
mod pack_operands;
mod simd;

pub use crate::gemm::{c32, c64, gemm};

pub enum Parallelism {
    None,
    #[cfg(feature = "rayon")]
    Rayon(usize),
}

pub(crate) struct Ptr<T>(*mut T);

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Ptr<T> {}

unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl<T> Ptr<T> {
    #[inline(always)]
    pub fn wrapping_offset(self, offset: isize) -> Self {
        Ptr::<T>(self.0.wrapping_offset(offset))
    }
    #[inline(always)]
    pub fn wrapping_add(self, offset: usize) -> Self {
        Ptr::<T>(self.0.wrapping_add(offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_gemm() {
        let mut mnks = vec![];
        mnks.push((64, 64, 4));
        mnks.push((256, 256, 256));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            dbg!(m, n, k);
            for alpha in [0.0, 1.0, 2.3] {
                for beta in [0.0, 1.0, 2.3] {
                    dbg!(alpha, beta);
                    let a_vec: Vec<f64> = (0..(m * k)).map(|_| rand::random()).collect();
                    let b_vec: Vec<f64> = (0..(k * n)).map(|_| rand::random()).collect();
                    let mut c_vec: Vec<f64> = (0..(m * n)).map(|_| rand::random()).collect();
                    let mut d_vec = c_vec.clone();

                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            c_vec.as_mut_ptr(),
                            m as isize,
                            1,
                            true,
                            a_vec.as_ptr(),
                            m as isize,
                            1,
                            b_vec.as_ptr(),
                            k as isize,
                            1,
                            alpha,
                            beta,
                            false,
                            false,
                            false,
                            #[cfg(feature = "rayon")]
                            Parallelism::Rayon(0),
                            #[cfg(not(feature = "rayon"))]
                            Parallelism::None,
                        );

                        gemm::gemm_fallback(
                            m,
                            n,
                            k,
                            d_vec.as_mut_ptr(),
                            m as isize,
                            1,
                            true,
                            a_vec.as_ptr(),
                            m as isize,
                            1,
                            b_vec.as_ptr(),
                            k as isize,
                            1,
                            alpha,
                            beta,
                        );
                    }
                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                        assert_approx_eq::assert_approx_eq!(c, d);
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx() {
        let mut mnks = vec![];
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            dbg!(m, n, k);

            let zero = c64::new(0.0, 0.0);
            let one = c64::new(1.0, 0.0);
            let arbitrary = c64::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                dbg!(conj_dst);
                                dbg!(conj_lhs);
                                dbg!(conj_rhs);
                                let a_vec: Vec<f64> =
                                    (0..(2 * m * k)).map(|_| rand::random()).collect();
                                let b_vec: Vec<f64> =
                                    (0..(2 * k * n)).map(|_| rand::random()).collect();
                                let mut c_vec: Vec<f64> =
                                    (0..(2 * m * n)).map(|_| rand::random()).collect();
                                let mut d_vec = c_vec.clone();

                                unsafe {
                                    gemm::gemm(
                                        m,
                                        n,
                                        k,
                                        c_vec.as_mut_ptr() as *mut c64,
                                        m as isize,
                                        1,
                                        true,
                                        a_vec.as_ptr() as *const c64,
                                        m as isize,
                                        1,
                                        b_vec.as_ptr() as *const c64,
                                        k as isize,
                                        1,
                                        alpha,
                                        beta,
                                        conj_dst,
                                        conj_lhs,
                                        conj_rhs,
                                        #[cfg(feature = "rayon")]
                                        Parallelism::Rayon(0),
                                        #[cfg(not(feature = "rayon"))]
                                        Parallelism::None,
                                    );

                                    gemm::gemm_cplx_fallback(
                                        m,
                                        n,
                                        k,
                                        d_vec.as_mut_ptr() as *mut c64,
                                        m as isize,
                                        1,
                                        true,
                                        a_vec.as_ptr() as *const c64,
                                        m as isize,
                                        1,
                                        b_vec.as_ptr() as *const c64,
                                        k as isize,
                                        1,
                                        alpha,
                                        beta,
                                        conj_dst,
                                        conj_lhs,
                                        conj_rhs,
                                    );
                                }
                                for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                    assert_approx_eq::assert_approx_eq!(c, d);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
