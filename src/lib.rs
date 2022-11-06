#![cfg_attr(feature = "nightly", feature(stdsimd), feature(avx512_target_feature))]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(rust_2018_idioms)]

mod cache;
mod gemm;
mod gemv;
mod gevv;
mod microkernel;
mod pack_operands;
mod simd;

pub use crate::gemm::gemm;

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

#[cfg(feature = "std")]
macro_rules! x86_feature_detected {
    ($tt: tt) => {
        is_x86_feature_detected!($tt)
    };
}

#[cfg(not(feature = "std"))]
macro_rules! x86_feature_detected {
    ($tt: tt) => {
        cfg!(feature = $tt)
    };
}

pub(crate) use x86_feature_detected;

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
}
