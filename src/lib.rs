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

#[inline(always)]
pub fn unroll<const N: usize>(mut f: impl FnMut(usize)) {
    if N > 0 {
        f(0)
    }
    if N > 1 {
        f(1)
    }
    if N > 2 {
        f(2)
    }
    if N > 3 {
        f(3)
    }
    if N > 4 {
        f(4)
    }
    if N > 5 {
        f(5)
    }
    if N > 6 {
        f(6)
    }
    if N > 7 {
        f(7)
    }
    if N > 8 {
        f(8)
    }
    if N > 9 {
        f(9)
    }
    if N > 10 {
        f(10)
    }
    if N > 11 {
        f(11)
    }
    if N > 12 {
        f(12)
    }
    if N > 13 {
        f(13)
    }
    if N > 14 {
        f(14)
    }
    if N > 15 {
        f(15)
    }
    if N > 16 {
        f(16)
    }
    if N > 17 {
        f(17)
    }
    if N > 18 {
        f(18)
    }
    if N > 19 {
        f(19)
    }
    if N > 20 {
        f(20)
    }
    if N > 21 {
        f(21)
    }
    if N > 22 {
        f(22)
    }
    if N > 23 {
        f(23)
    }
    if N > 24 {
        f(24)
    }
    if N > 25 {
        f(25)
    }
    if N > 26 {
        f(26)
    }
    if N > 27 {
        f(27)
    }
    if N > 28 {
        f(28)
    }
    if N > 29 {
        f(29)
    }
    if N > 30 {
        f(30)
    }
    if N > 31 {
        f(31)
    }
    if N > 32 {
        f(32)
    }
    if N > 33 {
        f(33)
    }
    if N > 34 {
        f(34)
    }
    if N > 35 {
        f(35)
    }
    if N > 36 {
        f(36)
    }
    if N > 37 {
        f(37)
    }
    if N > 38 {
        f(38)
    }
    if N > 39 {
        f(39)
    }
    if N > 40 {
        f(40)
    }
    if N > 41 {
        f(41)
    }
    if N > 42 {
        f(42)
    }
    if N > 43 {
        f(43)
    }
    if N > 44 {
        f(44)
    }
    if N > 45 {
        f(45)
    }
    if N > 46 {
        f(46)
    }
    if N > 47 {
        f(47)
    }
    if N > 48 {
        f(48)
    }
    if N > 49 {
        f(49)
    }
    if N > 50 {
        f(50)
    }
    if N > 51 {
        f(51)
    }
    if N > 52 {
        f(52)
    }
    if N > 53 {
        f(53)
    }
    if N > 54 {
        f(54)
    }
    if N > 55 {
        f(55)
    }
    if N > 56 {
        f(56)
    }
    if N > 57 {
        f(57)
    }
    if N > 58 {
        f(58)
    }
    if N > 59 {
        f(59)
    }
    if N > 60 {
        f(60)
    }
    if N > 61 {
        f(61)
    }
    if N > 62 {
        f(62)
    }
    if N > 63 {
        f(63)
    }
    if N > 64 {
        unreachable!()
    }
}

pub use crate::gemm::gemm;
pub use crate::gemm::gemm_req;

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
        use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};

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

        let n_threads = 1;

        for (m, n, k) in mnks {
            dbg!(m, n, k);
            for alpha in [0.0, 1.0, 2.3] {
                for beta in [0.0, 1.0, 2.3] {
                    dbg!(alpha, beta);
                    let a_vec: Vec<f64> = (0..(m * k)).map(|_| rand::random()).collect();
                    let b_vec: Vec<f64> = (0..(k * n)).map(|_| rand::random()).collect();
                    let mut c_vec: Vec<f64> = (0..(m * n)).map(|_| rand::random()).collect();
                    let mut d_vec = c_vec.clone();

                    let mut mem =
                        GlobalMemBuffer::new(gemm::gemm_req::<f64>(m, n, k, n_threads).unwrap());
                    let mut stack = DynStack::new(&mut mem);
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
                            n_threads,
                            stack.rb_mut(),
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
                            n_threads,
                            stack.rb_mut(),
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
