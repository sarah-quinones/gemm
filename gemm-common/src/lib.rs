#![cfg_attr(feature = "nightly", feature(stdsimd), feature(avx512_target_feature))]

pub mod cache;

pub mod gemm;
pub mod gemv;
pub mod gevv;

pub mod microkernel;
pub mod pack_operands;
pub mod simd;

#[derive(Copy, Clone, Debug)]
pub enum Parallelism {
    None,
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

#[cfg(not(feature = "std"))]
#[macro_export]
macro_rules! feature_detected {
    ($tt: tt) => {
        cfg!(feature = $tt)
    };
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[macro_export]
macro_rules! feature_detected {
    ($tt: tt) => {
        ::std::arch::is_x86_feature_detected!($tt)
    };
}
#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[macro_export]
macro_rules! feature_detected {
    ($tt: tt) => {
        ::std::arch::is_aarch64_feature_detected!($tt)
    };
}
