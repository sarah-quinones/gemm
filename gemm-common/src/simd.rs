pub trait Simd: Copy + Send + Sync {
    unsafe fn vectorize(f: impl FnOnce());
}

#[derive(Copy, Clone)]
pub struct Scalar;

impl Simd for Scalar {
    #[inline(always)]
    unsafe fn vectorize(f: impl FnOnce()) {
        f()
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[derive(Copy, Clone)]
    pub struct Sse;
    #[derive(Copy, Clone)]
    pub struct Avx;
    #[derive(Copy, Clone)]
    pub struct Fma;

    #[cfg(feature = "nightly")]
    #[derive(Copy, Clone)]
    pub struct Avx512f;

    impl Simd for Sse {
        #[inline]
        #[target_feature(enable = "sse,sse2")]
        unsafe fn vectorize(f: impl FnOnce()) {
            f()
        }
    }

    impl Simd for Avx {
        #[inline]
        #[target_feature(enable = "avx")]
        unsafe fn vectorize(f: impl FnOnce()) {
            f()
        }
    }

    impl Simd for Fma {
        #[inline]
        #[target_feature(enable = "fma")]
        unsafe fn vectorize(f: impl FnOnce()) {
            f()
        }
    }

    #[cfg(feature = "nightly")]
    impl Simd for Avx512f {
        #[inline]
        #[target_feature(enable = "avx512f")]
        unsafe fn vectorize(f: impl FnOnce()) {
            f()
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

#[cfg(target_arch = "wasm32")]
mod wasm32 {
    use super::*;

    #[derive(Copy, Clone)]
    pub struct Simd128;

    impl Simd for Simd128 {
        #[inline]
        #[target_feature(enable = "simd128")]
        unsafe fn vectorize(f: impl FnOnce()) {
            f()
        }
    }
}
#[cfg(target_arch = "wasm32")]
pub use wasm32::*;
