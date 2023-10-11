pub use bytemuck::Pod;
use half::f16;
use pulp::{cast, NullaryFnOnce};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

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
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

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

    pulp::simd_type! {
        pub struct V3 {
            pub sse: "sse",
            pub sse2: "sse2",
            pub fxsr: "fxsr",
            pub sse3: "sse3",
            pub ssse3: "ssse3",
            pub sse4_1: "sse4.1",
            pub sse4_2: "sse4.2",
            pub popcnt: "popcnt",
            pub avx: "avx",
            pub avx2: "avx2",
            pub bmi1: "bmi1",
            pub bmi2: "bmi2",
            pub fma: "fma",
            pub lzcnt: "lzcnt",
            pub f16c: "f16c",
        }
    }

    unsafe impl MixedSimd<f16, f16, f16, f32> for V3 {
        const SIMD_WIDTH: usize = 8;

        type LhsN = [f16; 8];
        type RhsN = [f16; 8];
        type DstN = [f16; 8];
        type AccN = [f32; 8];

        #[inline]
        fn try_new() -> Option<Self> {
            Self::try_new()
        }

        #[inline(always)]
        fn mult(self, lhs: f32, rhs: f32) -> f32 {
            lhs * rhs
        }

        #[inline(always)]
        fn mult_add(self, lhs: f32, rhs: f32, acc: f32) -> f32 {
            f32::mul_add(lhs, rhs, acc)
        }

        #[inline(always)]
        fn from_lhs(self, lhs: f16) -> f32 {
            unsafe { pulp::cast_lossy(_mm_cvtph_ps(self.sse2._mm_set1_epi16(cast(lhs)))) }
        }

        #[inline(always)]
        fn from_rhs(self, rhs: f16) -> f32 {
            unsafe { pulp::cast_lossy(_mm_cvtph_ps(self.sse2._mm_set1_epi16(cast(rhs)))) }
        }

        #[inline(always)]
        fn from_dst(self, dst: f16) -> f32 {
            unsafe { pulp::cast_lossy(_mm_cvtph_ps(self.sse2._mm_set1_epi16(cast(dst)))) }
        }

        #[inline(always)]
        fn into_dst(self, acc: f32) -> f16 {
            unsafe { pulp::cast_lossy(self.sse._mm_load_ss(&acc)) }
        }

        #[inline(always)]
        fn simd_mult_add(self, lhs: Self::AccN, rhs: Self::AccN, acc: Self::AccN) -> Self::AccN {
            cast(self.fma._mm256_fmadd_ps(cast(lhs), cast(rhs), cast(acc)))
        }

        #[inline(always)]
        fn simd_from_lhs(self, lhs: Self::LhsN) -> Self::AccN {
            unsafe { cast(_mm256_cvtph_ps(cast(lhs))) }
        }

        #[inline(always)]
        fn simd_from_rhs(self, rhs: Self::RhsN) -> Self::AccN {
            unsafe { cast(_mm256_cvtph_ps(cast(rhs))) }
        }

        #[inline(always)]
        fn simd_splat(self, lhs: f32) -> Self::AccN {
            cast(self.avx._mm256_set1_ps(lhs))
        }

        #[inline(always)]
        fn simd_from_dst(self, dst: Self::DstN) -> Self::AccN {
            unsafe { cast(_mm256_cvtph_ps(cast(dst))) }
        }

        #[inline(always)]
        fn simd_into_dst(self, acc: Self::AccN) -> Self::DstN {
            unsafe { cast(_mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(cast(acc))) }
        }

        #[inline(always)]
        fn vectorize<F: NullaryFnOnce>(self, f: F) -> F::Output {
            self.vectorize(f)
        }
    }

    unsafe impl MixedSimd<f32, f32, f32, f32> for V3 {
        const SIMD_WIDTH: usize = 8;

        type LhsN = [f32; 8];
        type RhsN = [f32; 8];
        type DstN = [f32; 8];
        type AccN = [f32; 8];

        #[inline]
        fn try_new() -> Option<Self> {
            Self::try_new()
        }

        #[inline(always)]
        fn mult(self, lhs: f32, rhs: f32) -> f32 {
            lhs * rhs
        }

        #[inline(always)]
        fn mult_add(self, lhs: f32, rhs: f32, acc: f32) -> f32 {
            f32::mul_add(lhs, rhs, acc)
        }

        #[inline(always)]
        fn from_lhs(self, lhs: f32) -> f32 {
            lhs
        }

        #[inline(always)]
        fn from_rhs(self, rhs: f32) -> f32 {
            rhs
        }

        #[inline(always)]
        fn from_dst(self, dst: f32) -> f32 {
            dst
        }

        #[inline(always)]
        fn into_dst(self, acc: f32) -> f32 {
            acc
        }

        #[inline(always)]
        fn simd_mult_add(self, lhs: Self::AccN, rhs: Self::AccN, acc: Self::AccN) -> Self::AccN {
            cast(self.fma._mm256_fmadd_ps(cast(lhs), cast(rhs), cast(acc)))
        }

        #[inline(always)]
        fn simd_from_lhs(self, lhs: Self::LhsN) -> Self::AccN {
            lhs
        }

        #[inline(always)]
        fn simd_from_rhs(self, rhs: Self::RhsN) -> Self::AccN {
            rhs
        }

        #[inline(always)]
        fn simd_splat(self, lhs: f32) -> Self::AccN {
            cast(self.avx._mm256_set1_ps(lhs))
        }

        #[inline(always)]
        fn simd_from_dst(self, dst: Self::DstN) -> Self::AccN {
            dst
        }

        #[inline(always)]
        fn simd_into_dst(self, acc: Self::AccN) -> Self::DstN {
            acc
        }

        #[inline(always)]
        fn vectorize<F: NullaryFnOnce>(self, f: F) -> F::Output {
            self.vectorize(f)
        }
    }

    unsafe impl MixedSimd<f64, f64, f64, f64> for V3 {
        const SIMD_WIDTH: usize = 4;

        type LhsN = [f64; 4];
        type RhsN = [f64; 4];
        type DstN = [f64; 4];
        type AccN = [f64; 4];

        #[inline]
        fn try_new() -> Option<Self> {
            Self::try_new()
        }

        #[inline(always)]
        fn mult(self, lhs: f64, rhs: f64) -> f64 {
            lhs * rhs
        }

        #[inline(always)]
        fn mult_add(self, lhs: f64, rhs: f64, acc: f64) -> f64 {
            f64::mul_add(lhs, rhs, acc)
        }

        #[inline(always)]
        fn from_lhs(self, lhs: f64) -> f64 {
            lhs
        }

        #[inline(always)]
        fn from_rhs(self, rhs: f64) -> f64 {
            rhs
        }

        #[inline(always)]
        fn from_dst(self, dst: f64) -> f64 {
            dst
        }

        #[inline(always)]
        fn into_dst(self, acc: f64) -> f64 {
            acc
        }

        #[inline(always)]
        fn simd_mult_add(self, lhs: Self::AccN, rhs: Self::AccN, acc: Self::AccN) -> Self::AccN {
            cast(self.fma._mm256_fmadd_pd(cast(lhs), cast(rhs), cast(acc)))
        }

        #[inline(always)]
        fn simd_from_lhs(self, lhs: Self::LhsN) -> Self::AccN {
            lhs
        }

        #[inline(always)]
        fn simd_from_rhs(self, rhs: Self::RhsN) -> Self::AccN {
            rhs
        }

        #[inline(always)]
        fn simd_splat(self, lhs: f64) -> Self::AccN {
            cast(self.avx._mm256_set1_pd(lhs))
        }

        #[inline(always)]
        fn simd_from_dst(self, dst: Self::DstN) -> Self::AccN {
            dst
        }

        #[inline(always)]
        fn simd_into_dst(self, acc: Self::AccN) -> Self::DstN {
            acc
        }

        #[inline(always)]
        fn vectorize<F: NullaryFnOnce>(self, f: F) -> F::Output {
            self.vectorize(f)
        }
    }
}

pub trait Boilerplate: Copy + Send + Sync + core::fmt::Debug + 'static {}
impl<T: Copy + Send + Sync + core::fmt::Debug + 'static> Boilerplate for T {}

pub unsafe trait MixedSimd<Lhs, Rhs, Dst, Acc>: Boilerplate {
    const SIMD_WIDTH: usize;

    type LhsN: Boilerplate;
    type RhsN: Boilerplate;
    type DstN: Boilerplate;
    type AccN: Boilerplate;

    fn try_new() -> Option<Self>;

    fn vectorize<F: NullaryFnOnce>(self, f: F) -> F::Output;

    fn mult(self, lhs: Acc, rhs: Acc) -> Acc;
    fn mult_add(self, lhs: Acc, rhs: Acc, acc: Acc) -> Acc;
    fn from_lhs(self, lhs: Lhs) -> Acc;
    fn from_rhs(self, rhs: Rhs) -> Acc;
    fn from_dst(self, dst: Dst) -> Acc;
    fn into_dst(self, acc: Acc) -> Dst;

    fn simd_mult_add(self, lhs: Self::AccN, rhs: Self::AccN, acc: Self::AccN) -> Self::AccN;
    fn simd_from_lhs(self, lhs: Self::LhsN) -> Self::AccN;
    fn simd_from_rhs(self, rhs: Self::RhsN) -> Self::AccN;
    fn simd_splat(self, lhs: Acc) -> Self::AccN;

    fn simd_from_dst(self, dst: Self::DstN) -> Self::AccN;
    fn simd_into_dst(self, acc: Self::AccN) -> Self::DstN;
}
