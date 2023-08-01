pub mod scalar {
    pub mod f32 {
        type T = f32;
        const N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value]
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] * rhs[0]]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0]]
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(, 2, x1x1, 1, 1);
        microkernel!(, 2, x1x2, 1, 2);
        microkernel!(, 2, x1x3, 1, 3);
        microkernel!(, 2, x1x4, 1, 4);

        microkernel!(, 2, x2x1, 2, 1);
        microkernel!(, 2, x2x2, 2, 2);
        microkernel!(, 2, x2x3, 2, 3);
        microkernel!(, 2, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f32;
        const N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["sse,sse2"], 2, x1x1, 1, 1);
        microkernel!(["sse,sse2"], 2, x1x2, 1, 2);
        microkernel!(["sse,sse2"], 2, x1x3, 1, 3);
        microkernel!(["sse,sse2"], 2, x1x4, 1, 4);

        microkernel!(["sse,sse2"], 2, x2x1, 2, 1);
        microkernel!(["sse,sse2"], 2, x2x2, 2, 2);
        microkernel!(["sse,sse2"], 2, x2x3, 2, 3);
        microkernel!(["sse,sse2"], 2, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f32;
        const N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["avx"], 2, x1x1, 1, 1);
        microkernel!(["avx"], 2, x1x2, 1, 2);
        microkernel!(["avx"], 2, x1x3, 1, 3);
        microkernel!(["avx"], 2, x1x4, 1, 4);

        microkernel!(["avx"], 2, x2x1, 2, 1);
        microkernel!(["avx"], 2, x2x2, 2, 2);
        microkernel!(["avx"], 2, x2x3, 2, 3);
        microkernel!(["avx"], 2, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod fma {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f32;
        const N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm256_fmadd_ps(transmute(a), transmute(b), transmute(c)))
        }

        microkernel!(["fma"], 2, x1x1, 1, 1);
        microkernel!(["fma"], 2, x1x2, 1, 2);
        microkernel!(["fma"], 2, x1x3, 1, 3);
        microkernel!(["fma"], 2, x1x4, 1, 4);

        microkernel!(["fma"], 2, x2x1, 2, 1);
        microkernel!(["fma"], 2, x2x2, 2, 2);
        microkernel!(["fma"], 2, x2x3, 2, 3);
        microkernel!(["fma"], 2, x2x4, 2, 4);

        microkernel!(["fma"], 2, x3x1, 3, 1);
        microkernel!(["fma"], 2, x3x2, 3, 2);
        microkernel!(["fma"], 2, x3x3, 3, 3);
        microkernel!(["fma"], 2, x3x4, 3, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
            [x3x1, x3x2, x3x3, x3x4,],
        }
    }
}

#[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512f {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f32;
        const N: usize = 16;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm512_fmadd_ps(transmute(a), transmute(b), transmute(c)))
        }

        microkernel!(["avx512f"], 4, x1x1, 1, 1);
        microkernel!(["avx512f"], 4, x1x2, 1, 2);
        microkernel!(["avx512f"], 4, x1x3, 1, 3);
        microkernel!(["avx512f"], 4, x1x4, 1, 4);
        microkernel!(["avx512f"], 4, x1x5, 1, 5);
        microkernel!(["avx512f"], 4, x1x6, 1, 6);
        microkernel!(["avx512f"], 4, x1x7, 1, 7);
        microkernel!(["avx512f"], 4, x1x8, 1, 8);

        microkernel!(["avx512f"], 4, x2x1, 2, 1);
        microkernel!(["avx512f"], 4, x2x2, 2, 2);
        microkernel!(["avx512f"], 4, x2x3, 2, 3);
        microkernel!(["avx512f"], 4, x2x4, 2, 4);
        microkernel!(["avx512f"], 4, x2x5, 2, 5);
        microkernel!(["avx512f"], 4, x2x6, 2, 6);
        microkernel!(["avx512f"], 4, x2x7, 2, 7);
        microkernel!(["avx512f"], 4, x2x8, 2, 8);

        microkernel!(["avx512f"], 4, x3x1, 3, 1);
        microkernel!(["avx512f"], 4, x3x2, 3, 2);
        microkernel!(["avx512f"], 4, x3x3, 3, 3);
        microkernel!(["avx512f"], 4, x3x4, 3, 4);
        microkernel!(["avx512f"], 4, x3x5, 3, 5);
        microkernel!(["avx512f"], 4, x3x6, 3, 6);
        microkernel!(["avx512f"], 4, x3x7, 3, 7);
        microkernel!(["avx512f"], 4, x3x8, 3, 8);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }
}

#[allow(dead_code)]
mod v128_common {
    pub mod f32 {
        pub type T = f32;
        pub const N: usize = 4;
        pub type Pack = [T; N];

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value, value, value]
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    pub mod f32 {
        use super::super::v128_common::f32::*;
        use core::arch::aarch64::*;
        use core::mem::transmute;

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f32(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f32(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_f32(transmute(c), transmute(a), transmute(b)))
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f32::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        microkernel!(["neon"], 2, x1x1, 1, 1);
        microkernel!(["neon"], 2, x1x2, 1, 2);
        microkernel!(["neon"], 2, x1x3, 1, 3);
        microkernel!(["neon"], 2, x1x4, 1, 4, 1, 4);
        microkernel!(["neon"], 2, x1x5, 1, 5);
        microkernel!(["neon"], 2, x1x6, 1, 6);
        microkernel!(["neon"], 2, x1x7, 1, 7);
        microkernel!(["neon"], 2, x1x8, 1, 8, 2, 4);

        microkernel!(["neon"], 2, x2x1, 2, 1);
        microkernel!(["neon"], 2, x2x2, 2, 2);
        microkernel!(["neon"], 2, x2x3, 2, 3);
        microkernel!(["neon"], 2, x2x4, 2, 4, 1, 4);
        microkernel!(["neon"], 2, x2x5, 2, 5);
        microkernel!(["neon"], 2, x2x6, 2, 6);
        microkernel!(["neon"], 2, x2x7, 2, 7);
        microkernel!(["neon"], 2, x2x8, 2, 8, 2, 4);

        microkernel!(["neon"], 2, x3x1, 3, 1);
        microkernel!(["neon"], 2, x3x2, 3, 2);
        microkernel!(["neon"], 2, x3x3, 3, 3);
        microkernel!(["neon"], 2, x3x4, 3, 4, 1, 4);
        microkernel!(["neon"], 2, x3x5, 3, 5);
        microkernel!(["neon"], 2, x3x6, 3, 6);
        microkernel!(["neon"], 2, x3x7, 3, 7);
        microkernel!(["neon"], 2, x3x8, 3, 8, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }

    pub mod f16 {
        use core::arch::{aarch64::uint16x8_t, asm};
        use core::mem::transmute;

        pub type T = half::f16;
        pub const N: usize = 8;
        pub type Pack = [T; N];

        #[allow(non_camel_case_types)]
        type float16x8_t = uint16x8_t;

        /// Floating point multiplication
        /// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FMUL--vector-)
        #[inline]
        pub unsafe fn vmulq_f16(a: float16x8_t, b: float16x8_t) -> float16x8_t {
            let result: float16x8_t;
            asm!(
                "fmul {0:v}.8h, {1:v}.8h, {2:v}.8h",
                out(vreg) result,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack, preserves_flags));
            result
        }

        /// Floating point addition
        /// [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FADD--vector-)
        #[inline]
        pub unsafe fn vaddq_f16(a: float16x8_t, b: float16x8_t) -> float16x8_t {
            let result: float16x8_t;
            asm!(
                "fadd {0:v}.8h, {1:v}.8h, {2:v}.8h",
                out(vreg) result,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack, preserves_flags));
            result
        }

        /// Fused multiply add [doc](https://developer.arm.com/documentation/dui0801/g/A64-SIMD-Vector-Instructions/FMLA--vector-)
        #[inline]
        pub unsafe fn vfmaq_f16(mut a: float16x8_t, b: float16x8_t, c: float16x8_t) -> float16x8_t {
            asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.8h",
                inout(vreg) a,
                in(vreg) b,
                in(vreg) c,
                options(nomem, nostack, preserves_flags));
            a
        }

        #[inline]
        pub unsafe fn vfmaq_laneq_f16<const LANE: i32>(
            mut a: float16x8_t,
            b: float16x8_t,
            c: float16x8_t,
        ) -> float16x8_t {
            match LANE {
                0 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[0]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                1 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[1]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                2 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[2]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                3 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[3]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                4 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[4]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                5 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[5]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                6 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[6]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                7 => asm!(
                "fmla {0:v}.8h, {1:v}.8h, {2:v}.h[7]",
                inout(vreg) a,
                in(vreg) b,
                in(vreg_low16) c,
                options(nomem, nostack, preserves_flags)),
                _ => unreachable!(),
            }
            a
        }

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value, value, value, value, value, value, value]
        }

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f16(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f16(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_f16(transmute(c), transmute(a), transmute(b)))
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f16::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        microkernel_f16!(["neon"], 2, x1x1, 1, 1);
        microkernel_f16!(["neon"], 2, x1x2, 1, 2);
        microkernel_f16!(["neon"], 2, x1x3, 1, 3);
        microkernel_f16!(["neon"], 2, x1x4, 1, 4, 1, 4);
        microkernel_f16!(["neon"], 2, x1x5, 1, 5);
        microkernel_f16!(["neon"], 2, x1x6, 1, 6);
        microkernel_f16!(["neon"], 2, x1x7, 1, 7);
        microkernel_f16!(["neon"], 2, x1x8, 1, 8, 2, 4);

        microkernel_f16!(["neon"], 2, x2x1, 2, 1);
        microkernel_f16!(["neon"], 2, x2x2, 2, 2);
        microkernel_f16!(["neon"], 2, x2x3, 2, 3);
        microkernel_f16!(["neon"], 2, x2x4, 2, 4, 1, 4);
        microkernel_f16!(["neon"], 2, x2x5, 2, 5);
        microkernel_f16!(["neon"], 2, x2x6, 2, 6);
        microkernel_f16!(["neon"], 2, x2x7, 2, 7);
        microkernel_f16!(["neon"], 2, x2x8, 2, 8, 2, 4);

        microkernel_f16!(["neon"], 2, x3x1, 3, 1);
        microkernel_f16!(["neon"], 2, x3x2, 3, 2);
        microkernel_f16!(["neon"], 2, x3x3, 3, 3);
        microkernel_f16!(["neon"], 2, x3x4, 3, 4, 1, 4);
        microkernel_f16!(["neon"], 2, x3x5, 3, 5);
        microkernel_f16!(["neon"], 2, x3x6, 3, 6);
        microkernel_f16!(["neon"], 2, x3x7, 3, 7);
        microkernel_f16!(["neon"], 2, x3x8, 3, 8, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }
}
