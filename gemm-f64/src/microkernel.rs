pub mod scalar {
    pub mod f64 {
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..1 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..1 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

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

        microkernel!(, x1x1, 1, 1);
        microkernel!(, x1x2, 1, 2);
        microkernel!(, x1x3, 1, 3);
        microkernel!(, x1x4, 1, 4);

        microkernel!(, x2x1, 2, 1);
        microkernel!(, x2x2, 2, 2);
        microkernel!(, x2x3, 2, 3);
        microkernel!(, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse {
    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 2;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..2 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..2 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["sse"], x1x1, 1, 1);
        microkernel!(["sse"], x1x2, 1, 2);
        microkernel!(["sse"], x1x3, 1, 3);
        microkernel!(["sse"], x1x4, 1, 4);

        microkernel!(["sse"], x2x1, 2, 1);
        microkernel!(["sse"], x2x2, 2, 2);
        microkernel!(["sse"], x2x3, 2, 3);
        microkernel!(["sse"], x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx {
    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..4 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..4 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["avx"], x1x1, 1, 1);
        microkernel!(["avx"], x1x2, 1, 2);
        microkernel!(["avx"], x1x3, 1, 3);
        microkernel!(["avx"], x1x4, 1, 4);

        microkernel!(["avx"], x2x1, 2, 1);
        microkernel!(["avx"], x2x2, 2, 2);
        microkernel!(["avx"], x2x3, 2, 3);
        microkernel!(["avx"], x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod fma {
    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..4 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..4 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm256_fmadd_pd(transmute(a), transmute(b), transmute(c)))
        }

        microkernel!(["fma"], x1x1, 1, 1);
        microkernel!(["fma"], x1x2, 1, 2);
        microkernel!(["fma"], x1x3, 1, 3);
        microkernel!(["fma"], x1x4, 1, 4);

        microkernel!(["fma"], x2x1, 2, 1);
        microkernel!(["fma"], x2x2, 2, 2);
        microkernel!(["fma"], x2x3, 2, 3);
        microkernel!(["fma"], x2x4, 2, 4);

        microkernel!(["fma"], x3x1, 3, 1);
        microkernel!(["fma"], x3x2, 3, 2);
        microkernel!(["fma"], x3x3, 3, 3);
        microkernel!(["fma"], x3x4, 3, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
            [x3x1, x3x2, x3x3, x3x4,],
        }
    }
}

#[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512f {
    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f64;
        const N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            transmute(_mm512_i64gather_pd::<8>(
                _mm512_setr_epi64(
                    0 * stride as i64,
                    1 * stride as i64,
                    2 * stride as i64,
                    3 * stride as i64,
                    4 * stride as i64,
                    5 * stride as i64,
                    6 * stride as i64,
                    7 * stride as i64,
                ),
                base as _,
            ))
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            _mm512_i64scatter_pd::<8>(
                base as _,
                _mm512_setr_epi64(
                    0 * stride as i64,
                    1 * stride as i64,
                    2 * stride as i64,
                    3 * stride as i64,
                    4 * stride as i64,
                    5 * stride as i64,
                    6 * stride as i64,
                    7 * stride as i64,
                ),
                transmute(p),
            );
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm512_fmadd_pd(transmute(a), transmute(b), transmute(c)))
        }

        microkernel!(["avx512f"], x1x1, 1, 1);
        microkernel!(["avx512f"], x1x2, 1, 2);
        microkernel!(["avx512f"], x1x3, 1, 3);
        microkernel!(["avx512f"], x1x4, 1, 4);
        microkernel!(["avx512f"], x1x5, 1, 5);
        microkernel!(["avx512f"], x1x6, 1, 6);
        microkernel!(["avx512f"], x1x7, 1, 7);
        microkernel!(["avx512f"], x1x8, 1, 8);

        microkernel!(["avx512f"], x2x1, 2, 1);
        microkernel!(["avx512f"], x2x2, 2, 2);
        microkernel!(["avx512f"], x2x3, 2, 3);
        microkernel!(["avx512f"], x2x4, 2, 4);
        microkernel!(["avx512f"], x2x5, 2, 5);
        microkernel!(["avx512f"], x2x6, 2, 6);
        microkernel!(["avx512f"], x2x7, 2, 7);
        microkernel!(["avx512f"], x2x8, 2, 8);

        microkernel!(["avx512f"], x3x1, 3, 1);
        microkernel!(["avx512f"], x3x2, 3, 2);
        microkernel!(["avx512f"], x3x3, 3, 3);
        microkernel!(["avx512f"], x3x4, 3, 4);
        microkernel!(["avx512f"], x3x5, 3, 5);
        microkernel!(["avx512f"], x3x6, 3, 6);
        microkernel!(["avx512f"], x3x7, 3, 7);
        microkernel!(["avx512f"], x3x8, 3, 8);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }
}

#[allow(dead_code)]
mod v128_common {
    pub mod f64 {
        use core::mem::MaybeUninit;

        pub type T = f64;
        pub const N: usize = 2;
        pub type Pack = [T; N];

        #[inline(always)]
        pub unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..2 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        pub unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..2 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value]
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    pub mod f64 {
        use super::super::v128_common::f64::*;
        use core::arch::aarch64::*;
        use core::mem::transmute;

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f64(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f64(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_f64(transmute(c), transmute(a), transmute(b)))
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f64::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        microkernel!(["neon"], x1x1, 1, 1);
        microkernel!(["neon"], x1x2, 1, 2, 1, 2);
        microkernel!(["neon"], x1x3, 1, 3);
        microkernel!(["neon"], x1x4, 1, 4, 2, 2);
        microkernel!(["neon"], x1x5, 1, 5);
        microkernel!(["neon"], x1x6, 1, 6, 3, 2);
        microkernel!(["neon"], x1x7, 1, 7);
        microkernel!(["neon"], x1x8, 1, 8, 4, 2);

        microkernel!(["neon"], x2x1, 2, 1);
        microkernel!(["neon"], x2x2, 2, 2, 1, 2);
        microkernel!(["neon"], x2x3, 2, 3);
        microkernel!(["neon"], x2x4, 2, 4, 2, 2);
        microkernel!(["neon"], x2x5, 2, 5);
        microkernel!(["neon"], x2x6, 2, 6, 3, 2);
        microkernel!(["neon"], x2x7, 2, 7);
        microkernel!(["neon"], x2x8, 2, 8, 4, 2);

        microkernel!(["neon"], x3x1, 3, 1);
        microkernel!(["neon"], x3x2, 3, 2, 1, 2);
        microkernel!(["neon"], x3x3, 3, 3);
        microkernel!(["neon"], x3x4, 3, 4, 2, 2);
        microkernel!(["neon"], x3x5, 3, 5);
        microkernel!(["neon"], x3x6, 3, 6, 3, 2);
        microkernel!(["neon"], x3x7, 3, 7);
        microkernel!(["neon"], x3x8, 3, 8, 4, 2);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }
}
