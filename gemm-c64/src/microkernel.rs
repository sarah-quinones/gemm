pub mod scalar {
    pub mod f64 {
        type T = f64;
        const N: usize = 2;
        const CPLX_N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value, value]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0], lhs[1] + rhs[1]]
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            [a[0], -a[1]]
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            [a[1], a[0]]
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, _a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            [
                a_re_im[0] * b_re[0] - a_re_im[1] * b_im[0],
                a_re_im[1] * b_re[0] + a_re_im[0] * b_im[0],
            ]
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                add(
                    c_re_im,
                    mul_cplx(a_re_im, a_im_re, b_re, [-b_im[0], -b_im[1]]),
                )
            } else {
                add(c_re_im, mul_cplx(a_re_im, a_im_re, b_re, b_im))
            }
        }

        microkernel_cplx!(, x1x1, 1, 1);
        microkernel_cplx!(, x1x2, 1, 2);
        microkernel_cplx!(, x1x3, 1, 3);
        microkernel_cplx!(, x1x4, 1, 4);

        microkernel_cplx!(, x2x1, 2, 1);
        microkernel_cplx!(, x2x2, 2, 2);
        microkernel_cplx!(, x2x3, 2, 3);
        microkernel_cplx!(, x2x4, 2, 4);

        microkernel_cplx_fn_array! {
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

        type T = f64;
        const N: usize = 4;
        const CPLX_N: usize = 2;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m256d = unsafe { transmute([0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64]) };
            transmute(_mm256_xor_pd(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm256_permute_pd::<0b0101>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm256_fmaddsub_pd(
                transmute(a_re_im),
                transmute(b_re),
                _mm256_mul_pd(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(_mm256_fmsubadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm256_fmsubadd_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm256_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm256_fmaddsub_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel_cplx!(["fma"], cplx_x1x1, 1, 1);
        microkernel_cplx!(["fma"], cplx_x1x2, 1, 2);
        microkernel_cplx!(["fma"], cplx_x1x3, 1, 3);

        microkernel_cplx!(["fma"], cplx_x2x1, 2, 1);
        microkernel_cplx!(["fma"], cplx_x2x2, 2, 2);
        microkernel_cplx!(["fma"], cplx_x2x3, 2, 3);

        microkernel_cplx!(["fma"], cplx_x3x1, 3, 1);
        microkernel_cplx!(["fma"], cplx_x3x2, 3, 2);
        microkernel_cplx!(["fma"], cplx_x3x3, 3, 3);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3,],
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
        const CPLX_N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m512i = unsafe {
                transmute([
                    0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64,
                ])
            };
            transmute(_mm512_xor_si512(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm512_permute_pd::<0b01010101>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm512_fmaddsub_pd(
                transmute(a_re_im),
                transmute(b_re),
                _mm512_mul_pd(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(_mm512_fmsubadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmsubadd_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm512_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmaddsub_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel_cplx!(["avx512f"], cplx_x1x1, 1, 1);
        microkernel_cplx!(["avx512f"], cplx_x1x2, 1, 2);
        microkernel_cplx!(["avx512f"], cplx_x1x3, 1, 3);
        microkernel_cplx!(["avx512f"], cplx_x1x4, 1, 4);

        microkernel_cplx!(["avx512f"], cplx_x2x1, 2, 1);
        microkernel_cplx!(["avx512f"], cplx_x2x2, 2, 2);
        microkernel_cplx!(["avx512f"], cplx_x2x3, 2, 3);
        microkernel_cplx!(["avx512f"], cplx_x2x4, 2, 4);

        microkernel_cplx!(["avx512f"], cplx_x3x1, 3, 1);
        microkernel_cplx!(["avx512f"], cplx_x3x2, 3, 2);
        microkernel_cplx!(["avx512f"], cplx_x3x3, 3, 3);
        microkernel_cplx!(["avx512f"], cplx_x3x4, 3, 4);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3, cplx_x1x4,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3, cplx_x2x4,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3, cplx_x3x4,],
        }
    }
}
