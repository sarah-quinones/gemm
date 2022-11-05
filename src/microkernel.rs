use crate::Ptr;

pub(crate) type MicroKernelFn<T> =
    unsafe fn(usize, usize, usize, Ptr<T>, Ptr<T>, Ptr<T>, isize, isize, isize, isize, isize, T, T);

macro_rules! microkernel {
    ($([$target: tt])?, $name: ident) => {
        #[inline]
        $(#[target_feature(enable = $target)])?
        // 0, 1, or 2 for generic alpha
        pub(crate) unsafe fn $name<const ALPHA: u8, const MR_DIV_N: usize, const NR: usize>(
            m: usize,
            n: usize,
            k: usize,
            dst: crate::Ptr<T>,
            packed_lhs: crate::Ptr<T>,
            packed_rhs: crate::Ptr<T>,
            dst_cs: isize,
            dst_rs: isize,
            lhs_cs: isize,
            rhs_rs: isize,
            rhs_cs: isize,
            alpha: T,
            beta: T,
        ) {
            assert!(ALPHA <= 2);

            let dst = dst.0;
            let mut packed_lhs = packed_lhs.0;
            let mut packed_rhs = packed_rhs.0;

            let mut accum_storage = [[splat(0.0); MR_DIV_N]; NR];
            let accum = accum_storage.as_mut_ptr() as *mut Pack;

            let mut lhs = [::core::mem::MaybeUninit::<Pack>::uninit(); MR_DIV_N];
            let mut rhs = ::core::mem::MaybeUninit::<Pack>::uninit();

            #[derive(Copy, Clone)]
            struct KernelIter {
                packed_lhs: *const T,
                packed_rhs: *const T,
                lhs_cs: isize,
                rhs_rs: isize,
                rhs_cs: isize,
                accum: *mut Pack,
                lhs: *mut ::core::mem::MaybeUninit<Pack>,
                rhs: *mut ::core::mem::MaybeUninit<Pack>,
            }

            impl KernelIter {
                #[inline(always)]
                unsafe fn execute<const MR_DIV_N: usize, const NR: usize>(self, iter: usize) {
                    let packed_lhs = self.packed_lhs.offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.offset(iter as isize * self.rhs_rs);

                    crate::unroll::<MR_DIV_N>(|m_iter|{
                        (*self.lhs.add(m_iter))
                            .write((packed_lhs.add(m_iter * N) as *const Pack).read_unaligned());
                    });

                    crate::unroll::<NR>(|n_iter| {
                        (*self.rhs).write(splat(*packed_rhs.offset(n_iter as isize * self.rhs_cs)));
                        let accum = self.accum.add(n_iter * MR_DIV_N);
                        crate::unroll::<MR_DIV_N>(|m_iter| {
                            let accum = &mut *accum.add(m_iter);
                            *accum = mul_add(
                                (*self.lhs.add(m_iter)).assume_init(),
                                (*self.rhs).assume_init(),
                                *accum,
                            );
                        });
                    });
                }
            }

            let k_unroll = k / 4;
            let k_leftover = k % 4;

            let mut depth = k_unroll;

            if depth != 0 {
                loop {
                    let iter = KernelIter {
                        packed_lhs,
                        packed_rhs,
                        lhs_cs,
                        rhs_rs,
                        rhs_cs,
                        accum,
                        lhs: lhs.as_mut_ptr(),
                        rhs: &mut rhs,
                    };

                    iter.execute::<MR_DIV_N, NR>(0);
                    iter.execute::<MR_DIV_N, NR>(1);
                    iter.execute::<MR_DIV_N, NR>(2);
                    iter.execute::<MR_DIV_N, NR>(3);

                    packed_lhs = packed_lhs.wrapping_offset(4 * lhs_cs);
                    packed_rhs = packed_rhs.wrapping_offset(4 * rhs_rs);

                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }

            depth = k_leftover;
            if depth != 0 {
                loop {
                    KernelIter {
                        packed_lhs,
                        packed_rhs,
                        lhs_cs,
                        rhs_rs,
                        rhs_cs,
                        accum,
                        lhs: lhs.as_mut_ptr(),
                        rhs: &mut rhs,
                    }
                    .execute::<MR_DIV_N, NR>(0);

                    packed_lhs = packed_lhs.wrapping_offset(lhs_cs);
                    packed_rhs = packed_rhs.wrapping_offset(rhs_rs);

                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }

            if m == MR_DIV_N * N && n == NR {
                let alpha = splat(alpha);
                let beta = splat(beta);
                if dst_rs == 1 {

                    if ALPHA == 2 {
                        crate::unroll::<NR>(|n_iter| {
                            crate::unroll::<MR_DIV_N>(|m_iter| {
                                let dst = dst.offset((m_iter * N) as isize + n_iter as isize * dst_cs) as *mut Pack;
                                dst.write_unaligned(add(
                                    mul(alpha, dst.read_unaligned()),
                                    mul(beta, *accum.add(m_iter + MR_DIV_N * n_iter)),
                                ));
                            });
                        });
                    } else if ALPHA == 1 {
                        crate::unroll::<NR>(|n_iter| {
                            crate::unroll::<MR_DIV_N>(|m_iter| {
                                let dst = dst.offset((m_iter * N) as isize + n_iter as isize * dst_cs) as *mut Pack;
                                dst.write_unaligned(mul_add(
                                    beta,
                                    *accum.add(m_iter + MR_DIV_N * n_iter),
                                    dst.read_unaligned(),
                                ));
                            });
                        });
                    } else {
                        crate::unroll::<NR>(|n_iter| {
                            crate::unroll::<MR_DIV_N>(|m_iter| {
                                let dst = dst.offset((m_iter * N) as isize + n_iter as isize * dst_cs) as *mut Pack;
                                dst.write_unaligned(mul(beta, *accum.add(m_iter + MR_DIV_N * n_iter)));
                            });
                        });
                    }
                } else {
                    if ALPHA == 2 {
                        crate::unroll::<NR>(|n_iter| {
                            crate::unroll::<MR_DIV_N>(|m_iter| {
                                let dst = dst.offset((m_iter * N) as isize * dst_rs + n_iter as isize * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    add(
                                        mul(alpha, gather(dst, dst_rs)),
                                        mul(beta, *accum.add(m_iter + MR_DIV_N * n_iter)),
                                    ),
                                );
                            });
                        });
                    } else if ALPHA == 1 {
                        crate::unroll::<NR>(|n_iter| {
                            crate::unroll::<MR_DIV_N>(|m_iter| {
                                let dst = dst.offset((m_iter * N) as isize * dst_rs + n_iter as isize * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    mul_add(
                                        beta,
                                        *accum.add(m_iter + MR_DIV_N * n_iter),
                                        gather(dst, dst_rs),
                                    ),
                                );
                            });
                        });
                    } else {
                        crate::unroll::<NR>(|n_iter| {
                            crate::unroll::<MR_DIV_N>(|m_iter| {
                                let dst = dst.offset((m_iter * N) as isize * dst_rs + n_iter as isize * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    mul(beta, *accum.add(m_iter + MR_DIV_N * n_iter)),
                                );
                            });
                        });
                    }
                }
            } else {
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const T;

                if ALPHA == 2 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * MR_DIV_N * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = alpha * *dst_ij + beta * *src_ij;
                        }
                    }
                } else if ALPHA == 1 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * MR_DIV_N * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = *dst_ij + beta * *src_ij;
                        }
                    }
                } else {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * MR_DIV_N * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = beta * *src_ij;
                        }
                    }
                }
            }
        }
    };
}

pub mod scalar {
    pub mod f32 {
        use core::mem::MaybeUninit;

        type T = f32;
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

        microkernel!(, ukr);
    }

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

        microkernel!(, ukr);
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
        use core::mem::MaybeUninit;

        type T = f32;
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

        microkernel!(["sse"], ukr);
    }

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

        microkernel!(["sse"], ukr);
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
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..8 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..8 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

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

        microkernel!(["avx"], ukr);
    }

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

        microkernel!(["avx"], ukr);
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
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..8 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..8 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

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

        microkernel!(["fma"], ukr);
    }

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

        microkernel!(["fma"], ukr);
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
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 16;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..16 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..16 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

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

        microkernel!(["avx512f"], ukr);
    }

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

        microkernel!(["avx512f"], ukr);
    }
}
