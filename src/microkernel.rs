use crate::Ptr;

pub(crate) type MicroKernelFn<T> =
    unsafe fn(usize, usize, usize, Ptr<T>, Ptr<T>, Ptr<T>, isize, isize, isize, isize, isize, T, T);

macro_rules! microkernel {
    ($([$target: tt])?, $name: ident, $mr_div_n: tt, $nr: tt) => {
        #[inline]
        $(#[target_feature(enable = $target)])?
        // 0, 1, or 2 for generic alpha
        pub(crate) unsafe fn $name<const ALPHA: u8>(
            m: usize,
            n: usize,
            k: usize,
            dst: $crate::Ptr<T>,
            packed_lhs: $crate::Ptr<T>,
            packed_rhs: $crate::Ptr<T>,
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

            let mut accum_storage = [[splat(0.0); $mr_div_n]; $nr];
            let accum = accum_storage.as_mut_ptr() as *mut Pack;

            let mut lhs = [::core::mem::MaybeUninit::<Pack>::uninit(); $mr_div_n];
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
                unsafe fn execute(self, iter: usize) {
                    let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {
                        (*self.lhs.add(M_ITER))
                            .write((packed_lhs.add(M_ITER * N) as *const Pack).read_unaligned());
                    });

                    seq_macro::seq!(N_ITER in 0..$nr {
                        (*self.rhs).write(splat(*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs)));
                        let accum = self.accum.add(N_ITER * $mr_div_n);
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let accum = &mut *accum.add(M_ITER);
                            *accum = mul_add(
                                (*self.lhs.add(M_ITER)).assume_init(),
                                (*self.rhs).assume_init(),
                                *accum,
                            );
                        }});
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

                    seq_macro::seq!(UNROLL_ITER in 0..4 {
                        iter.execute(UNROLL_ITER);
                    });

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
                    .execute(0);

                    packed_lhs = packed_lhs.wrapping_offset(lhs_cs);
                    packed_rhs = packed_rhs.wrapping_offset(rhs_rs);

                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }

            if m == $mr_div_n * N && n == $nr {
                let alpha = splat(alpha);
                let beta = splat(beta);
                if dst_rs == 1 {

                    if ALPHA == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(add(
                                    mul(alpha, dst.read_unaligned()),
                                    mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
                                ));
                            }});
                        });
                    } else if ALPHA == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(mul_add(
                                    beta,
                                    *accum.offset(M_ITER + $mr_div_n * N_ITER),
                                    dst.read_unaligned(),
                                ));
                            }});
                        });
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)));
                            }});
                        });
                    }
                } else {
                    if ALPHA == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize * dst_rs + N_ITER * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    add(
                                        mul(alpha, gather(dst, dst_rs)),
                                        mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
                                    ),
                                );
                            }});
                        });
                    } else if ALPHA == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize * dst_rs + N_ITER * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    mul_add(
                                        beta,
                                        *accum.offset(M_ITER + $mr_div_n * N_ITER),
                                        gather(dst, dst_rs),
                                    ),
                                );
                            }});
                        });
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize * dst_rs + N_ITER * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
                                );
                            }});
                        });
                    }
                }
            } else {
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const T;

                if ALPHA == 2 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = alpha * *dst_ij + beta * *src_ij;
                        }
                    }
                } else if ALPHA == 1 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = *dst_ij + beta * *src_ij;
                        }
                    }
                } else {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

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

        microkernel!(, x1x1, 1, 1);
        microkernel!(, x1x2, 1, 2);
        microkernel!(, x1x3, 1, 3);
        microkernel!(, x1x4, 1, 4);

        microkernel!(, x2x1, 2, 1);
        microkernel!(, x2x2, 2, 2);
        microkernel!(, x2x3, 2, 3);
        microkernel!(, x2x4, 2, 4);
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

        microkernel!(, x1x1, 1, 1);
        microkernel!(, x1x2, 1, 2);
        microkernel!(, x1x3, 1, 3);
        microkernel!(, x1x4, 1, 4);

        microkernel!(, x2x1, 2, 1);
        microkernel!(, x2x2, 2, 2);
        microkernel!(, x2x3, 2, 3);
        microkernel!(, x2x4, 2, 4);
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

        microkernel!(["sse"], x1x1, 1, 1);
        microkernel!(["sse"], x1x2, 1, 2);
        microkernel!(["sse"], x1x3, 1, 3);
        microkernel!(["sse"], x1x4, 1, 4);

        microkernel!(["sse"], x2x1, 2, 1);
        microkernel!(["sse"], x2x2, 2, 2);
        microkernel!(["sse"], x2x3, 2, 3);
        microkernel!(["sse"], x2x4, 2, 4);
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

        microkernel!(["sse"], x1x1, 1, 1);
        microkernel!(["sse"], x1x2, 1, 2);
        microkernel!(["sse"], x1x3, 1, 3);
        microkernel!(["sse"], x1x4, 1, 4);

        microkernel!(["sse"], x2x1, 2, 1);
        microkernel!(["sse"], x2x2, 2, 2);
        microkernel!(["sse"], x2x3, 2, 3);
        microkernel!(["sse"], x2x4, 2, 4);
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

        microkernel!(["avx"], x1x1, 1, 1);
        microkernel!(["avx"], x1x2, 1, 2);
        microkernel!(["avx"], x1x3, 1, 3);
        microkernel!(["avx"], x1x4, 1, 4);

        microkernel!(["avx"], x2x1, 2, 1);
        microkernel!(["avx"], x2x2, 2, 2);
        microkernel!(["avx"], x2x3, 2, 3);
        microkernel!(["avx"], x2x4, 2, 4);
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

        microkernel!(["avx"], x1x1, 1, 1);
        microkernel!(["avx"], x1x2, 1, 2);
        microkernel!(["avx"], x1x3, 1, 3);
        microkernel!(["avx"], x1x4, 1, 4);

        microkernel!(["avx"], x2x1, 2, 1);
        microkernel!(["avx"], x2x2, 2, 2);
        microkernel!(["avx"], x2x3, 2, 3);
        microkernel!(["avx"], x2x4, 2, 4);
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
    }
}
