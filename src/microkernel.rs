macro_rules! microkernel {
    ($name: ident, $mr_div_n: tt, $nr: tt) => {
        #[inline(always)]
        pub(crate) unsafe fn $name(
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
            alpha: T,
            beta: T,
            read_dst: bool,
        ) {
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
                accum: *mut Pack,
                lhs: *mut ::core::mem::MaybeUninit<Pack>,
                rhs: *mut ::core::mem::MaybeUninit<Pack>,
            }

            impl KernelIter {
                #[inline(always)]
                unsafe fn execute(self, iter: usize) {
                    let packed_lhs = self.packed_lhs.offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.offset(iter as isize * self.rhs_rs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {
                        (*self.lhs.add(M_ITER))
                            .write((packed_lhs.add(M_ITER * N) as *const Pack).read_unaligned());
                    });

                    seq_macro::seq!(N_ITER in 0..$nr {
                        (*self.rhs).write(splat(*packed_rhs.add(N_ITER)));
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
                        accum,
                        lhs: lhs.as_mut_ptr(),
                        rhs: &mut rhs,
                    };

                    seq_macro::seq!(UNROLL_ITER in 0..4 {
                        iter.execute(UNROLL_ITER);
                    });

                    packed_lhs = packed_lhs.offset(4 * lhs_cs);
                    packed_rhs = packed_rhs.offset(4 * rhs_rs);

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
                        accum,
                        lhs: lhs.as_mut_ptr(),
                        rhs: &mut rhs,
                    }
                    .execute(0);

                    packed_lhs = packed_lhs.offset(lhs_cs);
                    packed_rhs = packed_rhs.offset(rhs_rs);

                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }

            if m == $mr_div_n * N && n == $nr {
                let beta = splat(beta);
                if dst_rs == 1 {

                    if read_dst {
                        let alpha = splat(alpha);
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(add(
                                    mul(alpha, dst.read_unaligned()),
                                    mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
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
                    if read_dst {
                        let alpha = splat(alpha);
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs);
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
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs);
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

                if read_dst {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = alpha * *dst_ij + beta * *src_ij;
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

pub mod x256bit {
    pub mod f64 {
        use core::arch::x86_64::*;
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
            let mut out = MaybeUninit::<Pack>::uninit();
            _mm256_storeu_pd(out.as_mut_ptr() as *mut f64, _mm256_set1_pd(value));
            out.assume_init()
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            let mut out = MaybeUninit::<Pack>::uninit();
            _mm256_storeu_pd(
                out.as_mut_ptr() as *mut f64,
                _mm256_mul_pd(_mm256_loadu_pd(lhs.as_ptr()), _mm256_loadu_pd(rhs.as_ptr())),
            );
            out.assume_init()
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            let mut out = MaybeUninit::<Pack>::uninit();
            _mm256_storeu_pd(
                out.as_mut_ptr() as *mut f64,
                _mm256_add_pd(_mm256_loadu_pd(lhs.as_ptr()), _mm256_loadu_pd(rhs.as_ptr())),
            );
            out.assume_init()
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            let mut out = MaybeUninit::<Pack>::uninit();
            _mm256_storeu_pd(
                out.as_mut_ptr() as *mut f64,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(a.as_ptr()),
                    _mm256_loadu_pd(b.as_ptr()),
                    _mm256_loadu_pd(c.as_ptr()),
                ),
            );
            out.assume_init()
        }

        microkernel!(x1x1, 1, 1);
        microkernel!(x1x2, 1, 2);
        microkernel!(x1x3, 1, 3);
        microkernel!(x1x4, 1, 4);

        microkernel!(x2x1, 2, 1);
        microkernel!(x2x2, 2, 2);
        microkernel!(x2x3, 2, 3);
        microkernel!(x2x4, 2, 4);

        microkernel!(x3x1, 3, 1);
        microkernel!(x3x2, 3, 2);
        microkernel!(x3x3, 3, 3);
        microkernel!(x3x4, 3, 4);
    }
}
