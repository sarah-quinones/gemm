pub mod x512bit {
    pub mod f64 {
        type T = f64;
        const N: usize = 8;
        type Pack = packed_simd_2::f64x8;
        crate::ukr!();
    }
    pub mod f32 {
        type T = f32;
        const N: usize = 16;
        type Pack = packed_simd_2::f32x16;
        crate::ukr!();
    }
}

pub mod x256bit {
    pub mod f64 {
        type T = f64;
        const N: usize = 4;
        type Pack = packed_simd_2::f64x4;
        crate::ukr!();
    }
    pub mod f32 {
        type T = f32;
        const N: usize = 8;
        type Pack = packed_simd_2::f32x8;
        crate::ukr!();
    }
}

pub mod x128bit {
    pub mod f64 {
        type T = f64;
        const N: usize = 2;
        type Pack = packed_simd_2::f64x2;
        crate::ukr!();
    }
    pub mod f32 {
        type T = f32;
        const N: usize = 4;
        type Pack = packed_simd_2::f32x4;
        crate::ukr!();
    }
}

pub mod scalar {
    pub trait Scalar: Copy + core::ops::Mul<Output = Self> + core::ops::Add<Output = Self> {
        fn splat(value: Self) -> Self {
            value
        }
        fn mul_adde(a: Self, b: Self, c: Self) -> Self {
            a * b + c
        }
    }
    impl Scalar for f32 {}
    impl Scalar for f64 {}

    pub mod f64 {
        use super::Scalar;
        type T = f64;
        const N: usize = 1;
        type Pack = f64;
        crate::ukr!();
    }
    pub mod f32 {
        use super::Scalar;
        type T = f32;
        const N: usize = 1;
        type Pack = f32;
        crate::ukr!();
    }
}

#[macro_export]
macro_rules! ukr {
    () => {
        #[inline(always)]
        unsafe fn gather(src: *const T, stride: isize) -> Pack {
            let mut p = [::core::mem::MaybeUninit::<T>::uninit(); N];
            ::unroll_fn::unroll::<N, _>(|iter| {
                p.get_unchecked_mut(iter)
                    .write(*src.offset(iter as isize * stride));
            });
            ::core::mem::transmute(p)
        }
        #[inline(always)]
        unsafe fn scatter(dst: *mut T, stride: isize, p: Pack) {
            let p = ::core::mem::transmute::<Pack, [T; N]>(p);
            ::unroll_fn::unroll::<N, _>(|iter| *dst.offset(iter as isize * stride) = *p.get_unchecked(iter));
        }

        #[derive(Copy, Clone)]
        struct KernelIter<const MR_DIV_N: usize, const NR: usize> {
            packed_lhs: *const T,
            packed_rhs: *const T,
            lhs_cs: isize,
            rhs_rs: isize,
            accum: *mut Pack,
            lhs: *mut ::core::mem::MaybeUninit<Pack>,
            rhs: *mut ::core::mem::MaybeUninit<Pack>,
        }

        impl<const MR_DIV_N: usize, const NR: usize> KernelIter<MR_DIV_N, NR> {
            #[inline(always)]
            unsafe fn execute(self, iter: usize) {
                {
                    let packed_lhs = self.packed_lhs.offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.offset(iter as isize * self.rhs_rs);

                    ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter: usize| {
                        (*self.lhs.add(m_iter))
                            .write((packed_lhs.add(m_iter * N) as *const Pack).read_unaligned());
                    });
                    ::unroll_fn::unroll::<NR, _>(|n_iter: usize| {
                        (*self.rhs).write(Pack::splat(*packed_rhs.add(n_iter)));
                        let accum = self.accum.add(n_iter * MR_DIV_N);
                        ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter: usize| {
                            let accum = &mut *accum.add(m_iter);
                            *accum = Pack::mul_adde(
                                (*self.lhs.add(m_iter)).assume_init(),
                                (*self.rhs).assume_init(),
                                *accum,
                            );
                        });
                    });
                }
            }
        }

        #[derive(Copy, Clone)]
        struct DstSetInner {
            accum: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct DstSetOuter<const NR: usize> {
            accum: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct DstUpdateInner {
            accum: *const Pack,
            alpha: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct DstUpdateOuter<const NR: usize> {
            accum: *const Pack,
            alpha: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct ScatterDstSetInner {
            accum: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_rs: isize,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct ScatterDstSetOuter<const NR: usize> {
            accum: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_rs: isize,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct ScatterDstUpdateInner {
            accum: *const Pack,
            alpha: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_rs: isize,
            dst_cs: isize,
            mr_div_n: usize,
        }
        #[derive(Copy, Clone)]
        struct ScatterDstUpdateOuter<const NR: usize> {
            accum: *const Pack,
            alpha: *const Pack,
            beta: *const Pack,
            dst: *mut T,
            dst_rs: isize,
            dst_cs: isize,
            mr_div_n: usize,
        }

        impl DstSetInner {
            #[inline(always)]
            unsafe fn execute(self, n_iter: usize) {
                let dst = self.dst.offset(n_iter as isize * self.dst_cs) as *mut Pack;
                dst.write_unaligned(<Pack as ::core::ops::Mul>::mul(
                    *self.beta,
                    *self.accum.add(self.mr_div_n * n_iter),
                ));
            }
        }
        impl<const NR: usize> DstSetOuter<NR> {
            #[inline(always)]
            unsafe fn execute(self, m_iter: usize) {
                let inner = DstSetInner {
                    accum: self.accum.add(m_iter),
                    beta: self.beta,
                    dst: self.dst.add(m_iter * N),
                    dst_cs: self.dst_cs,
                    mr_div_n: self.mr_div_n,
                };
                ::unroll_fn::unroll::<NR, _>(|n_iter| {
                    inner.execute(n_iter);
                });
            }
        }

        impl DstUpdateInner {
            #[inline(always)]
            unsafe fn execute(self, n_iter: usize) {
                let dst = self.dst.offset(n_iter as isize * self.dst_cs) as *mut Pack;
                dst.write_unaligned(<Pack as ::core::ops::Add>::add(
                    <Pack as ::core::ops::Mul>::mul(*self.alpha, dst.read_unaligned()),
                    <Pack as ::core::ops::Mul>::mul(*self.beta, *self.accum.add(self.mr_div_n * n_iter)),
                ));
            }
        }
        impl<const NR: usize> DstUpdateOuter<NR> {
            #[inline(always)]
            unsafe fn execute(self, m_iter: usize) {
                let inner = DstUpdateInner {
                    accum: self.accum.add(m_iter),
                    alpha: self.alpha,
                    beta: self.beta,
                    dst: self.dst.add(m_iter * N),
                    dst_cs: self.dst_cs,
                    mr_div_n: self.mr_div_n,
                };
                ::unroll_fn::unroll::<NR, _>(|n_iter| {
                    inner.execute(n_iter);
                });
            }
        }

        impl ScatterDstSetInner {
            #[inline(always)]
            unsafe fn execute(self, n_iter: usize) {
                let dst = self.dst.offset(n_iter as isize * self.dst_cs);
                scatter(
                    dst,
                    self.dst_rs,
                    <Pack as ::core::ops::Mul>::mul(*self.beta, *self.accum.add(self.mr_div_n * n_iter)),
                );
            }
        }
        impl<const NR: usize> ScatterDstSetOuter<NR> {
            #[inline(always)]
            unsafe fn execute(self, m_iter: usize) {
                let inner = ScatterDstSetInner {
                    accum: self.accum.add(m_iter),
                    beta: self.beta,
                    dst: self.dst.add(m_iter * N),
                    dst_rs: self.dst_rs,
                    dst_cs: self.dst_cs,
                    mr_div_n: self.mr_div_n,
                };
                ::unroll_fn::unroll::<NR, _>(|n_iter| {
                    inner.execute(n_iter);
                });
            }
        }

        impl ScatterDstUpdateInner {
            #[inline(always)]
            unsafe fn execute(self, n_iter: usize) {
                let dst = self.dst.offset(n_iter as isize * self.dst_cs);
                scatter(
                    dst,
                    self.dst_rs,
                    <Pack as ::core::ops::Add>::add(
                        <Pack as ::core::ops::Mul>::mul(*self.alpha, gather(dst, self.dst_rs)),
                        <Pack as ::core::ops::Mul>::mul(*self.beta, *self.accum.add(self.mr_div_n * n_iter)),
                    ),
                );
            }
        }
        impl<const NR: usize> ScatterDstUpdateOuter<NR> {
            #[inline(always)]
            unsafe fn execute(self, m_iter: usize) {
                let inner = ScatterDstUpdateInner {
                    accum: self.accum.add(m_iter),
                    alpha: self.alpha,
                    beta: self.beta,
                    dst: self.dst.add(m_iter * N),
                    dst_rs: self.dst_rs,
                    dst_cs: self.dst_cs,
                    mr_div_n: self.mr_div_n,
                };
                ::unroll_fn::unroll::<NR, _>(|n_iter| {
                    inner.execute(n_iter);
                });
            }
        }

        struct Const<const OUTER_COUNT: usize>;

        trait ReduceAccum<const MR_DIV_N: usize, const NR: usize> {
            unsafe fn reduce(accum: *mut Pack);
        }

        impl<const MR_DIV_N: usize, const NR: usize, const OUTER_COUNT: usize> ReduceAccum<MR_DIV_N, NR>
            for Const<OUTER_COUNT>
        {
            default unsafe fn reduce(_accum: *mut Pack) {
                extern "Rust" {
                    fn unimplemented();
                }
                unimplemented()
            }
        }

        impl<const MR_DIV_N: usize, const NR: usize> ReduceAccum<MR_DIV_N, NR> for Const<1> {
            #[inline(always)]
            unsafe fn reduce(_accum: *mut Pack) {}
        }
        macro_rules! impl_reduce_accum {
            ($outer_count: expr) => {
                impl<const MR_DIV_N: usize, const NR: usize> ReduceAccum<MR_DIV_N, NR>
                    for Const<$outer_count>
                {
                    #[inline(always)]
                    unsafe fn reduce(accum: *mut Pack) {
                        const OUTER_COUNT: usize = $outer_count;

                        let accum0 = accum;
                        let accum1 = accum.add(MR_DIV_N * NR * OUTER_COUNT / 2);

                        <Const<{ OUTER_COUNT / 2 }> as ReduceAccum<MR_DIV_N, NR>>::reduce(accum0);
                        <Const<{ OUTER_COUNT - OUTER_COUNT / 2 }> as ReduceAccum<MR_DIV_N, NR>>::reduce(
                            accum1,
                        );

                        ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter| {
                            ::unroll_fn::unroll::<NR, _>(|n_iter| {
                                let iter = m_iter * NR + n_iter;
                                let accum1 = *accum1.add(iter);
                                let accum0 = &mut *accum0.add(iter);
                                *accum0 = <Pack as ::core::ops::Add>::add(*accum0, accum1);
                            });
                        });
                    }
                }
            };
        }

        seq_macro::seq!(N in 2..=32 {
            impl_reduce_accum!(N);
        });

        #[inline(always)]
        #[allow(dead_code)]
        pub(crate) unsafe fn ukr<
            const MR_DIV_N: usize,
            const NR: usize,
            const MULTIPLIER: usize,
            const K_UNROLL: usize,
        >(
            m :usize,
            n :usize,
            k :usize,
            dst :$crate::Ptr<T>,
            packed_lhs :$crate::Ptr<T>,
            packed_rhs :$crate::Ptr<T>,
            dst_cs :isize,
            dst_rs :isize,
            lhs_cs :isize,
            rhs_rs :isize,
            alpha :T,
            beta :T,
            read_dst :bool
        ) {
            let dst = dst.0;
            let mut packed_lhs = packed_lhs.0;
            let mut packed_rhs = packed_rhs.0;

            let mut accum_storage = [[[Pack::splat(0.0); MR_DIV_N]; NR]; MULTIPLIER];
            let accum = accum_storage.as_mut_ptr() as *mut Pack;

            let mut lhs = [::core::mem::MaybeUninit::<Pack>::uninit(); MR_DIV_N];
            let mut rhs = ::core::mem::MaybeUninit::<Pack>::uninit();

            let k_unroll = k / (K_UNROLL * MULTIPLIER);
            let k_leftover = k % (K_UNROLL * MULTIPLIER);

            let mut depth = k_unroll;
            if depth != 0 {
                loop {
                    ::unroll_fn::unroll::<K_UNROLL, _>(|unroll_iter| {
                        ::unroll_fn::unroll::<MULTIPLIER, _>(|mult_iter| {
                            KernelIter::<MR_DIV_N, NR> {
                                packed_lhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                accum: accum.add((NR * MR_DIV_N) * mult_iter),
                                lhs: lhs.as_mut_ptr(),
                                rhs: &mut rhs,
                            }
                            .execute(unroll_iter * MULTIPLIER + mult_iter);
                        });
                    });
                    packed_lhs = packed_lhs.offset((K_UNROLL * MULTIPLIER) as isize * lhs_cs);
                    packed_rhs = packed_rhs.offset((K_UNROLL * MULTIPLIER) as isize * rhs_rs);
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }

            depth = k_leftover;
            if depth != 0 {
                loop {
                    KernelIter::<MR_DIV_N, NR> {
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

            <Const<MULTIPLIER> as ReduceAccum<MR_DIV_N, NR>>::reduce(accum);

            if m == MR_DIV_N * N && n == NR {
                if dst_rs == 1 {
                    let beta = Pack::splat(beta);

                    if read_dst {
                        let alpha = Pack::splat(alpha);

                        let outer = DstUpdateOuter::<NR> {
                            accum,
                            alpha: &alpha,
                            beta: &beta,
                            dst,
                            dst_cs,
                            mr_div_n: MR_DIV_N,
                        };
                        ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter| outer.execute(m_iter));
                    } else {
                        let outer = DstSetOuter::<NR> {
                            accum,
                            beta: &beta,
                            dst,
                            dst_cs,
                            mr_div_n: MR_DIV_N,
                        };
                        ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter| outer.execute(m_iter));
                    }
                } else {
                    let beta = Pack::splat(beta);
                    if read_dst {
                        let alpha = Pack::splat(alpha);
                        let outer = ScatterDstUpdateOuter::<NR> {
                            accum,
                            alpha: &alpha,
                            beta: &beta,
                            dst,
                            dst_rs,
                            dst_cs,
                            mr_div_n: MR_DIV_N,
                        };
                        ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter| outer.execute(m_iter));
                    } else {
                        let outer = ScatterDstSetOuter::<NR> {
                            accum,
                            beta: &beta,
                            dst,
                            dst_rs,
                            dst_cs,
                            mr_div_n: MR_DIV_N,
                        };
                        ::unroll_fn::unroll::<MR_DIV_N, _>(|m_iter| outer.execute(m_iter));
                    }
                }
            } else {
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const T;
                if read_dst {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * (MR_DIV_N *N));

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = alpha * *dst_ij + beta * *src_ij;
                        }
                    }
                } else {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * (MR_DIV_N *N));

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
