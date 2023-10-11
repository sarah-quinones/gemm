use core::slice::from_raw_parts_mut;

use num_traits::{One, Zero};
use seq_macro::seq;

use crate::simd::{Boilerplate, MixedSimd, Simd};

#[inline(always)]
pub unsafe fn gemv<
    T: Copy
        + Zero
        + One
        + Send
        + Sync
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::cmp::PartialEq,
    S: Simd,
>(
    _simd: S,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    mul_add: impl Fn(T, T, T) -> T,
) {
    if !alpha.is_zero() {
        for col in 0..n {
            for row in 0..m {
                let dst = dst
                    .wrapping_offset(row as isize * dst_rs)
                    .wrapping_offset(col as isize * dst_cs);

                *dst = alpha * *dst;
            }
        }
    } else {
        for col in 0..n {
            for row in 0..m {
                let dst = dst
                    .wrapping_offset(row as isize * dst_rs)
                    .wrapping_offset(col as isize * dst_cs);

                *dst = T::zero();
            }
        }
    }

    macro_rules! do_work {
        ($n: tt) => {
            for depth in 0..k {
                seq!(COL in 0..$n {
                    let rhs~COL = beta * *rhs
                        .wrapping_offset(COL as isize * rhs_cs)
                        .wrapping_offset(depth as isize * rhs_rs);
                });
                for row in 0..m {
                    let lhs = *lhs
                        .wrapping_offset(depth as isize * lhs_cs)
                        .wrapping_offset(row as isize * lhs_rs);

                    seq!(COL in 0..$n {
                        {
                            let dst = dst
                                .wrapping_offset(COL as isize * dst_cs)
                                .wrapping_offset(row as isize * dst_rs);
                            *dst = mul_add(rhs~COL, lhs, *dst);
                        }
                    });
                }
            }
        }
    }
    match n {
        1 => do_work!(1),
        _ => unreachable!(),
    }
}

// dst, lhs are colmajor
// n is small
#[inline(always)]
pub unsafe fn mixed_gemv_colmajor<
    Lhs: Boilerplate,
    Rhs: Boilerplate,
    Dst: Boilerplate,
    Acc: Boilerplate,
    S: MixedSimd<Lhs, Rhs, Dst, Acc>,
>(
    simd: S,

    m: usize,
    n: usize,
    k: usize,

    dst: *mut Dst,
    dst_cs: isize,
    dst_rs: isize,

    lhs: *const Lhs,
    lhs_cs: isize,
    lhs_rs: isize,

    rhs: *const Rhs,
    rhs_cs: isize,
    rhs_rs: isize,

    beta: Acc,
) {
    macro_rules! implementation {
        ($name: ident, $dst_tuple: ty, $n: tt) => {
            #[inline(always)]
            unsafe fn $name<
                'a,
                Lhs: Boilerplate,
                Rhs: Boilerplate,
                Dst: Boilerplate,
                Acc: Boilerplate,
                S: MixedSimd<Lhs, Rhs, Dst, Acc>,
            >(
                simd: S,

                m: usize,
                k: usize,

                noalias_dst: $dst_tuple,

                lhs: *const Lhs,
                lhs_cs: isize,

                rhs: *const Rhs,
                rhs_cs: isize,
                rhs_rs: isize,

                beta: Acc,
            ) {
                struct Impl<'a, Lhs, Rhs, Dst, Acc, S> {
                    simd: S,

                    m: usize,
                    k: usize,

                    noalias_dst: $dst_tuple,

                    lhs: *const Lhs,
                    lhs_cs: isize,

                    rhs: *const Rhs,
                    rhs_cs: isize,
                    rhs_rs: isize,

                    beta: Acc,
                }

                impl<
                        Lhs: Boilerplate,
                        Rhs: Boilerplate,
                        Dst: Boilerplate,
                        Acc: Boilerplate,
                        S: MixedSimd<Lhs, Rhs, Dst, Acc>,
                    > pulp::NullaryFnOnce for Impl<'_, Lhs, Rhs, Dst, Acc, S>
                {
                    type Output = ();

                    #[inline(always)]
                    fn call(self) -> Self::Output {
                        unsafe {
                            let Self {
                                simd,
                                m,
                                k,
                                noalias_dst,
                                lhs,
                                lhs_cs,
                                rhs,
                                rhs_cs,
                                rhs_rs,
                                beta,
                            } = self;

                            let lane = S::SIMD_WIDTH;
                            seq!(N in 0..$n {
                                let dst~N = noalias_dst.N.as_mut_ptr();
                            });

                            let m_lane = m / lane * lane;

                            for col in 0..k {
                                let lhs = lhs.wrapping_offset(col as isize * lhs_cs);
                                seq!(N in 0..$n {
                                    let rhs~N = simd.from_rhs(*rhs.wrapping_offset(col as isize * rhs_rs + N * rhs_cs));
                                    let rhs_scalar~N = simd.mult(beta, rhs~N);
                                    let rhs~N = simd.simd_splat(rhs_scalar~N);
                                });

                                let mut row = 0usize;
                                while row < m_lane {
                                    seq!(N in 0..$n {
                                        let dst_ptr~N = dst~N.wrapping_add(row) as *mut S::DstN;
                                        let dst~N = *dst_ptr~N;
                                    });
                                    let lhs = simd.simd_from_lhs(*(lhs.wrapping_add(row) as *const S::LhsN));

                                    seq!(N in 0..$n {
                                        *dst_ptr~N = simd.simd_into_dst(simd.simd_mult_add(
                                            lhs,
                                            rhs~N,
                                            simd.simd_from_dst(dst~N),
                                        ));
                                    });

                                    row += m_lane;
                                }
                                while row < m {
                                    seq!(N in 0..$n {
                                        let dst_ptr~N = dst~N.wrapping_add(row);
                                        let dst~N = *dst_ptr~N;
                                    });
                                    let lhs = simd.from_lhs(*lhs.wrapping_add(row));
                                    seq!(N in 0..$n {
                                        *dst_ptr~N = simd.into_dst(simd.mult_add(
                                            lhs,
                                            rhs_scalar~N,
                                            simd.from_dst(dst~N),
                                        ));
                                    });
                                    row += 1;
                                }
                            }
                        }
                    }
                }

                simd.vectorize(Impl {
                    simd,
                    m,
                    k,
                    noalias_dst,
                    lhs,
                    lhs_cs,
                    rhs,
                    rhs_cs,
                    rhs_rs,
                    beta,
                })
            }
        };
    }
    implementation!(_1, (&'a mut [Dst],), 1);
    implementation!(_2, (&'a mut [Dst], &'a mut [Dst]), 2);
    implementation!(_3, (&'a mut [Dst], &'a mut [Dst], &'a mut [Dst]), 3);
    implementation!(
        _4,
        (&'a mut [Dst], &'a mut [Dst], &'a mut [Dst], &'a mut [Dst]),
        4
    );
    implementation!(
        _5,
        (
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
        ),
        5
    );
    implementation!(
        _6,
        (
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
        ),
        6
    );
    implementation!(
        _7,
        (
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
        ),
        7
    );
    implementation!(
        _8,
        (
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
            &'a mut [Dst],
        ),
        8
    );

    assert_eq!(lhs_rs, 1);
    assert_eq!(dst_rs, 1);

    if n == 0 {
        return;
    }

    assert!(n <= 8);

    match n {
        1 => _1(
            simd,
            m,
            k,
            (from_raw_parts_mut(dst as _, m),),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        2 => _2(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        3 => _3(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(2 * dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        4 => _4(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(2 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(3 * dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        5 => _5(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(2 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(3 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(4 * dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        6 => _6(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(2 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(3 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(4 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(5 * dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        7 => _7(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(2 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(3 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(4 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(5 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(6 * dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        8 => _8(
            simd,
            m,
            k,
            (
                from_raw_parts_mut(dst as _, m),
                from_raw_parts_mut(dst.wrapping_offset(dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(2 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(3 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(4 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(5 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(6 * dst_cs) as _, m),
                from_raw_parts_mut(dst.wrapping_offset(7 * dst_cs) as _, m),
            ),
            lhs,
            lhs_cs,
            rhs,
            rhs_cs,
            rhs_rs,
            beta,
        ),
        _ => unreachable!(),
    }
}
