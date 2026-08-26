use core::slice::from_raw_parts_mut;

use num_traits::{One, Zero};
use seq_macro::seq;

use crate::simd::{Boilerplate, MixedSimd, Simd};
#[cfg(feature = "rayon")]
use crate::{
    cache::DivCeil,
    gemm::{get_threading_threshold, par_for_each, CACHELINE_ALIGN},
    Ptr,
};
use crate::Parallelism;
#[cfg(feature = "rayon")]
use dyn_stack::{DynStack, MemBuffer, StackReq};

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
    Lhs: Boilerplate + One + Zero,
    Rhs: Boilerplate + One + Zero,
    Dst: Boilerplate + One + Zero,
    Acc: Boilerplate + One + Zero,
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

    alpha: Acc,
    beta: Acc,
) {
    #[inline(always)]
    unsafe fn implementation<
        'a,
        Lhs: Boilerplate + One + Zero,
        Rhs: Boilerplate + One + Zero,
        Dst: Boilerplate + One + Zero,
        Acc: Boilerplate + One + Zero,
        S: MixedSimd<Lhs, Rhs, Dst, Acc>,
    >(
        noalias_dst: (&'a mut [Dst],),
        simd: S,
        m: usize,
        k: usize,
        lhs: *const Lhs,
        lhs_cs: isize,
        rhs: *const Rhs,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha: Acc,
        beta: Acc,
    ) {
        #[allow(dead_code)]
        struct Impl<'a, Lhs, Rhs, Dst, Acc, S> {
            simd: S,
            m: usize,
            k: usize,
            noalias_dst: (&'a mut [Dst],),
            lhs: *const Lhs,
            lhs_cs: isize,
            rhs: *const Rhs,
            rhs_cs: isize,
            rhs_rs: isize,
            alpha: Acc,
            beta: Acc,
        }
        impl<
                Lhs: Boilerplate + One + Zero,
                Rhs: Boilerplate + One + Zero,
                Dst: Boilerplate + One + Zero,
                Acc: Boilerplate + One + Zero,
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
                        rhs_cs: _,
                        rhs_rs,
                        mut alpha,
                        beta,
                    } = self;

                    let lane = S::SIMD_WIDTH;
                    let dst = noalias_dst.0.as_mut_ptr();
                    let m_lane = m / lane * lane;
                    for col in 0..k {
                        let lhs = lhs.wrapping_offset(col as isize * lhs_cs);
                        let rhs = simd.from_rhs(*rhs.wrapping_offset(col as isize * rhs_rs));

                        let alpha_s = alpha;
                        let alpha_v = simd.simd_splat(alpha_s);

                        let rhs_scalar = simd.mult(beta, rhs);
                        let rhs = simd.simd_splat(rhs_scalar);

                        if alpha_s.is_zero() {
                            let mut row = 0usize;
                            while row < m_lane {
                                let dst_ptr = dst.wrapping_add(row) as *mut S::DstN;
                                let lhs =
                                    simd.simd_from_lhs(*(lhs.wrapping_add(row) as *const S::LhsN));
                                *dst_ptr = simd.simd_into_dst(simd.simd_mul(lhs, rhs));
                                row += lane;
                            }
                            while row < m {
                                let dst_ptr = dst.wrapping_add(row);
                                let lhs = simd.from_lhs(*lhs.wrapping_add(row));
                                *dst_ptr = simd.into_dst(simd.mult(lhs, rhs_scalar));
                                row += 1;
                            }
                        } else if alpha_s.is_one() {
                            let mut row = 0usize;
                            while row < m_lane {
                                let dst_ptr = dst.wrapping_add(row) as *mut S::DstN;
                                let dst = *dst_ptr;
                                let lhs =
                                    simd.simd_from_lhs(*(lhs.wrapping_add(row) as *const S::LhsN));
                                *dst_ptr = simd.simd_into_dst(simd.simd_mult_add(
                                    lhs,
                                    rhs,
                                    simd.simd_from_dst(dst),
                                ));
                                row += lane;
                            }
                            while row < m {
                                let dst_ptr = dst.wrapping_add(row);
                                let dst = *dst_ptr;
                                let lhs = simd.from_lhs(*lhs.wrapping_add(row));
                                *dst_ptr = simd.into_dst(simd.mult_add(
                                    lhs,
                                    rhs_scalar,
                                    simd.from_dst(dst),
                                ));
                                row += 1;
                            }
                        } else {
                            let mut row = 0usize;
                            while row < m_lane {
                                let dst_ptr = dst.wrapping_add(row) as *mut S::DstN;
                                let dst = *dst_ptr;
                                let lhs =
                                    simd.simd_from_lhs(*(lhs.wrapping_add(row) as *const S::LhsN));
                                *dst_ptr = simd.simd_into_dst(simd.simd_add(
                                    simd.simd_mul(lhs, rhs),
                                    simd.simd_mul(alpha_v, simd.simd_from_dst(dst)),
                                ));
                                row += lane;
                            }
                            while row < m {
                                let dst_ptr = dst.wrapping_add(row);
                                let dst = *dst_ptr;
                                let lhs = simd.from_lhs(*lhs.wrapping_add(row));
                                *dst_ptr = simd.into_dst(simd.add(
                                    simd.mult(lhs, rhs_scalar),
                                    simd.mult(alpha_s, simd.from_dst(dst)),
                                ));
                                row += 1;
                            }
                        }
                        alpha = Acc::one();
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
            alpha,
            beta,
        })
    }

    assert_eq!(lhs_rs, 1);
    assert_eq!(dst_rs, 1);

    if k == 0 {
        if alpha.is_one() {
            return;
        }
        if alpha.is_zero() {
            for j in 0..n {
                core::ptr::write_bytes(dst.wrapping_offset(j as isize * dst_cs), 0u8, m);
            }
            return;
        }

        for j in 0..n {
            let dst = dst.wrapping_offset(j as isize * dst_cs);
            for i in 0..m {
                let dst = dst.add(i);
                *dst = simd.into_dst(simd.mult(simd.from_dst(*dst), alpha));
            }
        }
    }

    for x in 0..n {
        implementation(
            (from_raw_parts_mut(
                dst.wrapping_offset(x as isize * dst_cs) as _,
                m,
            ),),
            simd,
            m,
            k,
            lhs,
            lhs_cs,
            rhs.wrapping_offset(rhs_cs * x as isize),
            rhs_cs,
            rhs_rs,
            alpha,
            beta,
        );
    }
}

// lhs is rowmajor
// rhs is colmajor
// n is small
#[inline(always)]
pub unsafe fn mixed_gemv_rowmajor<
    Lhs: Boilerplate + One + Zero,
    Rhs: Boilerplate + One + Zero,
    Dst: Boilerplate + One + Zero,
    Acc: Boilerplate + One + Zero,
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

    alpha: Acc,
    beta: Acc,
) {
    #[inline(always)]
    unsafe fn implementation<
        'a,
        Lhs: Boilerplate + One + Zero,
        Rhs: Boilerplate + One + Zero,
        Dst: Boilerplate + One + Zero,
        Acc: Boilerplate + One + Zero,
        S: MixedSimd<Lhs, Rhs, Dst, Acc>,
    >(
        simd: S,
        dst: *mut Dst,
        dst_rs: isize,
        m: usize,
        k: usize,
        lhs: *const Lhs,
        lhs_rs: isize,
        rhs: *const Rhs,
        alpha: Acc,
        beta: Acc,
    ) {
        #[allow(dead_code)]
        struct Impl<Lhs, Rhs, Dst, Acc, S> {
            simd: S,
            dst: *mut Dst,
            dst_rs: isize,
            m: usize,
            k: usize,
            lhs: *const Lhs,
            lhs_rs: isize,
            rhs: *const Rhs,
            alpha: Acc,
            beta: Acc,
        }
        impl<
                Lhs: Boilerplate + One + Zero,
                Rhs: Boilerplate + One + Zero,
                Dst: Boilerplate + One + Zero,
                Acc: Boilerplate + One + Zero,
                S: MixedSimd<Lhs, Rhs, Dst, Acc>,
            > pulp::NullaryFnOnce for Impl<Lhs, Rhs, Dst, Acc, S>
        {
            type Output = ();

            #[inline(always)]
            fn call(self) -> Self::Output {
                unsafe {
                    let Self {
                        simd,
                        dst,
                        dst_rs,
                        m,
                        k,
                        lhs,
                        lhs_rs,
                        rhs,
                        alpha,
                        beta,
                    } = self;

                    let lane = S::SIMD_WIDTH;
                    let lane8 = 8 * S::SIMD_WIDTH;

                    let k_lane = k / lane * lane;
                    let k_lane8 = k / lane8 * lane8;

                    for row in 0..m {
                        let lhs = lhs.wrapping_offset(row as isize * lhs_rs);

                        let mut depth = 0;

                        let mut acc0 = simd.simd_splat(Acc::zero());
                        let mut acc1 = simd.simd_splat(Acc::zero());
                        let mut acc2 = simd.simd_splat(Acc::zero());
                        let mut acc3 = simd.simd_splat(Acc::zero());
                        let mut acc4 = simd.simd_splat(Acc::zero());
                        let mut acc5 = simd.simd_splat(Acc::zero());
                        let mut acc6 = simd.simd_splat(Acc::zero());
                        let mut acc7 = simd.simd_splat(Acc::zero());

                        while depth < k_lane8 {
                            let lhs0 = *(lhs.wrapping_add(depth + lane * 0) as *const S::LhsN);
                            let rhs0 = *(rhs.wrapping_add(depth + lane * 0) as *const S::RhsN);
                            acc0 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs0),
                                simd.simd_from_rhs(rhs0),
                                acc0,
                            );

                            let lhs1 = *(lhs.wrapping_add(depth + lane * 1) as *const S::LhsN);
                            let rhs1 = *(rhs.wrapping_add(depth + lane * 1) as *const S::RhsN);
                            acc1 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs1),
                                simd.simd_from_rhs(rhs1),
                                acc1,
                            );

                            let lhs2 = *(lhs.wrapping_add(depth + lane * 2) as *const S::LhsN);
                            let rhs2 = *(rhs.wrapping_add(depth + lane * 2) as *const S::RhsN);
                            acc2 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs2),
                                simd.simd_from_rhs(rhs2),
                                acc2,
                            );

                            let lhs3 = *(lhs.wrapping_add(depth + lane * 3) as *const S::LhsN);
                            let rhs3 = *(rhs.wrapping_add(depth + lane * 3) as *const S::RhsN);
                            acc3 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs3),
                                simd.simd_from_rhs(rhs3),
                                acc3,
                            );

                            let lhs4 = *(lhs.wrapping_add(depth + lane * 4) as *const S::LhsN);
                            let rhs4 = *(rhs.wrapping_add(depth + lane * 4) as *const S::RhsN);
                            acc4 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs4),
                                simd.simd_from_rhs(rhs4),
                                acc4,
                            );

                            let lhs5 = *(lhs.wrapping_add(depth + lane * 5) as *const S::LhsN);
                            let rhs5 = *(rhs.wrapping_add(depth + lane * 5) as *const S::RhsN);
                            acc5 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs5),
                                simd.simd_from_rhs(rhs5),
                                acc5,
                            );

                            let lhs6 = *(lhs.wrapping_add(depth + lane * 6) as *const S::LhsN);
                            let rhs6 = *(rhs.wrapping_add(depth + lane * 6) as *const S::RhsN);
                            acc6 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs6),
                                simd.simd_from_rhs(rhs6),
                                acc6,
                            );

                            let lhs7 = *(lhs.wrapping_add(depth + lane * 7) as *const S::LhsN);
                            let rhs7 = *(rhs.wrapping_add(depth + lane * 7) as *const S::RhsN);
                            acc7 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs7),
                                simd.simd_from_rhs(rhs7),
                                acc7,
                            );

                            depth += lane8;
                        }

                        let acc0 = simd.simd_add(acc0, acc1);
                        let acc2 = simd.simd_add(acc2, acc3);
                        let acc4 = simd.simd_add(acc4, acc5);
                        let acc6 = simd.simd_add(acc6, acc7);

                        let acc0 = simd.simd_add(acc0, acc2);
                        let acc4 = simd.simd_add(acc4, acc6);

                        let mut acc0 = simd.simd_add(acc0, acc4);

                        while depth < k_lane {
                            let lhs0 = *(lhs.wrapping_add(depth) as *const S::LhsN);
                            let rhs0 = *(rhs.wrapping_add(depth) as *const S::RhsN);
                            acc0 = simd.simd_mult_add(
                                simd.simd_from_lhs(lhs0),
                                simd.simd_from_rhs(rhs0),
                                acc0,
                            );

                            depth += lane;
                        }

                        let acc_ptr = &acc0 as *const _ as *const Acc;
                        let mut acc0 = *acc_ptr;
                        for x in 1..S::SIMD_WIDTH {
                            acc0 = simd.add(acc0, *acc_ptr.add(x));
                        }

                        while depth < k {
                            let lhs0 = *(lhs.wrapping_add(depth + 0));
                            let rhs0 = *(rhs.wrapping_add(depth + 0));

                            acc0 = simd.mult_add(simd.from_lhs(lhs0), simd.from_rhs(rhs0), acc0);

                            depth += 1;
                        }

                        if alpha.is_zero() {
                            let dst = dst.wrapping_offset(dst_rs * row as isize);
                            *dst = simd.into_dst(simd.mult(acc0, beta));
                        } else {
                            let dst = dst.wrapping_offset(dst_rs * row as isize);
                            *dst =
                                simd.into_dst(simd.add(
                                    simd.mult(acc0, beta),
                                    simd.mult(simd.from_dst(*dst), alpha),
                                ));
                        }
                    }
                }
            }
        }

        simd.vectorize(Impl {
            simd,
            dst,
            dst_rs,
            m,
            k,
            lhs,
            lhs_rs,
            rhs,
            alpha,
            beta,
        })
    }

    assert_eq!(lhs_cs, 1);
    assert_eq!(rhs_rs, 1);

    for x in 0..n {
        implementation(
            simd,
            dst.wrapping_offset(x as isize * dst_cs),
            dst_rs,
            m,
            k,
            lhs,
            lhs_rs,
            rhs.wrapping_offset(rhs_cs * x as isize),
            alpha,
            beta,
        );
    }
}

// the serial kernel to run on one task, with the same argument order as
// `mixed_gemv_colmajor` / `mixed_gemv_rowmajor`.
type SerialGemv<Lhs, Rhs, Dst, Acc, S> = unsafe fn(
    S,
    usize,
    usize,
    usize,
    *mut Dst,
    isize,
    isize,
    *const Lhs,
    isize,
    isize,
    *const Rhs,
    isize,
    isize,
    Acc,
    Acc,
);

/// splits the evenly sized `n_items` into `n_tasks` ranges, and returns the start of the
/// range owned by `tid`. `tid == n_tasks` returns `n_items`, so a range is
/// `(range_start(tid), range_start(tid + 1))`.
#[cfg(feature = "rayon")]
#[inline(always)]
fn range_start(tid: usize, n_tasks: usize, n_items: usize, granularity: usize) -> usize {
    if tid >= n_tasks {
        return n_items;
    }

    let n_granules = n_items.msrv_div_ceil(granularity);
    let base = n_granules / n_tasks;
    let rem = n_granules % n_tasks;

    let granule = if tid < rem {
        tid * (base + 1)
    } else {
        rem + tid * base
    };

    Ord::min(granule * granularity, n_items)
}

/// splits `m` (the output length), falling back to a split along `k` with a reduction pass
/// when `m` is too short to keep every thread busy.
///
/// `serial` must be one of `mixed_gemv_colmajor` / `mixed_gemv_rowmajor`, and the layout
/// preconditions it asserts must already hold. both variants offset identically: a task
/// owning rows `[r0, r1)` and depths `[d0, d1)` reads `lhs + r0*lhs_rs + d0*lhs_cs` and
/// `rhs + d0*rhs_rs`, and writes `dst + r0*dst_rs`.
#[inline(always)]
unsafe fn gemv_parallel<
    Lhs: Boilerplate + One + Zero,
    Rhs: Boilerplate + One + Zero,
    Dst: Boilerplate + One + Zero,
    Acc: Boilerplate + One + Zero,
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

    alpha: Acc,
    beta: Acc,

    parallelism: Parallelism,
    axis: SplitAxis,
    serial: SerialGemv<Lhs, Rhs, Dst, Acc, S>,
) {
    let n_threads = match parallelism {
        Parallelism::None => 1,
        #[cfg(feature = "rayon")]
        Parallelism::Rayon(n_threads) => {
            let total_work = m.saturating_mul(n).saturating_mul(k);
            if total_work < get_threading_threshold() {
                1
            } else if n_threads == 0 {
                rayon::current_num_threads()
            } else {
                n_threads
            }
        }
    };

    let _ = axis;
    if n_threads <= 1 {
        serial(
            simd, m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, alpha,
            beta,
        );
        return;
    }

    #[cfg(not(feature = "rayon"))]
    unreachable!();

    #[cfg(feature = "rayon")]
    {
        // keep each chunk a multiple of the simd width so every task keeps the vectorized
        // main loop, and at least a cacheline wide so no two tasks share a `dst` cacheline.
        let gran = Ord::max(
            S::SIMD_WIDTH,
            CACHELINE_ALIGN / Ord::max(1, core::mem::size_of::<Dst>()),
        );

        let pad = Ord::max(1, CACHELINE_ALIGN / Ord::max(1, core::mem::size_of::<Dst>()));
        let partial_stride = m.msrv_next_multiple_of(pad);
        let partial_bytes = partial_stride.saturating_mul(core::mem::size_of::<Dst>());

        let (n_row_tasks, n_depth_tasks) =
            split_tasks(axis, n_threads, m, n, k, gran, partial_bytes);

        let dst = Ptr(dst);
        let lhs = Ptr(lhs as *mut Lhs);
        let rhs = Ptr(rhs as *mut Rhs);

        let row_start = |tid: usize| range_start(tid, n_row_tasks, m, gran);

        if n_depth_tasks == 1 {
            par_for_each(n_row_tasks, |tid| {
                // bind the `Ptr` itself so the closure captures the `Send`/`Sync` wrapper
                // rather than the bare `*mut Rhs` field.
                let rhs = rhs;

                let r0 = row_start(tid);
                let r1 = row_start(tid + 1);
                if r1 > r0 {
                    serial(
                        simd,
                        r1 - r0,
                        n,
                        k,
                        dst.wrapping_offset(r0 as isize * dst_rs).0,
                        dst_cs,
                        dst_rs,
                        lhs.wrapping_offset(r0 as isize * lhs_rs).0,
                        lhs_cs,
                        lhs_rs,
                        rhs.0,
                        rhs_cs,
                        rhs_rs,
                        alpha,
                        beta,
                    );
                }
            });
            return;
        }

        // `dst[row] = alpha * dst[row] + beta * sum_d lhs[row, d] * rhs[d]` for both variants,
        // so each depth slice accumulates `sum_{d in slice} lhs[row, d] * rhs[d]` into its own
        // scratch column (alpha = 0, beta = 1) and a second pass reduces them.
        let depth_start = |tid: usize| range_start(tid, n_depth_tasks, k, 1);

        let mut mem = MemBuffer::new(StackReq::new_aligned::<Dst>(
            n_depth_tasks * partial_stride,
            CACHELINE_ALIGN,
        ));
        let (partial_storage, _) = DynStack::new(&mut mem)
            .make_aligned_uninit::<Dst>(n_depth_tasks * partial_stride, CACHELINE_ALIGN);
        let partial = Ptr(partial_storage.as_mut_ptr() as *mut Dst);

        par_for_each(n_row_tasks * n_depth_tasks, |tid| {
            let i = tid / n_depth_tasks;
            let j = tid % n_depth_tasks;

            let r0 = row_start(i);
            let r1 = row_start(i + 1);
            let d0 = depth_start(j);
            let d1 = depth_start(j + 1);

            if r1 > r0 && d1 > d0 {
                serial(
                    simd,
                    r1 - r0,
                    1,
                    d1 - d0,
                    partial.wrapping_add(j * partial_stride + r0).0,
                    partial_stride as isize,
                    1,
                    lhs.wrapping_offset(r0 as isize * lhs_rs + d0 as isize * lhs_cs).0,
                    lhs_cs,
                    lhs_rs,
                    rhs.wrapping_offset(d0 as isize * rhs_rs).0,
                    rhs_cs,
                    rhs_rs,
                    Acc::zero(),
                    Acc::one(),
                );
            }
        });

        // `par_for_each` joins, so every partial column is complete by now.
        par_for_each(n_row_tasks, |tid| {
            let r0 = row_start(tid);
            let r1 = row_start(tid + 1);
            if r1 > r0 {
                reduce_partials(
                    simd,
                    r0,
                    r1,
                    dst,
                    dst_rs,
                    partial,
                    partial_stride,
                    n_depth_tasks,
                    alpha,
                    beta,
                );
            }
        });
    }
}

/// picks how many row tasks and depth tasks to carve the work into.
///
/// a depth split needs one scratch column of `Dst` per slice plus a reduction pass, so it is
/// bounded by both memory and a minimum useful slice of `k`. it also reassociates the sum over
/// `k` and the reduction assumes one output column, so it requires `n == 1` (true at every
/// call site).
#[cfg(feature = "rayon")]
#[inline(always)]
fn split_tasks(
    axis: SplitAxis,
    n_threads: usize,
    m: usize,
    n: usize,
    k: usize,
    gran: usize,
    partial_bytes: usize,
) -> (usize, usize) {
    let max_depth_tasks = if n == 1 {
        let by_mem = if partial_bytes == 0 {
            n_threads
        } else {
            MAX_PARTIAL_BYTES / partial_bytes
        };
        Ord::min(Ord::min(n_threads, k / MIN_DEPTH_PER_TASK), by_mem)
    } else {
        1
    };
    let max_row_tasks = Ord::max(1, Ord::min(n_threads, m.msrv_div_ceil(gran)));

    // colmajor walks `lhs` down the `k` loop one whole column at a time, so a row split leaves
    // every task reading only `m / n_threads` elements out of each column, a full column apart.
    // once those runs get short, the lost DRAM locality costs more than the reduction does, so
    // split the depth instead and hand each task one contiguous slab. rowmajor rows are
    // independent dot products, so its row split needs no scratch and no reduction at all;
    // there, depth is only a fallback for an `m` too short to fill the threads.
    let prefer_depth = match axis {
        SplitAxis::Depth => partial_bytes < n_threads.saturating_mul(MIN_ROW_RUN_BYTES),
        SplitAxis::Rows => false,
    };

    if prefer_depth && max_depth_tasks > 1 {
        (
            Ord::max(1, Ord::min(n_threads / max_depth_tasks, max_row_tasks)),
            max_depth_tasks,
        )
    } else if max_row_tasks < n_threads && max_depth_tasks > 1 {
        (
            max_row_tasks,
            Ord::max(1, Ord::min(n_threads / max_row_tasks, max_depth_tasks)),
        )
    } else {
        (max_row_tasks, 1)
    }
}

/// smallest depth slice worth handing to its own task.
#[cfg(feature = "rayon")]
const MIN_DEPTH_PER_TASK: usize = 128;

/// below this many bytes per row chunk, a colmajor row split reads too short a run from each
/// column to stream well, and a depth split wins despite needing a reduction.
#[cfg(feature = "rayon")]
const MIN_ROW_RUN_BYTES: usize = 64 * 1024;

/// ceiling on the scratch a depth split may allocate. large outputs hit this and fall back to
/// the row split, which is what they want anyway: their row chunks are already long.
#[cfg(feature = "rayon")]
const MAX_PARTIAL_BYTES: usize = 4 * 1024 * 1024;

/// which axis `gemv_parallel` should split first.
#[derive(Copy, Clone, PartialEq, Eq)]
enum SplitAxis {
    /// rowmajor: independent dot products, one store per row.
    Rows,
    /// colmajor: an axpy over `k` accumulating into `dst`.
    Depth,
}

/// `dst[row] = alpha * dst[row] + beta * sum_j partial[j][row]` for `row in r0..r1`,
/// matching the epilogue of both serial kernels.
#[cfg(feature = "rayon")]
#[inline(always)]
unsafe fn reduce_partials<
    Lhs: Boilerplate + One + Zero,
    Rhs: Boilerplate + One + Zero,
    Dst: Boilerplate + One + Zero,
    Acc: Boilerplate + One + Zero,
    S: MixedSimd<Lhs, Rhs, Dst, Acc>,
>(
    simd: S,
    r0: usize,
    r1: usize,
    dst: Ptr<Dst>,
    dst_rs: isize,
    partial: Ptr<Dst>,
    partial_stride: usize,
    n_depth_tasks: usize,
    alpha: Acc,
    beta: Acc,
) {
    #[allow(dead_code)]
    struct Impl<Lhs, Rhs, Dst, Acc, S> {
        simd: S,
        r0: usize,
        r1: usize,
        dst: Ptr<Dst>,
        dst_rs: isize,
        partial: Ptr<Dst>,
        partial_stride: usize,
        n_depth_tasks: usize,
        alpha: Acc,
        beta: Acc,
        __marker: core::marker::PhantomData<(Lhs, Rhs)>,
    }

    impl<
            Lhs: Boilerplate + One + Zero,
            Rhs: Boilerplate + One + Zero,
            Dst: Boilerplate + One + Zero,
            Acc: Boilerplate + One + Zero,
            S: MixedSimd<Lhs, Rhs, Dst, Acc>,
        > pulp::NullaryFnOnce for Impl<Lhs, Rhs, Dst, Acc, S>
    {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            unsafe {
                let Self {
                    simd,
                    r0,
                    r1,
                    dst,
                    dst_rs,
                    partial,
                    partial_stride,
                    n_depth_tasks,
                    alpha,
                    beta,
                    __marker: _,
                } = self;

                for row in r0..r1 {
                    let mut acc = Acc::zero();
                    for j in 0..n_depth_tasks {
                        acc = simd.add(
                            acc,
                            simd.from_dst(*partial.wrapping_add(j * partial_stride + row).0),
                        );
                    }

                    let dst = dst.wrapping_offset(row as isize * dst_rs).0;
                    *dst = if alpha.is_zero() {
                        simd.into_dst(simd.mult(acc, beta))
                    } else {
                        simd.into_dst(simd.add(
                            simd.mult(acc, beta),
                            simd.mult(simd.from_dst(*dst), alpha),
                        ))
                    };
                }
            }
        }
    }

    simd.vectorize(Impl::<Lhs, Rhs, Dst, Acc, S> {
        simd,
        r0,
        r1,
        dst,
        dst_rs,
        partial,
        partial_stride,
        n_depth_tasks,
        alpha,
        beta,
        __marker: core::marker::PhantomData,
    })
}

/// `mixed_gemv_colmajor`, splitting the work across `parallelism`.
///
/// same preconditions as `mixed_gemv_colmajor`: `lhs_rs == 1` and `dst_rs == 1`.
#[inline(always)]
pub unsafe fn mixed_gemv_colmajor_parallel<
    Lhs: Boilerplate + One + Zero,
    Rhs: Boilerplate + One + Zero,
    Dst: Boilerplate + One + Zero,
    Acc: Boilerplate + One + Zero,
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

    alpha: Acc,
    beta: Acc,

    parallelism: Parallelism,
) {
    gemv_parallel(
        simd,
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        parallelism,
        SplitAxis::Depth,
        mixed_gemv_colmajor::<Lhs, Rhs, Dst, Acc, S>,
    )
}

/// `mixed_gemv_rowmajor`, splitting the work across `parallelism`.
///
/// same preconditions as `mixed_gemv_rowmajor`: `lhs_cs == 1` and `rhs_rs == 1`.
#[inline(always)]
pub unsafe fn mixed_gemv_rowmajor_parallel<
    Lhs: Boilerplate + One + Zero,
    Rhs: Boilerplate + One + Zero,
    Dst: Boilerplate + One + Zero,
    Acc: Boilerplate + One + Zero,
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

    alpha: Acc,
    beta: Acc,

    parallelism: Parallelism,
) {
    gemv_parallel(
        simd,
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        parallelism,
        SplitAxis::Rows,
        mixed_gemv_rowmajor::<Lhs, Rhs, Dst, Acc, S>,
    )
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use super::*;
    use crate::gemm::set_threading_threshold;
    use crate::simd::Scalar;
    use alloc::vec::Vec;

    // deterministic data, so a failing case is reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        }
    }

    fn data(len: usize, seed: u64) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..len).map(|_| lcg.next()).collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            // the m-split is bit-exact; the k-split reassociates the sum over k.
            let tol = 1e-4 * f32::max(1.0, e.abs());
            assert!(
                (a - e).abs() <= tol,
                "{what}: mismatch at {i}: {a} vs {e}",
                what = what,
                i = i,
                a = a,
                e = e,
            );
        }
    }

    // (m, k) covering: pure m-split, m not a multiple of the chunk granularity, short m
    // with long k (forces the k-split), and the degenerate m == 0 / k == 0 cases.
    const SHAPES: &[(usize, usize)] = &[
        (0, 100),
        // m == 0 with a long depth: the k-split path with a zero-sized scratch buffer.
        (0, 4096),
        (100, 0),
        (1, 4096),
        (5, 1000),
        (33, 4096),
        (64, 4096),
        (63, 517),
        (1000, 65),
        (1024, 64),
        (2048, 512),
    ];

    const N_THREADS: &[usize] = &[1, 2, 3, 5, 8, 16, 128];
    const ALPHAS: &[f32] = &[0.0, 1.0, 2.5];
    const BETAS: &[f32] = &[0.0, 1.0, 2.5];

    #[test]
    fn gemv_colmajor_parallel_matches_serial() {
        // make every shape below cross the threading threshold.
        set_threading_threshold(0);

        for &(m, k) in SHAPES {
            // lhs is colmajor (lhs_rs == 1), dst is colmajor (dst_rs == 1).
            let lhs = data(m * k, 0x1234);
            let rhs = data(k, 0x5678);
            let dst_init = data(m, 0x9abc);

            for &alpha in ALPHAS {
                for &beta in BETAS {
                    let mut expected = dst_init.clone();
                    unsafe {
                        mixed_gemv_colmajor(
                            Scalar,
                            m,
                            1,
                            k,
                            expected.as_mut_ptr(),
                            m as isize,
                            1,
                            lhs.as_ptr(),
                            m as isize,
                            1,
                            rhs.as_ptr(),
                            k as isize,
                            1,
                            alpha,
                            beta,
                        );
                    }

                    for &n_threads in N_THREADS {
                        let mut actual = dst_init.clone();
                        unsafe {
                            mixed_gemv_colmajor_parallel(
                                Scalar,
                                m,
                                1,
                                k,
                                actual.as_mut_ptr(),
                                m as isize,
                                1,
                                lhs.as_ptr(),
                                m as isize,
                                1,
                                rhs.as_ptr(),
                                k as isize,
                                1,
                                alpha,
                                beta,
                                Parallelism::Rayon(n_threads),
                            );
                        }

                        assert_close(
                            &actual,
                            &expected,
                            &alloc::format!(
                                "colmajor m={m} k={k} alpha={alpha} beta={beta} threads={n_threads}"
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gemv_rowmajor_parallel_matches_serial() {
        set_threading_threshold(0);

        for &(m, k) in SHAPES {
            // lhs is rowmajor (lhs_cs == 1), rhs is colmajor (rhs_rs == 1).
            let lhs = data(m * k, 0x1234);
            let rhs = data(k, 0x5678);

            // also cover a strided dst, which the rowmajor kernel allows.
            for &dst_rs in &[1isize, 3] {
                let dst_init = data(m * dst_rs as usize, 0x9abc);

                for &alpha in ALPHAS {
                    for &beta in BETAS {
                        let mut expected = dst_init.clone();
                        unsafe {
                            mixed_gemv_rowmajor(
                                Scalar,
                                m,
                                1,
                                k,
                                expected.as_mut_ptr(),
                                (m * dst_rs as usize) as isize,
                                dst_rs,
                                lhs.as_ptr(),
                                1,
                                k as isize,
                                rhs.as_ptr(),
                                k as isize,
                                1,
                                alpha,
                                beta,
                            );
                        }

                        for &n_threads in N_THREADS {
                            let mut actual = dst_init.clone();
                            unsafe {
                                mixed_gemv_rowmajor_parallel(
                                    Scalar,
                                    m,
                                    1,
                                    k,
                                    actual.as_mut_ptr(),
                                    (m * dst_rs as usize) as isize,
                                    dst_rs,
                                    lhs.as_ptr(),
                                    1,
                                    k as isize,
                                    rhs.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    Parallelism::Rayon(n_threads),
                                );
                            }

                            assert_close(
                                &actual,
                                &expected,
                                &alloc::format!(
                                    "rowmajor m={m} k={k} dst_rs={dst_rs} alpha={alpha} beta={beta} threads={n_threads}"
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    // exercises the real decision function, so the strategy per variant stays pinned down.
    // sizes below are f16 llm-decode shapes on an 8-thread machine.
    #[test]
    fn split_strategy() {
        let gran = Ord::max(8, crate::gemm::CACHELINE_ALIGN / 2);
        let bytes = |m: usize, sz: usize| {
            let pad = Ord::max(1, crate::gemm::CACHELINE_ALIGN / sz);
            m.msrv_next_multiple_of(pad) * sz
        };

        // colmajor, 14336x4096 f16: a row split would leave each task ~3.5 KiB per column, so
        // this must take the depth split - one contiguous slab per task, no row split at all.
        assert_eq!(
            split_tasks(SplitAxis::Depth, 8, 14336, 1, 4096, gran, bytes(14336, 2)),
            (1, 8),
        );

        // rowmajor, same shape: rows are independent dot products, so split rows only.
        assert_eq!(
            split_tasks(SplitAxis::Rows, 8, 14336, 1, 4096, gran, bytes(14336, 2)),
            (8, 1),
        );

        // rowmajor with an `m` too short to fill the threads falls back to a depth split.
        let (rows, depth) = split_tasks(SplitAxis::Rows, 8, 128, 1, 16384, gran, bytes(128, 2));
        assert!(depth > 1, "expected a depth fallback, got {rows}x{depth}");
        assert!(rows * depth <= 8, "oversubscribed: {rows}x{depth}");

        // a long output already has long row chunks, so colmajor stays on the row split rather
        // than allocating a scratch column per depth slice.
        assert_eq!(
            split_tasks(SplitAxis::Depth, 8, 1 << 22, 1, 4096, gran, bytes(1 << 22, 2)),
            (8, 1),
        );

        // `n > 1` can never take a depth split: the reduction assumes one output column.
        // and a short `k` is never worth slicing.
        for axis in [SplitAxis::Rows, SplitAxis::Depth] {
            assert_eq!(split_tasks(axis, 8, 128, 2, 16384, gran, bytes(128, 2)).1, 1);
            assert_eq!(split_tasks(axis, 8, 64, 1, 100, gran, bytes(64, 2)).1, 1);
        }
    }
}
