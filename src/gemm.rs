use crate::{
    cache::{kernel_params, KernelParams},
    microkernel,
    pack_operands::{pack_lhs, pack_rhs},
    x86_feature_detected, Ptr,
};
use aligned_vec::CACHELINE_ALIGN;
use core::any::TypeId;
use dyn_stack::{DynStack, ReborrowMut, SizeOverflow, StackReq};
use num_traits::{One, Zero};
use seq_macro::seq;
use strength_reduce::StrengthReducedUsize;

#[inline(always)]
fn div_ceil(a: usize, b: usize) -> usize {
    let div = a / b;
    let rem = a % b;
    if rem == 0 {
        div
    } else {
        div + 1
    }
}

#[inline(always)]
unsafe fn gemm_basic_generic<
    T: Copy + Zero + One + Send + Sync + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
    const N: usize,
    const MR: usize,
    const NR: usize,
>(
    m: usize,
    n: usize,
    k: usize,
    mut dst: *mut T,
    mut dst_cs: isize,
    mut dst_rs: isize,
    read_dst: bool,
    mut lhs: *const T,
    lhs_cs: isize,
    mut lhs_rs: isize,
    mut rhs: *const T,
    mut rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    n_threads: usize,
    dispatcher: impl Copy
        + Send
        + Sync
        + Fn(
            usize,
            usize,
        ) -> unsafe fn(
            usize,
            usize,
            usize,
            Ptr<T>,
            Ptr<T>,
            Ptr<T>,
            isize,
            isize,
            isize,
            isize,
            isize,
            T,
            T,
            bool,
        ),
    mut stack: DynStack<'_>,
) {
    if m == 0 || n == 0 {
        return;
    }

    if dst_rs < 0 {
        dst = dst.wrapping_offset((m - 1) as isize * dst_rs);
        dst_rs = -dst_rs;
        lhs = lhs.wrapping_offset((m - 1) as isize * lhs_rs);
        lhs_rs = -lhs_rs;
    }

    if dst_cs < 0 {
        dst = dst.wrapping_offset((n - 1) as isize * dst_cs);
        dst_cs = -dst_cs;
        rhs = rhs.wrapping_offset((n - 1) as isize * rhs_cs);
        rhs_cs = -rhs_cs;
    }

    if k == 0 {
        if read_dst {
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
    }

    // n is greater than or equal to m, no need to handle gemv case directly
    // gevm case
    if m <= 4 {
        macro_rules! do_work {
            ($m: tt) => {
                if rhs_cs == 1 && dst_cs == 1 {
                    for depth in 0..k {
                        if depth == 0 {
                            if read_dst {
                                seq!(ROW in 0..$m {
                                    let lhs~ROW = *lhs
                                        .wrapping_offset(ROW as isize * lhs_rs)
                                        .wrapping_offset(depth as isize * lhs_cs);
                                });
                                for col in 0..n {
                                    let rhs = *rhs
                                        .wrapping_offset(depth as isize * rhs_rs)
                                        .wrapping_offset(col as isize);

                                    seq!(ROW in 0..$m {
                                        {
                                            let dst = dst
                                                .wrapping_offset(ROW as isize * dst_rs)
                                                .wrapping_offset(col as isize);
                                            *dst = alpha * *dst + beta * lhs~ROW * rhs;
                                        }
                                    });
                                }
                            } else {
                                seq!(ROW in 0..$m {
                                    let lhs~ROW = *lhs
                                        .wrapping_offset(ROW as isize * lhs_rs)
                                        .wrapping_offset(depth as isize * lhs_cs);
                                });
                                for col in 0..n {
                                    let rhs = *rhs
                                        .wrapping_offset(depth as isize * rhs_rs)
                                        .wrapping_offset(col as isize);

                                    seq!(ROW in 0..$m {
                                        {
                                            let dst = dst
                                                .wrapping_offset(ROW as isize * dst_rs)
                                                .wrapping_offset(col as isize);
                                            *dst = beta * lhs~ROW * rhs;
                                        }
                                    });
                                }
                            }
                        } else {
                            seq!(ROW in 0..$m {
                                let lhs~ROW = *lhs
                                    .wrapping_offset(ROW as isize * lhs_rs)
                                    .wrapping_offset(depth as isize * lhs_cs);
                            });
                            for col in 0..n {
                                let rhs = *rhs
                                    .wrapping_offset(depth as isize * rhs_rs)
                                    .wrapping_offset(col as isize);

                                seq!(ROW in 0..$m {
                                    {
                                        let dst = dst
                                            .wrapping_offset(ROW as isize * dst_rs)
                                            .wrapping_offset(col as isize);
                                        *dst = *dst + beta * lhs~ROW * rhs;
                                    }
                                });
                            }
                        }
                    }
                } else {
                    for depth in 0..k {
                        if depth == 0 {
                            if read_dst {
                                seq!(ROW in 0..$m {
                                    let lhs~ROW = *lhs
                                        .wrapping_offset(ROW as isize * lhs_rs)
                                        .wrapping_offset(depth as isize * lhs_cs);
                                });
                                for col in 0..n {
                                    let rhs = *rhs
                                        .wrapping_offset(depth as isize * rhs_rs)
                                        .wrapping_offset(col as isize * rhs_cs);

                                    seq!(ROW in 0..$m {
                                        {
                                            let dst = dst
                                                .wrapping_offset(ROW as isize * dst_rs)
                                                .wrapping_offset(col as isize * dst_cs);
                                            *dst = alpha * *dst + beta * lhs~ROW * rhs;
                                        }
                                    });
                                }
                            } else {
                                seq!(ROW in 0..$m {
                                    let lhs~ROW = *lhs
                                        .wrapping_offset(ROW as isize * lhs_rs)
                                        .wrapping_offset(depth as isize * lhs_cs);
                                });
                                for col in 0..n {
                                    let rhs = *rhs
                                        .wrapping_offset(depth as isize * rhs_rs)
                                        .wrapping_offset(col as isize * rhs_cs);

                                    seq!(ROW in 0..$m {
                                        {
                                            let dst = dst
                                                .wrapping_offset(ROW as isize * dst_rs)
                                                .wrapping_offset(col as isize * dst_cs);
                                            *dst = beta * lhs~ROW * rhs;
                                        }
                                    });
                                }
                            }
                        } else {
                            seq!(ROW in 0..$m {
                                let lhs~ROW = *lhs
                                    .wrapping_offset(ROW as isize * lhs_rs)
                                    .wrapping_offset(depth as isize * lhs_cs);
                            });
                            for col in 0..n {
                                let rhs = *rhs
                                    .wrapping_offset(depth as isize * rhs_rs)
                                    .wrapping_offset(col as isize * rhs_cs);

                                seq!(ROW in 0..$m {
                                    {
                                        let dst = dst
                                            .wrapping_offset(ROW as isize * dst_rs)
                                            .wrapping_offset(col as isize * dst_cs);
                                        *dst = *dst + beta * lhs~ROW * rhs;
                                    }
                                });
                            }
                        }
                    }
                }
            };
        }
        match m {
            1 => do_work!(1),
            2 => do_work!(2),
            3 => do_work!(3),
            4 => do_work!(4),
            _ => (),
        }
        return;
    }

    let KernelParams { kc, mc, nc } = kernel_params(m, n, k, MR, NR, core::mem::size_of::<T>());

    // use a single thread for small workloads
    let n_threads = if mc * nc * kc <= 48 * 48 * 256 {
        1
    } else {
        n_threads
    };
    let n_threads = StrengthReducedUsize::new(n_threads);

    let simd_align = CACHELINE_ALIGN;
    let simd_stride = CACHELINE_ALIGN / core::mem::size_of::<T>();

    let packed_rhs_stride = div_ceil(kc * NR, simd_stride) * simd_stride;
    let packed_lhs_stride = div_ceil(kc * MR, simd_stride) * simd_stride;

    let (mut packed_rhs_storage, mut stack) = stack
        .rb_mut()
        .make_aligned_uninit::<T>(packed_rhs_stride * (nc / NR), simd_align);

    let packed_rhs = packed_rhs_storage.as_mut_ptr() as *mut T;

    let (mut packed_lhs_storage, _) = stack
        .rb_mut()
        .make_aligned_uninit::<T>(n_threads.get() * packed_lhs_stride * (mc / MR), simd_align);

    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);
    let packed_rhs = Ptr(packed_rhs);
    let do_pack_rhs = m > MR;

    let mut col_outer = 0;
    while col_outer != n {
        let n_chunk = nc.min(n - col_outer);

        let mut depth_outer = 0;
        while depth_outer != k {
            let k_chunk = kc.min(k - depth_outer);

            if do_pack_rhs {
                pack_rhs::<T, NR>(
                    n_chunk,
                    k_chunk,
                    packed_rhs,
                    rhs.wrapping_offset(
                        depth_outer as isize * rhs_rs + col_outer as isize * rhs_cs,
                    ),
                    rhs_cs,
                    rhs_rs,
                    packed_rhs_stride,
                );
            }

            let packed_lhs = Ptr(packed_lhs_storage.as_mut_ptr() as *mut T);
            let n_col_mini_chunks = (n_chunk + (NR - 1)) / NR;

            let mut n_jobs = 0;
            let mut row_outer = 0;
            while row_outer != m {
                let m_chunk = mc.min(m - row_outer);
                let n_row_mini_chunks = (m_chunk + (MR - 1)) / MR;
                n_jobs += n_col_mini_chunks * n_row_mini_chunks;
                row_outer += m_chunk;
            }

            let func = move |tid| {
                let packed_lhs = packed_lhs
                    .wrapping_add(tid * packed_lhs_stride * (mc / MR).min(div_ceil(m, MR)));

                let min_jobs_per_thread = n_jobs / n_threads;
                let rem = n_jobs - n_threads.get() * min_jobs_per_thread;

                // thread `tid` takes min_jobs_per_thread or min_jobs_per_thread + 1
                let (job_start, job_end) = if tid < rem {
                    let start = tid * (min_jobs_per_thread + 1);
                    (start, start + min_jobs_per_thread + 1)
                } else {
                    // start = rem * (min_jobs_per_thread + 1) + (tid - rem) * min_jobs_per_thread;
                    let start = tid * min_jobs_per_thread + rem;
                    (start, start + min_jobs_per_thread)
                };

                let mut row_outer = 0;
                let mut job_id = 0;
                while row_outer != m {
                    let m_chunk = mc.min(m - row_outer);
                    let n_row_mini_chunks = (m_chunk + (MR - 1)) / MR;

                    let n_mini_jobs = n_col_mini_chunks * n_row_mini_chunks;

                    if job_id >= job_end {
                        return;
                    }
                    if job_id + n_mini_jobs < job_start {
                        row_outer += m_chunk;
                        job_id += n_mini_jobs;
                        continue;
                    }

                    pack_lhs::<T, MR>(
                        m_chunk,
                        k_chunk,
                        packed_lhs,
                        lhs.wrapping_offset(
                            row_outer as isize * lhs_rs + depth_outer as isize * lhs_cs,
                        ),
                        lhs_cs,
                        lhs_rs,
                        packed_lhs_stride,
                    );

                    let mut j = 0;
                    while j < n_col_mini_chunks {
                        let mut i = 0;
                        while i < n_row_mini_chunks {
                            let col_inner = NR * j;
                            let n_chunk_inner = NR.min(n_chunk - col_inner);

                            let row_inner = MR * i;
                            let m_chunk_inner = MR.min(m_chunk - row_inner);

                            if job_id < job_start || job_id >= job_end {
                                job_id += 1;
                                i += 1;
                                continue;
                            }
                            job_id += 1;

                            let dst = dst.wrapping_offset(
                                (row_outer + row_inner) as isize * dst_rs
                                    + (col_outer + col_inner) as isize * dst_cs,
                            );

                            let dispatcher = dispatcher;
                            let func = dispatcher((m_chunk_inner + (N - 1)) / N, n_chunk_inner);

                            func(
                                m_chunk_inner,
                                n_chunk_inner,
                                k_chunk,
                                dst,
                                packed_lhs.wrapping_add(i * packed_lhs_stride),
                                if do_pack_rhs {
                                    packed_rhs.wrapping_add(j * packed_rhs_stride)
                                } else {
                                    rhs.wrapping_offset(
                                        depth_outer as isize * rhs_rs
                                            + (col_outer + col_inner) as isize * rhs_cs,
                                    )
                                },
                                dst_cs,
                                dst_rs,
                                MR as isize,
                                if do_pack_rhs { NR as isize } else { rhs_rs },
                                if do_pack_rhs { 1 } else { rhs_cs },
                                if depth_outer == 0 { alpha } else { T::one() },
                                beta,
                                if depth_outer == 0 { read_dst } else { true },
                            );
                            i += 1;
                        }
                        j += 1;
                    }

                    row_outer += m_chunk;
                }
            };

            if n_threads.get() <= 1 {
                func(0);
            } else {
                use rayon::prelude::*;
                (0..n_threads.get()).into_par_iter().for_each(func);
            }
            depth_outer += k_chunk;
        }
        col_outer += n_chunk;
    }
}

#[inline(always)]
fn gemm_basic_req_generic<T>(
    mr: usize,
    nr: usize,
    m: usize,
    n: usize,
    k: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let KernelParams { kc, mc, nc } = kernel_params(m, n, k, mr, nr, core::mem::size_of::<T>());
    let simd_align = CACHELINE_ALIGN;
    let simd_stride = CACHELINE_ALIGN / core::mem::size_of::<T>();
    let packed_rhs_stride = div_ceil(kc * nr, simd_stride) * simd_stride;
    let packed_lhs_stride = div_ceil(kc * mr, simd_stride) * simd_stride;

    StackReq::try_new_aligned::<T>(packed_rhs_stride * (nc / nr), simd_align)?.try_and(
        StackReq::try_new_aligned::<T>(max_n_threads * packed_lhs_stride * (mc / mr), simd_align)?,
    )
}

macro_rules! gemm_def {
    ($ty: tt, $multiplier: expr) => {
        use super::*;
        type T = $ty;

        type GemmTy = (
            unsafe fn(
                usize,
                usize,
                usize,
                *mut T,
                isize,
                isize,
                bool,
                *const T,
                isize,
                isize,
                *const T,
                isize,
                isize,
                T,
                T,
                usize,
                DynStack<'_>,
            ),
            fn(usize, usize, usize, usize) -> Result<StackReq, SizeOverflow>,
        );

        fn init_gemm_fn() -> GemmTy {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if x86_feature_detected!("avx512f") {
                    return (avx512f::gemm_basic, avx512f::gemm_req);
                }
                if x86_feature_detected!("fma") {
                    (fma::gemm_basic, fma::gemm_req)
                } else if x86_feature_detected!("avx") {
                    (avx::gemm_basic, avx::gemm_req)
                } else if x86_feature_detected!("sse") {
                    (sse::gemm_basic, sse::gemm_req)
                } else {
                    (scalar::gemm_basic, scalar::gemm_req)
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                (scalar::gemm_basic, scalar::gemm_req)
            }
        }

        lazy_static::lazy_static! {
            static ref GEMM: GemmTy = init_gemm_fn();
        }

        pub fn gemm_req(
            m: usize,
            n: usize,
            k: usize,
            max_n_threads: usize,
        ) -> Result<StackReq, SizeOverflow> {
            if m > n {
                (GEMM.1)(n, m, k, max_n_threads)
            } else {
                (GEMM.1)(m, n, k, max_n_threads)
            }
        }

        #[inline]
        pub unsafe fn gemm_basic(
            m: usize,
            n: usize,
            k: usize,
            dst: *mut T,
            dst_cs: isize,
            dst_rs: isize,
            read_dst: bool,
            lhs: *const T,
            lhs_cs: isize,
            lhs_rs: isize,
            rhs: *const T,
            rhs_cs: isize,
            rhs_rs: isize,
            alpha: T,
            beta: T,
            n_threads: usize,
            stack: DynStack<'_>,
        ) {
            if m > n {
                (GEMM.0)(
                    n, m, k, dst, dst_rs, dst_cs, read_dst, rhs, rhs_rs, rhs_cs, lhs, lhs_rs,
                    lhs_cs, alpha, beta, n_threads, stack,
                )
            } else {
                (GEMM.0)(
                    m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                    rhs_rs, alpha, beta, n_threads, stack,
                )
            }
        }

        mod scalar {
            use super::*;
            const N: usize = 1;
            const MR: usize = 2 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                m: usize,
                n: usize,
                k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(MR, NR, m, n, k, max_n_threads)
            }

            #[inline(never)]
            pub unsafe fn gemm_basic(
                m: usize,
                n: usize,
                k: usize,
                dst: *mut T,
                dst_cs: isize,
                dst_rs: isize,
                read_dst: bool,
                lhs: *const T,
                lhs_cs: isize,
                lhs_rs: isize,
                rhs: *const T,
                rhs_cs: isize,
                rhs_rs: isize,
                alpha: T,
                beta: T,
                n_threads: usize,
                stack: DynStack<'_>,
            ) {
                use microkernel::scalar::$ty::*;
                gemm_basic_generic::<T, N, MR, NR>(
                    m,
                    n,
                    k,
                    dst,
                    dst_cs,
                    dst_rs,
                    read_dst,
                    lhs,
                    lhs_cs,
                    lhs_rs,
                    rhs,
                    rhs_cs,
                    rhs_rs,
                    alpha,
                    beta,
                    n_threads,
                    |mr_div_n, nr| match (mr_div_n, nr) {
                        (1, 1) => x1x1,
                        (1, 2) => x1x2,
                        (1, 3) => x1x3,
                        (1, 4) => x1x4,

                        (2, 1) => x2x1,
                        (2, 2) => x2x2,
                        (2, 3) => x2x3,
                        (2, 4) => x2x4,

                        _ => unreachable!(),
                    },
                    stack,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod sse {
            use super::*;
            const N: usize = 2 * $multiplier;
            const MR: usize = 2 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                m: usize,
                n: usize,
                k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(MR, NR, m, n, k, max_n_threads)
            }

            #[target_feature(enable = "sse")]
            #[inline(never)]
            pub unsafe fn gemm_basic(
                m: usize,
                n: usize,
                k: usize,
                dst: *mut T,
                dst_cs: isize,
                dst_rs: isize,
                read_dst: bool,
                lhs: *const T,
                lhs_cs: isize,
                lhs_rs: isize,
                rhs: *const T,
                rhs_cs: isize,
                rhs_rs: isize,
                alpha: T,
                beta: T,
                n_threads: usize,
                stack: DynStack<'_>,
            ) {
                use microkernel::sse::$ty::*;
                gemm_basic_generic::<T, N, MR, NR>(
                    m,
                    n,
                    k,
                    dst,
                    dst_cs,
                    dst_rs,
                    read_dst,
                    lhs,
                    lhs_cs,
                    lhs_rs,
                    rhs,
                    rhs_cs,
                    rhs_rs,
                    alpha,
                    beta,
                    n_threads,
                    |mr_div_n, nr| match (mr_div_n, nr) {
                        (1, 1) => x1x1,
                        (1, 2) => x1x2,
                        (1, 3) => x1x3,
                        (1, 4) => x1x4,

                        (2, 1) => x2x1,
                        (2, 2) => x2x2,
                        (2, 3) => x2x3,
                        (2, 4) => x2x4,

                        _ => unreachable!(),
                    },
                    stack,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod avx {
            use super::*;
            const N: usize = 4 * $multiplier;
            const MR: usize = 2 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                m: usize,
                n: usize,
                k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(MR, NR, m, n, k, max_n_threads)
            }

            #[target_feature(enable = "avx")]
            #[inline(never)]
            pub unsafe fn gemm_basic(
                m: usize,
                n: usize,
                k: usize,
                dst: *mut T,
                dst_cs: isize,
                dst_rs: isize,
                read_dst: bool,
                lhs: *const T,
                lhs_cs: isize,
                lhs_rs: isize,
                rhs: *const T,
                rhs_cs: isize,
                rhs_rs: isize,
                alpha: T,
                beta: T,
                n_threads: usize,
                stack: DynStack<'_>,
            ) {
                use microkernel::avx::$ty::*;
                gemm_basic_generic::<T, N, MR, NR>(
                    m,
                    n,
                    k,
                    dst,
                    dst_cs,
                    dst_rs,
                    read_dst,
                    lhs,
                    lhs_cs,
                    lhs_rs,
                    rhs,
                    rhs_cs,
                    rhs_rs,
                    alpha,
                    beta,
                    n_threads,
                    |mr_div_n, nr| match (mr_div_n, nr) {
                        (1, 1) => x1x1,
                        (1, 2) => x1x2,
                        (1, 3) => x1x3,
                        (1, 4) => x1x4,

                        (2, 1) => x2x1,
                        (2, 2) => x2x2,
                        (2, 3) => x2x3,
                        (2, 4) => x2x4,

                        _ => unreachable!(),
                    },
                    stack,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod fma {
            use super::*;
            const N: usize = 4 * $multiplier;
            const MR: usize = 3 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                m: usize,
                n: usize,
                k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(MR, NR, m, n, k, max_n_threads)
            }

            #[target_feature(enable = "fma")]
            #[inline(never)]
            pub unsafe fn gemm_basic(
                m: usize,
                n: usize,
                k: usize,
                dst: *mut T,
                dst_cs: isize,
                dst_rs: isize,
                read_dst: bool,
                lhs: *const T,
                lhs_cs: isize,
                lhs_rs: isize,
                rhs: *const T,
                rhs_cs: isize,
                rhs_rs: isize,
                alpha: T,
                beta: T,
                n_threads: usize,
                stack: DynStack<'_>,
            ) {
                use microkernel::fma::$ty::*;
                gemm_basic_generic::<T, N, MR, NR>(
                    m,
                    n,
                    k,
                    dst,
                    dst_cs,
                    dst_rs,
                    read_dst,
                    lhs,
                    lhs_cs,
                    lhs_rs,
                    rhs,
                    rhs_cs,
                    rhs_rs,
                    alpha,
                    beta,
                    n_threads,
                    |mr_div_n, nr| match (mr_div_n, nr) {
                        (1, 1) => x1x1,
                        (1, 2) => x1x2,
                        (1, 3) => x1x3,
                        (1, 4) => x1x4,

                        (2, 1) => x2x1,
                        (2, 2) => x2x2,
                        (2, 3) => x2x3,
                        (2, 4) => x2x4,

                        (3, 1) => x3x1,
                        (3, 2) => x3x2,
                        (3, 3) => x3x3,
                        (3, 4) => x3x4,

                        _ => unreachable!(),
                    },
                    stack,
                );
            }
        }

        #[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
        mod avx512f {
            use super::*;
            const N: usize = 8 * $multiplier;
            const MR: usize = 3 * N;
            const NR: usize = 8;

            pub fn gemm_req(
                m: usize,
                n: usize,
                k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(MR, NR, m, n, k, max_n_threads)
            }

            #[target_feature(enable = "avx512f")]
            #[inline(never)]
            pub unsafe fn gemm_basic(
                m: usize,
                n: usize,
                k: usize,
                dst: *mut T,
                dst_cs: isize,
                dst_rs: isize,
                read_dst: bool,
                lhs: *const T,
                lhs_cs: isize,
                lhs_rs: isize,
                rhs: *const T,
                rhs_cs: isize,
                rhs_rs: isize,
                alpha: T,
                beta: T,
                n_threads: usize,
                stack: DynStack<'_>,
            ) {
                use microkernel::avx512f::$ty::*;
                gemm_basic_generic::<T, N, MR, NR>(
                    m,
                    n,
                    k,
                    dst,
                    dst_cs,
                    dst_rs,
                    read_dst,
                    lhs,
                    lhs_cs,
                    lhs_rs,
                    rhs,
                    rhs_cs,
                    rhs_rs,
                    alpha,
                    beta,
                    n_threads,
                    |mr_div_n, nr| match (mr_div_n, nr) {
                        (1, 1) => x1x1,
                        (1, 2) => x1x2,
                        (1, 3) => x1x3,
                        (1, 4) => x1x4,
                        (1, 5) => x1x5,
                        (1, 6) => x1x6,
                        (1, 7) => x1x7,
                        (1, 8) => x1x8,

                        (2, 1) => x2x1,
                        (2, 2) => x2x2,
                        (2, 3) => x2x3,
                        (2, 4) => x2x4,
                        (2, 5) => x2x5,
                        (2, 6) => x2x6,
                        (2, 7) => x2x7,
                        (2, 8) => x2x8,

                        (3, 1) => x3x1,
                        (3, 2) => x3x2,
                        (3, 3) => x3x3,
                        (3, 4) => x3x4,
                        (3, 5) => x3x5,
                        (3, 6) => x3x6,
                        (3, 7) => x3x7,
                        (3, 8) => x3x8,

                        _ => unreachable!(),
                    },
                    stack,
                );
            }
        }
    };
}

mod f32 {
    gemm_def!(f32, 2);
}
mod f64 {
    gemm_def!(f64, 1);
}

pub fn gemm_req<T: 'static>(
    m: usize,
    n: usize,
    k: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        crate::gemm::f64::gemm_req(m, n, k, max_n_threads)
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        crate::gemm::f32::gemm_req(m, n, k, max_n_threads)
    } else {
        Ok(StackReq::default())
    }
}

#[inline]
pub unsafe fn gemm<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Send + Sync + 'static,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        crate::gemm::f64::gemm_basic(
            m,
            n,
            k,
            dst as *mut f64,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f64,
            lhs_cs,
            lhs_rs,
            rhs as *mut f64,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f64),
            *(&beta as *const T as *const f64),
            n_threads,
            stack,
        )
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        crate::gemm::f32::gemm_basic(
            m,
            n,
            k,
            dst as *mut f32,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut f32,
            lhs_cs,
            lhs_rs,
            rhs as *mut f32,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const f32),
            *(&beta as *const T as *const f32),
            n_threads,
            stack,
        )
    } else {
        gemm_fallback(
            m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs,
            alpha, beta, n_threads, stack,
        )
    }
}

#[inline(never)]
pub(crate) unsafe fn gemm_fallback<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    n_threads: usize,
    stack: DynStack<'_>,
) where
    T: Zero + Send + Sync,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    let _stack = stack;

    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);

    if n_threads == 1 {
        (0..m).for_each(|row| {
            (0..n).for_each(|col| {
                let mut accum = <T as Zero>::zero();
                for depth in 0..k {
                    let lhs = &*lhs
                        .wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs)
                        .0;

                    let rhs = &*rhs
                        .wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs)
                        .0;

                    accum = &accum + &(lhs * rhs);
                }
                accum = &accum * &beta;

                let dst = dst
                    .wrapping_offset(row as isize * dst_rs + col as isize * dst_cs)
                    .0;
                if read_dst {
                    accum = &accum + &(&alpha * &*dst);
                }
                *dst = accum
            });
        });
    } else {
        use rayon::prelude::*;
        (0..m).into_par_iter().for_each(|row| {
            (0..n).into_par_iter().for_each(|col| {
                let mut accum = <T as Zero>::zero();
                for depth in 0..k {
                    let lhs = &*lhs
                        .wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs)
                        .0;

                    let rhs = &*rhs
                        .wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs)
                        .0;

                    accum = &accum + &(lhs * rhs);
                }
                accum = &accum * &beta;

                let dst = dst
                    .wrapping_offset(row as isize * dst_rs + col as isize * dst_cs)
                    .0;
                if read_dst {
                    accum = &accum + &(&alpha * &*dst);
                }
                *dst = accum
            });
        });
    }
}
