use crate::{
    cache::{kernel_params, KernelParams},
    microkernel,
    pack_operands::{pack_lhs, pack_rhs},
    Ptr,
};
use dyn_stack::{DynStack, ReborrowMut, SizeOverflow, StackReq};
use num_traits::Zero;

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

#[inline(never)]
unsafe fn gemm_basic_generic<
    T: Copy + From<u8> + Send + Sync,
    F: Copy
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
            T,
            T,
            bool,
        ),
    const N: usize,
    const MR: usize,
    const NR: usize,
>(
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
    dispatcher: F,
    mut stack: DynStack,
) {
    if m == 0 || n == 0 {
        return;
    }

    let KernelParams { mut kc, mut mc, nc } = kernel_params(MR, NR, core::mem::size_of::<T>());
    while k < kc / 2 {
        kc /= 2;
        mc *= 2;
    }

    let packed_rhs_stride = kc * NR;
    let packed_lhs_stride = kc * MR;

    let simd_align = core::mem::size_of::<T>() * N;

    let (mut packed_rhs_storage, mut stack) = stack.rb_mut().make_aligned_uninit::<T>(
        packed_rhs_stride * (nc / NR).min(div_ceil(n, NR)),
        simd_align,
    );

    let packed_rhs = packed_rhs_storage.as_mut_ptr() as *mut T;

    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);
    let packed_rhs = Ptr(packed_rhs);

    let mut col_outer = 0;
    while col_outer != n {
        let n_chunk = nc.min(n - col_outer);

        let mut depth_outer = 0;
        while depth_outer != k {
            let k_chunk = kc.min(k - depth_outer);

            pack_rhs::<T>(
                NR,
                n_chunk,
                k_chunk,
                packed_rhs,
                rhs.wrapping_offset(depth_outer as isize * rhs_rs + col_outer as isize * rhs_cs),
                rhs_cs,
                rhs_rs,
                packed_rhs_stride,
            );

            let (mut packed_lhs_storage, _) = stack.rb_mut().make_aligned_uninit::<T>(
                n_threads * packed_lhs_stride * (mc / MR).min(div_ceil(m, MR)),
                simd_align,
            );

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
                let rem = n_jobs % n_threads;

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
                    if job_id + n_mini_jobs < job_start || job_id >= job_end {
                        row_outer += m_chunk;
                        job_id += n_mini_jobs;
                        continue;
                    }

                    pack_lhs::<T>(
                        MR,
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

                    for ij in 0..n_col_mini_chunks * n_row_mini_chunks {
                        let i = ij % n_row_mini_chunks;
                        let j = ij / n_row_mini_chunks;

                        let col_inner = NR * j;
                        let n_chunk_inner = NR.min(n_chunk - col_inner);

                        let row_inner = MR * i;
                        let m_chunk_inner = MR.min(m_chunk - row_inner);

                        if job_id < job_start || job_id >= job_end {
                            job_id += 1;
                            continue;
                        }
                        job_id += 1;

                        let dst = dst.wrapping_offset(
                            (row_outer + row_inner) as isize * dst_rs
                                + (col_outer + col_inner) as isize * dst_cs,
                        );

                        let func = dispatcher((m_chunk_inner + (N - 1)) / N, n_chunk_inner);
                        func(
                            m_chunk_inner,
                            n_chunk_inner,
                            k_chunk,
                            dst,
                            packed_lhs.wrapping_add(row_inner * kc),
                            packed_rhs.wrapping_add(col_inner * kc),
                            dst_cs,
                            dst_rs,
                            MR as isize,
                            NR as isize,
                            if depth_outer == 0 { alpha } else { 1_u8.into() },
                            beta,
                            if depth_outer == 0 { read_dst } else { true },
                        );
                    }

                    row_outer += m_chunk;
                }
            };

            if n_threads <= 1 {
                func(0);
            } else {
                use rayon::prelude::*;
                (0..n_threads).into_par_iter().for_each(func);
            }
            depth_outer += k_chunk;
        }
        col_outer += n_chunk;
    }
}

fn gemm_basic_req_generic<T>(
    n: usize,
    mr: usize,
    nr: usize,
    max_m: usize,
    max_n: usize,
    max_k: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let KernelParams { mut kc, mut mc, nc } = kernel_params(mr, nr, core::mem::size_of::<T>());
    while max_k < kc / 2 {
        kc /= 2;
        mc *= 2;
    }
    let packed_rhs_stride = kc * nr;
    let packed_lhs_stride = kc * mr;
    mc *= 2;
    let simd_align = core::mem::size_of::<T>() * n;

    StackReq::try_new_aligned::<T>(
        packed_rhs_stride * (nc / nr).min(div_ceil(max_n, nr)),
        simd_align,
    )?
    .try_and(StackReq::try_new_aligned::<T>(
        max_n_threads * packed_lhs_stride * (mc / mr).min(div_ceil(max_m, mr)),
        simd_align,
    )?)
}

macro_rules! gemm_def {
    ($ty: tt, $multiplier: expr) => {
        use super::*;
        type T = $ty;

        lazy_static::lazy_static! {
            static ref GEMM: (
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
                    DynStack,
                    ),
                fn(
                    usize,
                    usize,
                    usize,
                    usize,
                    ) -> Result<StackReq, SizeOverflow>,
                ) = {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    (avx2::gemm_basic, avx2::gemm_req)
                } else if is_x86_feature_detected!("avx") {
                    (avx::gemm_basic, avx::gemm_req)
                } else {
                    (sse::gemm_basic, sse::gemm_req)
                }
            };
        }

        pub fn gemm_req(
            max_m: usize,
            max_n: usize,
            max_k: usize,
            max_n_threads: usize,
        ) -> Result<StackReq, SizeOverflow> {
            (GEMM.1)(max_m, max_n, max_k, max_n_threads)
        }

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
            stack: DynStack,
        ) {
            (GEMM.0)(
                m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs,
                alpha, beta, n_threads, stack,
            )
        }

        mod sse {
            use super::*;
            const N: usize = 2 * $multiplier;
            const MR: usize = 2 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                max_m: usize,
                max_n: usize,
                max_k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(N, MR, NR, max_m, max_n, max_k, max_n_threads)
            }

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
                stack: DynStack,
            ) {
                use microkernel::sse::$ty::*;
                gemm_basic_generic::<T, _, N, MR, NR>(
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

        mod avx {
            use super::*;
            const N: usize = 4 * $multiplier;
            const MR: usize = 2 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                max_m: usize,
                max_n: usize,
                max_k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(N, MR, NR, max_m, max_n, max_k, max_n_threads)
            }

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
                stack: DynStack,
            ) {
                use microkernel::avx::$ty::*;
                gemm_basic_generic::<T, _, N, MR, NR>(
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

        mod avx2 {
            use super::*;
            const N: usize = 4 * $multiplier;
            const MR: usize = 3 * N;
            const NR: usize = 4;

            pub fn gemm_req(
                max_m: usize,
                max_n: usize,
                max_k: usize,
                max_n_threads: usize,
            ) -> Result<StackReq, SizeOverflow> {
                gemm_basic_req_generic::<T>(N, MR, NR, max_m, max_n, max_k, max_n_threads)
            }

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
                stack: DynStack,
            ) {
                use microkernel::avx2::$ty::*;
                gemm_basic_generic::<T, _, N, MR, NR>(
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
    };
}

mod f32 {
    gemm_def!(f32, 2);
}
mod f64 {
    gemm_def!(f64, 1);
}

fn unique_id<T>() -> usize {
    (unique_id::<T>) as usize
}

pub fn gemm_req<T>(
    max_m: usize,
    max_n: usize,
    max_k: usize,
    max_n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    if unique_id::<T>() == unique_id::<f64>() {
        crate::gemm::f64::gemm_req(max_m, max_n, max_k, max_n_threads)
    } else if unique_id::<T>() == unique_id::<f32>() {
        crate::gemm::f32::gemm_req(max_m, max_n, max_k, max_n_threads)
    } else {
        Ok(StackReq::new::<()>(0))
    }
}

#[inline(never)]
pub unsafe fn gemm_basic<T>(
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
    stack: DynStack,
) where
    T: Zero + Send + Sync,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    if unique_id::<T>() == unique_id::<f64>() {
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
    } else if unique_id::<T>() == unique_id::<f32>() {
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
        gemm_correct(
            m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs,
            alpha, beta, n_threads, stack,
        )
    }
}

#[inline(never)]
pub(crate) unsafe fn gemm_correct<T>(
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
    stack: DynStack,
) where
    T: Zero + Send + Sync,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    let _stack = stack;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);

    pool.install(|| {
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
    });
}
