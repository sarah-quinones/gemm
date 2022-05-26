use crate::{
    cache::{kernel_params, KernelParams},
    microkernel,
    pack_operands::{pack_lhs, pack_rhs},
};
use dyn_stack::{DynStack, ReborrowMut, SizeOverflow, StackReq};

type T = f64;
const N: usize = 4;
const MR: usize = 3 * N;
const NR: usize = 4;

pub fn gemm_req(
    max_m: usize,
    max_n: usize,
    max_k: usize,
    n_threads: usize,
) -> Result<StackReq, SizeOverflow> {
    let KernelParams { mut kc, mut mc, nc } = kernel_params(MR, NR, core::mem::size_of::<T>());
    while max_k < kc / 2 {
        kc /= 2;
        mc *= 2;
    }

    let packed_rhs_stride = kc * NR;
    let packed_lhs_stride = kc * MR;

    let simd_align = core::mem::size_of::<T>() * N;

    StackReq::try_new_aligned::<T>(
        packed_rhs_stride * (nc / NR).min(div_ceil(max_n, NR)),
        simd_align,
    )?
    .try_and(StackReq::new_aligned::<T>(
        n_threads * packed_lhs_stride * (mc / MR).min(div_ceil(max_m, MR)),
        simd_align,
    ))
}

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

    use crate::Ptr;

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
                let packed_lhs = packed_lhs.wrapping_add(tid * packed_lhs_stride * (mc / MR));

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

                        macro_rules! ukr {
                            ($fn: expr) => {
                                $fn(
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
                                    if depth_outer == 0 { alpha } else { 1.0 },
                                    beta,
                                    if depth_outer == 0 { read_dst } else { true },
                                )
                            };
                        }

                        use microkernel::x256bit::f64::*;
                        match ((m_chunk_inner + (N - 1)) / N, n_chunk_inner) {
                            (1, 1) => ukr!(x1x1),
                            (1, 2) => ukr!(x1x2),
                            (1, 3) => ukr!(x1x3),
                            (1, 4) => ukr!(x1x4),

                            (2, 1) => ukr!(x2x1),
                            (2, 2) => ukr!(x2x2),
                            (2, 3) => ukr!(x2x3),
                            (2, 4) => ukr!(x2x4),

                            (3, 1) => ukr!(x3x1),
                            (3, 2) => ukr!(x3x2),
                            (3, 3) => ukr!(x3x3),
                            (3, 4) => ukr!(x3x4),

                            _ => unreachable!(),
                        }
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

#[inline(never)]
pub unsafe fn gemm_correct(
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
    _stack: DynStack,
) {
    for row in 0..m {
        for col in 0..n {
            let mut accum = 0.0;
            for depth in 0..k {
                accum += *lhs.offset(row as isize * lhs_rs + depth as isize * lhs_cs)
                    * *rhs.offset(depth as isize * rhs_rs + col as isize * rhs_cs);
            }
            accum *= beta;

            let dst = dst.offset(row as isize * dst_rs + col as isize * dst_cs);
            if read_dst {
                accum += alpha * *dst;
            }
            *dst = accum
        }
    }
}
