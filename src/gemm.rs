use crate::{
    cache::{kernel_params, KernelParams, CACHE_INFO},
    gemv, gevv,
    microkernel::{self, MicroKernelFn},
    pack_operands::{pack_lhs, pack_rhs},
    simd::Simd,
    x86_feature_detected, Parallelism, Ptr,
};
use aligned_vec::CACHELINE_ALIGN;
use core::{any::TypeId, cell::RefCell};
use dyn_stack::GlobalMemBuffer;
use dyn_stack::{DynStack, StackReq};
use num_traits::{One, Zero};

thread_local! {
    static L2_SLAB: RefCell<GlobalMemBuffer> = RefCell::new(GlobalMemBuffer::new(
        StackReq::new_aligned::<u8>(CACHE_INFO[1].cache_bytes, CACHELINE_ALIGN)
    ));
}

#[inline(always)]
unsafe fn gemm_basic_generic<
    S: Simd,
    T: Copy
        + Zero
        + One
        + Send
        + Sync
        + core::fmt::Debug
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::cmp::PartialEq,
    const N: usize,
    const MR: usize,
    const NR: usize,
    const MR_DIV_N: usize,
>(
    simd: S,
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
    mut alpha: T,
    beta: T,
    mul_add: impl Copy + Fn(T, T, T) -> T,
    dispatcher_zero: &'static [[MicroKernelFn<T>; NR]; MR_DIV_N],
    dispatcher_one: &'static [[MicroKernelFn<T>; NR]; MR_DIV_N],
    dispatcher_generic: &'static [[MicroKernelFn<T>; NR]; MR_DIV_N],
    parallelism: Parallelism,
) {
    if m == 0 || n == 0 {
        return;
    }
    if !read_dst {
        alpha.set_zero();
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

    if k <= 2 {
        gevv::gevv(
            simd, m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, alpha,
            beta, mul_add,
        );
        return;
    }
    if m <= 4 && rhs_cs.wrapping_abs() <= rhs_rs.wrapping_abs() {
        gemv::gemv(
            simd, n, m, k, dst, dst_rs, dst_cs, rhs, rhs_rs, rhs_cs, lhs, lhs_rs, lhs_cs, alpha,
            beta, mul_add,
        );
        return;
    }
    if n <= 4 && lhs_rs.wrapping_abs() <= lhs_cs.wrapping_abs() {
        gemv::gemv(
            simd, m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, alpha,
            beta, mul_add,
        );
        return;
    }

    let KernelParams { kc, mc, nc } = kernel_params(m, n, k, MR, NR, core::mem::size_of::<T>());

    let simd_align = CACHELINE_ALIGN;

    let packed_rhs_stride = kc * NR;
    let packed_lhs_stride = kc * MR;

    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);
    let do_pack_rhs = m > 8 * MR && rhs_rs.abs() != 1;

    let mut mem = if do_pack_rhs {
        Some(GlobalMemBuffer::new(StackReq::new_aligned::<T>(
            packed_rhs_stride * (nc / NR),
            simd_align,
        )))
    } else {
        None
    };

    let mut packed_rhs_storage = mem.as_mut().map(|mem| {
        let stack = DynStack::new(mem);
        stack
            .make_aligned_uninit::<T>(packed_rhs_stride * (nc / NR), simd_align)
            .0
    });

    let packed_rhs = packed_rhs_storage
        .as_mut()
        .map(|storage| storage.as_mut_ptr() as *mut T)
        .unwrap_or(core::ptr::null_mut());
    let packed_rhs = Ptr(packed_rhs);

    let packed_rhs_rs = if do_pack_rhs { NR as isize } else { rhs_rs };
    let packed_rhs_cs = if do_pack_rhs { 1 } else { rhs_cs };

    let mut col_outer = 0;
    while col_outer != n {
        let n_chunk = nc.min(n - col_outer);

        let mut alpha = alpha;

        let mut depth_outer = 0;
        while depth_outer != k {
            let k_chunk = kc.min(k - depth_outer);
            let dispatcher = if alpha.is_zero() {
                dispatcher_zero
            } else if alpha.is_one() {
                dispatcher_one
            } else {
                dispatcher_generic
            };

            if do_pack_rhs {
                pack_rhs::<T, 1, NR, _>(
                    simd,
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

            let n_col_mini_chunks = (n_chunk + (NR - 1)) / NR;

            let mut n_jobs = 0;
            let mut row_outer = 0;
            while row_outer != m {
                let mut m_chunk = mc.min(m - row_outer);
                if m_chunk > N {
                    m_chunk = m_chunk / N * N;
                }
                let n_row_mini_chunks = (m_chunk + (MR - 1)) / MR;
                n_jobs += n_col_mini_chunks * n_row_mini_chunks;
                row_outer += m_chunk;
            }

            let n_threads = match parallelism {
                Parallelism::None => 1,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(n_threads) => {
                    let threading_threshold = 48 * 48 * 256;
                    if m * n_chunk * k_chunk <= threading_threshold {
                        1
                    } else {
                        if n_threads == 0 {
                            rayon::current_num_threads()
                        } else {
                            n_threads
                        }
                    }
                }
            };

            // use a single thread for small workloads

            let func = move |tid| {
                L2_SLAB.with(|mem| {
                    let mut mem = mem.borrow_mut();
                    let stack = DynStack::new(&mut **mem);

                    let (mut packed_lhs_storage, _) =
                        stack.make_aligned_uninit::<T>(packed_lhs_stride * (mc / MR), simd_align);

                    let packed_lhs = Ptr(packed_lhs_storage.as_mut_ptr() as *mut T);

                    let min_jobs_per_thread = n_jobs / n_threads;
                    let rem = n_jobs - n_threads * min_jobs_per_thread;

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
                        let mut m_chunk = mc.min(m - row_outer);
                        if m_chunk > N {
                            m_chunk = m_chunk / N * N;
                        }
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

                        let do_pack_lhs = (m_chunk % N != 0) || lhs_rs != 1 || n > 32 * NR;
                        let packed_lhs_cs = if do_pack_lhs { MR as isize } else { lhs_cs };

                        if do_pack_lhs {
                            pack_lhs::<T, N, MR, _>(
                                simd,
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
                        }

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

                                let func = dispatcher[(m_chunk_inner + (N - 1)) / N - 1]
                                    [n_chunk_inner - 1];

                                func(
                                    m_chunk_inner,
                                    n_chunk_inner,
                                    k_chunk,
                                    dst,
                                    if do_pack_lhs {
                                        packed_lhs.wrapping_add(i * packed_lhs_stride)
                                    } else {
                                        lhs.wrapping_offset(
                                            (row_outer + row_inner) as isize * lhs_rs
                                                + depth_outer as isize * lhs_cs,
                                        )
                                    },
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
                                    packed_lhs_cs,
                                    packed_rhs_rs,
                                    packed_rhs_cs,
                                    alpha,
                                    beta,
                                );
                                i += 1;
                            }
                            j += 1;
                        }

                        row_outer += m_chunk;
                    }
                });
            };

            match parallelism {
                Parallelism::None => func(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(_) => {
                    if n_threads == 1 {
                        func(0);
                    } else {
                        use rayon::prelude::*;
                        (0..n_threads).into_par_iter().for_each(func);
                    }
                }
            }

            alpha.set_one();
            depth_outer += k_chunk;
        }
        col_outer += n_chunk;
    }
}

macro_rules! gemm_def {
    ($ty: tt, $multiplier: expr) => {
        use super::*;
        type T = $ty;

        type GemmTy = unsafe fn(
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
            Parallelism,
        );

        fn init_gemm_fn() -> GemmTy {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if x86_feature_detected!("avx512f") {
                    return avx512f::gemm_basic;
                }
                if x86_feature_detected!("fma") {
                    fma::gemm_basic
                } else if x86_feature_detected!("avx") {
                    avx::gemm_basic
                } else if x86_feature_detected!("sse") {
                    sse::gemm_basic
                } else {
                    scalar::gemm_basic
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                scalar::gemm_basic
            }
        }

        lazy_static::lazy_static! {
            static ref GEMM: GemmTy = init_gemm_fn();
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
            parallelism: Parallelism,
        ) {
            if dst_cs.abs() < dst_rs.abs() {
                GEMM(
                    n,
                    m,
                    k,
                    dst,
                    dst_rs,
                    dst_cs,
                    read_dst,
                    rhs,
                    rhs_rs,
                    rhs_cs,
                    lhs,
                    lhs_rs,
                    lhs_cs,
                    alpha,
                    beta,
                    parallelism,
                )
            } else {
                GEMM(
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
                    parallelism,
                )
            }
        }

        mod scalar {
            use super::*;
            const N: usize = 1;
            const MR: usize = 2 * N;
            const NR: usize = 4;

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
                parallelism: Parallelism,
            ) {
                use microkernel::scalar::$ty::*;
                gemm_basic_generic::<_, T, N, MR, NR, { MR / N }>(
                    crate::simd::Scalar,
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
                    |a, b, c| a * b + c,
                    &[
                        [x1x1::<0>, x1x2::<0>, x1x3::<0>, x1x4::<0>],
                        [x2x1::<0>, x2x2::<0>, x2x3::<0>, x2x4::<0>],
                    ],
                    &[
                        [x1x1::<1>, x1x2::<1>, x1x3::<1>, x1x4::<1>],
                        [x2x1::<1>, x2x2::<1>, x2x3::<1>, x2x4::<1>],
                    ],
                    &[
                        [x1x1::<2>, x1x2::<2>, x1x3::<2>, x1x4::<2>],
                        [x2x1::<2>, x2x2::<2>, x2x3::<2>, x2x4::<2>],
                    ],
                    parallelism,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod sse {
            use super::*;
            const N: usize = 2 * $multiplier;
            const MR: usize = 2 * N;
            const NR: usize = 4;

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
                parallelism: Parallelism,
            ) {
                use microkernel::sse::$ty::*;
                gemm_basic_generic::<_, T, N, MR, NR, { MR / N }>(
                    crate::simd::Sse,
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
                    |a, b, c| a * b + c,
                    &[
                        [x1x1::<0>, x1x2::<0>, x1x3::<0>, x1x4::<0>],
                        [x2x1::<0>, x2x2::<0>, x2x3::<0>, x2x4::<0>],
                    ],
                    &[
                        [x1x1::<1>, x1x2::<1>, x1x3::<1>, x1x4::<1>],
                        [x2x1::<1>, x2x2::<1>, x2x3::<1>, x2x4::<1>],
                    ],
                    &[
                        [x1x1::<2>, x1x2::<2>, x1x3::<2>, x1x4::<2>],
                        [x2x1::<2>, x2x2::<2>, x2x3::<2>, x2x4::<2>],
                    ],
                    parallelism,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod avx {
            use super::*;
            const N: usize = 4 * $multiplier;
            const MR: usize = 2 * N;
            const NR: usize = 4;

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
                parallelism: Parallelism,
            ) {
                use microkernel::avx::$ty::*;
                gemm_basic_generic::<_, T, N, MR, NR, { MR / N }>(
                    crate::simd::Avx,
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
                    |a, b, c| a * b + c,
                    &[
                        [x1x1::<0>, x1x2::<0>, x1x3::<0>, x1x4::<0>],
                        [x2x1::<0>, x2x2::<0>, x2x3::<0>, x2x4::<0>],
                    ],
                    &[
                        [x1x1::<1>, x1x2::<1>, x1x3::<1>, x1x4::<1>],
                        [x2x1::<1>, x2x2::<1>, x2x3::<1>, x2x4::<1>],
                    ],
                    &[
                        [x1x1::<2>, x1x2::<2>, x1x3::<2>, x1x4::<2>],
                        [x2x1::<2>, x2x2::<2>, x2x3::<2>, x2x4::<2>],
                    ],
                    parallelism,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod fma {
            use super::*;
            const N: usize = 4 * $multiplier;
            const MR: usize = 3 * N;
            const NR: usize = 4;

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
                parallelism: Parallelism,
            ) {
                use microkernel::fma::$ty::*;
                gemm_basic_generic::<_, T, N, MR, NR, { MR / N }>(
                    crate::simd::Fma,
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
                    |a, b, c| <$ty>::mul_add(a, b, c),
                    &[
                        [x1x1::<0>, x1x2::<0>, x1x3::<0>, x1x4::<0>],
                        [x2x1::<0>, x2x2::<0>, x2x3::<0>, x2x4::<0>],
                        [x3x1::<0>, x3x2::<0>, x3x3::<0>, x3x4::<0>],
                    ],
                    &[
                        [x1x1::<1>, x1x2::<1>, x1x3::<1>, x1x4::<1>],
                        [x2x1::<1>, x2x2::<1>, x2x3::<1>, x2x4::<1>],
                        [x3x1::<1>, x3x2::<1>, x3x3::<1>, x3x4::<1>],
                    ],
                    &[
                        [x1x1::<2>, x1x2::<2>, x1x3::<2>, x1x4::<2>],
                        [x2x1::<2>, x2x2::<2>, x2x3::<2>, x2x4::<2>],
                        [x3x1::<2>, x3x2::<2>, x3x3::<2>, x3x4::<2>],
                    ],
                    parallelism,
                );
            }
        }

        #[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
        mod avx512f {
            use super::*;
            const N: usize = 8 * $multiplier;
            const MR: usize = 3 * N;
            const NR: usize = 8;

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
                parallelism: Parallelism,
            ) {
                use microkernel::avx512f::$ty::*;
                gemm_basic_generic::<_, T, N, MR, NR, { MR / N }>(
                    crate::simd::Avx512f,
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
                    |a, b, c| <$ty>::mul_add(a, b, c),
                    &[
                        [
                            x1x1::<0>, x1x2::<0>, x1x3::<0>, x1x4::<0>, x1x5::<0>, x1x6::<0>,
                            x1x7::<0>, x1x8::<0>,
                        ],
                        [
                            x2x1::<0>, x2x2::<0>, x2x3::<0>, x2x4::<0>, x2x5::<0>, x2x6::<0>,
                            x2x7::<0>, x2x8::<0>,
                        ],
                        [
                            x3x1::<0>, x3x2::<0>, x3x3::<0>, x3x4::<0>, x3x5::<0>, x3x6::<0>,
                            x3x7::<0>, x3x8::<0>,
                        ],
                    ],
                    &[
                        [
                            x1x1::<1>, x1x2::<1>, x1x3::<1>, x1x4::<1>, x1x5::<1>, x1x6::<1>,
                            x1x7::<1>, x1x8::<1>,
                        ],
                        [
                            x2x1::<1>, x2x2::<1>, x2x3::<1>, x2x4::<1>, x2x5::<1>, x2x6::<1>,
                            x2x7::<1>, x2x8::<1>,
                        ],
                        [
                            x3x1::<1>, x3x2::<1>, x3x3::<1>, x3x4::<1>, x3x5::<1>, x3x6::<1>,
                            x3x7::<1>, x3x8::<1>,
                        ],
                    ],
                    &[
                        [
                            x1x1::<2>, x1x2::<2>, x1x3::<2>, x1x4::<2>, x1x5::<2>, x1x6::<2>,
                            x1x7::<2>, x1x8::<2>,
                        ],
                        [
                            x2x1::<2>, x2x2::<2>, x2x3::<2>, x2x4::<2>, x2x5::<2>, x2x6::<2>,
                            x2x7::<2>, x2x8::<2>,
                        ],
                        [
                            x3x1::<2>, x3x2::<2>, x3x3::<2>, x3x4::<2>, x3x5::<2>, x3x6::<2>,
                            x3x7::<2>, x3x8::<2>,
                        ],
                    ],
                    parallelism,
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

/// dst := alpha×dst + beta×lhs×rhs
///
/// # Panics
///
/// Panics if `T` is not `f32` or `f64`
#[inline]
pub unsafe fn gemm<T: 'static>(
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
    parallelism: Parallelism,
) {
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
            parallelism,
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
            parallelism,
        )
    } else {
        unreachable!();
    }
}

#[inline(never)]
#[cfg(test)]
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
) where
    T: Zero + Send + Sync,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);

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
    return;
}
