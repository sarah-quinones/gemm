use crate::{
    cache::{kernel_params, KernelParams, CACHE_INFO},
    gemv, gevv,
    microkernel::{self, MicroKernelFn},
    pack_operands::{pack_lhs, pack_rhs},
    simd::Simd,
    Parallelism, Ptr,
};
use core::{any::TypeId, cell::RefCell};
use dyn_stack::GlobalMemBuffer;
use dyn_stack::{DynStack, StackReq};
use num_traits::{One, Zero};

// https://rust-lang.github.io/hashbrown/src/crossbeam_utils/cache_padded.rs.html#128-130
pub const CACHELINE_ALIGN: usize = {
    #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
    ))]
    {
        128
    }
    #[cfg(any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
    ))]
    {
        32
    }
    #[cfg(target_arch = "s390x")]
    {
        256
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64",
        target_arch = "s390x",
    )))]
    {
        64
    }
};

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
    mut alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    mul_add: impl Copy + Fn(T, T, T) -> T,
    dispatcher: &[[MicroKernelFn<T>; NR]; MR_DIV_N],
    parallelism: Parallelism,
) {
    if m == 0 || n == 0 {
        return;
    }
    if !read_dst {
        alpha.set_zero();
    }

    if !conj_dst && !conj_lhs && !conj_rhs {
        if k <= 2 {
            gevv::gevv(
                simd, m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs,
                alpha, beta, mul_add,
            );
            return;
        }
        if m <= 4 && rhs_cs.wrapping_abs() <= rhs_rs.wrapping_abs() {
            gemv::gemv(
                simd, n, m, k, dst, dst_rs, dst_cs, rhs, rhs_rs, rhs_cs, lhs, lhs_rs, lhs_cs,
                alpha, beta, mul_add,
            );
            return;
        }
        if n <= 4 && lhs_rs.wrapping_abs() <= lhs_cs.wrapping_abs() {
            gemv::gemv(
                simd, m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs,
                alpha, beta, mul_add,
            );
            return;
        }
    }

    let KernelParams { kc, mc, nc } = kernel_params(m, n, k, MR, NR, core::mem::size_of::<T>());

    let simd_align = CACHELINE_ALIGN;

    let packed_rhs_stride = kc * NR;
    let packed_lhs_stride = kc * MR;

    let dst = Ptr(dst);
    let lhs = Ptr(lhs as *mut T);
    let rhs = Ptr(rhs as *mut T);

    // on aarch64-neon, we always pack beyond a certain size, since the microkernel can use the
    // contiguity of the RHS with `vfmaq_laneq_[f32|f64]`
    #[cfg(target_arch = "aarch64")]
    let do_pack_rhs = m > 2 * MR;

    // no need to pack if the lhs is already contiguous-ish
    #[cfg(not(target_arch = "aarch64"))]
    let do_pack_rhs = m > 2 * MR && rhs_rs.abs() != 1;

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
        let mut conj_dst = conj_dst;

        let mut depth_outer = 0;
        while depth_outer != k {
            let k_chunk = kc.min(k - depth_outer);
            let alpha_status = if alpha.is_zero() {
                0
            } else if alpha.is_one() {
                1
            } else {
                2
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

                        let j_then_i = !do_pack_lhs;

                        if j_then_i {
                            let mut j = 0;
                            while j < n_col_mini_chunks {
                                let mut i = 0;
                                while i < n_row_mini_chunks {
                                    let col_inner = NR * j;
                                    let n_chunk_inner = NR.min(n_chunk - col_inner);

                                    let row_inner = MR * i;
                                    let m_chunk_inner = MR.min(m_chunk - row_inner);

                                    let inner_idx = &mut i;
                                    if job_id < job_start || job_id >= job_end {
                                        job_id += 1;
                                        *inner_idx += 1;
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
                                        dst.0,
                                        if do_pack_lhs {
                                            packed_lhs.wrapping_add(i * packed_lhs_stride).0
                                        } else {
                                            lhs.wrapping_offset(
                                                (row_outer + row_inner) as isize * lhs_rs
                                                    + depth_outer as isize * lhs_cs,
                                            )
                                            .0
                                        },
                                        if do_pack_rhs {
                                            packed_rhs.wrapping_add(j * packed_rhs_stride).0
                                        } else {
                                            rhs.wrapping_offset(
                                                depth_outer as isize * rhs_rs
                                                    + (col_outer + col_inner) as isize * rhs_cs,
                                            )
                                            .0
                                        },
                                        dst_cs,
                                        dst_rs,
                                        packed_lhs_cs,
                                        packed_rhs_rs,
                                        packed_rhs_cs,
                                        alpha,
                                        beta,
                                        alpha_status,
                                        conj_dst,
                                        conj_lhs,
                                        conj_rhs,
                                    );
                                    i += 1;
                                }
                                j += 1;
                            }
                        } else {
                            let mut i = 0;
                            while i < n_row_mini_chunks {
                                let mut j = 0;
                                while j < n_col_mini_chunks {
                                    let col_inner = NR * j;
                                    let n_chunk_inner = NR.min(n_chunk - col_inner);

                                    let row_inner = MR * i;
                                    let m_chunk_inner = MR.min(m_chunk - row_inner);

                                    let inner_idx = &mut j;
                                    if job_id < job_start || job_id >= job_end {
                                        job_id += 1;
                                        *inner_idx += 1;
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
                                        dst.0,
                                        if do_pack_lhs {
                                            packed_lhs.wrapping_add(i * packed_lhs_stride).0
                                        } else {
                                            lhs.wrapping_offset(
                                                (row_outer + row_inner) as isize * lhs_rs
                                                    + depth_outer as isize * lhs_cs,
                                            )
                                            .0
                                        },
                                        if do_pack_rhs {
                                            packed_rhs.wrapping_add(j * packed_rhs_stride).0
                                        } else {
                                            rhs.wrapping_offset(
                                                depth_outer as isize * rhs_rs
                                                    + (col_outer + col_inner) as isize * rhs_cs,
                                            )
                                            .0
                                        },
                                        dst_cs,
                                        dst_rs,
                                        packed_lhs_cs,
                                        packed_rhs_rs,
                                        packed_rhs_cs,
                                        alpha,
                                        beta,
                                        alpha_status,
                                        conj_dst,
                                        conj_lhs,
                                        conj_rhs,
                                    );
                                    j += 1;
                                }
                                i += 1;
                            }
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

            conj_dst = false;
            alpha.set_one();

            depth_outer += k_chunk;
        }
        col_outer += n_chunk;
    }
}

macro_rules! __inject_mod {
    ($module: ident, $ty: ident, $N: expr, $simd: ident) => {
        mod $module {
            use super::*;
            use microkernel::$module::$ty::*;
            const N: usize = $N;

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
                conj_dst: bool,
                conj_lhs: bool,
                conj_rhs: bool,
                parallelism: Parallelism,
            ) {
                gemm_basic_generic::<_, T, N, { MR_DIV_N * N }, NR, MR_DIV_N>(
                    crate::simd::$simd,
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
                    conj_dst,
                    conj_lhs,
                    conj_rhs,
                    |a, b, c| a * b + c,
                    &UKR,
                    parallelism,
                );
            }
        }
    };
}

macro_rules! __inject_mod_cplx {
    ($module: ident, $ty: ident, $N: expr, $simd: ident) => {
        paste::paste! {
            mod [<$module _cplx>] {
                use super::*;
                use microkernel::$module::$ty::*;
                const N: usize = $N;

                #[inline(never)]
                pub unsafe fn gemm_basic_cplx(
                    m: usize,
                    n: usize,
                    k: usize,
                    dst: *mut num_complex::Complex<T>,
                    dst_cs: isize,
                    dst_rs: isize,
                    read_dst: bool,
                    lhs: *const num_complex::Complex<T>,
                    lhs_cs: isize,
                    lhs_rs: isize,
                    rhs: *const num_complex::Complex<T>,
                    rhs_cs: isize,
                    rhs_rs: isize,
                    alpha: num_complex::Complex<T>,
                    beta: num_complex::Complex<T>,
                    conj_dst: bool,
                    conj_lhs: bool,
                    conj_rhs: bool,
                    parallelism: Parallelism,
                    ) {
                    gemm_basic_generic::<_, _, N, { CPLX_MR_DIV_N * N }, CPLX_NR, CPLX_MR_DIV_N>(
                        crate::simd::$simd,
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
                        conj_dst,
                        conj_lhs,
                        conj_rhs,
                        |a, b, c| a * b + c,
                        &CPLX_UKR,
                        parallelism,
                        );
                }
            }
        }
    };
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
            bool,
            bool,
            bool,
            Parallelism,
        );
        type GemmCplxTy = unsafe fn(
            usize,
            usize,
            usize,
            *mut num_complex::Complex<T>,
            isize,
            isize,
            bool,
            *const num_complex::Complex<T>,
            isize,
            isize,
            *const num_complex::Complex<T>,
            isize,
            isize,
            num_complex::Complex<T>,
            num_complex::Complex<T>,
            bool,
            bool,
            bool,
            Parallelism,
        );

        fn init_gemm_fn() -> GemmTy {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if feature_detected!("avx512f") {
                    return avx512f::gemm_basic;
                }
                if feature_detected!("fma") {
                    fma::gemm_basic
                } else if feature_detected!("avx") {
                    avx::gemm_basic
                } else if feature_detected!("sse") {
                    sse::gemm_basic
                } else {
                    scalar::gemm_basic
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if feature_detected!("neon") {
                    neon::gemm_basic
                } else {
                    scalar::gemm_basic
                }
            }

            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
            {
                scalar::gemm_basic
            }
        }

        fn init_gemm_cplx_fn() -> GemmCplxTy {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[cfg(feature = "nightly")]
                if feature_detected!("avx512f") {
                    return avx512f_cplx::gemm_basic_cplx;
                }
                if feature_detected!("fma") {
                    return fma_cplx::gemm_basic_cplx;
                }
            }

            scalar_cplx::gemm_basic_cplx
        }

        lazy_static::lazy_static! {
            pub static ref GEMM: GemmTy = init_gemm_fn();
        }

        lazy_static::lazy_static! {
            pub static ref GEMM_CPLX: GemmCplxTy = init_gemm_cplx_fn();
        }

        __inject_mod!(scalar, $ty, 1, Scalar);
        __inject_mod_cplx!(scalar, $ty, 1, Scalar);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        __inject_mod!(sse, $ty, 2 * $multiplier, Sse);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        __inject_mod!(avx, $ty, 4 * $multiplier, Avx);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        __inject_mod!(fma, $ty, 4 * $multiplier, Fma);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        __inject_mod_cplx!(fma, $ty, 2 * $multiplier, Fma);
        #[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
        __inject_mod!(avx512f, $ty, 8 * $multiplier, Avx512f);
        #[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
        __inject_mod_cplx!(avx512f, $ty, 4 * $multiplier, Avx512f);

        #[cfg(target_arch = "aarch64")]
        __inject_mod!(neon, $ty, 2 * $multiplier, Scalar);
    };
}

mod f32 {
    gemm_def!(f32, 2);
}
mod f64 {
    gemm_def!(f64, 1);
}

#[allow(non_camel_case_types)]
pub type c32 = num_complex::Complex32;
#[allow(non_camel_case_types)]
pub type c64 = num_complex::Complex64;

pub unsafe fn gemm_dispatch<T: 'static>(
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
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        crate::gemm::f64::GEMM(
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
            false,
            false,
            false,
            parallelism,
        )
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        crate::gemm::f32::GEMM(
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
            false,
            false,
            false,
            parallelism,
        )
    } else if TypeId::of::<T>() == TypeId::of::<c64>() {
        crate::gemm::f64::GEMM_CPLX(
            m,
            n,
            k,
            dst as *mut c64,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut c64,
            lhs_cs,
            lhs_rs,
            rhs as *mut c64,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const c64),
            *(&beta as *const T as *const c64),
            conj_dst,
            conj_lhs,
            conj_rhs,
            parallelism,
        )
    } else if TypeId::of::<T>() == TypeId::of::<c32>() {
        crate::gemm::f32::GEMM_CPLX(
            m,
            n,
            k,
            dst as *mut c32,
            dst_cs,
            dst_rs,
            read_dst,
            lhs as *mut c32,
            lhs_cs,
            lhs_rs,
            rhs as *mut c32,
            rhs_cs,
            rhs_rs,
            *(&alpha as *const T as *const c32),
            *(&beta as *const T as *const c32),
            conj_dst,
            conj_lhs,
            conj_rhs,
            parallelism,
        )
    } else {
        panic!();
    }
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
    mut dst: *mut T,
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
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: Parallelism,
) {
    // we want to transpose if the destination is column-oriented, since the microkernel prefers
    // column major matrices.
    let do_transpose = dst_cs.abs() < dst_rs.abs();

    let (
        m,
        n,
        mut dst_cs,
        mut dst_rs,
        mut lhs,
        lhs_cs,
        mut lhs_rs,
        mut rhs,
        mut rhs_cs,
        rhs_rs,
        conj_lhs,
        conj_rhs,
    ) = if do_transpose {
        (
            n, m, dst_rs, dst_cs, rhs, rhs_rs, rhs_cs, lhs, lhs_rs, lhs_cs, conj_rhs, conj_lhs,
        )
    } else {
        (
            m, n, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, conj_lhs, conj_rhs,
        )
    };

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

    gemm_dispatch(
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
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
    )
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

#[inline(never)]
#[cfg(test)]
pub(crate) unsafe fn gemm_cplx_fallback<T>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut num_complex::Complex<T>,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const num_complex::Complex<T>,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const num_complex::Complex<T>,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: num_complex::Complex<T>,
    beta: num_complex::Complex<T>,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
) where
    T: Zero + Send + Sync + std::clone::Clone + num_traits::Num + core::ops::Neg<Output = T>,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Sub<&'a T, Output = T>,
    for<'a> &'a T: core::ops::Mul<&'a T, Output = T>,
{
    (0..m).for_each(|row| {
        (0..n).for_each(|col| {
            let mut accum = num_complex::Complex::<T>::zero();
            for depth in 0..k {
                let lhs = &*lhs.wrapping_offset(row as isize * lhs_rs + depth as isize * lhs_cs);
                let rhs = &*rhs.wrapping_offset(depth as isize * rhs_rs + col as isize * rhs_cs);

                match (conj_lhs, conj_rhs) {
                    (true, true) => accum = &accum + &(lhs.conj() * rhs.conj()),
                    (true, false) => accum = &accum + &(lhs.conj() * rhs),
                    (false, true) => accum = &accum + &(lhs * rhs.conj()),
                    (false, false) => accum = &accum + &(lhs * rhs),
                }
            }
            accum = &accum * &beta;

            let dst = dst.wrapping_offset(row as isize * dst_rs + col as isize * dst_cs);
            if read_dst {
                match conj_dst {
                    true => accum = &accum + &(&alpha * (*dst).conj()),
                    false => accum = &accum + &(&alpha * &*dst),
                }
            }
            *dst = accum
        });
    });
    return;
}
