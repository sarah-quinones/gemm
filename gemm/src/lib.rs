#![cfg_attr(not(feature = "std"), no_std)]
#![warn(rust_2018_idioms)]

//! Enable the optional `spindle` feature for parallel GEMM inside
//! `spindle::with_lock`; otherwise GEMM uses Rayon directly.

mod gemm;

#[cfg(feature = "f16")]
pub use crate::gemm::f16;
pub use crate::gemm::{c32, c64, gemm};
pub use gemm_common::Parallelism;

pub use gemm_common::gemm::{
    get_lhs_packing_threshold_multi_thread, get_lhs_packing_threshold_single_thread,
    get_rhs_packing_threshold, get_threading_threshold, set_lhs_packing_threshold_multi_thread,
    set_lhs_packing_threshold_single_thread, set_rhs_packing_threshold, set_threading_threshold,
    DEFAULT_LHS_PACKING_THRESHOLD_MULTI_THREAD, DEFAULT_LHS_PACKING_THRESHOLD_SINGLE_THREAD,
    DEFAULT_RHS_PACKING_THRESHOLD, DEFAULT_THREADING_THRESHOLD,
};
pub use gemm_common::{get_wasm_simd128, set_wasm_simd128, DEFAULT_WASM_SIMD128};

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::{vec, vec::Vec};
    use num_traits::Float;

    #[cfg(all(
        feature = "rayon",
        any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos",
            target_os = "ios",
            target_os = "watchos",
            target_os = "windows",
            target_os = "freebsd",
        )
    ))]
    mod scheduling {
        #[cfg(feature = "spindle")]
        use crate::Parallelism;
        use gemm_common::gemm::par_for_each;
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

        /// Runs a test in a private Rayon pool with the requested worker count.
        fn with_pool(n_threads: usize, test: impl FnOnce() + Send) {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap()
                .install(test);
        }

        /// Verifies that a scheduling path executes every numbered job exactly once.
        fn assert_jobs(n_jobs: usize, run: impl FnOnce(&(dyn Fn(usize) + Send + Sync))) {
            let calls = (0..n_jobs).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
            let job = |idx: usize| {
                calls[idx].fetch_add(1, SeqCst);
            };

            run(&job);

            for (idx, calls) in calls.iter().enumerate() {
                assert_eq!(calls.load(SeqCst), 1, "job {idx} execution count");
            }
        }

        /// Checks packing and compute scheduling against an exact period-16 fixture.
        /// Positive integer products and sums stay below 2048, including the scaled
        /// destination, so f16 and f64 need no tolerance or second GEMM invocation.
        #[cfg(feature = "spindle")]
        fn assert_fixed_gemm<T: num_traits::Float + 'static>(convert: fn(f64) -> T) {
            const N: usize = 192;
            const K: usize = 64;
            for (m, row_major) in [(256, false), (95, true)] {
                let (lhs_cs, lhs_rs, rhs_cs, rhs_rs) = if row_major {
                    (1, K as isize, 1, N as isize)
                } else {
                    (m as isize, 1, K as isize, 1)
                };
                let lhs = (0..m * K)
                    .map(|idx| {
                        let i = idx / lhs_rs as usize % m;
                        let k = idx / lhs_cs as usize % K;
                        convert((1 + i % 4 + k % 4) as f64)
                    })
                    .collect::<Vec<_>>();
                let rhs = (0..K * N)
                    .map(|idx| {
                        let k = idx / rhs_rs as usize % K;
                        let j = idx / rhs_cs as usize % N;
                        convert((1 + j % 4 + (k / 4) % 4) as f64)
                    })
                    .collect::<Vec<_>>();
                let mut actual = (0..m * N)
                    .map(|idx| convert((2 * (idx % 11)) as f64))
                    .collect::<Vec<_>>();
                unsafe {
                    crate::gemm::gemm(
                        m,
                        N,
                        K,
                        actual.as_mut_ptr(),
                        m as isize,
                        1,
                        true,
                        lhs.as_ptr(),
                        lhs_cs,
                        lhs_rs,
                        rhs.as_ptr(),
                        rhs_cs,
                        rhs_rs,
                        convert(0.5),
                        convert(1.0),
                        false,
                        false,
                        false,
                        Parallelism::Rayon(0),
                    );
                }
                for (idx, actual) in actual.iter().enumerate() {
                    // All 16 depth pairs occur K/16 times; alpha * initial_dst = idx % 11.
                    let expected =
                        (K / 16) * (10 + 4 * (idx % m % 4)) * (10 + 4 * (idx / m % 4)) + idx % 11;
                    assert_eq!(actual.to_f64().unwrap(), expected as f64);
                }
            }
        }

        /// Runs one nested check on parent worker 1, not merely on job 1.
        /// Every parent participant must reach the barrier before any job can finish.
        #[cfg(feature = "spindle")]
        fn assert_worker_call(n_threads: usize, check: impl Fn() + Sync) {
            let ready = std::sync::Barrier::new(n_threads);
            let worker_calls = AtomicUsize::new(0);
            spindle::for_each_raw(n_threads, |_| {
                ready.wait();
                if spindle::current_thread_index() == Some(1) {
                    worker_calls.fetch_add(1, SeqCst);
                    check();
                }
            });
            assert_eq!(worker_calls.load(SeqCst), 1, "nested check on worker 1");
        }

        /// Checks every pool/job size outside a scope and, when enabled, inside one.
        #[test]
        fn test_par_for_each_scheduling() {
            for n_threads in [1, 2, 4] {
                with_pool(n_threads, || {
                    let check = |_in_scope: bool| {
                        for n_jobs in [0, 1, 2, 7] {
                            assert_jobs(n_jobs, |job| {
                                par_for_each(n_jobs, |idx| {
                                    #[cfg(feature = "spindle")]
                                    if _in_scope && n_jobs > 1 {
                                        assert!(spindle::current_thread_index().is_some());
                                    }
                                    job(idx);
                                })
                            });
                        }
                    };
                    check(false);
                    #[cfg(feature = "spindle")]
                    spindle::with_lock(n_threads, || check(true));
                });
            }
        }

        /// Requires a helper in an automatically created child scope. Parent jobs
        /// stay active until the first child callback, so the helper sees its request.
        #[cfg(feature = "spindle")]
        #[test]
        fn test_spindle_nested_helper() {
            with_pool(3, || {
                spindle::with_lock(3, || {
                    let ready = std::sync::Barrier::new(3);
                    let workers = AtomicUsize::new(0);
                    spindle::for_each_raw(3, |_| {
                        ready.wait();
                        if spindle::current_thread_index() == Some(1) {
                            assert_jobs(2, |job| {
                                par_for_each(2, |idx| {
                                    job(idx);
                                    let worker = spindle::current_thread_index()
                                        .expect("child job outside Spindle scope");
                                    assert!(worker < 2);
                                    let bit = 1 << worker;
                                    assert_eq!(workers.fetch_or(bit, SeqCst) & bit, 0);
                                    let deadline = std::time::Instant::now()
                                        + std::time::Duration::from_secs(10);
                                    while workers.load(SeqCst) != 3 {
                                        assert!(
                                            std::time::Instant::now() < deadline,
                                            "nested GEMM dispatch had no helper",
                                        );
                                        std::thread::yield_now();
                                    }
                                });
                            });
                        } else {
                            // Fail if the child never reaches its first callback.
                            let deadline =
                                std::time::Instant::now() + std::time::Duration::from_secs(10);
                            while workers.load(SeqCst) == 0 {
                                assert!(
                                    std::time::Instant::now() < deadline,
                                    "nested GEMM dispatch never started",
                                );
                                std::thread::yield_now();
                            }
                        }
                    });
                    assert_eq!(workers.load(SeqCst), 3);
                });
            });
        }

        /// Verifies default parallelism when a Spindle scope is narrower
        /// than its Rayon pool, including calls made by an active Spindle worker.
        #[cfg(feature = "spindle")]
        #[test]
        fn test_spindle_narrow_lock() {
            with_pool(8, || {
                spindle::with_lock(2, || {
                    let check = || {
                        assert_eq!(rayon::current_num_threads(), 8);
                        assert_eq!(spindle::max_num_threads(), 2);
                        par_for_each(0, |_| panic!("zero jobs must not execute"));
                        assert_jobs(8, |job| {
                            par_for_each(8, |idx| {
                                assert!(spindle::current_thread_index().is_some());
                                assert!(spindle::current_num_threads() <= 2);
                                job(idx);
                            });
                        });
                        assert_fixed_gemm(|value| value);
                        #[cfg(feature = "f16")]
                        assert_fixed_gemm(crate::f16::from_f64);
                    };
                    check();
                    assert_worker_call(2, check);
                });
            });
        }
    }

    #[test]
    fn test_gemm_f16() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((63, 2, 10));
        mnks.push((16, 2, 1));
        mnks.push((0, 0, 4));
        mnks.push((16, 1, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((64, 64, 0));
        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);

                        for colmajor in [true, false] {
                            let alpha = f16::from_f32(alpha);
                            let beta = f16::from_f32(beta);
                            let a_vec: Vec<f16> = (0..(m * k))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let b_vec: Vec<f16> = (0..(k * n))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let mut c_vec: Vec<f16> = (0..(m * n))
                                .map(|_| f16::from_f32(rand::random()))
                                .collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            let eps = f16::from_f32(1e-1);
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                let eps_rel = c.abs() * eps;
                                let eps_abs = eps;
                                let eps = if eps_rel > eps_abs { eps_rel } else { eps_abs };
                                assert_approx_eq::assert_approx_eq!(c, d, eps);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_f32() {
        set_wasm_simd128(true);

        let mut mnks = vec![];
        mnks.push((63, 2, 10));
        mnks.push((1, 2, 10));
        mnks.push((1, 63, 10));

        // large m to trigger parallelized rhs packing with big number of threads and small n
        mnks.push((2048, 255, 255));

        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 0));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);
                        for colmajor in [true, false] {
                            let a_vec: Vec<f32> = (0..(m * k)).map(|_| rand::random()).collect();
                            let b_vec: Vec<f32> = (0..(k * n)).map(|_| rand::random()).collect();
                            let mut c_vec: Vec<f32> =
                                (0..(m * n)).map(|_| rand::random()).collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                assert_approx_eq::assert_approx_eq!(c, d, 1e-3);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_f64() {
        set_wasm_simd128(true);

        let mut mnks = vec![];
        mnks.push((63, 2, 10));
        mnks.push((1, 2, 10));
        mnks.push((1, 63, 10));

        // large m to trigger parallelized rhs packing with big number of threads and small n
        mnks.push((2048, 255, 255));

        mnks.push((256, 256, 256));
        mnks.push((4096, 4096, 4));
        mnks.push((64, 64, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 0));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1, 1, 2));
        mnks.push((4, 4, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((63, 1, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);
            for parallelism in [
                Parallelism::None,
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(0),
                #[cfg(feature = "rayon")]
                Parallelism::Rayon(128),
            ] {
                for alpha in [0.0, 1.0, 2.3] {
                    for beta in [0.0, 1.0, 2.3] {
                        #[cfg(feature = "std")]
                        dbg!(alpha, beta, parallelism);
                        for colmajor in [true, false] {
                            let a_vec: Vec<f64> = (0..(m * k)).map(|_| rand::random()).collect();
                            let b_vec: Vec<f64> = (0..(k * n)).map(|_| rand::random()).collect();
                            let mut c_vec: Vec<f64> =
                                (0..(m * n)).map(|_| rand::random()).collect();
                            let mut d_vec = c_vec.clone();

                            unsafe {
                                gemm::gemm(
                                    m,
                                    n,
                                    k,
                                    c_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                    false,
                                    false,
                                    false,
                                    parallelism,
                                );

                                gemm::gemm_fallback(
                                    m,
                                    n,
                                    k,
                                    d_vec.as_mut_ptr(),
                                    if colmajor { m } else { 1 } as isize,
                                    if colmajor { 1 } else { n } as isize,
                                    true,
                                    a_vec.as_ptr(),
                                    m as isize,
                                    1,
                                    b_vec.as_ptr(),
                                    k as isize,
                                    1,
                                    alpha,
                                    beta,
                                );
                            }
                            for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                assert_approx_eq::assert_approx_eq!(c, d);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx32() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 4));
        mnks.push((64, 64, 0));
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            let zero = c32::new(0.0, 0.0);
            let one = c32::new(1.0, 0.0);
            let arbitrary = c32::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    #[cfg(feature = "std")]
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                #[cfg(feature = "std")]
                                dbg!(conj_dst);
                                #[cfg(feature = "std")]
                                dbg!(conj_lhs);
                                #[cfg(feature = "std")]
                                dbg!(conj_rhs);
                                for colmajor in [true, false] {
                                    let a_vec: Vec<f32> =
                                        (0..(2 * m * k)).map(|_| rand::random()).collect();
                                    let b_vec: Vec<f32> =
                                        (0..(2 * k * n)).map(|_| rand::random()).collect();
                                    let mut c_vec: Vec<f32> =
                                        (0..(2 * m * n)).map(|_| rand::random()).collect();
                                    let mut d_vec = c_vec.clone();

                                    unsafe {
                                        gemm::gemm(
                                            m,
                                            n,
                                            k,
                                            c_vec.as_mut_ptr() as *mut c32,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c32,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c32,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                            #[cfg(feature = "rayon")]
                                            Parallelism::Rayon(0),
                                            #[cfg(not(feature = "rayon"))]
                                            Parallelism::None,
                                        );

                                        gemm::gemm_cplx_fallback(
                                            m,
                                            n,
                                            k,
                                            d_vec.as_mut_ptr() as *mut c32,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c32,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c32,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                        );
                                    }
                                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                        assert_approx_eq::assert_approx_eq!(c, d, 1e-3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_gemm_cplx64() {
        let mut mnks = vec![];
        mnks.push((4, 4, 4));
        mnks.push((0, 64, 4));
        mnks.push((64, 0, 4));
        mnks.push((0, 0, 4));
        mnks.push((64, 64, 4));
        mnks.push((64, 64, 0));
        mnks.push((6, 3, 1));
        mnks.push((1, 1, 2));
        mnks.push((128, 128, 128));
        mnks.push((16, 1, 1));
        mnks.push((16, 2, 1));
        mnks.push((16, 3, 1));
        mnks.push((16, 4, 1));
        mnks.push((16, 1, 2));
        mnks.push((16, 2, 2));
        mnks.push((16, 3, 2));
        mnks.push((16, 4, 2));
        mnks.push((16, 16, 1));
        mnks.push((8, 16, 1));
        mnks.push((16, 8, 1));
        mnks.push((1024, 1024, 4));
        mnks.push((1024, 1024, 1));
        mnks.push((63, 1, 10));
        mnks.push((63, 2, 10));
        mnks.push((63, 3, 10));
        mnks.push((63, 4, 10));
        mnks.push((1, 63, 10));
        mnks.push((2, 63, 10));
        mnks.push((3, 63, 10));
        mnks.push((4, 63, 10));

        for (m, n, k) in mnks {
            #[cfg(feature = "std")]
            dbg!(m, n, k);

            let zero = c64::new(0.0, 0.0);
            let one = c64::new(1.0, 0.0);
            let arbitrary = c64::new(2.3, 4.1);
            for alpha in [zero, one, arbitrary] {
                for beta in [zero, one, arbitrary] {
                    #[cfg(feature = "std")]
                    dbg!(alpha, beta);
                    for conj_dst in [false, true] {
                        for conj_lhs in [false, true] {
                            for conj_rhs in [false, true] {
                                #[cfg(feature = "std")]
                                dbg!(conj_dst);
                                #[cfg(feature = "std")]
                                dbg!(conj_lhs);
                                #[cfg(feature = "std")]
                                dbg!(conj_rhs);
                                for colmajor in [true, false] {
                                    let a_vec: Vec<f64> =
                                        (0..(2 * m * k)).map(|_| rand::random()).collect();
                                    let b_vec: Vec<f64> =
                                        (0..(2 * k * n)).map(|_| rand::random()).collect();
                                    let mut c_vec: Vec<f64> =
                                        (0..(2 * m * n)).map(|_| rand::random()).collect();
                                    let mut d_vec = c_vec.clone();

                                    unsafe {
                                        gemm::gemm(
                                            m,
                                            n,
                                            k,
                                            c_vec.as_mut_ptr() as *mut c64,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c64,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c64,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                            #[cfg(feature = "rayon")]
                                            Parallelism::Rayon(0),
                                            #[cfg(not(feature = "rayon"))]
                                            Parallelism::None,
                                        );

                                        gemm::gemm_cplx_fallback(
                                            m,
                                            n,
                                            k,
                                            d_vec.as_mut_ptr() as *mut c64,
                                            if colmajor { m } else { 1 } as isize,
                                            if colmajor { 1 } else { n } as isize,
                                            true,
                                            a_vec.as_ptr() as *const c64,
                                            m as isize,
                                            1,
                                            b_vec.as_ptr() as *const c64,
                                            k as isize,
                                            1,
                                            alpha,
                                            beta,
                                            conj_dst,
                                            conj_lhs,
                                            conj_rhs,
                                        );
                                    }
                                    for (c, d) in c_vec.iter().zip(d_vec.iter()) {
                                        assert_approx_eq::assert_approx_eq!(c, d);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
