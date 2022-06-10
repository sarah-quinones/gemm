use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{uninit_mem_in_global, DynStack, ReborrowMut};
use gemm::gemm::{gemm_basic, gemm_req};
use nalgebra::DMatrix;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut mnks = vec![];
    mnks.push((256, 256, 256));
    mnks.push((1024, 1024, 1024));

    for (m, n, k) in mnks {
        {
            let a_vec = vec![0.0; m * k];
            let b_vec = vec![0.0; k * n];
            let mut c_vec = vec![0.0; m * n];

            let n_threads = rayon::current_num_threads();

            let mut mem = uninit_mem_in_global(gemm_req(m, n, k, n_threads).unwrap());
            let mut stack = DynStack::new(&mut mem);
            c.bench_function(&format!("lib-{}×{}×{}", m, n, k), |b| {
                b.iter(|| unsafe {
                    gemm_basic(
                        m,
                        n,
                        k,
                        c_vec.as_mut_ptr(),
                        m as isize,
                        1,
                        true,
                        a_vec.as_ptr(),
                        m as isize,
                        1,
                        b_vec.as_ptr(),
                        k as isize,
                        1,
                        0.0,
                        0.0,
                        n_threads,
                        stack.rb_mut(),
                    )
                })
            });
        }

        {
            let a_mat = DMatrix::<f64>::zeros(m, k);
            let b_mat = DMatrix::<f64>::zeros(k, n);
            let mut c_mat = DMatrix::<f64>::zeros(m, n);
            c.bench_function(&format!("alg-{}×{}×{}", m, n, k), |b| {
                b.iter(|| c_mat = &a_mat * &b_mat)
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
