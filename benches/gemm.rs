use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use gemm::{gemm, gemm_req};
use nalgebra::DMatrix;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut mnks = vec![];
    mnks.push((256, 256, 256));
    mnks.push((256, 512, 256));
    mnks.push((512, 256, 256));
    mnks.push((1024, 1024, 1024));

    for (m, n, k) in mnks {
        {
            let a_vec = vec![0.0; m * k];
            let b_vec = vec![0.0; k * n];
            let mut c_vec = vec![0.0; m * n];

            let n_threads = 12;

            let mut mem = GlobalMemBuffer::new(gemm_req::<f64>(m, n, k, n_threads).unwrap());
            let mut stack = DynStack::new(&mut mem);
            c.bench_function(&format!("gemm-{}×{}×{}", m, n, k), |b| {
                b.iter(|| unsafe {
                    gemm(
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
            c.bench_function(&format!("nalg-{}×{}×{}", m, n, k), |b| {
                b.iter(|| c_mat = &a_mat * &b_mat)
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
