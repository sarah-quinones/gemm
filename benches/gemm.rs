use criterion::{criterion_group, criterion_main, Criterion};
use dyn_stack::{uninit_mem, DynStack, ReborrowMut};
use gemm::gemm::{gemm_basic, gemm_req};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut mnks = vec![];
    mnks.push((256, 256, 256));
    mnks.push((1024, 1024, 1024));

    let mut mem = uninit_mem(gemm_req(12));
    let mut stack = DynStack::new(&mut mem);

    for (m, n, k) in mnks {
        let a_vec = vec![0.0; m * k];
        let b_vec = vec![0.0; k * n];
        let mut c_vec = vec![0.0; m * n];
        c.bench_function(&format!("{}×{}×{}", m, n, k), |b| {
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
                    12,
                    stack.rb_mut(),
                )
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
