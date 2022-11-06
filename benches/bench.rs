use criterion::{criterion_group, criterion_main, Criterion};
use gemm::gemm;
use nalgebra::DMatrix;
use std::time::Duration;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut mnks = vec![];
    mnks.push((896, 128, 128));
    mnks.push((256, 32, 256));
    mnks.push((48, 48, 256));
    mnks.push((52, 52, 256));
    mnks.push((256, 256, 256));
    mnks.push((256, 512, 256));
    mnks.push((512, 256, 256));
    mnks.push((1024, 1024, 1024));
    mnks.push((63, 1, 10));
    mnks.push((63, 2, 10));
    mnks.push((63, 3, 10));
    mnks.push((63, 4, 10));
    mnks.push((1, 63, 10));
    mnks.push((2, 63, 10));
    mnks.push((3, 63, 10));
    mnks.push((4, 63, 10));

    for (m, n, k) in mnks {
        {
            let a_vec = vec![0.0; m * k];
            let b_vec = vec![0.0; k * n];
            let mut c_vec = vec![0.0; m * n];

            for (dst_label, dst_cs, dst_rs) in [("n", m, 1), ("t", 1, n)] {
                for (lhs_label, lhs_cs, lhs_rs) in [("n", m, 1), ("t", 1, k)] {
                    for (rhs_label, rhs_cs, rhs_rs) in [("n", k, 1), ("t", 1, n)] {
                        c.bench_function(
                            &format!(
                                "{}{}{}-gemm-{}×{}×{}",
                                dst_label, lhs_label, rhs_label, m, n, k
                            ),
                            |b| {
                                b.iter(|| unsafe {
                                    gemm(
                                        m,
                                        n,
                                        k,
                                        c_vec.as_mut_ptr(),
                                        dst_cs as isize,
                                        dst_rs as isize,
                                        true,
                                        a_vec.as_ptr(),
                                        lhs_cs as isize,
                                        lhs_rs as isize,
                                        b_vec.as_ptr(),
                                        rhs_cs as isize,
                                        rhs_rs as isize,
                                        0.0,
                                        0.0,
                                    )
                                })
                            },
                        );
                    }
                }
            }
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

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = criterion_benchmark
);
criterion_main!(benches);
