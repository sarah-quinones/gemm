// Benchmark targeting thread oversubscription on thin / small matrices

use aligned_vec::{avec, AVec};
use diol::prelude::*;
use gemm::*;
use num_traits::One;

fn make_col_major<T: Copy + One>(rows: usize, cols: usize, reg: usize) -> (isize, isize, AVec<T>) {
    // column-major: row-stride 1, col-stride = padded number of rows.
    let col_stride = rows.next_multiple_of(reg);
    (1, col_stride as isize, avec![T::one(); cols * col_stride])
}

fn bench_gemm<T: One + Copy + 'static>(
    bencher: Bencher,
    list![par, m, n, k]: List![Parallelism, usize, usize, usize],
) {
    let reg = 64 / core::mem::size_of::<T>();

    let (dst_rs, dst_cs, mut dst) = make_col_major::<T>(m, n, reg);
    let (lhs_rs, lhs_cs, mut lhs) = make_col_major::<T>(m, k, reg);
    let (rhs_rs, rhs_cs, mut rhs) = make_col_major::<T>(k, n, reg);

    lhs.fill(unsafe { core::mem::zeroed() });
    rhs.fill(unsafe { core::mem::zeroed() });
    dst.fill(unsafe { core::mem::zeroed() });

    bencher.bench(|| unsafe {
        gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            dst_cs,
            dst_rs,
            true,
            lhs.as_ptr(),
            lhs_cs,
            lhs_rs,
            rhs.as_ptr(),
            rhs_cs,
            rhs_rs,
            T::one(),
            T::one(),
            false,
            false,
            false,
            par,
        );
    })
}

/// Thin / small shapes that clear the MAC threshold yet expose only a modest
/// number of tiles -- exactly where the old all-or-nothing gate oversubscribed.
fn shapes() -> Vec<(usize, usize, usize)> {
    vec![
        // tall & skinny: many rows, very few columns
        (4096, 2, 512),
        (4096, 4, 1024),
        (8192, 8, 512),
        // short & wide: very few rows, many columns
        (2, 4096, 512),
        (4, 4096, 1024),
        (8, 8192, 512),
        // small in m & n but fat reduction dim k
        (32, 32, 4096),
        (48, 48, 4096),
        (64, 64, 2048),
        // small-ish square: just over the threshold, few tiles
        (128, 128, 256),
        (192, 192, 256),
    ]
}

fn args() -> Vec<List![Parallelism, usize, usize, usize]> {
    let mut out = Vec::new();
    for (m, n, k) in shapes() {
        for par in [Parallelism::Rayon(0), Parallelism::None] {
            out.push(list![par, m, n, k]);
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let config = BenchConfig::from_args()?;

    gemm::set_wasm_simd128(true);

    let mut bench = Bench::new(&config);
    bench.register(bench_gemm::<f32>, args());
    bench.register(bench_gemm::<f64>, args());
    bench.register(bench_gemm::<c32>, args());
    bench.register(bench_gemm::<c64>, args());
    bench.run()?;

    Ok(())
}
