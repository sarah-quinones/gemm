pub fn run(m: usize, n: usize, k: usize, niters: usize) {
    let a_vec = vec![0f32; m * k];
    let b_vec = vec![0f32; k * n];
    let mut c_vec = vec![0f32; m * n];
    let (dst_rs, dst_cs) = (n, 1);
    let (lhs_rs, lhs_cs) = (k, 1);
    let (rhs_rs, rhs_cs) = (1, k);
    let mut run = move || unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c_vec.as_mut_ptr(),
            dst_cs as isize,
            dst_rs as isize,
            false,
            a_vec.as_ptr(),
            lhs_cs as isize,
            lhs_rs as isize,
            b_vec.as_ptr(),
            rhs_cs as isize,
            rhs_rs as isize,
            0f32,
            1f32,
            false,
            false,
            false,
            gemm::Parallelism::None,
        )
    };
    for _i in 0..10 {
        run();
    }
    let start = std::time::Instant::now();
    for _iter in 0..niters {
        run()
    }
    let duration = start.elapsed();
    println!("{:?}", duration / niters as u32);
}

fn main() {
    run(1, 768, 288, 1000)
}
