use num_traits::{One, Zero};
use seq_macro::seq;

use crate::simd::Simd;

#[inline(always)]
pub unsafe fn gemv<
    T: Copy
        + Zero
        + One
        + Send
        + Sync
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::cmp::PartialEq,
    S: Simd,
>(
    _simd: S,
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    mul_add: impl Fn(T, T, T) -> T,
) {
    if !alpha.is_zero() {
        for col in 0..n {
            for row in 0..m {
                let dst = dst
                    .wrapping_offset(row as isize * dst_rs)
                    .wrapping_offset(col as isize * dst_cs);

                *dst = alpha * *dst;
            }
        }
    } else {
        for col in 0..n {
            for row in 0..m {
                let dst = dst
                    .wrapping_offset(row as isize * dst_rs)
                    .wrapping_offset(col as isize * dst_cs);

                *dst = T::zero();
            }
        }
    }

    macro_rules! do_work {
        ($n: tt) => {
            for depth in 0..k {
                seq!(COL in 0..$n {
                    let rhs~COL = beta * *rhs
                        .wrapping_offset(COL as isize * rhs_cs)
                        .wrapping_offset(depth as isize * rhs_rs);
                });
                for row in 0..m {
                    let lhs = *lhs
                        .wrapping_offset(depth as isize * lhs_cs)
                        .wrapping_offset(row as isize * lhs_rs);

                    seq!(COL in 0..$n {
                        {
                            let dst = dst
                                .wrapping_offset(COL as isize * dst_cs)
                                .wrapping_offset(row as isize * dst_rs);
                            *dst = mul_add(rhs~COL, lhs, *dst);
                        }
                    });
                }
            }
        }
    }
    match n {
        1 => {
            use std::arch::x86_64::*;
            assert!(dst_rs == 1);
            assert!(rhs_rs == 1);
            let lhs_cs = lhs_cs as usize;
            let lhs_rs = lhs_rs as usize;
            if lhs_rs == 1 {
                let m_round = m / 8 * 8;
                let rhs = rhs as *const f32;
                let lhs = lhs as *const f32;
                let dst = dst as *mut f32;
                for depth in 0..k {
                    // TODO: beta
                    let rhs = *rhs.add(depth);
                    let rhs_ = _mm256_set1_ps(rhs);
                    let lhs = lhs.add(depth * lhs_cs);
                    for row in (0..m_round).step_by(8) {
                        let lhs = _mm256_loadu_ps(lhs.add(row));
                        let dst = dst.add(row);
                        let dst_ = _mm256_loadu_ps(dst);
                        let dst_ = _mm256_fmadd_ps(rhs_, lhs, dst_);
                        _mm256_storeu_ps(dst, dst_)
                    }
                    for row in m_round..m {
                        let lhs = lhs.add(row);
                        let dst = dst.add(row);
                        *dst = rhs * *lhs + *dst;
                    }
                }
            } else if lhs_cs == 1 {
                let k_round = k / 8 * 8;
                let rhs = rhs as *const f32;
                for row in 0..m {
                    let lhs = lhs.add(row * lhs_rs) as *const f32;
                    let dst = dst.add(row) as *mut f32;
                    for depth in (0..k_round).step_by(8) {
                        let rhs = _mm256_loadu_ps(rhs.add(depth));
                        let lhs = _mm256_loadu_ps(lhs.add(depth));
                        let s = _mm256_mul_ps(lhs, rhs);
                        let res = _mm256_extractf128_ps(s, 1);
                        let res = _mm_add_ps(res, _mm256_castps256_ps128(s));
                        let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
                        let res = _mm_add_ps(res, _mm_movehdup_ps(res));
                        *dst = *dst + _mm_cvtss_f32(res)
                    }
                    for depth in k_round..k {
                        // beta
                        let rhs = rhs.add(depth);
                        let lhs = lhs.add(depth);
                        *dst = *dst + *rhs * *lhs;
                    }
                }
            } else {
                for depth in 0..k {
                    let rhs = beta * *rhs.add(depth);
                    let lhs = lhs.add(depth * lhs_cs);
                    for row in 0..m {
                        let lhs = lhs.add(row * lhs_rs);
                        let dst = dst.add(row);
                        *dst = mul_add(rhs, *lhs, *dst);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
}
