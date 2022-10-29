use dyn_stack::DynStack;
use num_traits::{One, Zero};
use seq_macro::seq;

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
>(
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
    _stack: DynStack<'_>,
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
                            *dst =  mul_add(rhs~COL, lhs, *dst);
                        }
                    });
                }
            }
        }
        }
    match n {
        1 => do_work!(1),
        2 => do_work!(2),
        3 => do_work!(3),
        4 => do_work!(4),
        _ => unreachable!(),
    }
}
