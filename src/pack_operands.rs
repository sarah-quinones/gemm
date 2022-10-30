use crate::simd::Simd;

#[inline(always)]
unsafe fn pack_generic_inner_loop<T: Copy, const N: usize, const DST_WIDTH: usize>(
    mut dst: *mut T,
    mut src: *const T,
    src_rs: isize,
    src_cs: isize,
    src_width: usize,
    k: usize,
) {
    if src_width == DST_WIDTH {
        if src_rs == 1 {
            for _ in 0..k {
                let val = (src as *const [T; DST_WIDTH]).read();
                (dst as *mut [T; DST_WIDTH]).write(val);

                src = src.wrapping_offset(src_cs);
                dst = dst.add(DST_WIDTH);
            }
        } else {
            for _ in 0..k {
                for j in 0..DST_WIDTH {
                    *dst.add(j) = *src.offset(j as isize * src_rs);
                }
                src = src.wrapping_offset(src_cs);
                dst = dst.add(DST_WIDTH);
            }
        }
    } else if src_width == N {
        if src_rs == 1 {
            for _ in 0..k {
                let val = (src as *const [T; N]).read();
                (dst as *mut [T; N]).write(val);

                src = src.wrapping_offset(src_cs);
                dst = dst.add(DST_WIDTH);
            }
        } else {
            for _ in 0..k {
                for j in 0..N {
                    *dst.add(j) = *src.offset(j as isize * src_rs);
                }
                src = src.wrapping_offset(src_cs);
                dst = dst.add(DST_WIDTH);
            }
        }
    } else if src_width == 2 * N {
        if src_rs == 1 {
            for _ in 0..k {
                let val0 = (src as *const [T; N]).read();
                let val1 = (src.add(N) as *const [T; N]).read();
                (dst as *mut [T; N]).write(val0);
                (dst.add(N) as *mut [T; N]).write(val1);

                src = src.wrapping_offset(src_cs);
                dst = dst.add(DST_WIDTH);
            }
        } else {
            for _ in 0..k {
                for j in 0..2 * N {
                    *dst.add(j) = *src.offset(j as isize * src_rs);
                }
                src = src.wrapping_offset(src_cs);
                dst = dst.add(DST_WIDTH);
            }
        }
    } else {
        for _ in 0..k {
            for j in 0..src_width {
                *dst.add(j) = *src.offset(j as isize * src_rs);
            }
            for j in src_width..DST_WIDTH {
                *dst.add(j) = core::mem::zeroed::<T>();
            }
            src = src.wrapping_offset(src_cs);
            dst = dst.add(DST_WIDTH);
        }
    }
}

#[inline(always)]
unsafe fn pack_generic<T: Copy, const N: usize, const DST_WIDTH: usize>(
    m: usize,
    k: usize,
    mut dst: *mut T,
    mut src: *const T,
    src_cs: isize,
    src_rs: isize,
    dst_stride: usize,
) {
    let m_width = m / DST_WIDTH * DST_WIDTH;

    let mut i = 0;
    while i < m_width {
        pack_generic_inner_loop::<_, N, DST_WIDTH>(dst, src, src_rs, src_cs, DST_WIDTH, k);
        src = src.wrapping_offset(src_rs * DST_WIDTH as isize);
        dst = dst.add(dst_stride);

        i += DST_WIDTH;
    }
    if i < m {
        pack_generic_inner_loop::<_, N, DST_WIDTH>(dst, src, src_rs, src_cs, m - i, k);
    }
}

#[inline(never)]
pub(crate) unsafe fn pack_lhs<T: Copy, const N: usize, const MR: usize, S: Simd>(
    _: S,
    m: usize,
    k: usize,
    dst: crate::Ptr<T>,
    src: crate::Ptr<T>,
    src_cs: isize,
    src_rs: isize,
    dst_stride: usize,
) {
    let dst = dst.0;
    let src = src.0;
    S::vectorize(
        #[inline(always)]
        || pack_generic::<T, N, MR>(m, k, dst, src, src_cs, src_rs, dst_stride),
    );
}

#[inline(never)]
pub(crate) unsafe fn pack_rhs<T: Copy, const N: usize, const NR: usize, S: Simd>(
    _: S,
    n: usize,
    k: usize,
    dst: crate::Ptr<T>,
    src: crate::Ptr<T>,
    src_cs: isize,
    src_rs: isize,
    dst_stride: usize,
) {
    let dst = dst.0;
    let src = src.0;
    S::vectorize(
        #[inline(always)]
        || pack_generic::<T, N, NR>(n, k, dst, src, src_rs, src_cs, dst_stride),
    );
}
