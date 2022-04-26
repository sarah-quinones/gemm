#[inline(always)]
unsafe fn pack_generic_inner_loop<T: Copy>(
    mut dst: *mut T,
    dst_width: usize,
    mut src: *const T,
    src_rs: isize,
    src_cs: isize,
    src_width: usize,
    k: usize,
) {
    for _ in 0..k {
        for j in 0..src_width {
            *dst.add(j) = *src.offset(j as isize * src_rs);
        }
        for j in src_width..dst_width {
            *dst.add(j) = core::mem::zeroed::<T>();
        }
        src = src.wrapping_offset(src_cs);
        dst = dst.add(dst_width);
    }
}

#[inline(always)]
unsafe fn pack_generic<T: Copy>(
    m: usize,
    k: usize,
    mut dst: *mut T,
    dst_width: usize,
    mut src: *const T,
    src_cs: isize,
    src_rs: isize,
    dst_stride: usize,
) {
    let m_width = m / dst_width * dst_width;

    let mut i = 0;
    while i < m_width {
        pack_generic_inner_loop(dst, dst_width, src, src_rs, src_cs, dst_width, k);
        src = src.wrapping_offset(src_rs * dst_width as isize);
        dst = dst.add(dst_stride);

        i += dst_width;
    }
    if i < m {
        pack_generic_inner_loop(dst, dst_width, src, src_rs, src_cs, m - i, k);
    }
}

#[inline(always)]
pub(crate) unsafe fn pack_lhs<T: Copy>(
    mr: usize,
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
    pack_generic::<T>(m, k, dst, mr, src, src_cs, src_rs, dst_stride);
}

#[inline(always)]
pub(crate) unsafe fn pack_rhs<T: Copy>(
    nr: usize,
    n: usize,
    k: usize,
    dst: crate::Ptr<T>,
    src: crate::Ptr<T>,
    src_cs: isize,
    src_rs: isize,
    dst_stride: usize,
) {
    pack_lhs::<T>(nr, n, k, dst, src, src_rs, src_cs, dst_stride);
}
