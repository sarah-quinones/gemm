#[inline(always)]
fn prefetch(addr: *const ()) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(addr as _);
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        core::arch::x86::_mm_prefetch::<{ core::arch::x86::_MM_HINT_T0 }>(addr as _);
    }

    let _ = addr;
}

pub(crate) type MicroKernelFn<T> = unsafe fn(
    usize,
    usize,
    usize,
    *mut T,
    *const T,
    *const T,
    isize,
    isize,
    isize,
    isize,
    isize,
    T,
    T,
    u8,
    bool,
    bool,
    bool,
    *const T,
);

// microkernel_fn_array!{
// [ a, b, c, ],
// [ d, e, f, ],
// }
//
// expands to
// pub const UKR: [[[MicroKernelFn; 3]; 2]; 3] = [
// [
// [ a::<0>, b::<0>, c::<0>, ],
// [ d::<0>, e::<0>, f::<0>, ],
// ],
// [
// [ a::<1>, b::<1>, c::<1>, ],
// [ d::<1>, e::<1>, f::<1>, ],
// ],
// [
// [ a::<2>, b::<2>, c::<2>, ],
// [ d::<2>, e::<2>, f::<2>, ],
// ],
// ]
macro_rules! __one {
    (
        $tt: tt
    ) => {
        1
    };
}

macro_rules! __first {
    (
        $first: tt, $($rest: tt,)*
    ) => {
        $first
    };
}

macro_rules! __microkernel_fn_array_helper {
    (
        [ $($tt: tt,)* ]
    ) => {
        {
            let mut count = 0_usize;
            $(count += __one!($tt);)*
            count
        }
    }
}

macro_rules! __microkernel_fn_array_helper_nr {
    ($([
       $($ukr: ident,)*
    ],)*) => {
        {
            let counts = [$({
                let mut count = 0_usize;
                $(count += __one!($ukr);)*
                count
            },)*];

            counts[0]
        }
    }
}

macro_rules! microkernel_fn_array {
    ($([
       $($ukr: ident,)*
    ],)*) => {
       pub const MR_DIV_N: usize =
           __microkernel_fn_array_helper!([$([$($ukr,)*],)*]);
       pub const NR: usize =
           __microkernel_fn_array_helper_nr!($([$($ukr,)*],)*);

        pub const UKR: [[super::super::MicroKernelFn<T>; NR]; MR_DIV_N] =
            [ $([$($ukr,)*]),* ];
    };
}

macro_rules! microkernel_cplx_fn_array {
    ($([
       $($ukr: ident,)*
    ],)*) => {
       pub const CPLX_MR_DIV_N: usize =
           __microkernel_fn_array_helper!([$([$($ukr,)*],)*]);
       pub const CPLX_NR: usize =
           __microkernel_fn_array_helper_nr!($([$($ukr,)*],)*);

        pub const CPLX_UKR: [[super::super::MicroKernelFn<num_complex::Complex<T>>; CPLX_NR]; CPLX_MR_DIV_N] =
            [ $([$($ukr,)*]),* ];
    };
}

macro_rules! microkernel {
    ($([$target: tt])?, $name: ident, $mr_div_n: tt, $nr: tt $(, $nr_div_n: tt, $n: tt)?) => {
        #[inline]
        $(#[target_feature(enable = $target)])?
        // 0, 1, or 2 for generic alpha
        pub unsafe fn $name(
            m: usize,
            n: usize,
            k: usize,
            dst: *mut T,
            mut packed_lhs: *const T,
            mut packed_rhs: *const T,
            dst_cs: isize,
            dst_rs: isize,
            lhs_cs: isize,
            rhs_rs: isize,
            rhs_cs: isize,
            alpha: T,
            beta: T,
            alpha_status: u8,
            _conj_dst: bool,
            _conj_lhs: bool,
            _conj_rhs: bool,
            mut next_lhs: *const T,
        ) {
            let mut accum_storage = [[splat(0.0); $mr_div_n]; $nr];
            let accum = accum_storage.as_mut_ptr() as *mut Pack;

            let mut lhs = [::core::mem::MaybeUninit::<Pack>::uninit(); $mr_div_n];
            let mut rhs = ::core::mem::MaybeUninit::<Pack>::uninit();

            #[derive(Copy, Clone)]
            struct KernelIter {
                packed_lhs: *const T,
                packed_rhs: *const T,
                next_lhs: *const T,
                lhs_cs: isize,
                rhs_rs: isize,
                rhs_cs: isize,
                accum: *mut Pack,
                lhs: *mut Pack,
                rhs: *mut Pack,
            }

            seq_macro::seq!(N_ITER in 0..$nr {{
                seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                    let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *const ();
                    crate::microkernel::prefetch(dst);
                }});
            }});

            impl KernelIter {
                #[inline(always)]
                unsafe fn execute(self, iter: usize) {
                    let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);
                    let next_lhs = self.next_lhs.wrapping_offset(iter as isize * self.lhs_cs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        *self.lhs.add(M_ITER) = (packed_lhs.add(M_ITER * N) as *const Pack).read_unaligned();
                    }});

                    seq_macro::seq!(N_ITER in 0..$nr {{
                        *self.rhs = splat(*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs));
                        let accum = self.accum.add(N_ITER * $mr_div_n);
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let accum = &mut *accum.add(M_ITER);
                            *accum = mul_add(
                                *self.lhs.add(M_ITER),
                                *self.rhs,
                                *accum,
                                );
                        }});
                    }});

                    let _ = next_lhs;
                }

                $(
                    #[inline(always)]
                    unsafe fn execute_neon(self, iter: usize) {
                        debug_assert_eq!(self.rhs_cs, 1);
                        let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                        let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);

                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            *self.lhs.add(M_ITER) = (packed_lhs.add(M_ITER * N) as *const Pack).read_unaligned();
                        }});

                        seq_macro::seq!(N_ITER0 in 0..$nr_div_n {{
                            *self.rhs = (packed_rhs.wrapping_offset(N_ITER0 * $n) as *const Pack).read_unaligned();

                            seq_macro::seq!(N_ITER1 in 0..$n {{
                                const N_ITER: usize = N_ITER0 * $n + N_ITER1;
                                let accum = self.accum.add(N_ITER * $mr_div_n);
                                seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                    let accum = &mut *accum.add(M_ITER);
                                    *accum = mul_add_lane::<N_ITER1>(
                                        *self.lhs.add(M_ITER),
                                        *self.rhs,
                                        *accum,
                                        );
                                }});
                            }});
                        }});
                    }
                )?
            }

            let k_unroll = k / 4;
            let k_leftover = k % 4;

            loop {
                $(
                let _ = $nr_div_n;
                if rhs_cs == 1 {
                    let mut depth = k_unroll;
                    if depth != 0 {
                        loop {
                            let iter = KernelIter {
                                packed_lhs,
                                next_lhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                rhs_cs,
                                accum,
                                lhs: lhs.as_mut_ptr() as _,
                                rhs: &mut rhs as *mut _ as _,
                            };

                            seq_macro::seq!(UNROLL_ITER in 0..4 {{
                                iter.execute_neon(UNROLL_ITER);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset(4 * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset(4 * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset(4 * lhs_cs);

                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    depth = k_leftover;
                    if depth != 0 {
                        loop {
                            KernelIter {
                                packed_lhs,
                                next_lhs,
                                packed_rhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                rhs_cs,
                                accum,
                                lhs: lhs.as_mut_ptr() as _,
                                rhs: &mut rhs as *mut _ as _,
                            }
                            .execute_neon(0);

                            packed_lhs = packed_lhs.wrapping_offset(lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset(rhs_rs);
                            next_lhs = next_lhs.wrapping_offset(lhs_cs);

                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    break;
                }
                )?

                let mut depth = k_unroll;
                if depth != 0 {
                    loop {
                        let iter = KernelIter {
                            packed_lhs,
                            next_lhs,
                            packed_rhs,
                            lhs_cs,
                            rhs_rs,
                            rhs_cs,
                            accum,
                            lhs: lhs.as_mut_ptr() as _,
                            rhs: &mut rhs as *mut _ as _,
                        };

                        seq_macro::seq!(UNROLL_ITER in 0..4 {{
                            iter.execute(UNROLL_ITER);
                        }});

                        packed_lhs = packed_lhs.wrapping_offset(4 * lhs_cs);
                        packed_rhs = packed_rhs.wrapping_offset(4 * rhs_rs);
                        next_lhs = next_lhs.wrapping_offset(4 * lhs_cs);

                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
                depth = k_leftover;
                if depth != 0 {
                    loop {
                        KernelIter {
                            packed_lhs,
                            next_lhs,
                            packed_rhs,
                            lhs_cs,
                            rhs_rs,
                            rhs_cs,
                            accum,
                            lhs: lhs.as_mut_ptr() as _,
                            rhs: &mut rhs as *mut _ as _,
                        }
                        .execute(0);

                        packed_lhs = packed_lhs.wrapping_offset(lhs_cs);
                        packed_rhs = packed_rhs.wrapping_offset(rhs_rs);
                        next_lhs = next_lhs.wrapping_offset(lhs_cs);

                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
                break;
            }

            if m == $mr_div_n * N && n == $nr {
                let alpha = splat(alpha);
                let beta = splat(beta);
                if dst_rs == 1 {

                    if alpha_status == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(add(
                                        mul(alpha, dst.read_unaligned()),
                                        mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
                                        ));
                            }});
                        }});
                    } else if alpha_status == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(mul_add(
                                        beta,
                                        *accum.offset(M_ITER + $mr_div_n * N_ITER),
                                        dst.read_unaligned(),
                                        ));
                            }});
                        }});
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                dst.write_unaligned(mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)));
                            }});
                        }});
                    }
                } else {
                    if alpha_status == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize * dst_rs + N_ITER * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    add(
                                        mul(alpha, gather(dst, dst_rs)),
                                        mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
                                    ),
                                );
                            }});
                        }});
                    } else if alpha_status == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize * dst_rs + N_ITER * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    mul_add(
                                        beta,
                                        *accum.offset(M_ITER + $mr_div_n * N_ITER),
                                        gather(dst, dst_rs),
                                    ),
                                );
                            }});
                        }});
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize * dst_rs + N_ITER * dst_cs);
                                scatter(
                                    dst,
                                    dst_rs,
                                    mul(beta, *accum.offset(M_ITER + $mr_div_n * N_ITER)),
                                );
                            }});
                        }});
                    }
                }
            } else {
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const T;

                if alpha_status == 2 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = alpha * *dst_ij + beta * *src_ij;
                        }
                    }
                } else if alpha_status == 1 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = *dst_ij + beta * *src_ij;
                        }
                    }
                } else {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = beta * *src_ij;
                        }
                    }
                }
            }
        }
    };
}

macro_rules! microkernel_cplx {
    ($([$target: tt])?, $name: ident, $mr_div_n: tt, $nr: tt) => {
        #[inline]
        $(#[target_feature(enable = $target)])?
        // 0, 1, or 2 for generic alpha
        pub unsafe fn $name(
            m: usize,
            n: usize,
            k: usize,
            dst: *mut num_complex::Complex<T>,
            mut packed_lhs: *const num_complex::Complex<T>,
            mut packed_rhs: *const num_complex::Complex<T>,
            dst_cs: isize,
            dst_rs: isize,
            lhs_cs: isize,
            rhs_rs: isize,
            rhs_cs: isize,
            alpha: num_complex::Complex<T>,
            beta: num_complex::Complex<T>,
            alpha_status: u8,
            conj_dst: bool,
            conj_lhs: bool,
            conj_rhs: bool,
            mut next_lhs: *const num_complex::Complex<T>,
        ) {
            let mut accum_storage = [[splat(0.0); $mr_div_n]; $nr];
            let accum = accum_storage.as_mut_ptr() as *mut Pack;

            let conj_both_lhs_rhs = conj_lhs;
            let conj_rhs = conj_lhs != conj_rhs;

            let mut lhs_re_im = [::core::mem::MaybeUninit::<Pack>::uninit(); $mr_div_n];
            let mut lhs_im_re = [::core::mem::MaybeUninit::<Pack>::uninit(); $mr_div_n];
            let mut rhs_re = ::core::mem::MaybeUninit::<Pack>::uninit();
            let mut rhs_im = ::core::mem::MaybeUninit::<Pack>::uninit();

            #[derive(Copy, Clone)]
            struct KernelIter {
                packed_lhs: *const num_complex::Complex<T>,
                next_lhs: *const num_complex::Complex<T>,
                packed_rhs: *const num_complex::Complex<T>,
                lhs_cs: isize,
                rhs_rs: isize,
                rhs_cs: isize,
                accum: *mut Pack,
                lhs_re_im: *mut Pack,
                lhs_im_re: *mut Pack,
                rhs_re: *mut Pack,
                rhs_im: *mut Pack,
            }

            impl KernelIter {
                #[inline(always)]
                unsafe fn execute(self, iter: usize, conj_rhs: bool) {
                    let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);
                    let next_lhs = self.next_lhs.wrapping_offset(iter as isize * self.lhs_cs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        let tmp = (packed_lhs.add(M_ITER * CPLX_N) as *const Pack).read_unaligned();
                        *self.lhs_re_im.add(M_ITER) = tmp;
                        *self.lhs_im_re.add(M_ITER) = swap_re_im(tmp);
                    }});

                    seq_macro::seq!(N_ITER in 0..$nr {{
                        *self.rhs_re = splat((*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs)).re);
                        *self.rhs_im = splat((*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs)).im);

                        let accum = self.accum.add(N_ITER * $mr_div_n);
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let accum = &mut *accum.add(M_ITER);
                            *accum = mul_add_cplx(
                                *self.lhs_re_im.add(M_ITER),
                                *self.lhs_im_re.add(M_ITER),
                                *self.rhs_re,
                                *self.rhs_im,
                                *accum,
                                conj_rhs,
                                );
                        }});
                    }});

                    let _ = next_lhs;
                }
            }

            let k_unroll = k / 4;
            let k_leftover = k % 4;

            loop {
                if conj_rhs {
                    let mut depth = k_unroll;
                    if depth != 0 {
                        loop {
                            let iter = KernelIter {
                                packed_lhs,
                                next_lhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                rhs_cs,
                                accum,
                                lhs_re_im: lhs_re_im.as_mut_ptr() as _,
                                lhs_im_re: lhs_im_re.as_mut_ptr() as _,
                                rhs_re: &mut rhs_re as *mut _ as _,
                                rhs_im: &mut rhs_im as *mut _ as _,
                            };

                            seq_macro::seq!(UNROLL_ITER in 0..4 {{
                                iter.execute(UNROLL_ITER, true);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset(4 * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset(4 * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset(4 * lhs_cs);

                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    depth = k_leftover;
                    if depth != 0 {
                        loop {
                            KernelIter {
                                packed_lhs,
                                next_lhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                rhs_cs,
                                accum,
                                lhs_re_im: lhs_re_im.as_mut_ptr() as _,
                                lhs_im_re: lhs_im_re.as_mut_ptr() as _,
                                rhs_re: &mut rhs_re as *mut _ as _,
                                rhs_im: &mut rhs_im as *mut _ as _,
                            }
                            .execute(0, true);

                            packed_lhs = packed_lhs.wrapping_offset(lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset(rhs_rs);
                            next_lhs = next_lhs.wrapping_offset(lhs_cs);

                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    break;
                } else {
                    let mut depth = k_unroll;
                    if depth != 0 {
                        loop {
                            let iter = KernelIter {
                                next_lhs,
                                packed_lhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                rhs_cs,
                                accum,
                                lhs_re_im: lhs_re_im.as_mut_ptr() as _,
                                lhs_im_re: lhs_im_re.as_mut_ptr() as _,
                                rhs_re: &mut rhs_re as *mut _ as _,
                                rhs_im: &mut rhs_im as *mut _ as _,
                            };

                            seq_macro::seq!(UNROLL_ITER in 0..4 {{
                                iter.execute(UNROLL_ITER, false);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset(4 * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset(4 * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset(4 * lhs_cs);

                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    depth = k_leftover;
                    if depth != 0 {
                        loop {
                            KernelIter {
                                next_lhs,
                                packed_lhs,
                                packed_rhs,
                                lhs_cs,
                                rhs_rs,
                                rhs_cs,
                                accum,
                                lhs_re_im: lhs_re_im.as_mut_ptr() as _,
                                lhs_im_re: lhs_im_re.as_mut_ptr() as _,
                                rhs_re: &mut rhs_re as *mut _ as _,
                                rhs_im: &mut rhs_im as *mut _ as _,
                            }
                            .execute(0, false);

                            packed_lhs = packed_lhs.wrapping_offset(lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset(rhs_rs);
                            next_lhs = next_lhs.wrapping_offset(lhs_cs);

                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                    break;
                }
            }

            if conj_both_lhs_rhs {
                seq_macro::seq!(N_ITER in 0..$nr {{
                    let accum = accum.add(N_ITER * $mr_div_n);
                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        let accum = &mut *accum.add(M_ITER);
                        *accum = conj(*accum);
                    }});
                }});
            }

            if m == $mr_div_n * CPLX_N && n == $nr && dst_rs == 1 {
                let alpha_re = splat(alpha.re);
                let alpha_im = splat(alpha.im);
                let beta_re = splat(beta.re);
                let beta_im = splat(beta.im);

                if conj_dst {
                    if alpha_status == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * CPLX_N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    mul_cplx(conj(*dst), swap_re_im(conj(*dst)), alpha_re, alpha_im),
                                    mul_cplx(accum, swap_re_im(accum), beta_re, beta_im),
                                    );
                            }});
                        }});
                    } else if alpha_status == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * CPLX_N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    conj(*dst),
                                    mul_cplx(accum, swap_re_im(accum), beta_re, beta_im),
                                    );
                            }});
                        }});
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * CPLX_N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = mul_cplx(accum, swap_re_im(accum), beta_re, beta_im);
                            }});
                        }});
                    }
                } else {
                    if alpha_status == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * CPLX_N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    mul_cplx(*dst, swap_re_im(*dst), alpha_re, alpha_im),
                                    mul_cplx(accum, swap_re_im(accum), beta_re, beta_im),
                                );
                            }});
                        }});
                    } else if alpha_status == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * CPLX_N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    *dst,
                                    mul_cplx(accum, swap_re_im(accum), beta_re, beta_im),
                                );
                            }});
                        }});
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * CPLX_N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = mul_cplx(accum, swap_re_im(accum), beta_re, beta_im);
                            }});
                        }});
                    }
                }
            } else {
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const num_complex::Complex<T>;

                if conj_dst {
                    if alpha_status == 2 {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * CPLX_N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = alpha * (*dst_ij).conj() + beta * *src_ij;
                            }
                        }
                    } else if alpha_status == 1 {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * CPLX_N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = (*dst_ij).conj() + beta * *src_ij;
                            }
                        }
                    } else {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * CPLX_N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = beta * *src_ij;
                            }
                        }
                    }
                } else {
                    if alpha_status == 2 {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * CPLX_N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = alpha * *dst_ij + beta * *src_ij;
                            }
                        }
                    } else if alpha_status == 1 {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * CPLX_N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = *dst_ij + beta * *src_ij;
                            }
                        }
                    } else {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * CPLX_N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = beta * *src_ij;
                            }
                        }
                    }
                }
            }
        }
    };
}

pub mod scalar {
    pub mod f32 {
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..1 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..1 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value]
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] * rhs[0]]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0]]
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(, x1x1, 1, 1);
        microkernel!(, x1x2, 1, 2);
        microkernel!(, x1x3, 1, 3);
        microkernel!(, x1x4, 1, 4);

        microkernel!(, x2x1, 2, 1);
        microkernel!(, x2x2, 2, 2);
        microkernel!(, x2x3, 2, 3);
        microkernel!(, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
        pub use super::c32::*;
    }

    pub mod f64 {
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..1 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..1 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value]
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] * rhs[0]]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0]]
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(, x1x1, 1, 1);
        microkernel!(, x1x2, 1, 2);
        microkernel!(, x1x3, 1, 3);
        microkernel!(, x1x4, 1, 4);

        microkernel!(, x2x1, 2, 1);
        microkernel!(, x2x2, 2, 2);
        microkernel!(, x2x3, 2, 3);
        microkernel!(, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }

        pub use super::c64::*;
    }

    pub mod c32 {
        type T = f32;
        const N: usize = 2;
        const CPLX_N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value, value]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0], lhs[1] + rhs[1]]
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            [a[0], -a[1]]
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            [a[1], a[0]]
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, _a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            [
                a_re_im[0] * b_re[0] - a_re_im[1] * b_im[0],
                a_re_im[1] * b_re[0] + a_re_im[0] * b_im[0],
            ]
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                add(
                    c_re_im,
                    mul_cplx(a_re_im, a_im_re, b_re, [-b_im[0], -b_im[1]]),
                )
            } else {
                add(c_re_im, mul_cplx(a_re_im, a_im_re, b_re, b_im))
            }
        }

        microkernel_cplx!(, x1x1, 1, 1);
        microkernel_cplx!(, x1x2, 1, 2);
        microkernel_cplx!(, x1x3, 1, 3);
        microkernel_cplx!(, x1x4, 1, 4);

        microkernel_cplx!(, x2x1, 2, 1);
        microkernel_cplx!(, x2x2, 2, 2);
        microkernel_cplx!(, x2x3, 2, 3);
        microkernel_cplx!(, x2x4, 2, 4);

        microkernel_cplx_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
    pub mod c64 {
        type T = f64;
        const N: usize = 2;
        const CPLX_N: usize = 1;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            [value, value]
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            [lhs[0] + rhs[0], lhs[1] + rhs[1]]
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            [a[0], -a[1]]
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            [a[1], a[0]]
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, _a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            [
                a_re_im[0] * b_re[0] - a_re_im[1] * b_im[0],
                a_re_im[1] * b_re[0] + a_re_im[0] * b_im[0],
            ]
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                add(
                    c_re_im,
                    mul_cplx(a_re_im, a_im_re, b_re, [-b_im[0], -b_im[1]]),
                )
            } else {
                add(c_re_im, mul_cplx(a_re_im, a_im_re, b_re, b_im))
            }
        }

        microkernel_cplx!(, x1x1, 1, 1);
        microkernel_cplx!(, x1x2, 1, 2);
        microkernel_cplx!(, x1x3, 1, 3);
        microkernel_cplx!(, x1x4, 1, 4);

        microkernel_cplx!(, x2x1, 2, 1);
        microkernel_cplx!(, x2x2, 2, 2);
        microkernel_cplx!(, x2x3, 2, 3);
        microkernel_cplx!(, x2x4, 2, 4);

        microkernel_cplx_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..4 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..4 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["sse"], x1x1, 1, 1);
        microkernel!(["sse"], x1x2, 1, 2);
        microkernel!(["sse"], x1x3, 1, 3);
        microkernel!(["sse"], x1x4, 1, 4);

        microkernel!(["sse"], x2x1, 2, 1);
        microkernel!(["sse"], x2x2, 2, 2);
        microkernel!(["sse"], x2x3, 2, 3);
        microkernel!(["sse"], x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }

    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 2;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..2 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..2 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["sse"], x1x1, 1, 1);
        microkernel!(["sse"], x1x2, 1, 2);
        microkernel!(["sse"], x1x3, 1, 3);
        microkernel!(["sse"], x1x4, 1, 4);

        microkernel!(["sse"], x2x1, 2, 1);
        microkernel!(["sse"], x2x2, 2, 2);
        microkernel!(["sse"], x2x3, 2, 3);
        microkernel!(["sse"], x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..8 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..8 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["avx"], x1x1, 1, 1);
        microkernel!(["avx"], x1x2, 1, 2);
        microkernel!(["avx"], x1x3, 1, 3);
        microkernel!(["avx"], x1x4, 1, 4);

        microkernel!(["avx"], x2x1, 2, 1);
        microkernel!(["avx"], x2x2, 2, 2);
        microkernel!(["avx"], x2x3, 2, 3);
        microkernel!(["avx"], x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }

    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..4 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..4 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            add(mul(a, b), c)
        }

        microkernel!(["avx"], x1x1, 1, 1);
        microkernel!(["avx"], x1x2, 1, 2);
        microkernel!(["avx"], x1x3, 1, 3);
        microkernel!(["avx"], x1x4, 1, 4);

        microkernel!(["avx"], x2x1, 2, 1);
        microkernel!(["avx"], x2x2, 2, 2);
        microkernel!(["avx"], x2x3, 2, 3);
        microkernel!(["avx"], x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod fma {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 8;
        const CPLX_N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..8 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..8 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm256_fmadd_ps(transmute(a), transmute(b), transmute(c)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m256 = unsafe {
                transmute([
                    0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32,
                ])
            };
            transmute(_mm256_xor_ps(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm256_permute_ps::<0b10110001>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm256_fmaddsub_ps(
                transmute(a_re_im),
                transmute(b_re),
                _mm256_mul_ps(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(_mm256_fmsubadd_ps(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm256_fmsubadd_ps(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm256_fmaddsub_ps(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm256_fmaddsub_ps(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel!(["fma"], x1x1, 1, 1);
        microkernel!(["fma"], x1x2, 1, 2);
        microkernel!(["fma"], x1x3, 1, 3);
        microkernel!(["fma"], x1x4, 1, 4);

        microkernel!(["fma"], x2x1, 2, 1);
        microkernel!(["fma"], x2x2, 2, 2);
        microkernel!(["fma"], x2x3, 2, 3);
        microkernel!(["fma"], x2x4, 2, 4);

        microkernel!(["fma"], x3x1, 3, 1);
        microkernel!(["fma"], x3x2, 3, 2);
        microkernel!(["fma"], x3x3, 3, 3);
        microkernel!(["fma"], x3x4, 3, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
            [x3x1, x3x2, x3x3, x3x4,],
        }

        microkernel_cplx!(["fma"], cplx_x1x1, 1, 1);
        microkernel_cplx!(["fma"], cplx_x1x2, 1, 2);

        microkernel_cplx!(["fma"], cplx_x2x1, 2, 1);
        microkernel_cplx!(["fma"], cplx_x2x2, 2, 2);

        microkernel_cplx!(["fma"], cplx_x3x1, 3, 1);
        microkernel_cplx!(["fma"], cplx_x3x2, 3, 2);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2,],
            [cplx_x2x1, cplx_x2x2,],
            [cplx_x3x1, cplx_x3x2,],
        }
    }

    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f64;
        const N: usize = 4;
        const CPLX_N: usize = 2;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..4 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..4 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm256_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm256_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm256_fmadd_pd(transmute(a), transmute(b), transmute(c)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m256d = unsafe { transmute([0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64]) };
            transmute(_mm256_xor_pd(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm256_permute_pd::<0b0101>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm256_fmaddsub_pd(
                transmute(a_re_im),
                transmute(b_re),
                _mm256_mul_pd(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(_mm256_fmsubadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm256_fmsubadd_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm256_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm256_fmaddsub_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel!(["fma"], x1x1, 1, 1);
        microkernel!(["fma"], x1x2, 1, 2);
        microkernel!(["fma"], x1x3, 1, 3);
        microkernel!(["fma"], x1x4, 1, 4);

        microkernel!(["fma"], x2x1, 2, 1);
        microkernel!(["fma"], x2x2, 2, 2);
        microkernel!(["fma"], x2x3, 2, 3);
        microkernel!(["fma"], x2x4, 2, 4);

        microkernel!(["fma"], x3x1, 3, 1);
        microkernel!(["fma"], x3x2, 3, 2);
        microkernel!(["fma"], x3x3, 3, 3);
        microkernel!(["fma"], x3x4, 3, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
            [x3x1, x3x2, x3x3, x3x4,],
        }

        microkernel_cplx!(["fma"], cplx_x1x1, 1, 1);
        microkernel_cplx!(["fma"], cplx_x1x2, 1, 2);
        microkernel_cplx!(["fma"], cplx_x1x3, 1, 3);

        microkernel_cplx!(["fma"], cplx_x2x1, 2, 1);
        microkernel_cplx!(["fma"], cplx_x2x2, 2, 2);
        microkernel_cplx!(["fma"], cplx_x2x3, 2, 3);

        microkernel_cplx!(["fma"], cplx_x3x1, 3, 1);
        microkernel_cplx!(["fma"], cplx_x3x2, 3, 2);
        microkernel_cplx!(["fma"], cplx_x3x3, 3, 3);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3,],
        }
    }
}

#[cfg(all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512f {
    pub mod f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;
        use core::mem::MaybeUninit;

        type T = f32;
        const N: usize = 16;
        const CPLX_N: usize = 8;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..16 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..16 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_ps(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_mul_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_ps(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm512_fmadd_ps(transmute(a), transmute(b), transmute(c)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m512i = unsafe {
                transmute([
                    0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32,
                    0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32,
                ])
            };
            transmute(_mm512_xor_si512(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm512_permute_ps::<0b10110001>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm512_fmaddsub_ps(
                transmute(a_re_im),
                transmute(b_re),
                _mm512_mul_ps(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(_mm512_fmsubadd_ps(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmsubadd_ps(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm512_fmaddsub_ps(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmaddsub_ps(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel!(["avx512f"], x1x1, 1, 1);
        microkernel!(["avx512f"], x1x2, 1, 2);
        microkernel!(["avx512f"], x1x3, 1, 3);
        microkernel!(["avx512f"], x1x4, 1, 4);
        microkernel!(["avx512f"], x1x5, 1, 5);
        microkernel!(["avx512f"], x1x6, 1, 6);
        microkernel!(["avx512f"], x1x7, 1, 7);
        microkernel!(["avx512f"], x1x8, 1, 8);

        microkernel!(["avx512f"], x2x1, 2, 1);
        microkernel!(["avx512f"], x2x2, 2, 2);
        microkernel!(["avx512f"], x2x3, 2, 3);
        microkernel!(["avx512f"], x2x4, 2, 4);
        microkernel!(["avx512f"], x2x5, 2, 5);
        microkernel!(["avx512f"], x2x6, 2, 6);
        microkernel!(["avx512f"], x2x7, 2, 7);
        microkernel!(["avx512f"], x2x8, 2, 8);

        microkernel!(["avx512f"], x3x1, 3, 1);
        microkernel!(["avx512f"], x3x2, 3, 2);
        microkernel!(["avx512f"], x3x3, 3, 3);
        microkernel!(["avx512f"], x3x4, 3, 4);
        microkernel!(["avx512f"], x3x5, 3, 5);
        microkernel!(["avx512f"], x3x6, 3, 6);
        microkernel!(["avx512f"], x3x7, 3, 7);
        microkernel!(["avx512f"], x3x8, 3, 8);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }

        microkernel_cplx!(["avx512f"], cplx_x1x1, 1, 1);
        microkernel_cplx!(["avx512f"], cplx_x1x2, 1, 2);
        microkernel_cplx!(["avx512f"], cplx_x1x3, 1, 3);
        microkernel_cplx!(["avx512f"], cplx_x1x4, 1, 4);

        microkernel_cplx!(["avx512f"], cplx_x2x1, 2, 1);
        microkernel_cplx!(["avx512f"], cplx_x2x2, 2, 2);
        microkernel_cplx!(["avx512f"], cplx_x2x3, 2, 3);
        microkernel_cplx!(["avx512f"], cplx_x2x4, 2, 4);

        microkernel_cplx!(["avx512f"], cplx_x3x1, 3, 1);
        microkernel_cplx!(["avx512f"], cplx_x3x2, 3, 2);
        microkernel_cplx!(["avx512f"], cplx_x3x3, 3, 3);
        microkernel_cplx!(["avx512f"], cplx_x3x4, 3, 4);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3, cplx_x1x4,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3, cplx_x2x4,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3, cplx_x3x4,],
        }
    }

    pub mod f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use core::mem::transmute;

        type T = f64;
        const N: usize = 8;
        const CPLX_N: usize = 4;
        type Pack = [T; N];

        #[inline(always)]
        unsafe fn gather(base: *const T, stride: isize) -> Pack {
            transmute(_mm512_i64gather_pd::<8>(
                _mm512_setr_epi64(
                    0 * stride as i64,
                    1 * stride as i64,
                    2 * stride as i64,
                    3 * stride as i64,
                    4 * stride as i64,
                    5 * stride as i64,
                    6 * stride as i64,
                    7 * stride as i64,
                ),
                base as _,
            ))
        }

        #[inline(always)]
        unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            _mm512_i64scatter_pd::<8>(
                base as _,
                _mm512_setr_epi64(
                    0 * stride as i64,
                    1 * stride as i64,
                    2 * stride as i64,
                    3 * stride as i64,
                    4 * stride as i64,
                    5 * stride as i64,
                    6 * stride as i64,
                    7 * stride as i64,
                ),
                transmute(p),
            );
        }

        #[inline(always)]
        unsafe fn splat(value: T) -> Pack {
            transmute(_mm512_set1_pd(value))
        }

        #[inline(always)]
        unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_mul_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(_mm512_add_pd(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(_mm512_fmadd_pd(transmute(a), transmute(b), transmute(c)))
        }

        #[inline(always)]
        unsafe fn conj(a: Pack) -> Pack {
            const MASK: __m512i = unsafe {
                transmute([
                    0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64, 0.0_f64, -0.0_f64,
                ])
            };
            transmute(_mm512_xor_si512(MASK, transmute(a)))
        }

        #[inline(always)]
        unsafe fn swap_re_im(a: Pack) -> Pack {
            transmute(_mm512_permute_pd::<0b01010101>(transmute(a)))
        }

        #[inline(always)]
        unsafe fn mul_cplx(a_re_im: Pack, a_im_re: Pack, b_re: Pack, b_im: Pack) -> Pack {
            transmute(_mm512_fmaddsub_pd(
                transmute(a_re_im),
                transmute(b_re),
                _mm512_mul_pd(transmute(a_im_re), transmute(b_im)),
            ))
        }

        #[inline(always)]
        unsafe fn mul_add_cplx(
            a_re_im: Pack,
            a_im_re: Pack,
            b_re: Pack,
            b_im: Pack,
            c_re_im: Pack,
            conj_rhs: bool,
        ) -> Pack {
            if conj_rhs {
                transmute(_mm512_fmsubadd_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmsubadd_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            } else {
                transmute(_mm512_fmaddsub_pd(
                    transmute(a_re_im),
                    transmute(b_re),
                    _mm512_fmaddsub_pd(transmute(a_im_re), transmute(b_im), transmute(c_re_im)),
                ))
            }
        }

        microkernel!(["avx512f"], x1x1, 1, 1);
        microkernel!(["avx512f"], x1x2, 1, 2);
        microkernel!(["avx512f"], x1x3, 1, 3);
        microkernel!(["avx512f"], x1x4, 1, 4);
        microkernel!(["avx512f"], x1x5, 1, 5);
        microkernel!(["avx512f"], x1x6, 1, 6);
        microkernel!(["avx512f"], x1x7, 1, 7);
        microkernel!(["avx512f"], x1x8, 1, 8);

        microkernel!(["avx512f"], x2x1, 2, 1);
        microkernel!(["avx512f"], x2x2, 2, 2);
        microkernel!(["avx512f"], x2x3, 2, 3);
        microkernel!(["avx512f"], x2x4, 2, 4);
        microkernel!(["avx512f"], x2x5, 2, 5);
        microkernel!(["avx512f"], x2x6, 2, 6);
        microkernel!(["avx512f"], x2x7, 2, 7);
        microkernel!(["avx512f"], x2x8, 2, 8);

        microkernel!(["avx512f"], x3x1, 3, 1);
        microkernel!(["avx512f"], x3x2, 3, 2);
        microkernel!(["avx512f"], x3x3, 3, 3);
        microkernel!(["avx512f"], x3x4, 3, 4);
        microkernel!(["avx512f"], x3x5, 3, 5);
        microkernel!(["avx512f"], x3x6, 3, 6);
        microkernel!(["avx512f"], x3x7, 3, 7);
        microkernel!(["avx512f"], x3x8, 3, 8);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }

        microkernel_cplx!(["avx512f"], cplx_x1x1, 1, 1);
        microkernel_cplx!(["avx512f"], cplx_x1x2, 1, 2);
        microkernel_cplx!(["avx512f"], cplx_x1x3, 1, 3);
        microkernel_cplx!(["avx512f"], cplx_x1x4, 1, 4);

        microkernel_cplx!(["avx512f"], cplx_x2x1, 2, 1);
        microkernel_cplx!(["avx512f"], cplx_x2x2, 2, 2);
        microkernel_cplx!(["avx512f"], cplx_x2x3, 2, 3);
        microkernel_cplx!(["avx512f"], cplx_x2x4, 2, 4);

        microkernel_cplx!(["avx512f"], cplx_x3x1, 3, 1);
        microkernel_cplx!(["avx512f"], cplx_x3x2, 3, 2);
        microkernel_cplx!(["avx512f"], cplx_x3x3, 3, 3);
        microkernel_cplx!(["avx512f"], cplx_x3x4, 3, 4);

        microkernel_cplx_fn_array! {
            [cplx_x1x1, cplx_x1x2, cplx_x1x3, cplx_x1x4,],
            [cplx_x2x1, cplx_x2x2, cplx_x2x3, cplx_x2x4,],
            [cplx_x3x1, cplx_x3x2, cplx_x3x3, cplx_x3x4,],
        }
    }
}

#[allow(dead_code)]
mod v128_common {
    pub mod f32 {
        use core::mem::MaybeUninit;

        pub type T = f32;
        pub const N: usize = 4;
        pub type Pack = [T; N];

        #[inline(always)]
        pub unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..4 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        pub unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..4 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value, value, value]
        }
    }

    pub mod f64 {
        use core::mem::MaybeUninit;

        pub type T = f64;
        pub const N: usize = 2;
        pub type Pack = [T; N];

        #[inline(always)]
        pub unsafe fn gather(base: *const T, stride: isize) -> Pack {
            let mut p = MaybeUninit::<Pack>::uninit();
            let ptr = p.as_mut_ptr() as *mut T;
            seq_macro::seq!(ITER in 0..2 {
                *ptr.add(ITER) = *base.offset(ITER * stride);
            });
            p.assume_init()
        }

        #[inline(always)]
        pub unsafe fn scatter(base: *mut T, stride: isize, p: Pack) {
            let ptr = p.as_ptr();
            seq_macro::seq!(ITER in 0..2 {
                *base.offset(ITER * stride) = *ptr.add(ITER);
            });
        }

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value]
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    pub mod f32 {
        use super::super::v128_common::f32::*;
        use core::arch::aarch64::*;
        use core::mem::transmute;

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f32(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f32(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_f32(transmute(c), transmute(a), transmute(b)))
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f32::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        microkernel!(["neon"], x1x1, 1, 1);
        microkernel!(["neon"], x1x2, 1, 2);
        microkernel!(["neon"], x1x3, 1, 3);
        microkernel!(["neon"], x1x4, 1, 4, 1, 4);
        microkernel!(["neon"], x1x5, 1, 5);
        microkernel!(["neon"], x1x6, 1, 6);
        microkernel!(["neon"], x1x7, 1, 7);
        microkernel!(["neon"], x1x8, 1, 8, 2, 4);

        microkernel!(["neon"], x2x1, 2, 1);
        microkernel!(["neon"], x2x2, 2, 2);
        microkernel!(["neon"], x2x3, 2, 3);
        microkernel!(["neon"], x2x4, 2, 4, 1, 4);
        microkernel!(["neon"], x2x5, 2, 5);
        microkernel!(["neon"], x2x6, 2, 6);
        microkernel!(["neon"], x2x7, 2, 7);
        microkernel!(["neon"], x2x8, 2, 8, 2, 4);

        microkernel!(["neon"], x3x1, 3, 1);
        microkernel!(["neon"], x3x2, 3, 2);
        microkernel!(["neon"], x3x3, 3, 3);
        microkernel!(["neon"], x3x4, 3, 4, 1, 4);
        microkernel!(["neon"], x3x5, 3, 5);
        microkernel!(["neon"], x3x6, 3, 6);
        microkernel!(["neon"], x3x7, 3, 7);
        microkernel!(["neon"], x3x8, 3, 8, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }
    pub mod f64 {
        use super::super::v128_common::f64::*;
        use core::arch::aarch64::*;
        use core::mem::transmute;

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f64(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f64(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_f64(transmute(c), transmute(a), transmute(b)))
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f64::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        microkernel!(["neon"], x1x1, 1, 1);
        microkernel!(["neon"], x1x2, 1, 2, 1, 2);
        microkernel!(["neon"], x1x3, 1, 3);
        microkernel!(["neon"], x1x4, 1, 4, 2, 2);
        microkernel!(["neon"], x1x5, 1, 5);
        microkernel!(["neon"], x1x6, 1, 6, 3, 2);
        microkernel!(["neon"], x1x7, 1, 7);
        microkernel!(["neon"], x1x8, 1, 8, 4, 2);

        microkernel!(["neon"], x2x1, 2, 1);
        microkernel!(["neon"], x2x2, 2, 2, 1, 2);
        microkernel!(["neon"], x2x3, 2, 3);
        microkernel!(["neon"], x2x4, 2, 4, 2, 2);
        microkernel!(["neon"], x2x5, 2, 5);
        microkernel!(["neon"], x2x6, 2, 6, 3, 2);
        microkernel!(["neon"], x2x7, 2, 7);
        microkernel!(["neon"], x2x8, 2, 8, 4, 2);

        microkernel!(["neon"], x3x1, 3, 1);
        microkernel!(["neon"], x3x2, 3, 2, 1, 2);
        microkernel!(["neon"], x3x3, 3, 3);
        microkernel!(["neon"], x3x4, 3, 4, 2, 2);
        microkernel!(["neon"], x3x5, 3, 5);
        microkernel!(["neon"], x3x6, 3, 6, 3, 2);
        microkernel!(["neon"], x3x7, 3, 7);
        microkernel!(["neon"], x3x8, 3, 8, 4, 2);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4, x1x5, x1x6, x1x7, x1x8,],
            [x2x1, x2x2, x2x3, x2x4, x2x5, x2x6, x2x7, x2x8,],
            [x3x1, x3x2, x3x3, x3x4, x3x5, x3x6, x3x7, x3x8,],
        }
    }
}
