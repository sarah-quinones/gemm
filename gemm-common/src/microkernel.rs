pub type MicroKernelFn<T> = unsafe fn(
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
#[macro_export]
macro_rules! __one {
    (
        $tt: tt
    ) => {
        1
    };
}

#[macro_export]
macro_rules! __microkernel_fn_array_helper {
    (
        [ $($tt: tt,)* ]
    ) => {
        {
            let mut count = 0_usize;
            $(count += $crate::__one!($tt);)*
            count
        }
    }
}

#[macro_export]
macro_rules! __microkernel_fn_array_helper_nr {
    ($([
       $($ukr: ident,)*
    ],)*) => {
        {
            let counts = [$({
                let mut count = 0_usize;
                $(count += $crate::__one!($ukr);)*
                count
            },)*];

            counts[0]
        }
    }
}

#[macro_export]
macro_rules! microkernel_fn_array {
    ($([
       $($ukr: ident,)*
    ],)*) => {
       pub const MR_DIV_N: usize =
           $crate::__microkernel_fn_array_helper!([$([$($ukr,)*],)*]);
       pub const NR: usize =
           $crate::__microkernel_fn_array_helper_nr!($([$($ukr,)*],)*);

        pub const UKR: [[$crate::microkernel::MicroKernelFn<T>; NR]; MR_DIV_N] =
            [ $([$($ukr,)*]),* ];
    };
}

#[macro_export]
macro_rules! microkernel_cplx_fn_array {
    ($([
       $($ukr: ident,)*
    ],)*) => {
       pub const CPLX_MR_DIV_N: usize =
           $crate::__microkernel_fn_array_helper!([$([$($ukr,)*],)*]);
       pub const CPLX_NR: usize =
           $crate::__microkernel_fn_array_helper_nr!($([$($ukr,)*],)*);

        pub const CPLX_UKR: [[$crate::microkernel::MicroKernelFn<num_complex::Complex<T>>; CPLX_NR]; CPLX_MR_DIV_N] =
            [ $([$($ukr,)*]),* ];
    };
}

#[macro_export]
macro_rules! microkernel {
    ($([$target: tt])?, $unroll: tt, $name: ident, $mr_div_n: tt, $nr: tt $(, $nr_div_n: tt, $n: tt)?) => {
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
            let mut accum_storage = [[splat(::core::mem::zeroed()); $mr_div_n]; $nr];
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

            impl KernelIter {
                #[inline(always)]
                unsafe fn execute(self, iter: usize) {
                    let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);
                    let next_lhs = self.next_lhs.wrapping_offset(iter as isize * self.lhs_cs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        *self.lhs.add(M_ITER) = *(packed_lhs.add(M_ITER * N) as *const Pack);
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
                            *self.lhs.add(M_ITER) = *(packed_lhs.add(M_ITER * N) as *const Pack);
                        }});

                        seq_macro::seq!(N_ITER0 in 0..$nr_div_n {{
                            *self.rhs = *(packed_rhs.wrapping_offset(N_ITER0 * $n) as *const Pack);

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

            let k_unroll = k / $unroll;
            let k_leftover = k % $unroll;

            let mut main_loop = {
                #[inline(always)]
                || {
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

                                    seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                        iter.execute_neon(UNROLL_ITER);
                                    }});

                                    packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                                    packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                                    next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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

                                seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                    iter.execute(UNROLL_ITER);
                                }});

                                packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                                packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                                next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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
                }
            };

            if rhs_rs == 1 {
                main_loop();
            } else {
                main_loop();
            }

            if m == $mr_div_n * N && n == $nr && dst_rs == 1  {
                let alpha = splat(alpha);
                let beta = splat(beta);
                if alpha_status == 2 {
                    seq_macro::seq!(N_ITER in 0..$nr {{
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                            dst.write_unaligned(add(
                                    mul(alpha, *dst),
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
                                    *dst,
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
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const T;

                if alpha_status == 2 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = scalar_add(scalar_mul(alpha, *dst_ij), scalar_mul(beta, *src_ij));
                        }
                    }
                } else if alpha_status == 1 {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = scalar_mul_add(beta, *src_ij, *dst_ij);
                        }
                    }
                } else {
                    for j in 0..n {
                        let dst_j = dst.offset(dst_cs * j as isize);
                        let src_j = src.add(j * $mr_div_n * N);

                        for i in 0..m {
                            let dst_ij = dst_j.offset(dst_rs * i as isize);
                            let src_ij = src_j.add(i);

                            *dst_ij = scalar_mul(beta, *src_ij);
                        }
                    }
                }
            }

        }
    };
}

#[macro_export]
macro_rules! microkernel_cplx_2step {
    ($([$target: tt])?, $unroll: tt, $name: ident, $mr_div_n: tt, $nr: tt) => {
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

            let (neg_conj_rhs, conj_all, neg_all) = match (conj_lhs, conj_rhs) {
                (true, true) => (true, false, true),
                (false, true) => (false, true, false),
                (true, false) => (false, false, false),
                (false, false) => (true, true, true),
            };

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
                unsafe fn execute(self, iter: usize, neg_conj_rhs: bool) {
                    let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);
                    let next_lhs = self.next_lhs.wrapping_offset(iter as isize * self.lhs_cs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        let tmp = *(packed_lhs.add(M_ITER * CPLX_N) as *const Pack);
                        *self.lhs_re_im.add(M_ITER) = tmp;
                        *self.lhs_im_re.add(M_ITER) = swap_re_im(tmp);
                    }});

                    seq_macro::seq!(N_ITER in 0..$nr {{
                        *self.rhs_re = splat((*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs)).re);

                        let accum = self.accum.add(N_ITER * $mr_div_n);
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let accum = &mut *accum.add(M_ITER);
                            *accum = mul_add_cplx_step0(
                                *self.lhs_re_im.add(M_ITER),
                                *self.rhs_re,
                                *accum,
                                neg_conj_rhs,
                                );
                        }});
                    }});

                    seq_macro::seq!(N_ITER in 0..$nr {{
                        *self.rhs_im = splat((*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs)).im);

                        let accum = self.accum.add(N_ITER * $mr_div_n);
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let accum = &mut *accum.add(M_ITER);
                            *accum = mul_add_cplx_step1(
                                *self.lhs_im_re.add(M_ITER),
                                *self.rhs_im,
                                *accum,
                                neg_conj_rhs,
                                );
                        }});
                    }});

                    let _ = next_lhs;
                }
            }

            let k_unroll = k / $unroll;
            let k_leftover = k % $unroll;

            let mut main_loop = {
                #[inline(always)]
                || {
                    loop {
                        if neg_conj_rhs {
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

                                    seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                        iter.execute(UNROLL_ITER, true);
                                    }});

                                    packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                                    packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                                    next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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

                                    seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                        iter.execute(UNROLL_ITER, false);
                                    }});

                                    packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                                    packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                                    next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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
                }
            };

            if rhs_rs == 1 {
                main_loop();
            } else {
                main_loop();
            }

            if conj_all && neg_all {
                seq_macro::seq!(N_ITER in 0..$nr {{
                    let accum = accum.add(N_ITER * $mr_div_n);
                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        let accum = &mut *accum.add(M_ITER);
                        *accum = neg_conj(*accum);
                    }});
                }});
            } else if !conj_all && neg_all {
                seq_macro::seq!(N_ITER in 0..$nr {{
                    let accum = accum.add(N_ITER * $mr_div_n);
                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        let accum = &mut *accum.add(M_ITER);
                        *accum = neg(*accum);
                    }});
                }});
            } else if conj_all && !neg_all {
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

#[macro_export]
macro_rules! microkernel_cplx {
    ($([$target: tt])?, $unroll: tt, $name: ident, $mr_div_n: tt, $nr: tt) => {
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
                        let tmp = *(packed_lhs.add(M_ITER * CPLX_N) as *const Pack);
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

            let k_unroll = k / $unroll;
            let k_leftover = k % $unroll;

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

                            seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                iter.execute(UNROLL_ITER, true);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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

                            seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                iter.execute(UNROLL_ITER, false);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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

#[macro_export]
macro_rules! microkernel_cplx_packed {
    ($([$target: tt])?, $unroll: tt, $name: ident, $mr_div_n: tt, $nr: tt) => {
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
            conj_dst: bool,
            conj_lhs: bool,
            conj_rhs: bool,
            mut next_lhs: *const T,
        ) {
            let mut accum_storage = [[core::mem::zeroed::<Pack>(); $mr_div_n]; $nr];
            let accum = accum_storage.as_mut_ptr() as *mut Pack;

            let conj_both_lhs_rhs = conj_lhs;
            let conj_rhs = conj_lhs != conj_rhs;

            let mut lhs = [::core::mem::MaybeUninit::<Pack>::uninit(); $mr_div_n];
            let mut rhs = ::core::mem::MaybeUninit::<Pack>::uninit();

            #[derive(Copy, Clone)]
            struct KernelIter {
                packed_lhs: *const T,
                next_lhs: *const T,
                packed_rhs: *const T,
                lhs_cs: isize,
                rhs_rs: isize,
                rhs_cs: isize,
                accum: *mut Pack,
                lhs: *mut Pack,
                rhs: *mut Pack,
            }

            impl KernelIter {
                #[inline(always)]
                unsafe fn execute(self, iter: usize, conj_rhs: bool) {
                    let packed_lhs = self.packed_lhs.wrapping_offset(iter as isize * self.lhs_cs);
                    let packed_rhs = self.packed_rhs.wrapping_offset(iter as isize * self.rhs_rs);
                    let next_lhs = self.next_lhs.wrapping_offset(iter as isize * self.lhs_cs);

                    seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                        *self.lhs.add(M_ITER) = *(packed_lhs.add(M_ITER * N) as *const Pack);
                    }});

                    seq_macro::seq!(N_ITER in 0..$nr {{
                        *self.rhs = splat(*packed_rhs.wrapping_offset(N_ITER * self.rhs_cs));

                        let accum = self.accum.add(N_ITER * $mr_div_n);
                        seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                            let accum = &mut *accum.add(M_ITER);
                            *accum = mul_add_cplx(
                                *self.lhs.add(M_ITER),
                                *self.rhs,
                                *accum,
                                conj_rhs,
                            );
                        }});
                    }});

                    let _ = next_lhs;
                }
            }

            let k_unroll = k / $unroll;
            let k_leftover = k % $unroll;

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
                                lhs: lhs.as_mut_ptr() as _,
                                rhs: &mut rhs as *mut _ as _,
                            };

                            seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                iter.execute(UNROLL_ITER, true);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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
                                lhs: lhs.as_mut_ptr() as _,
                                rhs: &mut rhs as *mut _ as _,
                            };

                            seq_macro::seq!(UNROLL_ITER in 0..$unroll {{
                                iter.execute(UNROLL_ITER, false);
                            }});

                            packed_lhs = packed_lhs.wrapping_offset($unroll * lhs_cs);
                            packed_rhs = packed_rhs.wrapping_offset($unroll * rhs_rs);
                            next_lhs = next_lhs.wrapping_offset($unroll * lhs_cs);

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
                                lhs: lhs.as_mut_ptr() as _,
                                rhs: &mut rhs as *mut _ as _,
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

            if m == $mr_div_n * N && n == $nr && dst_rs == 1 {
                let alpha = splat(alpha);
                let beta = splat(beta);

                if conj_dst {
                    if alpha_status == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    mul_cplx(conj(*dst), alpha),
                                    mul_cplx(accum, beta),
                                );
                            }});
                        }});
                    } else if alpha_status == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    conj(*dst),
                                    mul_cplx(accum, beta),
                                    );
                            }});
                        }});
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = mul_cplx(accum, beta);
                            }});
                        }});
                    }
                } else {
                    if alpha_status == 2 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    mul_cplx(*dst, alpha),
                                    mul_cplx(accum, beta),
                                );
                            }});
                        }});
                    } else if alpha_status == 1 {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = add(
                                    *dst,
                                    mul_cplx(accum, beta),
                                );
                            }});
                        }});
                    } else {
                        seq_macro::seq!(N_ITER in 0..$nr {{
                            seq_macro::seq!(M_ITER in 0..$mr_div_n {{
                                let dst = dst.offset(M_ITER * N as isize + N_ITER * dst_cs) as *mut Pack;
                                let accum = *accum.offset(M_ITER + $mr_div_n * N_ITER);
                                *dst = mul_cplx(accum, beta);
                            }});
                        }});
                    }
                }
            } else {
                let src = accum_storage; // write to stack
                let src = src.as_ptr() as *const T;

                if conj_dst {
                    if alpha_status == 2 {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = alpha * (*dst_ij).conj() + beta * *src_ij;
                            }
                        }
                    } else if alpha_status == 1 {
                        for j in 0..n {
                            let dst_j = dst.offset(dst_cs * j as isize);
                            let src_j = src.add(j * $mr_div_n * N);

                            for i in 0..m {
                                let dst_ij = dst_j.offset(dst_rs * i as isize);
                                let src_ij = src_j.add(i);

                                *dst_ij = (*dst_ij).conj() + beta * *src_ij;
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
                } else {
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
        }
    };
}
