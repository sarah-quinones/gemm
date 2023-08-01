#[cfg(target_arch = "aarch64")]
pub mod neon {
    pub mod f16 {
        use half::binary16::arch::aarch64::{vfmaq_f16, vaddq_f16, vmulq_f16, vfmaq_laneq_f16};
        use core::mem::transmute;

        pub type T = half::f16;
        pub const N: usize = 8;
        pub type Pack = [T; N];

        #[inline(always)]
        pub unsafe fn splat(value: T) -> Pack {
            [value, value, value, value, value, value, value, value]
        }

        #[inline(always)]
        pub unsafe fn mul(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vmulq_f16(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn add(lhs: Pack, rhs: Pack) -> Pack {
            transmute(vaddq_f16(transmute(lhs), transmute(rhs)))
        }

        #[inline(always)]
        pub unsafe fn mul_add(a: Pack, b: Pack, c: Pack) -> Pack {
            let out = transmute(vfmaq_f16(transmute(c), transmute(a), transmute(b)));
            std::hint::black_box(out)
            // out
        }

        #[inline(always)]
        pub unsafe fn mul_add_lane<const LANE: i32>(a: Pack, b: Pack, c: Pack) -> Pack {
            transmute(vfmaq_laneq_f16::<LANE>(
                transmute(c),
                transmute(a),
                transmute(b),
            ))
        }

        microkernel_f16!(["neon"], 2, x1x1, 1, 1);
        microkernel_f16!(["neon"], 2, x1x2, 1, 2);
        microkernel_f16!(["neon"], 2, x1x3, 1, 3);
        microkernel_f16!(["neon"], 2, x1x4, 1, 4);

        microkernel_f16!(["neon"], 2, x2x1, 2, 1);
        microkernel_f16!(["neon"], 2, x2x2, 2, 2);
        microkernel_f16!(["neon"], 2, x2x3, 2, 3);
        microkernel_f16!(["neon"], 2, x2x4, 2, 4);

        microkernel_fn_array! {
            [x1x1, x1x2, x1x3, x1x4,],
            [x2x1, x2x2, x2x3, x2x4,],
        }
    }
}
