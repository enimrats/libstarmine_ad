#[inline]
pub(crate) fn dot_product_i32_prefix<const N: usize>(
    lhs: &[i32; N],
    rhs: &[i32; N],
    len: usize,
) -> i64 {
    debug_assert!(len <= N);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dot_product_i32_prefix_neon(lhs, rhs, len);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_i32_prefix_scalar(lhs, rhs, len)
    }
}

#[inline]
pub(crate) fn dual_dot_product_i32_prefix<const N: usize>(
    samples: &[i32; N],
    coeff_a: &[i32; N],
    coeff_b: &[i32; N],
    len: usize,
) -> (i64, i64) {
    debug_assert!(len <= N);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dual_dot_product_i32_prefix_neon(samples, coeff_a, coeff_b, len);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dual_dot_product_i32_prefix_scalar(samples, coeff_a, coeff_b, len)
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_product_i32_prefix_scalar<const N: usize>(
    lhs: &[i32; N],
    rhs: &[i32; N],
    len: usize,
) -> i64 {
    lhs.iter()
        .zip(rhs.iter())
        .take(len)
        .fold(0i64, |acc, (&l, &r)| acc + i64::from(l) * i64::from(r))
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dual_dot_product_i32_prefix_scalar<const N: usize>(
    samples: &[i32; N],
    coeff_a: &[i32; N],
    coeff_b: &[i32; N],
    len: usize,
) -> (i64, i64) {
    samples
        .iter()
        .zip(coeff_a.iter())
        .zip(coeff_b.iter())
        .take(len)
        .fold((0i64, 0i64), |(sum_a, sum_b), ((&sample, &a), &b)| {
            let sample = i64::from(sample);
            (sum_a + sample * i64::from(a), sum_b + sample * i64::from(b))
        })
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_i32_prefix_neon<const N: usize>(
    lhs: &[i32; N],
    rhs: &[i32; N],
    len: usize,
) -> i64 {
    use core::arch::aarch64::{
        int32x4_t, int64x2_t, vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_high_s32, vget_low_s32,
        vld1q_s32, vmull_s32,
    };

    let mut acc: int64x2_t = vdupq_n_s64(0);
    let mut i = 0;

    while i + 4 <= len {
        let left: int32x4_t = unsafe { vld1q_s32(lhs.as_ptr().add(i)) };
        let right: int32x4_t = unsafe { vld1q_s32(rhs.as_ptr().add(i)) };
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(left), vget_low_s32(right)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(left), vget_high_s32(right)));
        i += 4;
    }

    let mut sum = vaddvq_s64(acc);
    while i < len {
        sum += i64::from(lhs[i]) * i64::from(rhs[i]);
        i += 1;
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dual_dot_product_i32_prefix_neon<const N: usize>(
    samples: &[i32; N],
    coeff_a: &[i32; N],
    coeff_b: &[i32; N],
    len: usize,
) -> (i64, i64) {
    use core::arch::aarch64::{
        int32x4_t, int64x2_t, vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_high_s32, vget_low_s32,
        vld1q_s32, vmull_s32,
    };

    let mut acc_a: int64x2_t = vdupq_n_s64(0);
    let mut acc_b: int64x2_t = vdupq_n_s64(0);
    let mut i = 0;

    while i + 4 <= len {
        let sample: int32x4_t = unsafe { vld1q_s32(samples.as_ptr().add(i)) };
        let coeff_a_v: int32x4_t = unsafe { vld1q_s32(coeff_a.as_ptr().add(i)) };
        let coeff_b_v: int32x4_t = unsafe { vld1q_s32(coeff_b.as_ptr().add(i)) };

        acc_a = vaddq_s64(
            acc_a,
            vmull_s32(vget_low_s32(sample), vget_low_s32(coeff_a_v)),
        );
        acc_a = vaddq_s64(
            acc_a,
            vmull_s32(vget_high_s32(sample), vget_high_s32(coeff_a_v)),
        );

        acc_b = vaddq_s64(
            acc_b,
            vmull_s32(vget_low_s32(sample), vget_low_s32(coeff_b_v)),
        );
        acc_b = vaddq_s64(
            acc_b,
            vmull_s32(vget_high_s32(sample), vget_high_s32(coeff_b_v)),
        );

        i += 4;
    }

    let mut sum_a = vaddvq_s64(acc_a);
    let mut sum_b = vaddvq_s64(acc_b);
    while i < len {
        let sample = i64::from(samples[i]);
        sum_a += sample * i64::from(coeff_a[i]);
        sum_b += sample * i64::from(coeff_b[i]);
        i += 1;
    }

    (sum_a, sum_b)
}
