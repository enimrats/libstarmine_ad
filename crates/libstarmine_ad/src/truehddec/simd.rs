// SPDX-License-Identifier: Apache-2.0
// Derived from `truehdd`; modified for `libstarmine_ad`.

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

#[inline]
pub(crate) fn apply_matrix_31ea(
    rematrix_buffer: &mut [[i32; 16]],
    m_coeff: &[i32; 16],
    prefix: usize,
    matrix_ch: usize,
    mask: i32,
    block: &crate::truehddec::structs::block::Block,
    pmi: usize,
    block_size: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return apply_matrix_31ea_neon(
            rematrix_buffer,
            m_coeff,
            prefix,
            matrix_ch,
            mask,
            block,
            pmi,
            block_size,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for blki in 0..block_size {
            let acc = dot_product_i32_prefix_scalar(&rematrix_buffer[blki], m_coeff, prefix);
            rematrix_buffer[blki][matrix_ch] =
                (((acc >> 18) as i32) & mask) + block.bypassed_lsb_at(blki, pmi);
        }
    }
}

#[inline]
pub(crate) fn apply_matrix_31eb(
    rematrix_buffer: &mut [[i32; 16]],
    m_coeff: &[i32; 16],
    prefix: usize,
    matrix_ch: usize,
    mask: i32,
    block: &crate::truehddec::structs::block::Block,
    pmi: usize,
    block_size: usize,
    primitive_matrices: usize,
    dither_scale: i64,
    dither_table: &[i32; 256],
    dither_index_mask: usize,
    decoded_sample_len: usize,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return apply_matrix_31eb_neon(
            rematrix_buffer,
            m_coeff,
            prefix,
            matrix_ch,
            mask,
            block,
            pmi,
            block_size,
            primitive_matrices,
            dither_scale,
            dither_table,
            dither_index_mask,
            decoded_sample_len,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for blki in 0..block_size {
            let blki_abs = blki + decoded_sample_len;
            let mut acc = dot_product_i32_prefix_scalar(&rematrix_buffer[blki], m_coeff, prefix);
            let dither_index = (primitive_matrices - pmi) * (2 * blki_abs + 1) + blki_abs;

            if dither_scale != 0 {
                acc +=
                    (dither_table[dither_index & dither_index_mask] as i64) << (11 + dither_scale);
            }

            rematrix_buffer[blki][matrix_ch] =
                (((acc >> 18) as i32) & mask) + block.bypassed_lsb_at(blki, pmi);
        }
    }
}

#[inline]
pub(crate) fn apply_matrix_31ec(
    rematrix_buffer: &mut [[i32; 16]],
    m_coeff: &[i32; 16],
    delta_cf: &[i32; 16],
    prefix: usize,
    matrix_ch: usize,
    mask: i32,
    block: &crate::truehddec::structs::block::Block,
    pmi: usize,
    block_size: usize,
    primitive_matrices: usize,
    dither_scale: u64,
    dither_table: &[i32; 256],
    dither_index_mask: usize,
    decoded_sample_len: usize,
    samples_per_au_recip: i64,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return apply_matrix_31ec_neon(
            rematrix_buffer,
            m_coeff,
            delta_cf,
            prefix,
            matrix_ch,
            mask,
            block,
            pmi,
            block_size,
            primitive_matrices,
            dither_scale,
            dither_table,
            dither_index_mask,
            decoded_sample_len,
            samples_per_au_recip,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for blki in 0..block_size {
            let blki_abs = blki + decoded_sample_len;
            let (mut acc, acc_delta) = dual_dot_product_i32_prefix_scalar(
                &rematrix_buffer[blki],
                m_coeff,
                delta_cf,
                prefix,
            );

            let dither_index = (primitive_matrices - pmi) * (2 * blki_abs + 1) + blki_abs;

            if dither_scale != 0 {
                acc +=
                    (dither_table[dither_index & dither_index_mask] as i64) << (11 + dither_scale);
            }

            acc += (acc_delta >> 18) * (blki_abs as i64) * (samples_per_au_recip << 2);

            rematrix_buffer[blki][matrix_ch] =
                (((acc >> 18) as i32) & mask) + block.bypassed_lsb_at(blki, pmi);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_matrix_31ea_neon(
    rematrix_buffer: &mut [[i32; 16]],
    m_coeff: &[i32; 16],
    prefix: usize,
    matrix_ch: usize,
    mask: i32,
    block: &crate::truehddec::structs::block::Block,
    pmi: usize,
    block_size: usize,
) {
    use core::arch::aarch64::{
        vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_high_s32, vget_low_s32, vld1q_s32, vmull_s32,
    };

    let mut c_array = *m_coeff;
    for i in prefix..16 {
        c_array[i] = 0;
    }
    let c0 = vld1q_s32(c_array.as_ptr());
    let c1 = vld1q_s32(c_array.as_ptr().add(4));
    let c2 = vld1q_s32(c_array.as_ptr().add(8));
    let c3 = vld1q_s32(c_array.as_ptr().add(12));

    for blki in 0..block_size {
        let rb = rematrix_buffer[blki].as_ptr();
        let r0 = vld1q_s32(rb);
        let r1 = vld1q_s32(rb.add(4));
        let r2 = vld1q_s32(rb.add(8));
        let r3 = vld1q_s32(rb.add(12));

        let mut acc = vdupq_n_s64(0);
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r0), vget_low_s32(c0)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r0), vget_high_s32(c0)));
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r1), vget_low_s32(c1)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r1), vget_high_s32(c1)));
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r2), vget_low_s32(c2)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r2), vget_high_s32(c2)));
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r3), vget_low_s32(c3)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r3), vget_high_s32(c3)));

        let sum = vaddvq_s64(acc);
        rematrix_buffer[blki][matrix_ch] =
            (((sum >> 18) as i32) & mask) + block.bypassed_lsb_at(blki, pmi);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_matrix_31eb_neon(
    rematrix_buffer: &mut [[i32; 16]],
    m_coeff: &[i32; 16],
    prefix: usize,
    matrix_ch: usize,
    mask: i32,
    block: &crate::truehddec::structs::block::Block,
    pmi: usize,
    block_size: usize,
    primitive_matrices: usize,
    dither_scale: i64,
    dither_table: &[i32; 256],
    dither_index_mask: usize,
    decoded_sample_len: usize,
) {
    use core::arch::aarch64::{
        vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_high_s32, vget_low_s32, vld1q_s32, vmull_s32,
    };

    let mut c_array = *m_coeff;
    for i in prefix..16 {
        c_array[i] = 0;
    }
    let c0 = vld1q_s32(c_array.as_ptr());
    let c1 = vld1q_s32(c_array.as_ptr().add(4));
    let c2 = vld1q_s32(c_array.as_ptr().add(8));
    let c3 = vld1q_s32(c_array.as_ptr().add(12));

    for blki in 0..block_size {
        let rb = rematrix_buffer[blki].as_ptr();
        let r0 = vld1q_s32(rb);
        let r1 = vld1q_s32(rb.add(4));
        let r2 = vld1q_s32(rb.add(8));
        let r3 = vld1q_s32(rb.add(12));

        let mut acc = vdupq_n_s64(0);
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r0), vget_low_s32(c0)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r0), vget_high_s32(c0)));
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r1), vget_low_s32(c1)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r1), vget_high_s32(c1)));
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r2), vget_low_s32(c2)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r2), vget_high_s32(c2)));
        acc = vaddq_s64(acc, vmull_s32(vget_low_s32(r3), vget_low_s32(c3)));
        acc = vaddq_s64(acc, vmull_s32(vget_high_s32(r3), vget_high_s32(c3)));

        let mut sum = vaddvq_s64(acc);

        let blki_abs = blki + decoded_sample_len;
        let dither_index = (primitive_matrices - pmi) * (2 * blki_abs + 1) + blki_abs;
        if dither_scale != 0 {
            sum += (dither_table[dither_index & dither_index_mask] as i64) << (11 + dither_scale);
        }

        rematrix_buffer[blki][matrix_ch] =
            (((sum >> 18) as i32) & mask) + block.bypassed_lsb_at(blki, pmi);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn apply_matrix_31ec_neon(
    rematrix_buffer: &mut [[i32; 16]],
    m_coeff: &[i32; 16],
    delta_cf: &[i32; 16],
    prefix: usize,
    matrix_ch: usize,
    mask: i32,
    block: &crate::truehddec::structs::block::Block,
    pmi: usize,
    block_size: usize,
    primitive_matrices: usize,
    dither_scale: u64,
    dither_table: &[i32; 256],
    dither_index_mask: usize,
    decoded_sample_len: usize,
    samples_per_au_recip: i64,
) {
    use core::arch::aarch64::{
        vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_high_s32, vget_low_s32, vld1q_s32, vmull_s32,
    };

    let mut c_array = *m_coeff;
    let mut d_array = *delta_cf;
    for i in prefix..16 {
        c_array[i] = 0;
        d_array[i] = 0;
    }
    let c0 = vld1q_s32(c_array.as_ptr());
    let c1 = vld1q_s32(c_array.as_ptr().add(4));
    let c2 = vld1q_s32(c_array.as_ptr().add(8));
    let c3 = vld1q_s32(c_array.as_ptr().add(12));

    let d0 = vld1q_s32(d_array.as_ptr());
    let d1 = vld1q_s32(d_array.as_ptr().add(4));
    let d2 = vld1q_s32(d_array.as_ptr().add(8));
    let d3 = vld1q_s32(d_array.as_ptr().add(12));

    for blki in 0..block_size {
        let rb = rematrix_buffer[blki].as_ptr();
        let r0 = vld1q_s32(rb);
        let r1 = vld1q_s32(rb.add(4));
        let r2 = vld1q_s32(rb.add(8));
        let r3 = vld1q_s32(rb.add(12));

        let mut acc_a = vdupq_n_s64(0);
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_low_s32(r0), vget_low_s32(c0)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_high_s32(r0), vget_high_s32(c0)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_low_s32(r1), vget_low_s32(c1)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_high_s32(r1), vget_high_s32(c1)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_low_s32(r2), vget_low_s32(c2)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_high_s32(r2), vget_high_s32(c2)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_low_s32(r3), vget_low_s32(c3)));
        acc_a = vaddq_s64(acc_a, vmull_s32(vget_high_s32(r3), vget_high_s32(c3)));

        let mut acc_b = vdupq_n_s64(0);
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_low_s32(r0), vget_low_s32(d0)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_high_s32(r0), vget_high_s32(d0)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_low_s32(r1), vget_low_s32(d1)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_high_s32(r1), vget_high_s32(d1)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_low_s32(r2), vget_low_s32(d2)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_high_s32(r2), vget_high_s32(d2)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_low_s32(r3), vget_low_s32(d3)));
        acc_b = vaddq_s64(acc_b, vmull_s32(vget_high_s32(r3), vget_high_s32(d3)));

        let mut sum_a = vaddvq_s64(acc_a);
        let sum_b = vaddvq_s64(acc_b);

        let blki_abs = blki + decoded_sample_len;
        let dither_index = (primitive_matrices - pmi) * (2 * blki_abs + 1) + blki_abs;
        if dither_scale != 0 {
            sum_a += (dither_table[dither_index & dither_index_mask] as i64) << (11 + dither_scale);
        }

        sum_a += (sum_b >> 18) * (blki_abs as i64) * (samples_per_au_recip << 2);

        rematrix_buffer[blki][matrix_ch] =
            (((sum_a >> 18) as i32) & mask) + block.bypassed_lsb_at(blki, pmi);
    }
}
