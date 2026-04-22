// SPDX-License-Identifier: Apache-2.0
// Derived from `truehdd`; modified for `libstarmine_ad`.

use crate::truehddec::process::{MAX_PRESENTATIONS, PresentationMap, PresentationType};
use crate::truehddec::simd::{dot_product_i32_prefix, dual_dot_product_i32_prefix};
use crate::truehddec::structs::access_unit::AccessUnit;
use crate::truehddec::structs::block::Block;
use crate::truehddec::structs::channel::ChannelLabel;
use crate::truehddec::structs::oamd::ObjectAudioMetadataPayload;
use crate::truehddec::utils::dither::fill_dither_31eb;
use crate::truehddec::utils::errors::{DecodeError, Result};
use log::{info, trace};

/// Decodes access units to PCM audio samples.
///
/// Converts parsed [`AccessUnit`] structures into 24-bit PCM audio data.
#[derive(Default)]
pub struct Decoder {
    state: DecoderState,
}

impl Decoder {
    /// Decodes an access unit to PCM audio samples.
    ///
    /// Updates the internal decode state for the requested presentation.
    pub fn decode_presentation(
        &mut self,
        access_unit: &AccessUnit,
        presentation: usize,
    ) -> Result<()> {
        self.state.decode_access_unit(access_unit, presentation)?;
        Ok(())
    }

    /// Borrows the most recently decoded PCM view from the decoder state.
    pub fn decoded_access_unit(&mut self) -> DecodedAccessUnit<'_> {
        let sample_length = self.state.samples_per_au - self.state.zero_samples;
        let channel_count = self.state.substream_state[self.state.presentation].max_matrix_chan + 1;
        let is_duplicate = self.state.has_duplicate_timing && self.state.has_duplicate_sample;
        let substream_info_changed = self.state.substream_info_changed;
        self.state.substream_info_changed = false;

        DecodedAccessUnit {
            sampling_frequency: self.state.sampling_frequency,
            sample_length,
            channel_count,
            pcm_data: &self.state.output_buffer,
            channel_labels: &self.state.channel_labels,
            oamd: &self.state.oamd,
            is_duplicate,
            substream_info_changed,
        }
    }

    /// Sets the failure level for validation errors.
    ///
    /// - `log::Level::Error`: Only fail on Error level messages (default)  
    /// - `log::Level::Warn`: Fail on Warning level and above (strict mode)
    pub fn set_fail_level(&mut self, level: log::Level) {
        self.state.fail_level = level;
    }
}

/// The result of decoding an access unit to PCM audio.
///
/// Contains 24-bit signed integer samples in sample-major ordering
/// (`pcm_data[sample_index][channel_index]`) with associated metadata.
#[derive(Debug)]
pub struct DecodedAccessUnit<'a> {
    /// Sampling frequency in Hz.
    ///
    /// This is the sampling frequency used for the audio data.
    pub sampling_frequency: u32,

    /// Number of valid samples in this access unit.
    ///
    /// This indicates how many samples in the `pcm_data` array contain
    /// valid audio data. The remaining samples should be ignored.
    pub sample_length: usize,

    /// Channel count for the audio data.
    ///
    /// This is determined by the stream configuration and indicates how many
    /// channels are present in the audio data.
    pub channel_count: usize,

    /// PCM audio samples organized as `[sample_index][channel_index]`.
    ///
    /// Contains 24-bit signed integer samples with sample-major ordering.
    /// - Array dimensions: [160 samples][16 channels]
    /// - Valid data length: Determined by `sample_length`
    /// - Channel count: Determined by stream configuration
    pub pcm_data: &'a [[i32; 16]; 160],

    /// Channel labels for the audio data.
    ///
    /// Contains labels for each channel in the audio data, providing
    /// descriptive names for each channel.
    pub channel_labels: &'a [ChannelLabel],

    /// Optional object audio metadata payload.
    ///
    /// Contains spatial audio metadata when present in the stream.
    pub oamd: &'a [ObjectAudioMetadataPayload],

    /// Indicates whether this access unit is a duplicate of the previous one.
    ///
    /// This is `true` when both the output timing and the decoded audio sample
    /// checksum match the previous access unit.
    /// Downstream applications may safely discard this frame.
    pub is_duplicate: bool,

    /// Indicates whether this access unit triggered a substream info change.
    ///
    /// This is `true` when substream_info or extended_substream_info changed,
    /// indicating that channel layout may have changed requiring new output files.
    pub substream_info_changed: bool,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DecoderSubstreamState {
    pub restart_sync_word: u16,
    pub output_timing: u16,
    pub min_chan: usize,
    pub max_chan: usize,
    pub max_matrix_chan: usize,
    pub dither_shift: u32,
    pub dither_seed: u32,
    pub lossless_check_i32: i32,
    pub lossless_check_i32_prev_au: i32,
    pub lossless_check_i32_accum: i32,
    pub ch_assign: [usize; 16],

    pub block_size: usize,

    pub primitive_matrices: usize,
    pub matrix_ch: [u8; 16],
    pub frac_bits: [u8; 16],
    pub cf_shift_code: [i8; 16],
    pub dither_scale: [u8; 16],
    pub delta_precision: [u8; 16],
    pub delta_cf: [[i32; 16]; 16],
    pub m_coeff: [[i32; 16]; 16],

    pub output_shift: [i8; 16],
    pub quantiser_step_size: [u32; 16],

    pub order: [[usize; 16]; 2],
    pub coeff_q: [[i32; 16]; 2],
    pub coeff: [[[i32; 8]; 16]; 2],
    pub coeff_state: [[[i32; 8]; 16]; 2],

    pub dither_table: [i32; 256],
    pub decoded_sample_len: usize,
}

impl Default for DecoderSubstreamState {
    fn default() -> Self {
        Self {
            restart_sync_word: 0,
            output_timing: 0,
            min_chan: 0,
            max_chan: 0,
            max_matrix_chan: 0,
            dither_shift: 0,
            dither_seed: 0,
            lossless_check_i32: 0,
            lossless_check_i32_prev_au: 0,
            lossless_check_i32_accum: 0,
            ch_assign: [0; 16],

            block_size: 8,

            primitive_matrices: 0,
            matrix_ch: [0; 16],
            frac_bits: [0; 16],
            cf_shift_code: [0; 16],
            dither_scale: [0; 16],
            delta_precision: [0; 16],
            delta_cf: [[0; 16]; 16],
            m_coeff: [[0; 16]; 16],

            output_shift: [0; 16],
            quantiser_step_size: [0; 16],

            order: [[0; 16]; 2],
            coeff_q: [[0; 16]; 2],
            coeff: [[[0; 8]; 16]; 2],
            coeff_state: [[[0; 8]; 16]; 2],

            dither_table: [0; 256],
            decoded_sample_len: 0,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct DecoderState {
    pub fail_level: log::Level,

    pub valid: bool,
    pub counter: usize,
    pub has_valid_branch: bool,
    pub has_duplicate_timing: bool,
    pub has_duplicate_sample: bool,

    pub sampling_frequency: u32,
    pub samples_per_au: usize,

    pub presentation_map: Option<PresentationMap>,
    pub presentation: usize,

    pub channel_labels: Vec<ChannelLabel>,

    pub substreams: usize,
    pub substream_mask: u8,
    pub substream_info: u8,
    pub extended_substream_info: u8,

    pub substream_index: usize,
    pub substream_state: [DecoderSubstreamState; MAX_PRESENTATIONS],

    pub rematrix_buffer: [[i32; 16]; 160],
    pub output_buffer: [[i32; 16]; 160],
    pub zero_samples: usize,
    pub oamd: Vec<ObjectAudioMetadataPayload>,
    pub substream_info_changed: bool,
}

impl Default for DecoderState {
    fn default() -> Self {
        Self {
            fail_level: log::Level::Error,
            valid: false,
            counter: 0,
            has_valid_branch: false,
            has_duplicate_timing: false,
            has_duplicate_sample: false,
            sampling_frequency: 0,
            samples_per_au: 0,
            presentation_map: None,
            presentation: 0,
            channel_labels: vec![],
            substreams: 0,
            substream_mask: 0,
            substream_info: 0,
            extended_substream_info: 0,
            substream_index: 0,
            substream_state: [DecoderSubstreamState::default(); MAX_PRESENTATIONS],
            rematrix_buffer: [[0; 16]; 160],
            output_buffer: [[0; 16]; 160],
            zero_samples: 0,
            oamd: Vec::with_capacity(4),
            substream_info_changed: false,
        }
    }
}

impl DecoderState {
    pub fn substream_state_mut(&mut self) -> Result<&mut DecoderSubstreamState> {
        Ok(&mut self.substream_state[self.substream_index])
    }

    pub fn substream_state(&self) -> Result<&DecoderSubstreamState> {
        Ok(&self.substream_state[self.substream_index])
    }
    pub fn decode_access_unit(
        &mut self,
        access_unit: &AccessUnit,
        presentation: usize,
    ) -> Result<()> {
        access_unit.update_decoder_state(self)?;

        if !self.valid {
            self.update_presentation(presentation)?;
            self.channel_labels = access_unit
                .get_channel_labels(self.presentation)
                .unwrap_or_default();
        }

        self.has_duplicate_timing = false;
        self.has_duplicate_sample = false;
        self.oamd.clear();

        for i in 0..=self.presentation {
            if (self.substream_mask >> i) & 1 == 0 {
                continue;
            }

            let substream_segment = &access_unit.substream_segment[i];
            if i == presentation
                && let Some(terminator) = &substream_segment.terminator
                && terminator.zero_samples_indicated
            {
                self.zero_samples = terminator.zero_samples as usize;
            }

            if i == 3
                && let Some(extra_data) = &access_unit.extra_data
                && let Some(evo_frame) = &extra_data.evo_frame
            {
                for evo_payload in &evo_frame.evo_payloads {
                    if evo_payload.evo_payload_id == 11 {
                        let smploffst =
                            evo_payload.evo_payload_config.smploffst.unwrap_or_default() as u64;
                        let mut oamd =
                            ObjectAudioMetadataPayload::read(&evo_payload.evo_payload_byte)?;
                        oamd.evo_sample_offset = smploffst;
                        self.oamd.push(oamd);
                    }
                }
            }

            self.substream_index = i;
            let ss_state = &mut self.substream_state[self.substream_index];
            ss_state.decoded_sample_len = 0;

            for block in &substream_segment.block {
                block.update_decoder_state(self)?;
                self.decode(block)?;
            }
        }

        self.valid = true;
        self.counter += 1;

        Ok(())
    }

    fn update_presentation(&mut self, presentation: usize) -> Result<()> {
        let Some(presentation_map) = self.presentation_map else {
            return Err((DecodeError::PresentationMapNotInitialized).into());
        };

        let mut presentations = [false; MAX_PRESENTATIONS];
        presentations[..=presentation]
            .iter_mut()
            .for_each(|p| *p = true);

        self.substream_mask =
            presentation_map.substream_mask_by_required_presentations(&presentations);
        match presentation_map.presentation_type_by_index(presentation) {
            PresentationType::Invalid => {
                if !self.valid {
                    let Some(max_independent) = presentation_map.max_independent_presentation()
                    else {
                        return Err((DecodeError::NoPresentationAvailable).into());
                    };
                    info!(
                        "Presentation {presentation} is not available, using presentation {max_independent}"
                    );
                    self.presentation = max_independent;
                }
            }
            PresentationType::CopyOf(copy_index) => {
                if !self.valid {
                    info!("Presentation {presentation} is a copy of presentation {copy_index}")
                }
                self.presentation = copy_index;
            }
            _ => {
                self.presentation = presentation;
            }
        };

        Ok(())
    }

    pub fn reset_decoder_substream_state(&mut self) {
        let ss_state = &mut self.substream_state[self.substream_index];
        *ss_state = DecoderSubstreamState {
            lossless_check_i32_prev_au: ss_state.lossless_check_i32_prev_au,
            ..Default::default()
        }
    }

    fn decode(&mut self, block: &Block) -> Result<()> {
        let DecoderSubstreamState {
            restart_sync_word,
            min_chan,
            max_chan,
            max_matrix_chan,
            dither_shift,
            // TODO: max_lsbs
            ch_assign,

            block_size,

            primitive_matrices,
            matrix_ch,
            dither_scale,
            delta_cf,

            output_shift,
            quantiser_step_size,

            order,
            coeff,
            coeff_q,
            ..
        } = *self.substream_state()?;

        let samples_per_au = self.samples_per_au;

        let ss_state = &mut self.substream_state[self.substream_index];

        let decoded_sample_len = &mut ss_state.decoded_sample_len;
        let dither_seed = &mut ss_state.dither_seed;
        let coeff_state = &mut ss_state.coeff_state;
        let m_coeff = &mut ss_state.m_coeff;
        let mut quantiser_masks_i64 = [0i64; 16];
        let mut quantiser_masks_i32 = [0i32; 16];

        for chi in 0..16 {
            let mask = !((1i32 << quantiser_step_size[chi]) - 1);
            quantiser_masks_i32[chi] = mask;
            quantiser_masks_i64[chi] = i64::from(mask);
        }

        let (max_val, min_val) = if restart_sync_word == 0x31EC {
            (1 << 31, -(1 << 31))
        } else {
            (1 << 23, -(1 << 23))
        };

        // recorrelation
        {
            let rematrix_buffer = &mut self.rematrix_buffer[*decoded_sample_len..];

            #[allow(clippy::needless_range_loop)]
            for chi in min_chan..=max_chan {
                let mut fir_history = coeff_state[0][chi];
                let mut iir_history = coeff_state[1][chi];
                let fir_order = order[0][chi];
                let iir_order = order[1][chi];
                let coeff_q_shift = coeff_q[0][chi];
                let quantiser_mask = quantiser_masks_i64[chi];
                let fir_coeff = &coeff[0][chi];
                let iir_coeff = &coeff[1][chi];

                for blki in 0..block_size {
                    let audio_data = i64::from(block.block_data_at(blki, chi));
                    let acc = dot_product_i32_prefix(fir_coeff, &fir_history, fir_order)
                        + dot_product_i32_prefix(iir_coeff, &iir_history, iir_order);

                    let pred = acc >> coeff_q_shift;
                    let fir_state = audio_data + (pred & quantiser_mask);
                    let iir_state = fir_state - pred;

                    if fir_state >= max_val {
                        return Err((DecodeError::RecorrelatorPositiveSaturation(fir_state)).into());
                    } else if fir_state < min_val {
                        return Err((DecodeError::RecorrelatorNegativeSaturation(fir_state)).into());
                    }

                    if !(min_val..max_val).contains(&iir_state) {
                        if restart_sync_word == 0x31EC {
                            return Err((DecodeError::FilterBInputTooWide32(iir_state)).into());
                        } else {
                            return Err((DecodeError::FilterBInputTooWide24(iir_state)).into());
                        }
                    }

                    let fir_state = fir_state as i32;
                    let iir_state = iir_state as i32;
                    push_history_front(&mut fir_history, fir_state);
                    push_history_front(&mut iir_history, iir_state);
                    rematrix_buffer[blki][chi] = fir_state;
                }

                coeff_state[0][chi] = fir_history;
                coeff_state[1][chi] = iir_history;
            }
        }

        // lossless matrix
        if self.substream_index == self.presentation {
            let dither_table = &mut ss_state.dither_table;
            let rematrix_buffer = &mut self.rematrix_buffer[*decoded_sample_len..];

            match restart_sync_word {
                0x31EA => {
                    for blki in 0..block_size {
                        let rematrix_buffer = &mut rematrix_buffer[blki];
                        let dither_seed_shr7 = *dither_seed >> 7;

                        rematrix_buffer[max_matrix_chan + 1] =
                            (((*dither_seed >> 15) as i8) << dither_shift) as i32;
                        rematrix_buffer[max_matrix_chan + 2] =
                            ((dither_seed_shr7 as i8) << dither_shift) as i32;

                        *dither_seed =
                            (dither_seed_shr7 ^ (dither_seed_shr7 << 5) ^ (*dither_seed << 16))
                                & 0x7FFFFF;

                        for pmi in 0..primitive_matrices {
                            let matrix_ch = matrix_ch[pmi] as usize;
                            let acc = dot_product_i32_prefix(
                                rematrix_buffer,
                                &m_coeff[pmi],
                                max_matrix_chan + 3,
                            );

                            rematrix_buffer[matrix_ch] = (((acc >> 18) as i32)
                                & quantiser_masks_i32[matrix_ch])
                                + block.bypassed_lsb_at(blki, pmi);
                        }
                    }
                }
                0x31EB => {
                    let dither_len = samples_per_au.next_power_of_two();
                    let dither_index_mask = dither_len - 1;
                    if *decoded_sample_len == 0 {
                        fill_dither_31eb(&mut dither_table[..dither_len], dither_seed);
                    }

                    for blki in 0..block_size {
                        let rematrix_buffer = &mut rematrix_buffer[blki];
                        let blki_abs = blki + *decoded_sample_len;

                        for pmi in 0..primitive_matrices {
                            let dither_scale = dither_scale[pmi] as i64;
                            let matrix_ch = matrix_ch[pmi] as usize;
                            let mut acc = dot_product_i32_prefix(
                                rematrix_buffer,
                                &m_coeff[pmi],
                                max_matrix_chan + 1,
                            );

                            let dither_index =
                                (primitive_matrices - pmi) * (2 * blki_abs + 1) + blki_abs;

                            if dither_scale != 0 {
                                acc += (dither_table[dither_index & dither_index_mask] as i64)
                                    << (11 + dither_scale);
                            }

                            rematrix_buffer[matrix_ch] = (((acc >> 18) as i32)
                                & quantiser_masks_i32[matrix_ch])
                                + block.bypassed_lsb_at(blki, pmi);
                        }
                    }
                }
                0x31EC => {
                    let dither_len = samples_per_au.next_power_of_two();
                    let dither_index_mask = dither_len - 1;
                    if *decoded_sample_len == 0 {
                        fill_dither_31eb(&mut dither_table[..dither_len], dither_seed);
                    }

                    let samples_per_au_recip = (1 << 16) / samples_per_au as i64;

                    for blki in 0..block_size {
                        let rematrix_buffer = &mut rematrix_buffer[blki];
                        let blki_abs = blki + *decoded_sample_len;

                        for pmi in 0..primitive_matrices {
                            let dither_scale = dither_scale[pmi] as u64;
                            let matrix_ch = matrix_ch[pmi] as usize;
                            let (mut acc, acc_delta) = dual_dot_product_i32_prefix(
                                rematrix_buffer,
                                &m_coeff[pmi],
                                &delta_cf[pmi],
                                max_matrix_chan + 1,
                            );

                            let dither_index =
                                (primitive_matrices - pmi) * (2 * blki_abs + 1) + blki_abs;

                            if dither_scale != 0 {
                                acc += (dither_table[dither_index & dither_index_mask] as i64)
                                    << (11 + dither_scale);
                            }

                            acc +=
                                (acc_delta >> 18) * (blki_abs as i64) * (samples_per_au_recip << 2);

                            rematrix_buffer[matrix_ch] = (((acc >> 18) as i32)
                                & quantiser_masks_i32[matrix_ch])
                                + block.bypassed_lsb_at(blki, pmi);
                        }
                    }

                    if *decoded_sample_len + block_size == samples_per_au {
                        for pmi in 0..primitive_matrices {
                            for chi in 0..=max_matrix_chan {
                                m_coeff[pmi][chi] += delta_cf[pmi][chi];
                            }
                        }
                    }
                }
                _ => {}
            }

            // remap
            {
                let output_buffer = &mut self.output_buffer[*decoded_sample_len..];

                if *decoded_sample_len == 0 {
                    ss_state.lossless_check_i32 = 0;
                }

                let mut lossless_check_data = 0;

                for blki in 0..block_size {
                    let sample = &rematrix_buffer[blki];
                    let output = &mut output_buffer[blki];
                    output.fill(0);

                    for chi in 0..=max_matrix_chan {
                        let ch_assign = ch_assign[chi];
                        let output = &mut output[ch_assign];

                        *output = sample[chi];

                        let output_shift = output_shift[chi];
                        if output_shift < 0 {
                            *output >>= -output_shift;
                        } else {
                            *output <<= output_shift;
                        }

                        lossless_check_data ^= (*output & 0xFFFFFF) << (chi & 7);
                    }
                }

                ss_state.lossless_check_i32 ^= lossless_check_data;
                ss_state.lossless_check_i32_accum ^= lossless_check_data;

                if *decoded_sample_len + block_size == samples_per_au {
                    trace!(
                        "AU {}: lossless_check_i32: {:08X}, lossless_check_i32_prev_au {:08X}",
                        self.counter,
                        ss_state.lossless_check_i32,
                        ss_state.lossless_check_i32_prev_au
                    );

                    if self.has_duplicate_timing
                        && ss_state.lossless_check_i32 == ss_state.lossless_check_i32_prev_au
                    {
                        self.has_duplicate_sample = true;
                        info!(
                            "AU {}: duplicate samples at branch, should be discarded",
                            self.counter
                        );
                    }

                    ss_state.lossless_check_i32_prev_au = ss_state.lossless_check_i32;
                }
            }
        }

        *decoded_sample_len += block_size;

        Ok(())
    }
}

#[inline]
fn push_history_front(history: &mut [i32; 8], sample: i32) {
    history[7] = history[6];
    history[6] = history[5];
    history[5] = history[4];
    history[4] = history[3];
    history[3] = history[2];
    history[2] = history[1];
    history[1] = history[0];
    history[0] = sample;
}
