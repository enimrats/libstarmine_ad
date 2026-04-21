// SPDX-License-Identifier: Apache-2.0

use super::render_input::{
    RenderInputFrame, RenderMetadata, RenderMetadataElement, RenderMetadataObject,
    RenderMetadataUpdate,
};
use super::types::{BedChannel, ObjectAnchor, Vec3};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vabsq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32, vmulq_f32, vst1q_f32,
};
use thiserror::Error;

const RENDER_TIMESLOT_SAMPLES: usize = 64;
const RENDER_714_CHANNELS: usize = 12;
const RENDER_714_LFE_INDEX: usize = 3;
const LFE_SEND_MINUS_10_DB: f32 = 0.316_227_76;
const LFE_LOW_PASS_HZ: f32 = 120.0;
const LOW_PASS_REFERENCE_Q: f32 = 0.707_106_77;
const RENDER_ENVIRONMENT_SIZE: Vec3 = Vec3 {
    x: 10.0,
    y: 7.0,
    z: 10.0,
};
const SCREEN_SIZE_X: f32 = 0.9;
const SCREEN_SIZE_Z: f32 = 0.486;
const INFINITE_DISTANCE_FALLBACK: f32 = 100.0;
const ROOM_CENTER: Vec3 = Vec3 {
    x: 0.5,
    y: 0.5,
    z: 0.0,
};
const ZERO_VEC3: Vec3 = Vec3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

/// Fixed output channel order used by [`Renderer714`].
pub const RENDER_714_CHANNEL_ORDER: [BedChannel; RENDER_714_CHANNELS] = [
    BedChannel::FrontLeft,
    BedChannel::FrontRight,
    BedChannel::Center,
    BedChannel::LowFrequencyEffects,
    BedChannel::RearLeft,
    BedChannel::RearRight,
    BedChannel::SurroundLeft,
    BedChannel::SurroundRight,
    BedChannel::TopFrontLeft,
    BedChannel::TopFrontRight,
    BedChannel::TopRearLeft,
    BedChannel::TopRearRight,
];

const RENDER_714_CHANNEL_POSITIONS: [Vec3; RENDER_714_CHANNELS] = [
    Vec3 {
        x: -0.707_106_77,
        y: 0.0,
        z: 1.0,
    },
    Vec3 {
        x: 0.707_106_77,
        y: 0.0,
        z: 1.0,
    },
    Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    },
    Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    },
    Vec3 {
        x: -0.707_106_77,
        y: 0.0,
        z: -1.0,
    },
    Vec3 {
        x: 0.707_106_77,
        y: 0.0,
        z: -1.0,
    },
    Vec3 {
        x: -1.0,
        y: 0.0,
        z: -0.483_689_52,
    },
    Vec3 {
        x: 1.0,
        y: 0.0,
        z: -0.483_689_52,
    },
    Vec3 {
        x: -1.0,
        y: 1.0,
        z: 0.483_689_52,
    },
    Vec3 {
        x: 1.0,
        y: 1.0,
        z: 0.483_689_52,
    },
    Vec3 {
        x: -0.707_106_77,
        y: 1.0,
        z: -1.0,
    },
    Vec3 {
        x: 0.707_106_77,
        y: 1.0,
        z: -1.0,
    },
];

#[derive(Debug, Clone, PartialEq)]
/// One rendered 7.1.4 frame.
pub struct Render714Frame {
    pub sample_rate: u32,
    pub channel_order: Vec<BedChannel>,
    pub channels: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
/// Debug view of one source state inside a rendered timeslot.
pub struct Render714SourceDebug {
    pub object_index: usize,
    pub static_channel: Option<BedChannel>,
    pub position: Vec3,
    pub gain: f32,
    pub size: f32,
    pub lfe: bool,
    pub position_valid: bool,
}

#[derive(Debug, Clone, PartialEq)]
/// Debug payload emitted by [`Renderer714::push_frame_with_debug`].
pub struct Render714TimeslotDebug {
    pub sample_offset: usize,
    pub sources: Vec<Render714SourceDebug>,
}

impl Render714Frame {
    /// Number of samples carried by each output channel.
    pub fn samples_per_channel(&self) -> usize {
        self.channels.first().map(Vec::len).unwrap_or(0)
    }

    /// Number of output channels. For this renderer it is always `12`.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Errors returned by the 7.1.4 renderer.
pub enum Render714Error {
    #[error("missing-oamd")]
    MissingOamd,
    #[error("oamd-state-uninitialized")]
    OamdStateUninitialized,
    #[error("object-count-mismatch expected={expected} provided={provided}")]
    ObjectCountMismatch { expected: usize, provided: usize },
    #[error("bed-channel-count-mismatch expected={expected} provided={provided}")]
    BedChannelCountMismatch { expected: usize, provided: usize },
    #[error("unsupported-sample-count {0}")]
    UnsupportedSampleCount(usize),
    #[error("unsupported-bed-channel {0:?}")]
    UnsupportedBedChannel(BedChannel),
    #[error("sample-rate-changed expected={expected} provided={provided}")]
    SampleRateChanged { expected: u32, provided: u32 },
}

#[derive(Debug, Clone, Copy)]
struct RenderInputChannelRef<'a> {
    channel: BedChannel,
    samples: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
struct RenderInputFrameRef<'a> {
    sample_rate: u32,
    bed_channels: &'a [RenderInputChannelRef<'a>],
    object_channels: &'a [&'a [f32]],
    metadata_updates: &'a [RenderMetadataUpdate],
}

impl RenderInputFrameRef<'_> {
    fn samples_per_channel(&self) -> usize {
        self.bed_channels
            .first()
            .map(|channel| channel.samples.len())
            .or_else(|| self.object_channels.first().map(|channel| channel.len()))
            .unwrap_or(0)
    }
}

#[derive(Debug)]
/// Stateful 7.1.4 renderer.
///
/// Feed [`RenderInputFrame`] values in stream order. The renderer keeps object metadata, limiter
/// state, and LFE low-pass history across frames, so it must be reset after seeks or any other
/// discontinuity.
pub struct Renderer714 {
    sample_rate: Option<u32>,
    limiter_gain: f32,
    stream_sample_offset: i64,
    metadata_timeslot_phase: usize,
    metadata_timeslot_stream_sample: i64,
    lfe_lowpass: Option<BiquadLowpassState>,
    metadata: MetadataRendererState,
    limiter_pending: Vec<Vec<f32>>,
    pending_timeslot_debug: Vec<Render714TimeslotDebug>,
}

impl Default for Renderer714 {
    fn default() -> Self {
        Self {
            sample_rate: None,
            limiter_gain: 1.0,
            stream_sample_offset: 0,
            metadata_timeslot_phase: 0,
            metadata_timeslot_stream_sample: 0,
            lfe_lowpass: None,
            metadata: MetadataRendererState::default(),
            limiter_pending: vec![Vec::new(); RENDER_714_CHANNELS],
            pending_timeslot_debug: Vec::new(),
        }
    }
}

impl Renderer714 {
    /// Create a fresh renderer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all cross-frame render state.
    pub fn reset(&mut self) {
        self.sample_rate = None;
        self.limiter_gain = 1.0;
        self.stream_sample_offset = 0;
        self.metadata_timeslot_phase = 0;
        self.metadata_timeslot_stream_sample = 0;
        self.lfe_lowpass = None;
        self.metadata.reset();
        for channel in &mut self.limiter_pending {
            channel.clear();
        }
        self.pending_timeslot_debug.clear();
    }

    /// Render one input frame to 7.1.4 float PCM.
    ///
    /// The output limiter finalizes audio in 64-sample blocks. Calls that end mid-block may
    /// return fewer samples than they consumed; call [`Self::flush`] after the final input to
    /// retrieve any trailing partial block.
    pub fn push_frame(
        &mut self,
        frame: &RenderInputFrame,
    ) -> Result<Render714Frame, Render714Error> {
        let mut bed_channels = Vec::new();
        let mut object_channels = Vec::new();
        let input = render_input_from_frame(frame, &mut bed_channels, &mut object_channels);
        self.push_frame_impl(&input)
    }

    /// Render one input frame and also capture the effective source state for every render
    /// timeslot.
    ///
    /// Debug rows follow the samples returned by this call. If the limiter buffers a partial
    /// block, its debug row is deferred until the corresponding audio is emitted.
    pub fn push_frame_with_debug(
        &mut self,
        frame: &RenderInputFrame,
    ) -> Result<(Render714Frame, Vec<Render714TimeslotDebug>), Render714Error> {
        let mut bed_channels = Vec::new();
        let mut object_channels = Vec::new();
        let input = render_input_from_frame(frame, &mut bed_channels, &mut object_channels);
        self.push_frame_with_debug_impl(&input)
    }

    /// Emit the final short limiter block after the last input frame in a stream.
    ///
    /// This should only be used at end of stream. If more input follows, call [`Self::reset`]
    /// first to avoid mixing two independent limiter blocks together.
    pub fn flush(&mut self) -> Option<Render714Frame> {
        self.flush_impl().map(|(frame, _)| frame)
    }

    /// Like [`Self::flush`], but also returns deferred timeslot debug rows.
    pub fn flush_with_debug(&mut self) -> Option<(Render714Frame, Vec<Render714TimeslotDebug>)> {
        self.flush_impl()
    }

    fn push_frame_impl(
        &mut self,
        frame: &RenderInputFrameRef<'_>,
    ) -> Result<Render714Frame, Render714Error> {
        let (rendered, _) = self.push_frame_common_impl(frame)?;
        Ok(rendered)
    }

    fn push_frame_with_debug_impl(
        &mut self,
        frame: &RenderInputFrameRef<'_>,
    ) -> Result<(Render714Frame, Vec<Render714TimeslotDebug>), Render714Error> {
        self.push_frame_common_impl(frame)
    }

    fn push_frame_common_impl(
        &mut self,
        frame: &RenderInputFrameRef<'_>,
    ) -> Result<(Render714Frame, Vec<Render714TimeslotDebug>), Render714Error> {
        let samples = frame.samples_per_channel();
        let mut channels = vec![vec![0.0f32; samples]; RENDER_714_CHANNELS];
        let mut debug = Vec::new();
        self.push_frame_parts_impl(frame, &mut channels, &mut debug)?;
        let (channels, debug) = self.finalize_output(channels, debug, frame.sample_rate);
        Ok((
            Render714Frame {
                sample_rate: frame.sample_rate,
                channel_order: RENDER_714_CHANNEL_ORDER.to_vec(),
                channels,
            },
            debug,
        ))
    }

    fn push_frame_parts_impl(
        &mut self,
        frame: &RenderInputFrameRef<'_>,
        channels: &mut [Vec<f32>],
        debug: &mut Vec<Render714TimeslotDebug>,
    ) -> Result<(), Render714Error> {
        match self.sample_rate {
            Some(sample_rate) if sample_rate != frame.sample_rate => {
                return Err(Render714Error::SampleRateChanged {
                    expected: sample_rate,
                    provided: frame.sample_rate,
                });
            }
            None => {
                self.sample_rate = Some(frame.sample_rate);
                self.lfe_lowpass = Some(BiquadLowpassState::new(frame.sample_rate));
            }
            Some(_) => {}
        }

        let frame_stream_start = self.stream_sample_offset;
        let mut metadata_updates = frame.metadata_updates.iter().collect::<Vec<_>>();
        metadata_updates.sort_by_key(|update| update.sample_offset);
        let mut next_metadata_update = 0usize;
        if !self.metadata.initialized && metadata_updates.is_empty() {
            return Err(Render714Error::MissingOamd);
        }
        while let Some(update) = metadata_updates.get(next_metadata_update) {
            if update.sample_offset != 0 {
                break;
            }
            let was_initialized = self.metadata.initialized;
            self.metadata.apply_update(
                &update.metadata,
                frame_stream_start + i64::from(update.sample_offset),
            );
            if !was_initialized {
                self.metadata_timeslot_phase = 0;
                self.metadata_timeslot_stream_sample =
                    frame_stream_start + i64::from(update.sample_offset);
            }
            self.validate_dynamic_object_count(frame.object_channels.len())?;
            next_metadata_update += 1;
        }
        if next_metadata_update > 0 {
            self.metadata_timeslot_phase = 0;
            self.metadata_timeslot_stream_sample = frame_stream_start;
        }

        let samples = validate_render_input_sample_counts(frame)?;
        if samples == 0 {
            return Err(Render714Error::UnsupportedSampleCount(samples));
        }
        debug.reserve(samples.div_ceil(RENDER_TIMESLOT_SAMPLES) + 1);

        if self.metadata.initialized {
            self.validate_dynamic_object_count(frame.object_channels.len())?;
        }

        prepare_render_channels(channels, samples);

        let mut sample_offset = 0usize;
        while sample_offset < samples {
            while let Some(update) = metadata_updates.get(next_metadata_update) {
                if usize::from(update.sample_offset) > sample_offset {
                    break;
                }
                let was_initialized = self.metadata.initialized;
                self.metadata.apply_update(
                    &update.metadata,
                    frame_stream_start + i64::from(update.sample_offset),
                );
                if !was_initialized {
                    self.metadata_timeslot_phase = 0;
                    self.metadata_timeslot_stream_sample =
                        frame_stream_start + i64::from(update.sample_offset);
                }
                self.validate_dynamic_object_count(frame.object_channels.len())?;
                next_metadata_update += 1;
            }

            if !self.metadata.initialized {
                // Keep startup output silent until the first metadata payload becomes active.
                let sample_end = metadata_updates
                    .get(next_metadata_update)
                    .map(|update| usize::from(update.sample_offset).min(samples))
                    .unwrap_or(samples);
                if sample_end <= sample_offset {
                    return Err(Render714Error::OamdStateUninitialized);
                }
                sample_offset = sample_end;
                continue;
            }

            if self.metadata_timeslot_phase == 0 {
                self.metadata.update_timeslot(
                    self.metadata_timeslot_stream_sample,
                    RENDER_TIMESLOT_SAMPLES,
                );
                debug.push(self.capture_timeslot_debug(sample_offset));
            }
            let mut sample_end = sample_offset
                + (RENDER_TIMESLOT_SAMPLES - self.metadata_timeslot_phase)
                    .min(samples - sample_offset);
            if let Some(update) = metadata_updates.get(next_metadata_update) {
                let update_offset = usize::from(update.sample_offset);
                if update_offset > sample_offset {
                    sample_end = sample_end.min(update_offset.min(samples));
                }
            }
            let timeslot_samples = sample_end - sample_offset;

            mix_bed_objects_segment_to_714(
                frame.bed_channels,
                &self.metadata.bed_channels,
                self.metadata.bed_sources(),
                channels,
                sample_offset,
                timeslot_samples,
            )?;

            for (object_index, object_samples) in frame.object_channels.iter().enumerate() {
                let source = &self.metadata.dynamic_sources[object_index];
                render_object_timeslot_to_714(
                    &object_samples[sample_offset..sample_end],
                    channels,
                    sample_offset,
                    source,
                );
            }

            self.metadata_timeslot_phase += timeslot_samples;
            if self.metadata_timeslot_phase == RENDER_TIMESLOT_SAMPLES {
                self.metadata_timeslot_phase = 0;
                self.metadata_timeslot_stream_sample += RENDER_TIMESLOT_SAMPLES as i64;
            }
            sample_offset = sample_end;
        }

        self.stream_sample_offset += samples as i64;

        if let Some(lfe_lowpass) = self.lfe_lowpass.as_mut() {
            lfe_lowpass.process_in_place(&mut channels[RENDER_714_LFE_INDEX]);
        }

        Ok(())
    }

    fn validate_dynamic_object_count(
        &self,
        provided_dynamic_objects: usize,
    ) -> Result<(), Render714Error> {
        // Metadata bed object cardinality does not necessarily match the decoded PCM bed-channel
        // count, so we only validate the dynamic object side here.
        let expected_dynamic_objects = self.metadata.dynamic_object_count();
        if expected_dynamic_objects != provided_dynamic_objects {
            return Err(Render714Error::ObjectCountMismatch {
                expected: expected_dynamic_objects,
                provided: provided_dynamic_objects,
            });
        }
        Ok(())
    }

    fn finalize_output(
        &mut self,
        channels: Vec<Vec<f32>>,
        debug: Vec<Render714TimeslotDebug>,
        sample_rate: u32,
    ) -> (Vec<Vec<f32>>, Vec<Render714TimeslotDebug>) {
        if limiter_disabled() {
            return (channels, debug);
        }

        for (pending, channel) in self.limiter_pending.iter_mut().zip(channels.iter()) {
            pending.extend_from_slice(channel);
        }
        self.pending_timeslot_debug.extend(debug);
        self.take_ready_output(sample_rate)
    }

    fn take_ready_output(
        &mut self,
        sample_rate: u32,
    ) -> (Vec<Vec<f32>>, Vec<Render714TimeslotDebug>) {
        let ready_samples = self.limiter_pending.first().map(Vec::len).unwrap_or(0)
            / RENDER_TIMESLOT_SAMPLES
            * RENDER_TIMESLOT_SAMPLES;

        if ready_samples == 0 {
            return (vec![Vec::new(); RENDER_714_CHANNELS], Vec::new());
        }

        let mut channels = Vec::with_capacity(RENDER_714_CHANNELS);
        for pending in &mut self.limiter_pending {
            let remainder = pending.split_off(ready_samples);
            channels.push(std::mem::replace(pending, remainder));
        }

        apply_output_limiter_blocks(&mut channels, &mut self.limiter_gain, sample_rate);

        let mut debug = self
            .pending_timeslot_debug
            .drain(..ready_samples / RENDER_TIMESLOT_SAMPLES)
            .collect::<Vec<_>>();
        for (index, row) in debug.iter_mut().enumerate() {
            row.sample_offset = index * RENDER_TIMESLOT_SAMPLES;
        }

        (channels, debug)
    }

    fn flush_impl(&mut self) -> Option<(Render714Frame, Vec<Render714TimeslotDebug>)> {
        if limiter_disabled() {
            return None;
        }

        let pending_samples = self.limiter_pending.first().map(Vec::len).unwrap_or(0);
        if pending_samples == 0 {
            return None;
        }

        let sample_rate = self.sample_rate?;
        let mut channels = Vec::with_capacity(RENDER_714_CHANNELS);
        for pending in &mut self.limiter_pending {
            channels.push(std::mem::take(pending));
        }

        apply_output_limiter_partial(&mut channels, &mut self.limiter_gain, sample_rate);

        let mut debug = self.pending_timeslot_debug.drain(..).collect::<Vec<_>>();
        for (index, row) in debug.iter_mut().enumerate() {
            row.sample_offset = index * RENDER_TIMESLOT_SAMPLES;
        }

        Some((
            Render714Frame {
                sample_rate,
                channel_order: RENDER_714_CHANNEL_ORDER.to_vec(),
                channels,
            },
            debug,
        ))
    }

    fn capture_timeslot_debug(&self, sample_offset: usize) -> Render714TimeslotDebug {
        let mut sources = Vec::with_capacity(
            self.metadata.bed_sources.len() + self.metadata.dynamic_sources.len(),
        );

        for (object_index, (channel, source)) in self
            .metadata
            .bed_channels
            .iter()
            .copied()
            .zip(self.metadata.bed_sources.iter())
            .enumerate()
        {
            sources.push(Render714SourceDebug {
                object_index,
                static_channel: Some(channel),
                position: absolute_position(bed_channel_position(channel)),
                gain: source.gain,
                size: 0.0,
                lfe: matches!(
                    channel,
                    BedChannel::LowFrequencyEffects | BedChannel::LowFrequencyEffects2
                ),
                position_valid: true,
            });
        }

        for (dynamic_index, source) in self.metadata.dynamic_sources.iter().enumerate() {
            sources.push(Render714SourceDebug {
                object_index: self.metadata.bed_sources.len() + dynamic_index,
                static_channel: None,
                position: absolute_position(source.cubical_position),
                gain: source.gain,
                size: source.size,
                lfe: false,
                position_valid: source.position_valid,
            });
        }

        Render714TimeslotDebug {
            sample_offset,
            sources,
        }
    }
}

fn prepare_render_channels(channels: &mut [Vec<f32>], samples: usize) {
    debug_assert_eq!(channels.len(), RENDER_714_CHANNELS);
    for channel in channels.iter_mut() {
        channel.resize(samples, 0.0);
        channel.fill(0.0);
    }
}

#[derive(Debug, Clone)]
struct DynamicSourceState {
    gain: f32,
    size: f32,
    cubical_position: Vec3,
    position_valid: bool,
}

impl Default for DynamicSourceState {
    fn default() -> Self {
        Self {
            gain: 0.707,
            size: 0.0,
            cubical_position: ZERO_VEC3,
            position_valid: false,
        }
    }
}

#[derive(Debug, Clone)]
struct BedSourceState {
    gain: f32,
}

impl Default for BedSourceState {
    fn default() -> Self {
        Self { gain: 0.707 }
    }
}

#[derive(Debug, Clone)]
struct BiquadLowpassState {
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadLowpassState {
    fn new(sample_rate: u32) -> Self {
        let w0 = std::f32::consts::TAU * LFE_LOW_PASS_HZ / sample_rate as f32;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (LOW_PASS_REFERENCE_Q + LOW_PASS_REFERENCE_Q);
        let divisor = 1.0 / (1.0 + alpha);
        let a1 = -2.0 * cos_w0 * divisor;
        let a2 = (1.0 - alpha) * divisor;
        let b1 = (1.0 - cos_w0) * divisor;
        let b2 = b1.abs() * 0.5;
        Self {
            a1,
            a2,
            b0: b2,
            b1,
            b2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process_in_place(&mut self, samples: &mut [f32]) {
        for sample in samples {
            let current = *sample;
            *sample = self.b2 * self.x2 + self.b1 * self.x1 + self.b0 * current
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.y2 = self.y1;
            self.y1 = *sample;
            self.x2 = self.x1;
            self.x1 = current;
        }
    }
}

#[derive(Debug, Default)]
struct MetadataRendererState {
    initialized: bool,
    payload_stream_sample: i64,
    object_count: usize,
    bed_or_isf_objects: usize,
    bed_channels: Vec<BedChannel>,
    bed_sources: Vec<BedSourceState>,
    elements: Vec<ElementRendererState>,
    dynamic_sources: Vec<DynamicSourceState>,
}

impl MetadataRendererState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn apply_update(&mut self, metadata: &RenderMetadata, payload_stream_sample: i64) {
        self.initialized = true;
        self.payload_stream_sample = payload_stream_sample;
        self.object_count = metadata.object_count;
        self.bed_or_isf_objects = metadata.bed_or_isf_objects;
        self.bed_channels = metadata.bed_channels.clone();
        // TODO: Revalidate multi-bed instance handling when a stream with multiple non-LFE bed
        // instances is available.

        let dynamic_object_count = metadata.dynamic_object_count();
        if self.dynamic_sources.len() != dynamic_object_count {
            self.dynamic_sources = vec![DynamicSourceState::default(); dynamic_object_count];
        }
        if self.bed_sources.len() != metadata.bed_channels.len() {
            self.bed_sources = vec![BedSourceState::default(); metadata.bed_channels.len()];
        }

        if self.elements.len() != metadata.elements.len() {
            self.elements = vec![ElementRendererState::default(); metadata.elements.len()];
        }

        for (index, element) in metadata.elements.iter().enumerate() {
            self.elements[index].apply_element(
                element,
                metadata.object_count,
                metadata.bed_or_isf_objects,
            );
        }
    }

    fn dynamic_object_count(&self) -> usize {
        self.dynamic_sources.len()
    }

    fn bed_sources(&self) -> &[BedSourceState] {
        &self.bed_sources
    }

    fn update_timeslot(&mut self, timeslot_stream_sample: i64, timeslot_samples: usize) {
        let adjusted = timeslot_stream_sample - self.payload_stream_sample;
        let mut element_index = 0usize;
        for index in (0..self.elements.len()).rev() {
            if self.elements[index].min_offset >= 0 && self.elements[index].min_offset <= adjusted {
                element_index = index;
                break;
            }
        }
        if let Some(element) = self.elements.get_mut(element_index) {
            element.update_sources(
                adjusted,
                timeslot_samples,
                self.bed_or_isf_objects,
                &mut self.bed_sources,
                &mut self.dynamic_sources,
            );
        }
    }
}

#[derive(Debug, Clone)]
struct ElementRendererState {
    min_offset: i64,
    block_used: Vec<bool>,
    update_last: Vec<bool>,
    update_now: Vec<bool>,
    block_offsets: Vec<i64>,
    ramp_duration: Vec<i64>,
    info_blocks: Vec<Vec<ObjectInfoBlockState>>,
    future: Vec<Vec3>,
    future_distance: i64,
}

impl Default for ElementRendererState {
    fn default() -> Self {
        Self {
            min_offset: -1,
            block_used: Vec::new(),
            update_last: Vec::new(),
            update_now: Vec::new(),
            block_offsets: Vec::new(),
            ramp_duration: Vec::new(),
            info_blocks: Vec::new(),
            future: Vec::new(),
            future_distance: 0,
        }
    }
}

impl ElementRendererState {
    fn apply_element(
        &mut self,
        element: &RenderMetadataElement,
        object_count: usize,
        bed_or_isf_objects: usize,
    ) {
        let block_count = element.block_updates.len();
        let dynamic_object_count = object_count.saturating_sub(bed_or_isf_objects);
        if self.block_used.len() != block_count || self.info_blocks.len() != object_count {
            self.block_used = vec![false; block_count];
            self.update_last = vec![false; dynamic_object_count];
            self.update_now = vec![false; dynamic_object_count];
            self.block_offsets = vec![0; block_count];
            self.ramp_duration = vec![0; block_count];
            self.info_blocks =
                vec![vec![ObjectInfoBlockState::default(); block_count]; object_count];
            self.future = vec![ZERO_VEC3; dynamic_object_count];
            self.future_distance = 0;
        } else {
            self.block_used.fill(false);
        }

        for (index, update) in element.block_updates.iter().enumerate() {
            self.block_offsets[index] = update.offset;
            self.ramp_duration[index] = update.ramp_duration;
        }
        self.min_offset = self.block_offsets.first().copied().unwrap_or(-1);

        for object_index in 0..object_count {
            for block_index in 0..block_count {
                self.info_blocks[object_index][block_index].apply_parsed_update(
                    &element.object_blocks[object_index][block_index],
                    object_index < bed_or_isf_objects,
                );
            }
        }
    }

    fn update_sources(
        &mut self,
        timecode: i64,
        timeslot_samples: usize,
        bed_or_isf_objects: usize,
        bed_sources: &mut [BedSourceState],
        dynamic_sources: &mut [DynamicSourceState],
    ) {
        if self.min_offset < 0 || self.block_used.is_empty() {
            return;
        }

        self.update_last.copy_from_slice(&self.update_now);

        for block_index in 0..self.block_used.len() {
            if self.block_used[block_index] || timecode <= self.block_offsets[block_index] {
                continue;
            }

            self.block_used[block_index] = true;
            self.future_distance =
                self.ramp_duration[block_index] - (timecode - self.block_offsets[block_index]);
            for (bed_index, source) in bed_sources.iter_mut().enumerate() {
                let info_block = &mut self.info_blocks[bed_index][block_index];
                info_block.update_bed_source_state(source);
            }
            for (dynamic_index, source) in dynamic_sources.iter_mut().enumerate() {
                let object_index = dynamic_index + bed_or_isf_objects;
                let info_block = &mut self.info_blocks[object_index][block_index];
                self.update_now[dynamic_index] = info_block.valid_position;
                self.future[dynamic_index] = info_block.update_source_state(source);
                if self.update_now[dynamic_index] && self.future_distance <= 0 {
                    source.cubical_position = self.future[dynamic_index];
                    source.position_valid = true;
                }
            }
        }

        if self.future_distance > 0 {
            let t = (timeslot_samples as f32 / self.future_distance as f32).min(1.0);
            for (dynamic_index, source) in dynamic_sources.iter_mut().enumerate() {
                if self.update_now[dynamic_index] {
                    source.cubical_position =
                        if self.update_last[dynamic_index] && source.position_valid {
                            lerp_vec3(source.cubical_position, self.future[dynamic_index], t)
                        } else {
                            self.future[dynamic_index]
                        };
                    source.position_valid = true;
                }
            }
            self.future_distance -= timeslot_samples as i64;
        }
    }
}

#[derive(Debug, Clone)]
struct ObjectInfoBlockState {
    valid_position: bool,
    differential_position: bool,
    gain: Option<f32>,
    distance: Option<f32>,
    size: Option<f32>,
    screen_factor: f32,
    depth_factor: f32,
    anchor: ObjectAnchor,
    position: Vec3,
    last_precise: Vec3,
}

impl Default for ObjectInfoBlockState {
    fn default() -> Self {
        Self {
            valid_position: false,
            differential_position: false,
            gain: None,
            distance: None,
            size: None,
            screen_factor: 1.0,
            depth_factor: 1.0,
            anchor: ObjectAnchor::Room,
            position: ZERO_VEC3,
            last_precise: ZERO_VEC3,
        }
    }
}

impl ObjectInfoBlockState {
    fn apply_parsed_update(&mut self, parsed: &RenderMetadataObject, bed_or_isf_object: bool) {
        self.gain = parsed.gain;
        self.valid_position = parsed.position_valid;
        self.differential_position = parsed.differential_position;
        if let Some(position) = parsed.position {
            self.position = position;
        }
        self.distance = parsed.distance.map(|distance| {
            if distance.is_infinite() {
                INFINITE_DISTANCE_FALLBACK
            } else {
                distance
            }
        });
        self.size = parsed.size;
        self.anchor = parsed.anchor;
        self.screen_factor = parsed.screen_factor;
        self.depth_factor = parsed.depth_factor;

        if bed_or_isf_object {
            self.anchor = ObjectAnchor::Speaker;
        }
    }

    fn update_source_state(&mut self, source: &mut DynamicSourceState) -> Vec3 {
        if let Some(gain) = self.gain {
            source.gain = gain;
        }
        if let Some(size) = self.size {
            source.size = size;
        }

        if self.valid_position && self.anchor != ObjectAnchor::Speaker {
            let mut position = self.position;
            if self.differential_position {
                position = Vec3 {
                    x: (self.last_precise.x + position.x).clamp(0.0, 1.0),
                    y: (self.last_precise.y + position.y).clamp(0.0, 1.0),
                    z: (self.last_precise.z + position.z).clamp(0.0, 1.0),
                };
                self.position = position;
            } else {
                self.last_precise = position;
            }

            position = match self.anchor {
                ObjectAnchor::Room => room_anchored_position(position, self.distance),
                ObjectAnchor::Screen => {
                    screen_anchored_position(position, self.screen_factor, self.depth_factor)
                }
                ObjectAnchor::Speaker => position,
            };
            return Vec3 {
                x: position.x * 2.0 - 1.0,
                y: position.z,
                z: position.y * -2.0 + 1.0,
            };
        }

        source.cubical_position
    }

    fn update_bed_source_state(&mut self, source: &mut BedSourceState) {
        if let Some(gain) = self.gain {
            source.gain = gain;
        }
    }
}

#[cfg(test)]
fn mix_bed_objects_to_714(
    input_bed_channels: &[RenderInputChannelRef<'_>],
    metadata_bed_channels: &[BedChannel],
    bed_sources: &[BedSourceState],
    output: &mut [Vec<f32>],
) -> Result<(), Render714Error> {
    let samples = output.first().map(Vec::len).unwrap_or(0);
    mix_bed_objects_segment_to_714(
        input_bed_channels,
        metadata_bed_channels,
        bed_sources,
        output,
        0,
        samples,
    )
}

fn mix_bed_objects_segment_to_714(
    input_bed_channels: &[RenderInputChannelRef<'_>],
    metadata_bed_channels: &[BedChannel],
    bed_sources: &[BedSourceState],
    output: &mut [Vec<f32>],
    output_offset: usize,
    samples: usize,
) -> Result<(), Render714Error> {
    if metadata_bed_channels.len() != bed_sources.len() {
        return Err(Render714Error::BedChannelCountMismatch {
            expected: metadata_bed_channels.len(),
            provided: bed_sources.len(),
        });
    }

    for (bed_channel, source) in metadata_bed_channels
        .iter()
        .copied()
        .zip(bed_sources.iter())
    {
        let input_channel = find_matching_input_bed_channel(bed_channel, input_bed_channels)
            .ok_or(Render714Error::BedChannelCountMismatch {
                expected: metadata_bed_channels
                    .iter()
                    .filter(|candidate| bed_channel_matches_input(bed_channel, **candidate))
                    .count(),
                provided: input_bed_channels
                    .iter()
                    .filter(|candidate| bed_channel_matches_input(bed_channel, candidate.channel))
                    .count(),
            })?;
        let input_samples = &input_channel.samples[output_offset..output_offset + samples];
        match bed_channel {
            BedChannel::LowFrequencyEffects | BedChannel::LowFrequencyEffects2 => mix_full_channel(
                input_samples,
                &mut output[RENDER_714_LFE_INDEX][output_offset..output_offset + samples],
                source.gain * LFE_SEND_MINUS_10_DB,
            ),
            BedChannel::TopSurroundLeft
            | BedChannel::TopSurroundRight
            | BedChannel::WideLeft
            | BedChannel::WideRight => mix_folded_bed_channel_to_714(
                input_samples,
                output,
                output_offset,
                bed_channel,
                source.gain,
            ),
            _ => {
                let output_channel = map_bed_channel_to_714(bed_channel)
                    .ok_or(Render714Error::UnsupportedBedChannel(bed_channel))?;
                mix_full_channel(
                    input_samples,
                    &mut output[output_channel][output_offset..output_offset + samples],
                    source.gain,
                );
            }
        }
    }
    Ok(())
}

fn find_matching_input_bed_channel<'a>(
    requested: BedChannel,
    input_bed_channels: &[RenderInputChannelRef<'a>],
) -> Option<RenderInputChannelRef<'a>> {
    input_bed_channels
        .iter()
        .copied()
        .find(|channel| channel.channel == requested)
        .or_else(|| {
            if is_low_frequency_bed(requested) {
                input_bed_channels
                    .iter()
                    .copied()
                    .find(|channel| is_low_frequency_bed(channel.channel))
            } else {
                None
            }
        })
}

fn bed_channel_matches_input(requested: BedChannel, provided: BedChannel) -> bool {
    requested == provided || (is_low_frequency_bed(requested) && is_low_frequency_bed(provided))
}

fn is_low_frequency_bed(channel: BedChannel) -> bool {
    matches!(
        channel,
        BedChannel::LowFrequencyEffects | BedChannel::LowFrequencyEffects2
    )
}

fn mix_folded_bed_channel_to_714(
    input: &[f32],
    output: &mut [Vec<f32>],
    output_offset: usize,
    channel: BedChannel,
    gain: f32,
) {
    let source = DynamicSourceState {
        gain,
        size: 0.0,
        cubical_position: bed_channel_position(channel),
        position_valid: true,
    };
    render_object_timeslot_to_714(input, output, output_offset, &source);
}

fn map_bed_channel_to_714(channel: BedChannel) -> Option<usize> {
    match channel {
        BedChannel::FrontLeft => Some(0),
        BedChannel::FrontRight => Some(1),
        BedChannel::Center => Some(2),
        BedChannel::LowFrequencyEffects => Some(3),
        BedChannel::RearLeft => Some(4),
        BedChannel::RearRight => Some(5),
        BedChannel::SurroundLeft => Some(6),
        BedChannel::SurroundRight => Some(7),
        BedChannel::TopFrontLeft => Some(8),
        BedChannel::TopFrontRight => Some(9),
        BedChannel::TopRearLeft => Some(10),
        BedChannel::TopRearRight => Some(11),
        BedChannel::TopSurroundLeft
        | BedChannel::TopSurroundRight
        | BedChannel::WideLeft
        | BedChannel::WideRight
        | BedChannel::LowFrequencyEffects2 => None,
    }
}

fn bed_channel_position(channel: BedChannel) -> Vec3 {
    match channel {
        BedChannel::FrontLeft => Vec3 {
            x: -1.0,
            y: 0.0,
            z: 1.0,
        },
        BedChannel::FrontRight => Vec3 {
            x: 1.0,
            y: 0.0,
            z: 1.0,
        },
        BedChannel::Center => Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        BedChannel::LowFrequencyEffects | BedChannel::LowFrequencyEffects2 => Vec3 {
            x: -1.0,
            y: -1.0,
            z: 1.0,
        },
        BedChannel::SurroundLeft => Vec3 {
            x: -1.0,
            y: 0.0,
            z: 0.0,
        },
        BedChannel::SurroundRight => Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        BedChannel::RearLeft => Vec3 {
            x: -1.0,
            y: 0.0,
            z: -1.0,
        },
        BedChannel::RearRight => Vec3 {
            x: 1.0,
            y: 0.0,
            z: -1.0,
        },
        BedChannel::TopFrontLeft => Vec3 {
            x: -1.0,
            y: 1.0,
            z: 1.0,
        },
        BedChannel::TopFrontRight => Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
        BedChannel::TopSurroundLeft => Vec3 {
            x: -1.0,
            y: 1.0,
            z: 0.0,
        },
        BedChannel::TopSurroundRight => Vec3 {
            x: 1.0,
            y: 1.0,
            z: 0.0,
        },
        BedChannel::TopRearLeft => Vec3 {
            x: -1.0,
            y: 1.0,
            z: -1.0,
        },
        BedChannel::TopRearRight => Vec3 {
            x: 1.0,
            y: 1.0,
            z: -1.0,
        },
        BedChannel::WideLeft => Vec3 {
            x: -1.0,
            y: 0.0,
            z: 0.677_419,
        },
        BedChannel::WideRight => Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.677_419,
        },
    }
}

fn render_object_timeslot_to_714(
    input: &[f32],
    output: &mut [Vec<f32>],
    output_offset: usize,
    source: &DynamicSourceState,
) {
    if input.is_empty() || source.gain == 0.0 {
        return;
    }

    let mut bottom_front_left = None;
    let mut bottom_front_right = None;
    let mut bottom_rear_left = None;
    let mut bottom_rear_right = None;
    let mut top_front_left = None;
    let mut top_front_right = None;
    let mut top_rear_left = None;
    let mut top_rear_right = None;
    let mut closest_top = 66.0f32;
    let mut closest_bottom = -69.0f32;
    let mut closest_top_front = 82.0f32;
    let mut closest_top_rear = -84.0f32;
    let mut closest_bottom_front = 65.0f32;
    let mut closest_bottom_rear = -2665.0f32;

    let direction = source.cubical_position;
    for (channel_index, channel_position) in RENDER_714_CHANNEL_POSITIONS.iter().enumerate() {
        if channel_index == RENDER_714_LFE_INDEX {
            continue;
        }
        let channel_y = channel_position.y;
        let channel_z = channel_position.z;
        if channel_y <= direction.y {
            if closest_bottom < channel_y {
                closest_bottom = channel_y;
                closest_bottom_front = f32::INFINITY;
                closest_bottom_rear = f32::NEG_INFINITY;
            }
            if closest_bottom == channel_y {
                if channel_z <= direction.z {
                    if closest_bottom_rear < channel_z {
                        closest_bottom_rear = channel_z;
                    }
                } else if closest_bottom_front > channel_z {
                    closest_bottom_front = channel_z;
                }
            }
        } else {
            if closest_top > channel_y {
                closest_top = channel_y;
                closest_top_front = f32::INFINITY;
                closest_top_rear = f32::NEG_INFINITY;
            }
            if closest_top == channel_y {
                if channel_z <= direction.z {
                    if closest_top_rear < channel_z {
                        closest_top_rear = channel_z;
                    }
                } else if closest_top_front > channel_z {
                    closest_top_front = channel_z;
                }
            }
        }
    }

    for (channel_index, channel_position) in RENDER_714_CHANNEL_POSITIONS.iter().enumerate() {
        if channel_index == RENDER_714_LFE_INDEX {
            continue;
        }
        if channel_position.y == closest_bottom {
            if channel_position.z == closest_bottom_front {
                assign_lr(
                    channel_index,
                    &mut bottom_front_left,
                    &mut bottom_front_right,
                    direction.x,
                    channel_position.x,
                );
            }
            if channel_position.z == closest_bottom_rear {
                assign_lr(
                    channel_index,
                    &mut bottom_rear_left,
                    &mut bottom_rear_right,
                    direction.x,
                    channel_position.x,
                );
            }
        }
        if channel_position.y == closest_top {
            if channel_position.z == closest_top_front {
                assign_lr(
                    channel_index,
                    &mut top_front_left,
                    &mut top_front_right,
                    direction.x,
                    channel_position.x,
                );
            }
            if channel_position.z == closest_top_rear {
                assign_lr(
                    channel_index,
                    &mut top_rear_left,
                    &mut top_rear_right,
                    direction.x,
                    channel_position.x,
                );
            }
        }
    }

    fix_incomplete_layer(
        &mut top_front_left,
        &mut top_front_right,
        &mut top_rear_left,
        &mut top_rear_right,
    );

    if bottom_front_left.is_none()
        && bottom_front_right.is_none()
        && bottom_rear_left.is_none()
        && bottom_rear_right.is_none()
    {
        bottom_front_left = top_front_left;
        bottom_front_right = top_front_right;
        bottom_rear_left = top_rear_left;
        bottom_rear_right = top_rear_right;
    } else {
        fix_incomplete_layer(
            &mut bottom_front_left,
            &mut bottom_front_right,
            &mut bottom_rear_left,
            &mut bottom_rear_right,
        );
    }

    if top_front_left.is_none()
        || top_front_right.is_none()
        || top_rear_left.is_none()
        || top_rear_right.is_none()
    {
        top_front_left = bottom_front_left;
        top_front_right = bottom_front_right;
        top_rear_left = bottom_rear_left;
        top_rear_right = bottom_rear_right;
    }

    let (
        Some(bottom_front_left),
        Some(bottom_front_right),
        Some(bottom_rear_left),
        Some(bottom_rear_right),
        Some(top_front_left),
        Some(top_front_right),
        Some(top_rear_left),
        Some(top_rear_right),
    ) = (
        bottom_front_left,
        bottom_front_right,
        bottom_rear_left,
        bottom_rear_right,
        top_front_left,
        top_front_right,
        top_rear_left,
        top_rear_right,
    )
    else {
        return;
    };

    let mut layer_bottom = 1.0f32;
    let mut layer_top = 0.0f32;
    if top_front_left != bottom_front_left {
        let bottom_y = RENDER_714_CHANNEL_POSITIONS[bottom_front_left].y;
        let top_y = RENDER_714_CHANNEL_POSITIONS[top_front_left].y;
        layer_top = (direction.y - bottom_y) / (top_y - bottom_y);
        layer_bottom = 1.0 - layer_top;
    }

    let front_bottom = ratio(
        RENDER_714_CHANNEL_POSITIONS[bottom_rear_left].z,
        RENDER_714_CHANNEL_POSITIONS[bottom_front_left].z,
        direction.z,
    );
    let front_top = ratio(
        RENDER_714_CHANNEL_POSITIONS[top_rear_left].z,
        RENDER_714_CHANNEL_POSITIONS[top_front_left].z,
        direction.z,
    );
    let rear_bottom = 1.0 - front_bottom;
    let rear_top = 1.0 - front_top;

    let size = source.size;
    let mut inner_volume_3d = source.gain;
    if size != 0.0 {
        inner_volume_3d *= 1.0 - size;
        let extra_channel_volume = source.gain * (size / RENDER_714_CHANNELS as f32).sqrt();
        for (channel_index, channel_output) in output.iter_mut().enumerate() {
            if channel_index != RENDER_714_LFE_INDEX {
                mix_segment(input, channel_output, output_offset, extra_channel_volume);
            }
        }
    }

    let front_bottom = front_bottom * layer_bottom * inner_volume_3d;
    let rear_bottom = rear_bottom * layer_bottom * inner_volume_3d;
    let front_top = front_top * layer_top * inner_volume_3d;
    let rear_top = rear_top * layer_top * inner_volume_3d;

    if front_bottom != 0.0 {
        let blend = ratio(
            RENDER_714_CHANNEL_POSITIONS[bottom_front_left].x,
            RENDER_714_CHANNEL_POSITIONS[bottom_front_right].x,
            direction.x,
        );
        mix_segment(
            input,
            &mut output[bottom_front_left],
            output_offset,
            (front_bottom * (1.0 - blend)).sqrt(),
        );
        mix_segment(
            input,
            &mut output[bottom_front_right],
            output_offset,
            (front_bottom * blend).sqrt(),
        );
    }
    if rear_bottom != 0.0 {
        let blend = ratio(
            RENDER_714_CHANNEL_POSITIONS[bottom_rear_left].x,
            RENDER_714_CHANNEL_POSITIONS[bottom_rear_right].x,
            direction.x,
        );
        mix_segment(
            input,
            &mut output[bottom_rear_left],
            output_offset,
            (rear_bottom * (1.0 - blend)).sqrt(),
        );
        mix_segment(
            input,
            &mut output[bottom_rear_right],
            output_offset,
            (rear_bottom * blend).sqrt(),
        );
    }
    if front_top != 0.0 {
        let blend = ratio(
            RENDER_714_CHANNEL_POSITIONS[top_front_left].x,
            RENDER_714_CHANNEL_POSITIONS[top_front_right].x,
            direction.x,
        );
        mix_segment(
            input,
            &mut output[top_front_left],
            output_offset,
            (front_top * (1.0 - blend)).sqrt(),
        );
        mix_segment(
            input,
            &mut output[top_front_right],
            output_offset,
            (front_top * blend).sqrt(),
        );
    }
    if rear_top != 0.0 {
        let blend = ratio(
            RENDER_714_CHANNEL_POSITIONS[top_rear_left].x,
            RENDER_714_CHANNEL_POSITIONS[top_rear_right].x,
            direction.x,
        );
        mix_segment(
            input,
            &mut output[top_rear_left],
            output_offset,
            (rear_top * (1.0 - blend)).sqrt(),
        );
        mix_segment(
            input,
            &mut output[top_rear_right],
            output_offset,
            (rear_top * blend).sqrt(),
        );
    }
}

fn assign_lr(
    channel: usize,
    left: &mut Option<usize>,
    right: &mut Option<usize>,
    position_x: f32,
    channel_x: f32,
) {
    if channel_x == position_x {
        *left = Some(channel);
        *right = Some(channel);
    } else if channel_x < position_x {
        if left.is_none_or(|existing| RENDER_714_CHANNEL_POSITIONS[existing].x < channel_x) {
            *left = Some(channel);
        }
    } else if right.is_none_or(|existing| RENDER_714_CHANNEL_POSITIONS[existing].x > channel_x) {
        *right = Some(channel);
    }
}

fn fix_incomplete_layer(
    front_left: &mut Option<usize>,
    front_right: &mut Option<usize>,
    rear_left: &mut Option<usize>,
    rear_right: &mut Option<usize>,
) {
    if front_left.is_some() || front_right.is_some() {
        if front_left.is_none() {
            *front_left = *front_right;
        }
        if front_right.is_none() {
            *front_right = *front_left;
        }
        if rear_left.is_none() && rear_right.is_none() {
            *rear_left = *front_left;
            *rear_right = *front_right;
        }
    }
    if rear_left.is_some() || rear_right.is_some() {
        if rear_left.is_none() {
            *rear_left = *rear_right;
        }
        if rear_right.is_none() {
            *rear_right = *rear_left;
        }
        if front_left.is_none() && front_right.is_none() {
            *front_left = *rear_left;
            *front_right = *rear_right;
        }
    }
}

fn ratio(a: f32, b: f32, x: f32) -> f32 {
    if a == b { 0.0 } else { (x - a) / (b - a) }
}

#[cfg(test)]
fn apply_output_limiter(channels: &mut [Vec<f32>], last_gain: &mut f32, sample_rate: u32) {
    apply_output_limiter_blocks(channels, last_gain, sample_rate);
}

fn apply_output_limiter_blocks(channels: &mut [Vec<f32>], last_gain: &mut f32, sample_rate: u32) {
    let Some(samples) = channels.first().map(Vec::len) else {
        return;
    };
    debug_assert_eq!(samples % RENDER_TIMESLOT_SAMPLES, 0);

    let mut block_start = 0usize;
    while block_start < samples {
        apply_output_limiter_block(
            channels,
            last_gain,
            sample_rate,
            block_start,
            block_start + RENDER_TIMESLOT_SAMPLES,
        );
        block_start += RENDER_TIMESLOT_SAMPLES;
    }
}

fn apply_output_limiter_partial(channels: &mut [Vec<f32>], last_gain: &mut f32, sample_rate: u32) {
    let Some(samples) = channels.first().map(Vec::len) else {
        return;
    };
    if samples == 0 {
        return;
    }

    apply_output_limiter_block(channels, last_gain, sample_rate, 0, samples);
}

fn apply_output_limiter_block(
    channels: &mut [Vec<f32>],
    last_gain: &mut f32,
    sample_rate: u32,
    block_start: usize,
    block_end: usize,
) {
    let decay = (block_end - block_start) as f32 / sample_rate as f32;
    let mut max = 0.0f32;
    for channel in channels.iter() {
        max = max.max(max_abs_slice(&channel[block_start..block_end]));
    }

    if max * *last_gain > 1.0 {
        *last_gain = 0.9 / max;
    }

    if *last_gain != 1.0 {
        for channel in channels.iter_mut() {
            scale_slice_in_place(&mut channel[block_start..block_end], *last_gain);
        }
    }

    *last_gain += decay;
    if *last_gain > 1.0 {
        *last_gain = 1.0;
    }
}

fn mix_full_channel(input: &[f32], output: &mut [f32], gain: f32) {
    if gain == 0.0 {
        return;
    }
    mix_add_scaled(input, output, gain);
}

fn limiter_disabled() -> bool {
    #[cfg(debug_assertions)]
    {
        std::env::var_os("STARMINE_AD_DISABLE_LIMITER").is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

fn mix_segment(input: &[f32], output: &mut [f32], offset: usize, gain: f32) {
    if gain == 0.0 {
        return;
    }
    mix_add_scaled(input, &mut output[offset..offset + input.len()], gain);
}

fn mix_add_scaled(input: &[f32], output: &mut [f32], gain: f32) {
    debug_assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    unsafe {
        mix_add_scaled_neon(input.as_ptr(), output.as_mut_ptr(), input.len(), gain);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (target, sample) in output.iter_mut().zip(input.iter().copied()) {
            *target += sample * gain;
        }
    }
}

fn max_abs_slice(samples: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        max_abs_slice_neon(samples.as_ptr(), samples.len())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut max = 0.0f32;
        for sample in samples.iter().copied() {
            max = max.max(sample.abs());
        }
        max
    }
}

fn scale_slice_in_place(samples: &mut [f32], gain: f32) {
    if gain == 1.0 {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        scale_slice_in_place_neon(samples.as_mut_ptr(), samples.len(), gain);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for sample in samples {
            *sample *= gain;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn mix_add_scaled_neon(input: *const f32, output: *mut f32, len: usize, gain: f32) {
    let gainv = vdupq_n_f32(gain);
    let mut index = 0usize;
    while index + 4 <= len {
        let mixed = vfmaq_f32(
            vld1q_f32(output.add(index)),
            vld1q_f32(input.add(index)),
            gainv,
        );
        vst1q_f32(output.add(index), mixed);
        index += 4;
    }
    while index < len {
        *output.add(index) += *input.add(index) * gain;
        index += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn max_abs_slice_neon(input: *const f32, len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 4 <= len {
        acc = vmaxq_f32(acc, vabsq_f32(vld1q_f32(input.add(index))));
        index += 4;
    }
    let mut max = vmaxvq_f32(acc);
    while index < len {
        max = max.max((*input.add(index)).abs());
        index += 1;
    }
    max
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn scale_slice_in_place_neon(output: *mut f32, len: usize, gain: f32) {
    let gainv = vdupq_n_f32(gain);
    let mut index = 0usize;
    while index + 4 <= len {
        vst1q_f32(
            output.add(index),
            vmulq_f32(vld1q_f32(output.add(index)), gainv),
        );
        index += 4;
    }
    while index < len {
        *output.add(index) *= gain;
        index += 1;
    }
}

fn absolute_position(cubical_position: Vec3) -> Vec3 {
    Vec3 {
        x: cubical_position.x * RENDER_ENVIRONMENT_SIZE.x,
        y: cubical_position.y * RENDER_ENVIRONMENT_SIZE.y,
        z: cubical_position.z * RENDER_ENVIRONMENT_SIZE.z,
    }
}

fn room_anchored_position(position: Vec3, distance: Option<f32>) -> Vec3 {
    let Some(distance) = distance else {
        return position;
    };
    let intersect = map_to_cube(position);
    let distance_factor = vec_length(intersect) / distance;
    vec_add(
        vec_mul_scalar(intersect, distance_factor),
        vec_mul_scalar(ROOM_CENTER, 1.0 - distance_factor),
    )
}

fn screen_anchored_position(position: Vec3, screen_factor: f32, depth_factor: f32) -> Vec3 {
    let reference = Vec3 {
        x: (position.x - 0.5) * SCREEN_SIZE_X + 0.5,
        y: position.y,
        z: (position.z + 1.0) * SCREEN_SIZE_Z,
    };
    let screen_multiplier = Vec3 {
        x: screen_factor,
        y: 1.0,
        z: screen_factor,
    };
    let depth = position.y.powf(depth_factor);
    let depth_multiplier = Vec3 {
        x: depth,
        y: 1.0,
        z: depth,
    };

    vec_add(
        vec_mul_components(
            depth_multiplier,
            vec_sub(
                vec_add(vec_mul_components(screen_multiplier, position), reference),
                vec_mul_components(screen_multiplier, reference),
            ),
        ),
        vec_sub(reference, vec_mul_components(depth_multiplier, reference)),
    )
}

fn map_to_cube(vector: Vec3) -> Vec3 {
    let max = vector.x.abs().max(vector.y.abs()).max(vector.z.abs());
    if max == 0.0 {
        ZERO_VEC3
    } else {
        vec_mul_scalar(vector, 1.0 / max)
    }
}

fn lerp_vec3(from: Vec3, to: Vec3, t: f32) -> Vec3 {
    Vec3 {
        x: from.x + (to.x - from.x) * t,
        y: from.y + (to.y - from.y) * t,
        z: from.z + (to.z - from.z) * t,
    }
}

fn vec_add(lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3 {
        x: lhs.x + rhs.x,
        y: lhs.y + rhs.y,
        z: lhs.z + rhs.z,
    }
}

fn vec_sub(lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3 {
        x: lhs.x - rhs.x,
        y: lhs.y - rhs.y,
        z: lhs.z - rhs.z,
    }
}

fn vec_mul_scalar(vector: Vec3, scalar: f32) -> Vec3 {
    Vec3 {
        x: vector.x * scalar,
        y: vector.y * scalar,
        z: vector.z * scalar,
    }
}

fn vec_mul_components(lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3 {
        x: lhs.x * rhs.x,
        y: lhs.y * rhs.y,
        z: lhs.z * rhs.z,
    }
}

fn vec_length(vector: Vec3) -> f32 {
    (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).sqrt()
}

fn render_input_from_frame<'a>(
    frame: &'a RenderInputFrame,
    bed_channels: &'a mut Vec<RenderInputChannelRef<'a>>,
    object_channels: &'a mut Vec<&'a [f32]>,
) -> RenderInputFrameRef<'a> {
    bed_channels.clear();
    bed_channels.reserve(frame.bed_channels.len());
    for channel in &frame.bed_channels {
        bed_channels.push(RenderInputChannelRef {
            channel: channel.channel,
            samples: &channel.samples,
        });
    }

    object_channels.clear();
    object_channels.reserve(frame.object_channels.len());
    for channel in &frame.object_channels {
        object_channels.push(channel.as_slice());
    }

    RenderInputFrameRef {
        sample_rate: frame.sample_rate,
        bed_channels,
        object_channels,
        metadata_updates: &frame.metadata_updates,
    }
}

fn validate_render_input_sample_counts(
    frame: &RenderInputFrameRef<'_>,
) -> Result<usize, Render714Error> {
    let samples = frame.samples_per_channel();

    for channel in frame.bed_channels {
        if channel.samples.len() != samples {
            return Err(Render714Error::UnsupportedSampleCount(
                channel.samples.len(),
            ));
        }
    }

    for channel in frame.object_channels {
        if channel.len() != samples {
            return Err(Render714Error::UnsupportedSampleCount(channel.len()));
        }
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::{
        BedSourceState, DynamicSourceState, ElementRendererState, ObjectInfoBlockState,
        RENDER_714_CHANNEL_ORDER, RENDER_714_CHANNELS, Render714Frame, RenderInputChannelRef,
        RenderInputFrame, RenderMetadataUpdate, Renderer714, apply_output_limiter,
        map_bed_channel_to_714, mix_bed_objects_to_714, render_object_timeslot_to_714,
    };
    use crate::eac3dec::{
        OamdBlockUpdate, OamdElement, OamdElementKind, OamdObjectBlock, OamdObjectElement,
        OamdPayload,
    };
    use crate::renderer::{BedChannel, RenderInputChannel, RenderMetadata, Vec3};

    fn concat_rendered_frames(frames: &[Render714Frame]) -> Vec<Vec<f32>> {
        let mut merged = vec![Vec::new(); RENDER_714_CHANNELS];
        for frame in frames {
            for (output, channel) in merged.iter_mut().zip(frame.channels.iter()) {
                output.extend_from_slice(channel);
            }
        }
        merged
    }

    fn assert_channels_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
        assert_eq!(actual.len(), expected.len());
        for (channel_index, (actual_channel, expected_channel)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(actual_channel.len(), expected_channel.len());
            for (sample_index, (actual_sample, expected_sample)) in actual_channel
                .iter()
                .zip(expected_channel.iter())
                .enumerate()
            {
                assert!(
                    (actual_sample - expected_sample).abs() < 1e-6,
                    "channel={channel_index} sample={sample_index} actual={actual_sample} expected={expected_sample}",
                );
            }
        }
    }

    fn metadata_updates_from_oamd(oamd: Option<OamdPayload>) -> Vec<RenderMetadataUpdate> {
        oamd.as_ref()
            .map(|payload| vec![RenderMetadataUpdate::from_oamd_payload(payload, Some(0))])
            .unwrap_or_default()
    }

    #[test]
    fn render_714_channel_order_is_stable() {
        assert_eq!(RENDER_714_CHANNEL_ORDER.len(), RENDER_714_CHANNELS);
        assert_eq!(RENDER_714_CHANNEL_ORDER[0], BedChannel::FrontLeft);
        assert_eq!(RENDER_714_CHANNEL_ORDER[3], BedChannel::LowFrequencyEffects);
        assert_eq!(RENDER_714_CHANNEL_ORDER[11], BedChannel::TopRearRight);
    }

    #[test]
    fn direct_bed_mapping_matches_714_layout() {
        assert_eq!(map_bed_channel_to_714(BedChannel::FrontLeft), Some(0));
        assert_eq!(map_bed_channel_to_714(BedChannel::RearRight), Some(5));
        assert_eq!(map_bed_channel_to_714(BedChannel::TopRearLeft), Some(10));
        assert_eq!(map_bed_channel_to_714(BedChannel::WideLeft), None);
    }

    #[test]
    fn bed_gain_updates_apply_from_metadata_blocks() {
        let mut element = ElementRendererState {
            min_offset: 0,
            block_used: vec![false],
            update_last: vec![],
            update_now: vec![],
            block_offsets: vec![0],
            ramp_duration: vec![0],
            info_blocks: vec![vec![ObjectInfoBlockState {
                gain: Some(0.5),
                ..ObjectInfoBlockState::default()
            }]],
            future: vec![],
            future_distance: 0,
        };
        let mut bed_sources = vec![BedSourceState::default()];
        let mut dynamic_sources = Vec::new();

        element.update_sources(1, 64, 1, &mut bed_sources, &mut dynamic_sources);

        assert_eq!(bed_sources[0].gain, 0.5);
    }

    #[test]
    fn limiter_matches_limiter_only_attack() {
        let mut channels = vec![vec![2.0f32; 64], vec![0.5f32; 64]];
        let mut gain = 1.0;

        apply_output_limiter(&mut channels, &mut gain, 48_000);

        assert!((channels[0][0] - 0.9).abs() < 1e-6);
        assert!((channels[1][0] - 0.225).abs() < 1e-6);
        assert!((gain - 0.451_333_34).abs() < 1e-6);
    }

    #[test]
    fn generic_render_input_accepts_non_timeslot_aligned_lengths() {
        let frame = RenderInputFrame {
            sample_rate: 48_000,
            bed_channels: vec![RenderInputChannel {
                channel: BedChannel::FrontLeft,
                samples: vec![0.25; 65],
            }],
            object_channels: Vec::new(),
            metadata_updates: metadata_updates_from_oamd(Some(OamdPayload {
                version: 0,
                object_count: 1,
                alternate_object_present: false,
                element_count: 0,
                beds: 1,
                bed_instances: 1,
                bed_or_isf_objects: 1,
                dynamic_objects: 0,
                isf_in_use: false,
                isf_index: None,
                bed_assignment: vec![vec![BedChannel::FrontLeft]],
                elements: Vec::new(),
            })),
        };

        let mut renderer = Renderer714::new();
        let rendered = renderer
            .push_frame(&frame)
            .expect("non-timeslot-aligned input should render");
        let flushed = renderer
            .flush()
            .expect("trailing partial block should be flushable");

        assert_eq!(rendered.samples_per_channel(), 64);
        assert_eq!(rendered.channels[0].len(), 64);
        assert_eq!(flushed.samples_per_channel(), 1);
        assert_eq!(flushed.channels[0].len(), 1);
    }

    #[test]
    fn generic_render_input_rejects_mismatched_channel_lengths() {
        let frame = RenderInputFrame {
            sample_rate: 48_000,
            bed_channels: vec![
                RenderInputChannel {
                    channel: BedChannel::FrontLeft,
                    samples: vec![0.25; 65],
                },
                RenderInputChannel {
                    channel: BedChannel::FrontRight,
                    samples: vec![0.25; 64],
                },
            ],
            object_channels: Vec::new(),
            metadata_updates: metadata_updates_from_oamd(Some(OamdPayload {
                version: 0,
                object_count: 2,
                alternate_object_present: false,
                element_count: 0,
                beds: 2,
                bed_instances: 1,
                bed_or_isf_objects: 2,
                dynamic_objects: 0,
                isf_in_use: false,
                isf_index: None,
                bed_assignment: vec![vec![BedChannel::FrontLeft, BedChannel::FrontRight]],
                elements: Vec::new(),
            })),
        };

        let err = Renderer714::new()
            .push_frame(&frame)
            .expect_err("mismatched channel lengths should fail");
        assert_eq!(err.to_string(), "unsupported-sample-count 64");
    }

    #[test]
    fn metadata_update_mid_frame_remixes_bed_output() {
        let metadata = |sample_offset: u16, bed_channel: BedChannel| RenderMetadataUpdate {
            sample_offset,
            metadata: RenderMetadata {
                object_count: 1,
                bed_or_isf_objects: 1,
                bed_channels: vec![bed_channel],
                elements: Vec::new(),
            },
        };
        let frame = RenderInputFrame {
            sample_rate: 48_000,
            bed_channels: vec![
                RenderInputChannel {
                    channel: BedChannel::FrontLeft,
                    samples: vec![0.25; 128],
                },
                RenderInputChannel {
                    channel: BedChannel::FrontRight,
                    samples: vec![0.5; 128],
                },
            ],
            object_channels: Vec::new(),
            metadata_updates: vec![
                metadata(0, BedChannel::FrontLeft),
                metadata(64, BedChannel::FrontRight),
            ],
        };

        let rendered = Renderer714::new()
            .push_frame(&frame)
            .expect("bed metadata update should split the frame render");

        assert!(rendered.channels[0][0] > 0.0);
        assert!(rendered.channels[1][0].abs() < 1e-6);
        assert!(rendered.channels[0][80].abs() < 1e-6);
        assert!(rendered.channels[1][80] > 0.0);
    }

    #[test]
    fn out_of_order_metadata_updates_render_like_sorted_updates() {
        let metadata = |sample_offset: u16, bed_channel: BedChannel| RenderMetadataUpdate {
            sample_offset,
            metadata: RenderMetadata {
                object_count: 1,
                bed_or_isf_objects: 1,
                bed_channels: vec![bed_channel],
                elements: Vec::new(),
            },
        };
        let bed_channels = vec![
            RenderInputChannel {
                channel: BedChannel::FrontLeft,
                samples: vec![0.25; 128],
            },
            RenderInputChannel {
                channel: BedChannel::FrontRight,
                samples: vec![0.5; 128],
            },
        ];
        let sorted = RenderInputFrame {
            sample_rate: 48_000,
            bed_channels: bed_channels.clone(),
            object_channels: Vec::new(),
            metadata_updates: vec![
                metadata(0, BedChannel::FrontLeft),
                metadata(64, BedChannel::FrontRight),
            ],
        };
        let unsorted = RenderInputFrame {
            sample_rate: 48_000,
            bed_channels,
            object_channels: Vec::new(),
            metadata_updates: vec![
                metadata(64, BedChannel::FrontRight),
                metadata(0, BedChannel::FrontLeft),
            ],
        };

        let expected = Renderer714::new()
            .push_frame(&sorted)
            .expect("sorted metadata updates should render");
        let actual = Renderer714::new()
            .push_frame(&unsorted)
            .expect("out-of-order metadata updates should be normalized");

        assert_channels_close(&actual.channels, &expected.channels);
    }

    #[test]
    fn first_metadata_update_can_start_late_in_frame() {
        let frame = RenderInputFrame {
            sample_rate: 48_000,
            bed_channels: vec![
                RenderInputChannel {
                    channel: BedChannel::FrontLeft,
                    samples: vec![0.0; 64],
                },
                RenderInputChannel {
                    channel: BedChannel::FrontRight,
                    samples: vec![1.0; 64],
                },
            ],
            object_channels: Vec::new(),
            metadata_updates: vec![RenderMetadataUpdate {
                sample_offset: 32,
                metadata: RenderMetadata {
                    object_count: 1,
                    bed_or_isf_objects: 1,
                    bed_channels: vec![BedChannel::FrontRight],
                    elements: Vec::new(),
                },
            }],
        };

        let rendered = Renderer714::new()
            .push_frame(&frame)
            .expect("initial metadata update with non-zero sample offset should render");

        for sample in 0..32 {
            assert!(rendered.channels[0][sample].abs() < 1e-6);
            assert!(rendered.channels[1][sample].abs() < 1e-6);
        }
        assert!(rendered.channels[0][48].abs() < 1e-6);
        assert!(rendered.channels[1][48] > 0.0);
    }

    #[test]
    fn mix_bed_objects_matches_sources_by_channel_label() {
        let input_bed_channels = [
            RenderInputChannelRef {
                channel: BedChannel::FrontRight,
                samples: &[1.0],
            },
            RenderInputChannelRef {
                channel: BedChannel::FrontLeft,
                samples: &[2.0],
            },
        ];
        let metadata_bed_channels = [BedChannel::FrontLeft, BedChannel::FrontRight];
        let bed_sources = vec![BedSourceState { gain: 0.5 }, BedSourceState { gain: 0.25 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("reordered labels should render");

        assert_eq!(output[0][0], 1.0);
        assert_eq!(output[1][0], 0.25);
    }

    #[test]
    fn mix_bed_objects_reuses_labeled_input_for_duplicate_metadata_beds() {
        let input_bed_channels = [RenderInputChannelRef {
            channel: BedChannel::FrontLeft,
            samples: &[2.0],
        }];
        let metadata_bed_channels = [BedChannel::FrontLeft, BedChannel::FrontLeft];
        let bed_sources = vec![BedSourceState { gain: 0.5 }, BedSourceState { gain: 0.25 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("duplicate metadata beds should reuse the labeled input");

        assert_eq!(output[0][0], 1.5);
    }

    #[test]
    fn mix_bed_objects_folds_wide_beds_like_static_objects() {
        let input_bed_channels = [RenderInputChannelRef {
            channel: BedChannel::WideLeft,
            samples: &[1.0],
        }];
        let metadata_bed_channels = [BedChannel::WideLeft];
        let bed_sources = vec![BedSourceState { gain: 0.5 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("wide beds should fold to 7.1.4");

        let mut expected = vec![vec![0.0; 1]; RENDER_714_CHANNELS];
        render_object_timeslot_to_714(
            &[1.0],
            &mut expected,
            0,
            &DynamicSourceState {
                gain: 0.5,
                size: 0.0,
                cubical_position: super::bed_channel_position(BedChannel::WideLeft),
                position_valid: true,
            },
        );

        assert_channels_close(&output, &expected);
        assert!(output[0][0] > 0.0);
        assert!(output[6][0] > 0.0);
    }

    #[test]
    fn mix_bed_objects_folds_top_side_beds_like_static_objects() {
        let input_bed_channels = [RenderInputChannelRef {
            channel: BedChannel::TopSurroundLeft,
            samples: &[1.0],
        }];
        let metadata_bed_channels = [BedChannel::TopSurroundLeft];
        let bed_sources = vec![BedSourceState { gain: 0.5 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("top-side beds should fold to 7.1.4");

        let mut expected = vec![vec![0.0; 1]; RENDER_714_CHANNELS];
        render_object_timeslot_to_714(
            &[1.0],
            &mut expected,
            0,
            &DynamicSourceState {
                gain: 0.5,
                size: 0.0,
                cubical_position: super::bed_channel_position(BedChannel::TopSurroundLeft),
                position_valid: true,
            },
        );

        assert_channels_close(&output, &expected);
        assert!(output[8][0] > 0.0);
        assert!(output[10][0] > 0.0);
    }

    #[test]
    fn mix_bed_objects_merges_lfe2_into_primary_lfe() {
        let input_bed_channels = [RenderInputChannelRef {
            channel: BedChannel::LowFrequencyEffects2,
            samples: &[1.0],
        }];
        let metadata_bed_channels = [BedChannel::LowFrequencyEffects2];
        let bed_sources = vec![BedSourceState { gain: 0.5 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("LFE2 should merge into the primary LFE output");

        assert!((output[3][0] - 0.5 * super::LFE_SEND_MINUS_10_DB).abs() < 1e-6);
        for (index, channel) in output.iter().enumerate() {
            if index != 3 {
                assert_eq!(channel[0], 0.0);
            }
        }
    }

    #[test]
    fn mix_bed_objects_accepts_lfe_alias_when_metadata_uses_lfe2() {
        let input_bed_channels = [RenderInputChannelRef {
            channel: BedChannel::LowFrequencyEffects,
            samples: &[1.0],
        }];
        let metadata_bed_channels = [BedChannel::LowFrequencyEffects2];
        let bed_sources = vec![BedSourceState { gain: 0.5 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("LFE should satisfy LFE2 metadata assignments");

        assert!((output[3][0] - 0.5 * super::LFE_SEND_MINUS_10_DB).abs() < 1e-6);
    }

    #[test]
    fn mix_bed_objects_accepts_lfe_alias_when_input_uses_lfe2() {
        let input_bed_channels = [RenderInputChannelRef {
            channel: BedChannel::LowFrequencyEffects2,
            samples: &[1.0],
        }];
        let metadata_bed_channels = [BedChannel::LowFrequencyEffects];
        let bed_sources = vec![BedSourceState { gain: 0.5 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("LFE2 should satisfy LFE metadata assignments");

        assert!((output[3][0] - 0.5 * super::LFE_SEND_MINUS_10_DB).abs() < 1e-6);
    }

    #[test]
    fn mix_bed_objects_prefers_exact_lfe_matches_when_both_inputs_exist() {
        let input_bed_channels = [
            RenderInputChannelRef {
                channel: BedChannel::LowFrequencyEffects2,
                samples: &[2.0],
            },
            RenderInputChannelRef {
                channel: BedChannel::LowFrequencyEffects,
                samples: &[1.0],
            },
        ];
        let metadata_bed_channels = [
            BedChannel::LowFrequencyEffects,
            BedChannel::LowFrequencyEffects2,
        ];
        let bed_sources = vec![BedSourceState { gain: 0.5 }, BedSourceState { gain: 0.25 }];
        let mut output = vec![vec![0.0; 1]; RENDER_714_CHANNELS];

        mix_bed_objects_to_714(
            &input_bed_channels,
            &metadata_bed_channels,
            &bed_sources,
            &mut output,
        )
        .expect("LFE and LFE2 metadata should use their exact inputs when both are present");

        let expected_lfe = (1.0 * 0.5 + 2.0 * 0.25) * super::LFE_SEND_MINUS_10_DB;
        assert!((output[3][0] - expected_lfe).abs() < 1e-6);
    }

    #[test]
    fn split_render_matches_unsplit_render_with_non_aligned_chunks() {
        fn test_oamd_payload() -> OamdPayload {
            let block0 = OamdObjectBlock {
                basic_info_status: 1,
                gain: Some(0.2),
                ..OamdObjectBlock::default()
            };
            let block1 = OamdObjectBlock {
                basic_info_status: 1,
                gain: Some(0.8),
                ..OamdObjectBlock::default()
            };
            OamdPayload {
                version: 0,
                object_count: 1,
                alternate_object_present: false,
                element_count: 1,
                beds: 0,
                bed_instances: 0,
                bed_or_isf_objects: 0,
                dynamic_objects: 1,
                isf_in_use: false,
                isf_index: None,
                bed_assignment: Vec::new(),
                elements: vec![OamdElement {
                    element_index: 0,
                    byte_length: 0,
                    kind: OamdElementKind::Object(OamdObjectElement {
                        sample_offset: 0,
                        block_updates: vec![
                            OamdBlockUpdate {
                                offset: 0,
                                ramp_duration: 0,
                            },
                            OamdBlockUpdate {
                                offset: 64,
                                ramp_duration: 0,
                            },
                        ],
                        object_blocks: vec![vec![block0, block1]],
                    }),
                }],
            }
        }

        fn test_frame(samples: usize, oamd: Option<OamdPayload>) -> RenderInputFrame {
            RenderInputFrame {
                sample_rate: 48_000,
                bed_channels: Vec::new(),
                object_channels: vec![vec![1.0; samples]],
                metadata_updates: metadata_updates_from_oamd(oamd),
            }
        }

        let payload = test_oamd_payload();

        let mut unsplit_renderer = Renderer714::new();
        let unsplit = unsplit_renderer
            .push_frame(&test_frame(192, Some(payload.clone())))
            .expect("unsplit render should succeed");

        let mut split_renderer = Renderer714::new();
        let split_a = split_renderer
            .push_frame(&test_frame(65, Some(payload)))
            .expect("first split render should succeed");
        let split_b = split_renderer
            .push_frame(&test_frame(127, None))
            .expect("second split render should succeed");

        let split = concat_rendered_frames(&[split_a, split_b]);
        assert_channels_close(&split, &unsplit.channels);
    }

    #[test]
    fn split_render_matches_unsplit_render_when_object_ramp_spans_partial_timeslot() {
        fn test_oamd_payload() -> OamdPayload {
            let left = OamdObjectBlock {
                basic_info_status: 1,
                gain: Some(1.0),
                render_info_status: 1,
                position: Some(Vec3 {
                    x: 0.0,
                    y: 0.5,
                    z: 0.5,
                }),
                ..OamdObjectBlock::default()
            };
            let right = OamdObjectBlock {
                basic_info_status: 1,
                gain: Some(1.0),
                render_info_status: 1,
                position: Some(Vec3 {
                    x: 1.0,
                    y: 0.5,
                    z: 0.5,
                }),
                ..OamdObjectBlock::default()
            };
            OamdPayload {
                version: 0,
                object_count: 1,
                alternate_object_present: false,
                element_count: 1,
                beds: 0,
                bed_instances: 0,
                bed_or_isf_objects: 0,
                dynamic_objects: 1,
                isf_in_use: false,
                isf_index: None,
                bed_assignment: Vec::new(),
                elements: vec![OamdElement {
                    element_index: 0,
                    byte_length: 0,
                    kind: OamdElementKind::Object(OamdObjectElement {
                        sample_offset: 0,
                        block_updates: vec![
                            OamdBlockUpdate {
                                offset: 0,
                                ramp_duration: 0,
                            },
                            OamdBlockUpdate {
                                offset: 64,
                                ramp_duration: 128,
                            },
                        ],
                        object_blocks: vec![vec![left, right]],
                    }),
                }],
            }
        }

        fn test_frame(samples: usize, oamd: Option<OamdPayload>) -> RenderInputFrame {
            RenderInputFrame {
                sample_rate: 48_000,
                bed_channels: Vec::new(),
                object_channels: vec![vec![1.0; samples]],
                metadata_updates: metadata_updates_from_oamd(oamd),
            }
        }

        let payload = test_oamd_payload();

        let mut unsplit_renderer = Renderer714::new();
        let unsplit = unsplit_renderer
            .push_frame(&test_frame(192, Some(payload.clone())))
            .expect("unsplit render should succeed");

        let mut split_renderer = Renderer714::new();
        let split_a = split_renderer
            .push_frame(&test_frame(160, Some(payload)))
            .expect("first split render should succeed");
        let split_b = split_renderer
            .push_frame(&test_frame(32, None))
            .expect("second split render should succeed");

        let split = concat_rendered_frames(&[split_a, split_b]);
        assert_channels_close(&split, &unsplit.channels);
    }

    #[test]
    fn split_render_matches_unsplit_render_when_limiter_peak_arrives_late() {
        fn bed_payload() -> OamdPayload {
            OamdPayload {
                version: 0,
                object_count: 1,
                alternate_object_present: false,
                element_count: 0,
                beds: 1,
                bed_instances: 1,
                bed_or_isf_objects: 1,
                dynamic_objects: 0,
                isf_in_use: false,
                isf_index: None,
                bed_assignment: vec![vec![BedChannel::FrontLeft]],
                elements: Vec::new(),
            }
        }

        fn bed_frame(samples: Vec<f32>, oamd: Option<OamdPayload>) -> RenderInputFrame {
            RenderInputFrame {
                sample_rate: 48_000,
                bed_channels: vec![RenderInputChannel {
                    channel: BedChannel::FrontLeft,
                    samples,
                }],
                object_channels: Vec::new(),
                metadata_updates: metadata_updates_from_oamd(oamd),
            }
        }

        let mut samples = vec![0.4; 32];
        samples.extend(vec![2.0; 32]);

        let payload = bed_payload();

        let mut unsplit_renderer = Renderer714::new();
        let unsplit = unsplit_renderer
            .push_frame(&bed_frame(samples.clone(), Some(payload.clone())))
            .expect("unsplit render should succeed");

        let mut split_renderer = Renderer714::new();
        let split_a = split_renderer
            .push_frame(&bed_frame(samples[..32].to_vec(), Some(payload)))
            .expect("first split render should succeed");
        let split_b = split_renderer
            .push_frame(&bed_frame(samples[32..].to_vec(), None))
            .expect("second split render should succeed");

        assert_eq!(split_a.samples_per_channel(), 0);
        let split = concat_rendered_frames(&[split_a, split_b]);
        assert_channels_close(&split, &unsplit.channels);
    }
}
