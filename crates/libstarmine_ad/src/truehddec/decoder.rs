use crate::renderer::{
    BedChannel, ObjectAnchor, RenderMetadata, RenderMetadataBlockUpdate, RenderMetadataElement,
    RenderMetadataObject, RenderMetadataUpdate, Vec3,
};
use crate::truehddec::process::{decode::Decoder, parse::Parser};
use crate::truehddec::structs::channel::ChannelLabel;
use crate::truehddec::structs::oamd::{
    ExtendedObjectElement, GAIN_MINUS_INFINITY, ObjectAudioMetadataPayload, ObjectElement,
    ObjectInfoBlock,
};
use std::any::Any;
use std::fmt;
use thiserror::Error;

const PRESENTATION_INDEX_ATMOS: usize = 3;
const PCM_I24_SCALE: f32 = 1.0 / 8_388_608.0;
const BED_CHANNELS: [BedChannel; 17] = [
    BedChannel::FrontLeft,
    BedChannel::FrontRight,
    BedChannel::Center,
    BedChannel::LowFrequencyEffects,
    BedChannel::SurroundLeft,
    BedChannel::SurroundRight,
    BedChannel::RearLeft,
    BedChannel::RearRight,
    BedChannel::TopFrontLeft,
    BedChannel::TopFrontRight,
    BedChannel::TopSurroundLeft,
    BedChannel::TopSurroundRight,
    BedChannel::TopRearLeft,
    BedChannel::TopRearRight,
    BedChannel::WideLeft,
    BedChannel::WideRight,
    BedChannel::LowFrequencyEffects2,
];

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("truehd-{kind} {message}")]
pub struct TrueHdError {
    kind: TrueHdErrorKind,
    message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrueHdErrorKind {
    Parse,
    Decode,
    UnsupportedLayout,
    InvalidMetadata,
}

impl TrueHdErrorKind {
    const fn as_str(self) -> &'static str {
        match self {
            TrueHdErrorKind::Parse => "parse",
            TrueHdErrorKind::Decode => "decode",
            TrueHdErrorKind::UnsupportedLayout => "unsupported-layout",
            TrueHdErrorKind::InvalidMetadata => "invalid-metadata",
        }
    }
}

impl fmt::Display for TrueHdErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TrueHdError {
    pub(crate) fn parse(error: impl fmt::Display) -> Self {
        Self {
            kind: TrueHdErrorKind::Parse,
            message: error.to_string(),
        }
    }

    pub(crate) fn decode(error: impl fmt::Display) -> Self {
        Self {
            kind: TrueHdErrorKind::Decode,
            message: error.to_string(),
        }
    }

    fn unsupported_layout(message: impl Into<String>) -> Self {
        Self {
            kind: TrueHdErrorKind::UnsupportedLayout,
            message: message.into(),
        }
    }

    fn invalid_metadata(message: impl Into<String>) -> Self {
        Self {
            kind: TrueHdErrorKind::InvalidMetadata,
            message: message.into(),
        }
    }

    pub(crate) fn kind_name(&self) -> &'static str {
        self.kind.as_str()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectPcmFrame {
    pub sample_rate: u32,
    pub bed_channel_order: Vec<BedChannel>,
    pub bed_channels: Vec<Vec<f32>>,
    pub object_channels: Vec<Vec<f32>>,
    pub metadata_updates: Vec<RenderMetadataUpdate>,
}

impl ObjectPcmFrame {
    /// Number of samples carried by each decoded channel in this frame.
    pub fn samples_per_channel(&self) -> usize {
        self.bed_channels
            .first()
            .map(|channel| channel.len())
            .or_else(|| self.object_channels.first().map(Vec::len))
            .unwrap_or(0)
    }

    /// Number of decoded bed channels in this frame.
    pub fn bed_channel_count(&self) -> usize {
        self.bed_channels.len()
    }

    /// Number of decoded dynamic object channels in this frame.
    pub fn object_count(&self) -> usize {
        self.object_channels.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObjectPcmPushResult {
    pub access_units_seen: u64,
    pub frames_seen: u64,
    pub substream_info_changed: bool,
    pub pcm: ObjectPcmFrame,
}

#[derive(Debug, Clone, PartialEq)]
struct LayoutState {
    bed_channel_order: Vec<BedChannel>,
    dynamic_object_count: usize,
}

#[derive(Default)]
pub struct ObjectPcmDecoder {
    access_units_seen: u64,
    frames_seen: u64,
    parser: Parser,
    decoder: Decoder,
    layout: Option<LayoutState>,
}

impl ObjectPcmDecoder {
    /// Create a fresh TrueHD object decoder for presentation 3 streams.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all cross-access-unit decode state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Configure how strictly parser/decoder validation messages fail.
    pub fn set_fail_level(&mut self, level: log::Level) {
        self.parser.set_fail_level(level);
        self.decoder.set_fail_level(level);
    }

    /// Number of access units accepted since the last reset.
    pub fn access_units_seen(&self) -> u64 {
        self.access_units_seen
    }

    /// Number of decoded object frames emitted since the last reset.
    pub fn frames_seen(&self) -> u64 {
        self.frames_seen
    }

    /// Decode one complete TrueHD access unit into bed/object PCM.
    ///
    /// Returns `Ok(None)` when the access unit decodes successfully but does not expose
    /// any dynamic object channels for the current presentation.
    pub fn push_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<Option<ObjectPcmPushResult>, TrueHdError> {
        let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.parser.parse(access_unit)
        }))
        .map_err(|panic| TrueHdError::parse(format!("panic: {}", panic_message(panic))))?
        .map_err(TrueHdError::parse)?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.decoder
                .decode_presentation(&parsed, PRESENTATION_INDEX_ATMOS)
        }))
        .map_err(|panic| TrueHdError::decode(format!("panic: {}", panic_message(panic))))?
        .map_err(TrueHdError::decode)?;
        let cached_layout = self.layout.clone();
        let decoded = self.decoder.decoded_access_unit();

        self.access_units_seen += 1;

        let layout = Self::resolve_layout(
            cached_layout.as_ref(),
            decoded.channel_labels,
            decoded.channel_count,
            decoded.oamd,
        )?;
        let bed_channels = layout.bed_channel_order.len();
        if layout.dynamic_object_count == 0 {
            self.layout = Some(layout);
            return Ok(None);
        }
        if decoded.channel_count != bed_channels + layout.dynamic_object_count {
            return Err(TrueHdError::unsupported_layout(format!(
                "channel-count mismatch decoded={} bed={} dynamic={}",
                decoded.channel_count, bed_channels, layout.dynamic_object_count
            )));
        }

        let (bed_pcm, object_pcm) = deinterleave_bed_object_channels(
            decoded.pcm_data,
            decoded.sample_length,
            bed_channels,
            layout.dynamic_object_count,
        );
        let metadata_updates = decoded
            .oamd
            .iter()
            .map(render_metadata_update_from_oamd)
            .collect::<Result<Vec<_>, _>>()?;

        let pcm = ObjectPcmFrame {
            sample_rate: decoded.sampling_frequency,
            bed_channel_order: layout.bed_channel_order.clone(),
            bed_channels: bed_pcm,
            object_channels: object_pcm,
            metadata_updates,
        };

        self.layout = Some(layout);
        self.frames_seen += 1;
        Ok(Some(ObjectPcmPushResult {
            access_units_seen: self.access_units_seen,
            frames_seen: self.frames_seen,
            substream_info_changed: decoded.substream_info_changed,
            pcm,
        }))
    }

    fn resolve_layout(
        cached_layout: Option<&LayoutState>,
        channel_labels: &[ChannelLabel],
        channel_count: usize,
        oamd_payloads: &[ObjectAudioMetadataPayload],
    ) -> Result<LayoutState, TrueHdError> {
        if let Some(layout) = layout_from_oamd_payloads(oamd_payloads)? {
            return Ok(layout);
        }

        if let Some(layout) = layout_from_channel_labels(channel_labels, channel_count)? {
            return Ok(layout);
        }

        if let Some(layout) = cached_layout.cloned() {
            if layout.bed_channel_order.len() + layout.dynamic_object_count == channel_count {
                return Ok(layout);
            }
        }

        Err(TrueHdError::unsupported_layout(format!(
            "unable to resolve layout for channel_count={channel_count} labels={channel_labels:?}"
        )))
    }
}

fn layout_from_oamd_payloads(
    oamd_payloads: &[ObjectAudioMetadataPayload],
) -> Result<Option<LayoutState>, TrueHdError> {
    let Some(first) = oamd_payloads.first() else {
        return Ok(None);
    };

    let first_layout = layout_from_oamd(first)?;
    for payload in &oamd_payloads[1..] {
        let layout = layout_from_oamd(payload)?;
        if layout != first_layout {
            return Err(TrueHdError::unsupported_layout(
                "multiple OAMD payloads in one access unit disagree on bed/object layout",
            ));
        }
    }

    Ok(Some(first_layout))
}

fn layout_from_oamd(oamd: &ObjectAudioMetadataPayload) -> Result<LayoutState, TrueHdError> {
    if oamd.program_assignment.num_isf_objects != 0 {
        return Err(TrueHdError::unsupported_layout(format!(
            "ISF objects are not supported (count={})",
            oamd.program_assignment.num_isf_objects
        )));
    }

    let bed_channel_order = oamd
        .program_assignment
        .bed_assignment
        .iter()
        .flat_map(|bed| bed.to_index_vec())
        .map(|index| {
            BED_CHANNELS.get(index).copied().ok_or_else(|| {
                TrueHdError::unsupported_layout(format!("unsupported bed channel index {index}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(LayoutState {
        bed_channel_order,
        dynamic_object_count: oamd.program_assignment.num_dynamic_objects,
    })
}

fn layout_from_channel_labels(
    channel_labels: &[ChannelLabel],
    channel_count: usize,
) -> Result<Option<LayoutState>, TrueHdError> {
    if channel_labels.is_empty() || channel_labels.len() > channel_count {
        return Ok(None);
    }

    let bed_channel_order = channel_labels
        .iter()
        .copied()
        .map(|label| {
            bed_channel_from_label(label).ok_or_else(|| {
                TrueHdError::unsupported_layout(format!("unsupported TrueHD bed label {label:?}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let dynamic_object_count = channel_count.saturating_sub(bed_channel_order.len());
    Ok(Some(LayoutState {
        bed_channel_order,
        dynamic_object_count,
    }))
}

fn bed_channel_from_label(label: ChannelLabel) -> Option<BedChannel> {
    match label {
        ChannelLabel::L => Some(BedChannel::FrontLeft),
        ChannelLabel::R => Some(BedChannel::FrontRight),
        ChannelLabel::C => Some(BedChannel::Center),
        ChannelLabel::LFE => Some(BedChannel::LowFrequencyEffects),
        ChannelLabel::Ls => Some(BedChannel::SurroundLeft),
        ChannelLabel::Rs => Some(BedChannel::SurroundRight),
        ChannelLabel::Lb => Some(BedChannel::RearLeft),
        ChannelLabel::Rb => Some(BedChannel::RearRight),
        ChannelLabel::Tfl => Some(BedChannel::TopFrontLeft),
        ChannelLabel::Tfr => Some(BedChannel::TopFrontRight),
        ChannelLabel::Tsl => Some(BedChannel::TopSurroundLeft),
        ChannelLabel::Tsr => Some(BedChannel::TopSurroundRight),
        ChannelLabel::Tbl => Some(BedChannel::TopRearLeft),
        ChannelLabel::Tbr => Some(BedChannel::TopRearRight),
        ChannelLabel::Lw => Some(BedChannel::WideLeft),
        ChannelLabel::Rw => Some(BedChannel::WideRight),
        ChannelLabel::LFE2 => Some(BedChannel::LowFrequencyEffects2),
        _ => None,
    }
}

fn deinterleave_bed_object_channels(
    pcm_data: &[[i32; 16]; 160],
    sample_length: usize,
    bed_channel_count: usize,
    object_count: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut bed_channels = vec![vec![0.0; sample_length]; bed_channel_count];
    let mut object_channels = vec![vec![0.0; sample_length]; object_count];

    for (sample_index, row) in pcm_data.iter().take(sample_length).enumerate() {
        for channel_index in 0..bed_channel_count {
            bed_channels[channel_index][sample_index] = row[channel_index] as f32 * PCM_I24_SCALE;
        }
        for object_index in 0..object_count {
            object_channels[object_index][sample_index] =
                row[bed_channel_count + object_index] as f32 * PCM_I24_SCALE;
        }
    }

    (bed_channels, object_channels)
}

fn render_metadata_update_from_oamd(
    oamd: &ObjectAudioMetadataPayload,
) -> Result<RenderMetadataUpdate, TrueHdError> {
    let sample_offset = oamd
        .object_element
        .as_ref()
        .map(|element| element.md_update_info.sample_offset)
        .unwrap_or_default()
        + usize::try_from(oamd.evo_sample_offset).map_err(|_| {
            TrueHdError::invalid_metadata(format!(
                "evo sample offset does not fit usize: {}",
                oamd.evo_sample_offset
            ))
        })?;

    let sample_offset = u16::try_from(sample_offset).map_err(|_| {
        TrueHdError::invalid_metadata(format!("sample offset out of range: {sample_offset}"))
    })?;

    Ok(RenderMetadataUpdate {
        sample_offset,
        metadata: render_metadata_from_oamd(oamd)?,
    })
}

fn render_metadata_from_oamd(
    oamd: &ObjectAudioMetadataPayload,
) -> Result<RenderMetadata, TrueHdError> {
    if oamd.program_assignment.num_isf_objects != 0 {
        return Err(TrueHdError::unsupported_layout(format!(
            "ISF objects are not supported (count={})",
            oamd.program_assignment.num_isf_objects
        )));
    }

    let bed_channels = oamd
        .program_assignment
        .bed_assignment
        .iter()
        .flat_map(|bed| bed.to_index_vec())
        .map(|index| {
            BED_CHANNELS.get(index).copied().ok_or_else(|| {
                TrueHdError::unsupported_layout(format!("unsupported bed channel index {index}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let elements = if let Some(object_element) = &oamd.object_element {
        vec![render_metadata_element_from_oamd(oamd, object_element)?]
    } else {
        Vec::new()
    };

    Ok(RenderMetadata {
        object_count: oamd.object_count,
        bed_or_isf_objects: oamd.program_assignment.beds_or_isf_count(),
        bed_channels,
        elements,
    })
}

fn render_metadata_element_from_oamd(
    oamd: &ObjectAudioMetadataPayload,
    object_element: &ObjectElement,
) -> Result<RenderMetadataElement, TrueHdError> {
    if object_element.md_update_info.num_obj_info_blocks > 1 {
        return Err(TrueHdError::invalid_metadata(format!(
            "multi-block OAMD is not supported yet (blocks={})",
            object_element.md_update_info.num_obj_info_blocks
        )));
    }

    let block_updates = object_element
        .md_update_info
        .block_update_info
        .iter()
        .map(|update| {
            // TODO: A zero intra-element offset is less wrong than reinterpreting
            // block_offset_factor_bits as a sample offset.
            RenderMetadataBlockUpdate {
                offset: 0,
                ramp_duration: i64::from(update.ramp_duration),
            }
        })
        .collect();

    let object_blocks = object_element
        .object_data
        .iter()
        .enumerate()
        .map(|(object_index, blocks)| {
            blocks
                .iter()
                .enumerate()
                .map(|(block_index, block)| {
                    render_metadata_object_from_block(oamd, object_index, block_index, block)
                })
                .collect()
        })
        .collect();

    Ok(RenderMetadataElement {
        block_updates,
        object_blocks,
    })
}

fn render_metadata_object_from_block(
    oamd: &ObjectAudioMetadataPayload,
    object_index: usize,
    block_index: usize,
    block: &ObjectInfoBlock,
) -> RenderMetadataObject {
    let gain = match block.object_basic_info.object_gain {
        GAIN_MINUS_INFINITY => Some(0.0),
        value => Some(db_to_gain(value as f32)),
    };

    let anchor = if block.b_object_in_bed_or_isf {
        ObjectAnchor::Speaker
    } else if block.object_render_info.b_object_use_screen_ref {
        ObjectAnchor::Screen
    } else {
        ObjectAnchor::Room
    };

    let position = if block.b_object_not_active || block.b_object_in_bed_or_isf {
        None
    } else {
        Some(position_from_block(
            oamd.extended_object_element.as_ref(),
            object_index,
            block_index,
            block,
        ))
    };

    let distance = if !block.object_render_info.b_object_distance_specified {
        None
    } else if block.object_render_info.b_object_at_infinity {
        Some(f32::INFINITY)
    } else {
        // TODO: finite distance
        None
    };

    RenderMetadataObject {
        gain,
        anchor,
        position_valid: !block.b_object_not_active && !block.b_object_in_bed_or_isf,
        differential_position: block.object_render_info.b_differential_position_specified,
        position,
        distance,
        size: Some(block.object_render_info.object_size[0] as f32),
        screen_factor: block.object_render_info.screen_factor as f32,
        depth_factor: block.object_render_info.depth_factor as f32,
    }
}

fn position_from_block(
    extended_object_element: Option<&ExtendedObjectElement>,
    object_index: usize,
    block_index: usize,
    block: &ObjectInfoBlock,
) -> Vec3 {
    let mut position = block.object_render_info.pos3d;

    if let Some(extended_object_element) = extended_object_element
        && let Some(object_blocks) = extended_object_element.ext_prec_pos_block.get(object_index)
        && let Some(extended) = object_blocks.get(block_index)
    {
        position[0] += extended.ext_prec_pos3d_x;
        position[1] += extended.ext_prec_pos3d_y;
        position[2] += extended.ext_prec_pos3d_z;
    }

    Vec3 {
        x: position[0].clamp(0.0, 1.0) as f32,
        y: position[1].clamp(0.0, 1.0) as f32,
        z: position[2].clamp(-1.0, 1.0) as f32,
    }
}

fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

fn panic_message(panic: Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::bed_channel_from_label;
    use crate::renderer::BedChannel;
    use crate::truehddec::structs::channel::ChannelLabel;

    #[test]
    fn truehd_label_map_matches_renderer_channels() {
        assert_eq!(
            bed_channel_from_label(ChannelLabel::L),
            Some(BedChannel::FrontLeft)
        );
        assert_eq!(
            bed_channel_from_label(ChannelLabel::R),
            Some(BedChannel::FrontRight)
        );
        assert_eq!(
            bed_channel_from_label(ChannelLabel::LFE2),
            Some(BedChannel::LowFrequencyEffects2)
        );
        assert_eq!(bed_channel_from_label(ChannelLabel::Cb), None);
    }
}
