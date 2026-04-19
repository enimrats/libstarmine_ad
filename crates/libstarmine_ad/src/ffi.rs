use std::ffi::c_char;
use std::ptr;
use std::slice;

use crate::joc::JocObjectDecoderState;
use crate::metadata::{BedChannel, MetadataParseState, ParsedEmdfPayloadData};
use crate::syncframe::{
    AccessUnitInfo, CoreDecodeState, ParseError, decode_core_pcm_frame_with_state,
    inspect_access_unit_with_metadata_state,
};
use crate::{
    Decoder, PushResult, RENDER_714_CHANNEL_ORDER, Render714Error, Render714Frame, Renderer714,
};

const STARMINE_AD_RENDER_714_CHANNELS: usize = 12;

#[repr(C)]
/// C ABI snapshot for one parsed access unit.
pub struct StarmineAdAccessUnitInfo {
    pub frame_size: u32,
    pub bitstream_id: u8,
    pub frame_type: u8,
    pub substreamid: u8,
    pub sample_rate: u32,
    pub num_blocks: u8,
    pub channel_mode: u8,
    pub channels: u8,
    pub lfe_on: u8,
    pub addbsi_present: u8,
    pub extension_type_a: u8,
    pub complexity_index_type_a: u8,
    pub emdf_block_count: u32,
    pub payload_count: u32,
    pub joc_payload_count: u32,
    pub oamd_payload_count: u32,
    pub has_first_emdf_sync_offset: u8,
    pub first_emdf_sync_offset: u32,
    pub frames_seen: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Stable C ABI enum for speaker / bed channel identifiers.
pub enum StarmineAdBedChannel {
    Unknown = -1,
    FrontLeft = 0,
    FrontRight = 1,
    Center = 2,
    LowFrequencyEffects = 3,
    SurroundLeft = 4,
    SurroundRight = 5,
    RearLeft = 6,
    RearRight = 7,
    TopFrontLeft = 8,
    TopFrontRight = 9,
    TopSurroundLeft = 10,
    TopSurroundRight = 11,
    TopRearLeft = 12,
    TopRearRight = 13,
    WideLeft = 14,
    WideRight = 15,
    LowFrequencyEffects2 = 16,
}

impl From<BedChannel> for StarmineAdBedChannel {
    fn from(channel: BedChannel) -> Self {
        match channel {
            BedChannel::FrontLeft => Self::FrontLeft,
            BedChannel::FrontRight => Self::FrontRight,
            BedChannel::Center => Self::Center,
            BedChannel::LowFrequencyEffects => Self::LowFrequencyEffects,
            BedChannel::SurroundLeft => Self::SurroundLeft,
            BedChannel::SurroundRight => Self::SurroundRight,
            BedChannel::RearLeft => Self::RearLeft,
            BedChannel::RearRight => Self::RearRight,
            BedChannel::TopFrontLeft => Self::TopFrontLeft,
            BedChannel::TopFrontRight => Self::TopFrontRight,
            BedChannel::TopSurroundLeft => Self::TopSurroundLeft,
            BedChannel::TopSurroundRight => Self::TopSurroundRight,
            BedChannel::TopRearLeft => Self::TopRearLeft,
            BedChannel::TopRearRight => Self::TopRearRight,
            BedChannel::WideLeft => Self::WideLeft,
            BedChannel::WideRight => Self::WideRight,
            BedChannel::LowFrequencyEffects2 => Self::LowFrequencyEffects2,
        }
    }
}

#[repr(C)]
/// Borrowed view of one rendered 7.1.4 PCM frame.
pub struct StarmineAdRender714Frame {
    pub has_frame: u8,
    pub sample_rate: u32,
    pub samples_per_channel: usize,
    pub channel_count: usize,
    pub channels: [*const f32; STARMINE_AD_RENDER_714_CHANNELS],
    pub channel_order: [StarmineAdBedChannel; STARMINE_AD_RENDER_714_CHANNELS],
}

impl StarmineAdRender714Frame {
    fn empty() -> Self {
        Self {
            has_frame: 0,
            sample_rate: 0,
            samples_per_channel: 0,
            channel_count: 0,
            channels: [ptr::null(); STARMINE_AD_RENDER_714_CHANNELS],
            channel_order: [StarmineAdBedChannel::Unknown; STARMINE_AD_RENDER_714_CHANNELS],
        }
    }
}

impl From<&Render714Frame> for StarmineAdRender714Frame {
    fn from(frame: &Render714Frame) -> Self {
        let mut result = Self::empty();
        let channel_count = frame.channels.len().min(STARMINE_AD_RENDER_714_CHANNELS);

        result.has_frame = 1;
        result.sample_rate = frame.sample_rate;
        result.samples_per_channel = frame.samples_per_channel();
        result.channel_count = channel_count;

        for (index, channel) in frame.channels.iter().take(channel_count).enumerate() {
            result.channels[index] = channel.as_ptr();
        }
        for (index, channel) in frame.channel_order.iter().take(channel_count).enumerate() {
            result.channel_order[index] = (*channel).into();
        }

        result
    }
}

impl StarmineAdRender714Frame {
    fn from_channel_storage(
        sample_rate: u32,
        samples_per_channel: usize,
        channels: &[Vec<f32>; STARMINE_AD_RENDER_714_CHANNELS],
    ) -> Self {
        let mut result = Self::empty();

        result.has_frame = 1;
        result.sample_rate = sample_rate;
        result.samples_per_channel = samples_per_channel;
        result.channel_count = STARMINE_AD_RENDER_714_CHANNELS;

        for (index, channel) in channels.iter().enumerate() {
            result.channels[index] = channel.as_ptr();
            result.channel_order[index] = RENDER_714_CHANNEL_ORDER[index].into();
        }

        result
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// C ABI status code returned by every exported function.
pub enum StarmineAdStatus {
    Ok = 0,
    NullPointer = -1,
    ShortPacket = -2,
    BadSyncword = -3,
    NotEac3 = -4,
    InvalidHeader = -5,
    TruncatedFrame = -6,
    TrailingData = -7,
    UnsupportedFeature = -8,
    MissingOamd = -9,
    OamdStateUninitialized = -10,
    ObjectCountMismatch = -11,
    UnsupportedSampleCount = -12,
    UnsupportedBedChannel = -13,
    SampleRateChanged = -14,
}

impl StarmineAdStatus {
    fn from_parse_error(error: ParseError) -> Self {
        match error {
            ParseError::ShortPacket => Self::ShortPacket,
            ParseError::BadSyncword => Self::BadSyncword,
            ParseError::NotEac3 => Self::NotEac3,
            ParseError::InvalidHeader(_) => Self::InvalidHeader,
            ParseError::TruncatedFrame { .. } => Self::TruncatedFrame,
            ParseError::TrailingData { .. } => Self::TrailingData,
            ParseError::UnsupportedFeature(_) => Self::UnsupportedFeature,
        }
    }

    fn from_render_error(error: Render714Error) -> Self {
        match error {
            Render714Error::MissingOamd => Self::MissingOamd,
            Render714Error::OamdStateUninitialized => Self::OamdStateUninitialized,
            Render714Error::ObjectCountMismatch { .. } => Self::ObjectCountMismatch,
            Render714Error::UnsupportedSampleCount(_) => Self::UnsupportedSampleCount,
            Render714Error::UnsupportedBedChannel(_) => Self::UnsupportedBedChannel,
            Render714Error::SampleRateChanged { .. } => Self::SampleRateChanged,
        }
    }
}

impl StarmineAdAccessUnitInfo {
    fn from_parts(info: &AccessUnitInfo, frames_seen: u64) -> Self {
        let AccessUnitInfo {
            frame_size,
            bitstream_id,
            frame_type,
            substreamid,
            sample_rate,
            num_blocks,
            channel_mode,
            channels,
            lfe_on,
            addbsi_present,
            extension_type_a,
            complexity_index_type_a,
            emdf_block_count,
            first_emdf_sync_offset,
            ..
        } = info;

        Self {
            frame_size: *frame_size as u32,
            bitstream_id: *bitstream_id,
            frame_type: match frame_type {
                crate::FrameType::Independent => 0,
                crate::FrameType::Dependent => 1,
                crate::FrameType::Ac3Convert => 2,
            },
            substreamid: *substreamid,
            sample_rate: *sample_rate,
            num_blocks: *num_blocks,
            channel_mode: *channel_mode,
            channels: *channels,
            lfe_on: if *lfe_on { 1 } else { 0 },
            addbsi_present: if *addbsi_present { 1 } else { 0 },
            extension_type_a: if *extension_type_a { 1 } else { 0 },
            complexity_index_type_a: *complexity_index_type_a,
            emdf_block_count: *emdf_block_count as u32,
            payload_count: info.payloads().count() as u32,
            joc_payload_count: info.joc_payload_count() as u32,
            oamd_payload_count: info.oamd_payload_count() as u32,
            has_first_emdf_sync_offset: if first_emdf_sync_offset.is_some() {
                1
            } else {
                0
            },
            first_emdf_sync_offset: first_emdf_sync_offset.unwrap_or_default() as u32,
            frames_seen,
        }
    }
}

impl From<&PushResult> for StarmineAdAccessUnitInfo {
    fn from(result: &PushResult) -> Self {
        Self::from_parts(&result.info, result.frames_seen)
    }
}

static STATUS_OK: &[u8] = b"ok\0";
static STATUS_NULL_POINTER: &[u8] = b"null-pointer\0";
static STATUS_SHORT_PACKET: &[u8] = b"short-packet\0";
static STATUS_BAD_SYNCWORD: &[u8] = b"bad-syncword\0";
static STATUS_NOT_EAC3: &[u8] = b"not-eac3\0";
static STATUS_INVALID_HEADER: &[u8] = b"invalid-header\0";
static STATUS_TRUNCATED_FRAME: &[u8] = b"truncated-frame\0";
static STATUS_TRAILING_DATA: &[u8] = b"trailing-data\0";
static STATUS_UNSUPPORTED_FEATURE: &[u8] = b"unsupported-feature\0";
static STATUS_MISSING_OAMD: &[u8] = b"missing-oamd\0";
static STATUS_OAMD_STATE_UNINITIALIZED: &[u8] = b"oamd-state-uninitialized\0";
static STATUS_OBJECT_COUNT_MISMATCH: &[u8] = b"object-count-mismatch\0";
static STATUS_UNSUPPORTED_SAMPLE_COUNT: &[u8] = b"unsupported-sample-count\0";
static STATUS_UNSUPPORTED_BED_CHANNEL: &[u8] = b"unsupported-bed-channel\0";
static STATUS_SAMPLE_RATE_CHANGED: &[u8] = b"sample-rate-changed\0";

#[derive(Debug)]
pub struct StarmineAdRenderer714Handle {
    frames_seen: u64,
    core_state: CoreDecodeState,
    joc_state: JocObjectDecoderState,
    metadata_state: MetadataParseState,
    renderer: Renderer714,
    object_channels: Vec<Vec<f32>>,
    last_rendered_has_frame: bool,
    last_rendered_sample_rate: u32,
    last_rendered_samples_per_channel: usize,
    last_rendered_channels: [Vec<f32>; STARMINE_AD_RENDER_714_CHANNELS],
}

impl Default for StarmineAdRenderer714Handle {
    fn default() -> Self {
        Self {
            frames_seen: 0,
            core_state: CoreDecodeState::default(),
            joc_state: JocObjectDecoderState::default(),
            metadata_state: MetadataParseState::default(),
            renderer: Renderer714::default(),
            object_channels: Vec::new(),
            last_rendered_has_frame: false,
            last_rendered_sample_rate: 0,
            last_rendered_samples_per_channel: 0,
            last_rendered_channels: std::array::from_fn(|_| Vec::new()),
        }
    }
}

impl StarmineAdRenderer714Handle {
    fn reset(&mut self) {
        self.frames_seen = 0;
        self.core_state.reset();
        self.joc_state.reset();
        self.metadata_state.reset();
        self.renderer.reset();
        self.object_channels.clear();
        self.clear_last_rendered();
    }

    fn clear_last_rendered(&mut self) {
        self.last_rendered_has_frame = false;
        self.last_rendered_sample_rate = 0;
        self.last_rendered_samples_per_channel = 0;
    }

    fn push_access_unit(&mut self, access_unit: &[u8]) -> Result<AccessUnitInfo, StarmineAdStatus> {
        self.clear_last_rendered();

        let info = inspect_access_unit_with_metadata_state(access_unit, &mut self.metadata_state)
            .map_err(StarmineAdStatus::from_parse_error)?;

        if access_unit.len() < info.frame_size {
            return Err(StarmineAdStatus::TruncatedFrame);
        }
        if access_unit.len() != info.frame_size {
            return Err(StarmineAdStatus::TrailingData);
        }

        let joc = info.payloads().find_map(|payload| match &payload.parsed {
            ParsedEmdfPayloadData::Joc(joc) => Some(joc),
            _ => None,
        });

        if let Some(joc) = joc {
            let core = decode_core_pcm_frame_with_state(access_unit, &info, &mut self.core_state)
                .map_err(StarmineAdStatus::from_parse_error)?;
            self.joc_state
                .decode_frame_into(&core, joc, &mut self.object_channels)
                .map_err(StarmineAdStatus::from_parse_error)?;
            let oamd = info.payloads().find_map(|payload| match &payload.parsed {
                ParsedEmdfPayloadData::Oamd(oamd) => Some(oamd),
                _ => None,
            });
            let oamd_sample_offset = info.payloads().find_map(|payload| match &payload.parsed {
                ParsedEmdfPayloadData::Oamd(_) => payload.info.sample_offset,
                _ => None,
            });

            self.renderer
                .render_into_channels(
                    &core,
                    &self.object_channels,
                    oamd,
                    oamd_sample_offset,
                    &mut self.last_rendered_channels,
                )
                .map_err(StarmineAdStatus::from_render_error)?;
            self.last_rendered_has_frame = true;
            self.last_rendered_sample_rate = core.sample_rate;
            self.last_rendered_samples_per_channel = core.samples_per_channel();
        }

        self.frames_seen += 1;
        Ok(info)
    }
}

#[unsafe(no_mangle)]
/// Create a new C ABI decoder handle.
pub extern "C" fn starmine_ad_decoder_new() -> *mut Decoder {
    Box::into_raw(Box::new(Decoder::new()))
}

#[unsafe(no_mangle)]
/// Destroy a decoder handle previously returned by [`starmine_ad_decoder_new`].
pub unsafe extern "C" fn starmine_ad_decoder_free(decoder: *mut Decoder) {
    if !decoder.is_null() {
        unsafe {
            drop(Box::from_raw(decoder));
        }
    }
}

#[unsafe(no_mangle)]
/// Reset a decoder handle after a seek or discontinuity.
pub unsafe extern "C" fn starmine_ad_decoder_reset(decoder: *mut Decoder) -> StarmineAdStatus {
    let Some(decoder) = (unsafe { decoder.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    decoder.reset();
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Parse one complete access unit through the C ABI.
pub unsafe extern "C" fn starmine_ad_decoder_push_access_unit(
    decoder: *mut Decoder,
    data: *const u8,
    len: usize,
    out_info: *mut StarmineAdAccessUnitInfo,
) -> StarmineAdStatus {
    let Some(decoder) = (unsafe { decoder.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    if data.is_null() {
        return StarmineAdStatus::NullPointer;
    }

    let access_unit = unsafe { slice::from_raw_parts(data, len) };
    match decoder.push_access_unit(access_unit) {
        Ok(result) => {
            if let Some(out_info) = unsafe { out_info.as_mut() } {
                *out_info = StarmineAdAccessUnitInfo::from(&result);
            }
            StarmineAdStatus::Ok
        }
        Err(error) => StarmineAdStatus::from_parse_error(error),
    }
}

#[unsafe(no_mangle)]
/// Create a stateful 7.1.4 renderer handle.
pub extern "C" fn starmine_ad_renderer_714_new() -> *mut StarmineAdRenderer714Handle {
    Box::into_raw(Box::new(StarmineAdRenderer714Handle::default()))
}

#[unsafe(no_mangle)]
/// Destroy a renderer handle created by [`starmine_ad_renderer_714_new`].
pub unsafe extern "C" fn starmine_ad_renderer_714_free(renderer: *mut StarmineAdRenderer714Handle) {
    if !renderer.is_null() {
        unsafe {
            drop(Box::from_raw(renderer));
        }
    }
}

#[unsafe(no_mangle)]
/// Reset a renderer handle after a seek or discontinuity.
pub unsafe extern "C" fn starmine_ad_renderer_714_reset(
    renderer: *mut StarmineAdRenderer714Handle,
) -> StarmineAdStatus {
    let Some(renderer) = (unsafe { renderer.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    renderer.reset();
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Decode one access unit and, when possible, render it to 7.1.4 float PCM.
pub unsafe extern "C" fn starmine_ad_renderer_714_push_access_unit(
    renderer: *mut StarmineAdRenderer714Handle,
    data: *const u8,
    len: usize,
    out_info: *mut StarmineAdAccessUnitInfo,
    out_frame: *mut StarmineAdRender714Frame,
) -> StarmineAdStatus {
    let Some(renderer) = (unsafe { renderer.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    if data.is_null() {
        return StarmineAdStatus::NullPointer;
    }

    if let Some(out_frame) = unsafe { out_frame.as_mut() } {
        *out_frame = StarmineAdRender714Frame::empty();
    }

    let access_unit = unsafe { slice::from_raw_parts(data, len) };
    match renderer.push_access_unit(access_unit) {
        Ok(info) => {
            if let Some(out_info) = unsafe { out_info.as_mut() } {
                *out_info = StarmineAdAccessUnitInfo::from_parts(&info, renderer.frames_seen);
            }
            if let Some(out_frame) = unsafe { out_frame.as_mut() } {
                if renderer.last_rendered_has_frame {
                    *out_frame = StarmineAdRender714Frame::from_channel_storage(
                        renderer.last_rendered_sample_rate,
                        renderer.last_rendered_samples_per_channel,
                        &renderer.last_rendered_channels,
                    );
                }
            }
            StarmineAdStatus::Ok
        }
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
/// Convert a status code to a stable C string.
pub extern "C" fn starmine_ad_status_string(status: StarmineAdStatus) -> *const c_char {
    match status {
        StarmineAdStatus::Ok => STATUS_OK.as_ptr(),
        StarmineAdStatus::NullPointer => STATUS_NULL_POINTER.as_ptr(),
        StarmineAdStatus::ShortPacket => STATUS_SHORT_PACKET.as_ptr(),
        StarmineAdStatus::BadSyncword => STATUS_BAD_SYNCWORD.as_ptr(),
        StarmineAdStatus::NotEac3 => STATUS_NOT_EAC3.as_ptr(),
        StarmineAdStatus::InvalidHeader => STATUS_INVALID_HEADER.as_ptr(),
        StarmineAdStatus::TruncatedFrame => STATUS_TRUNCATED_FRAME.as_ptr(),
        StarmineAdStatus::TrailingData => STATUS_TRAILING_DATA.as_ptr(),
        StarmineAdStatus::UnsupportedFeature => STATUS_UNSUPPORTED_FEATURE.as_ptr(),
        StarmineAdStatus::MissingOamd => STATUS_MISSING_OAMD.as_ptr(),
        StarmineAdStatus::OamdStateUninitialized => STATUS_OAMD_STATE_UNINITIALIZED.as_ptr(),
        StarmineAdStatus::ObjectCountMismatch => STATUS_OBJECT_COUNT_MISMATCH.as_ptr(),
        StarmineAdStatus::UnsupportedSampleCount => STATUS_UNSUPPORTED_SAMPLE_COUNT.as_ptr(),
        StarmineAdStatus::UnsupportedBedChannel => STATUS_UNSUPPORTED_BED_CHANNEL.as_ptr(),
        StarmineAdStatus::SampleRateChanged => STATUS_SAMPLE_RATE_CHANGED.as_ptr(),
    }
    .cast::<c_char>()
}

#[unsafe(no_mangle)]
/// Initialize an info struct to zero / empty defaults.
pub extern "C" fn starmine_ad_access_unit_info_init(
    out_info: *mut StarmineAdAccessUnitInfo,
) -> StarmineAdStatus {
    let Some(out_info) = ptr::NonNull::new(out_info) else {
        return StarmineAdStatus::NullPointer;
    };
    unsafe {
        out_info.as_ptr().write(StarmineAdAccessUnitInfo {
            frame_size: 0,
            bitstream_id: 0,
            frame_type: 0,
            substreamid: 0,
            sample_rate: 0,
            num_blocks: 0,
            channel_mode: 0,
            channels: 0,
            lfe_on: 0,
            addbsi_present: 0,
            extension_type_a: 0,
            complexity_index_type_a: 0,
            emdf_block_count: 0,
            payload_count: 0,
            joc_payload_count: 0,
            oamd_payload_count: 0,
            has_first_emdf_sync_offset: 0,
            first_emdf_sync_offset: 0,
            frames_seen: 0,
        });
    }
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Initialize a render-frame struct to the empty / no-output state.
pub extern "C" fn starmine_ad_render_714_frame_init(
    out_frame: *mut StarmineAdRender714Frame,
) -> StarmineAdStatus {
    let Some(out_frame) = ptr::NonNull::new(out_frame) else {
        return StarmineAdStatus::NullPointer;
    };
    unsafe {
        out_frame.as_ptr().write(StarmineAdRender714Frame::empty());
    }
    StarmineAdStatus::Ok
}
