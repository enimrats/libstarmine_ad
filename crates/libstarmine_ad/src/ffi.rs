use std::ffi::c_char;
use std::ptr;
use std::slice;

use crate::eac3dec::{
    AccessUnitInfo, CoreDecodeState, CorePcmFrame, Decoder as Eac3Decoder, FrameType,
    JocObjectDecoderState, MetadataParseState, ParseError, ParsedEmdfPayloadData, PushResult,
    decode_core_pcm_frame_with_state_into, inspect_access_unit_with_metadata_state,
    render_input_from_eac3_parts,
};
use crate::renderer::{BedChannel, Render714Error, Render714Frame, Renderer714};
use crate::truehddec::{
    ObjectPcmDecoder as TrueHdDecoder, ObjectPcmFrame as TrueHdObjectPcmFrame,
    ObjectPcmPushResult as TrueHdObjectPcmPushResult, TrueHdError,
};

const STARMINE_AD_RENDER_714_CHANNELS: usize = 12;

#[repr(C)]
/// C ABI snapshot for one parsed E-AC-3 access unit.
pub struct StarmineAdEac3AccessUnitInfo {
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
/// C ABI snapshot for one decoded TrueHD access unit.
pub struct StarmineAdTrueHdAccessUnitInfo {
    pub has_frame: u8,
    pub substream_info_changed: u8,
    pub sample_rate: u32,
    pub samples_per_channel: u32,
    pub bed_channel_count: u32,
    pub object_count: u32,
    pub metadata_update_count: u32,
    pub access_units_seen: u64,
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
/// Borrowed view of one decoded object-PCM frame.
pub struct StarmineAdObjectPcmFrame {
    pub has_frame: u8,
    pub sample_rate: u32,
    pub samples_per_channel: usize,
    pub bed_channel_count: usize,
    pub object_count: usize,
    pub bed_channel_order: *const StarmineAdBedChannel,
    pub bed_channels: *const *const f32,
    pub object_channels: *const *const f32,
}

impl StarmineAdObjectPcmFrame {
    fn empty() -> Self {
        Self {
            has_frame: 0,
            sample_rate: 0,
            samples_per_channel: 0,
            bed_channel_count: 0,
            object_count: 0,
            bed_channel_order: ptr::null(),
            bed_channels: ptr::null(),
            object_channels: ptr::null(),
        }
    }

    fn from_handle(handle: &StarmineAdTrueHdDecoderHandle) -> Self {
        let Some(frame) = handle.last_pcm.as_ref() else {
            return Self::empty();
        };

        Self {
            has_frame: 1,
            sample_rate: frame.sample_rate,
            samples_per_channel: frame.samples_per_channel(),
            bed_channel_count: handle.last_bed_channel_order.len(),
            object_count: handle.last_object_channel_ptrs.len(),
            bed_channel_order: slice_ptr(&handle.last_bed_channel_order),
            bed_channels: slice_ptr(&handle.last_bed_channel_ptrs),
            object_channels: slice_ptr(&handle.last_object_channel_ptrs),
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
    BedChannelCountMismatch = -15,
    TrueHdParse = -16,
    TrueHdDecode = -17,
    TrueHdUnsupportedLayout = -18,
    TrueHdInvalidMetadata = -19,
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
            Render714Error::BedChannelCountMismatch { .. } => Self::BedChannelCountMismatch,
            Render714Error::UnsupportedSampleCount(_) => Self::UnsupportedSampleCount,
            Render714Error::UnsupportedBedChannel(_) => Self::UnsupportedBedChannel,
            Render714Error::SampleRateChanged { .. } => Self::SampleRateChanged,
        }
    }

    fn from_truehd_error(error: &TrueHdError) -> Self {
        match error.kind_name() {
            "extract" | "parse" => Self::TrueHdParse,
            "decode" => Self::TrueHdDecode,
            "unsupported-layout" => Self::TrueHdUnsupportedLayout,
            "invalid-metadata" => Self::TrueHdInvalidMetadata,
            _ => Self::TrueHdParse,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StarmineAdStatus, starmine_ad_status_string};
    use crate::renderer::Render714Error;
    use std::ffi::CStr;

    #[test]
    fn bed_channel_count_mismatch_uses_bed_status() {
        let status = StarmineAdStatus::from_render_error(Render714Error::BedChannelCountMismatch {
            expected: 2,
            provided: 1,
        });
        assert_eq!(status, StarmineAdStatus::BedChannelCountMismatch);
    }

    #[test]
    fn bed_channel_count_mismatch_status_string_is_stable() {
        let ptr = starmine_ad_status_string(StarmineAdStatus::BedChannelCountMismatch);
        let value = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("valid utf-8");
        assert_eq!(value, "bed-channel-count-mismatch");
    }
}

impl StarmineAdEac3AccessUnitInfo {
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
                FrameType::Independent => 0,
                FrameType::Dependent => 1,
                FrameType::Ac3Convert => 2,
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

impl From<&PushResult> for StarmineAdEac3AccessUnitInfo {
    fn from(result: &PushResult) -> Self {
        Self::from_parts(&result.info, result.frames_seen)
    }
}

impl StarmineAdTrueHdAccessUnitInfo {
    fn empty() -> Self {
        Self {
            has_frame: 0,
            substream_info_changed: 0,
            sample_rate: 0,
            samples_per_channel: 0,
            bed_channel_count: 0,
            object_count: 0,
            metadata_update_count: 0,
            access_units_seen: 0,
            frames_seen: 0,
        }
    }

    fn from_result(result: &TrueHdObjectPcmPushResult) -> Self {
        Self {
            has_frame: 1,
            substream_info_changed: if result.substream_info_changed { 1 } else { 0 },
            sample_rate: result.pcm.sample_rate,
            samples_per_channel: result.pcm.samples_per_channel() as u32,
            bed_channel_count: result.pcm.bed_channel_count() as u32,
            object_count: result.pcm.object_count() as u32,
            metadata_update_count: result.pcm.metadata_updates.len() as u32,
            access_units_seen: result.access_units_seen,
            frames_seen: result.frames_seen,
        }
    }

    fn from_decoder_no_frame(decoder: &TrueHdDecoder) -> Self {
        Self {
            access_units_seen: decoder.access_units_seen(),
            frames_seen: decoder.frames_seen(),
            ..Self::empty()
        }
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
static STATUS_BED_CHANNEL_COUNT_MISMATCH: &[u8] = b"bed-channel-count-mismatch\0";
static STATUS_TRUEHD_PARSE: &[u8] = b"truehd-parse\0";
static STATUS_TRUEHD_DECODE: &[u8] = b"truehd-decode\0";
static STATUS_TRUEHD_UNSUPPORTED_LAYOUT: &[u8] = b"truehd-unsupported-layout\0";
static STATUS_TRUEHD_INVALID_METADATA: &[u8] = b"truehd-invalid-metadata\0";

#[derive(Debug)]
pub struct StarmineAdEac3Renderer714Handle {
    frames_seen: u64,
    core_state: CoreDecodeState,
    core_pcm: CorePcmFrame,
    joc_state: JocObjectDecoderState,
    metadata_state: MetadataParseState,
    renderer: Renderer714,
    object_channels: Vec<Vec<f32>>,
    last_rendered: Option<Render714Frame>,
}

impl Default for StarmineAdEac3Renderer714Handle {
    fn default() -> Self {
        Self {
            frames_seen: 0,
            core_state: CoreDecodeState::default(),
            core_pcm: CorePcmFrame {
                sample_rate: 0,
                fullband_channel_order: Vec::new(),
                fullband_channels: Vec::new(),
                lfe_channel: None,
            },
            joc_state: JocObjectDecoderState::default(),
            metadata_state: MetadataParseState::default(),
            renderer: Renderer714::default(),
            object_channels: Vec::new(),
            last_rendered: None,
        }
    }
}

impl StarmineAdEac3Renderer714Handle {
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
        self.last_rendered = None;
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
            decode_core_pcm_frame_with_state_into(
                access_unit,
                &info,
                &mut self.core_state,
                &mut self.core_pcm,
            )
            .map_err(StarmineAdStatus::from_parse_error)?;
            self.joc_state
                .decode_frame_into(&self.core_pcm, joc, &mut self.object_channels)
                .map_err(StarmineAdStatus::from_parse_error)?;
            let oamd_payloads = info
                .payloads()
                .filter_map(|payload| match &payload.parsed {
                    ParsedEmdfPayloadData::Oamd(oamd) => Some((oamd, payload.info.sample_offset)),
                    _ => None,
                })
                .collect::<Vec<_>>();

            let input =
                render_input_from_eac3_parts(&self.core_pcm, &self.object_channels, &oamd_payloads);
            self.last_rendered = Some(
                self.renderer
                    .push_frame(&input)
                    .map_err(StarmineAdStatus::from_render_error)?,
            );
        }

        self.frames_seen += 1;
        Ok(info)
    }

    fn flush(&mut self) {
        self.clear_last_rendered();
        self.last_rendered = self.renderer.flush();
    }
}

#[derive(Default)]
pub struct StarmineAdTrueHdDecoderHandle {
    decoder: TrueHdDecoder,
    last_pcm: Option<TrueHdObjectPcmFrame>,
    last_bed_channel_order: Vec<StarmineAdBedChannel>,
    last_bed_channel_ptrs: Vec<*const f32>,
    last_object_channel_ptrs: Vec<*const f32>,
}

impl StarmineAdTrueHdDecoderHandle {
    fn reset(&mut self) {
        self.decoder.reset();
        self.clear_last_pcm();
    }

    fn clear_last_pcm(&mut self) {
        self.last_pcm = None;
        self.last_bed_channel_order.clear();
        self.last_bed_channel_ptrs.clear();
        self.last_object_channel_ptrs.clear();
    }

    fn set_last_pcm(&mut self, pcm: TrueHdObjectPcmFrame) {
        self.last_pcm = Some(pcm);
        self.last_bed_channel_order.clear();
        self.last_bed_channel_ptrs.clear();
        self.last_object_channel_ptrs.clear();

        if let Some(frame) = self.last_pcm.as_ref() {
            self.last_bed_channel_order.extend(
                frame
                    .bed_channel_order
                    .iter()
                    .copied()
                    .map(StarmineAdBedChannel::from),
            );
            self.last_bed_channel_ptrs
                .extend(frame.bed_channels.iter().map(|channel| channel.as_ptr()));
            self.last_object_channel_ptrs
                .extend(frame.object_channels.iter().map(|channel| channel.as_ptr()));
        }
    }

    fn push_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<StarmineAdTrueHdAccessUnitInfo, StarmineAdStatus> {
        self.clear_last_pcm();
        validate_truehd_access_unit(access_unit)?;

        match self.decoder.push_access_unit(access_unit) {
            Ok(Some(result)) => {
                let info = StarmineAdTrueHdAccessUnitInfo::from_result(&result);
                self.set_last_pcm(result.pcm);
                Ok(info)
            }
            Ok(None) => Ok(StarmineAdTrueHdAccessUnitInfo::from_decoder_no_frame(
                &self.decoder,
            )),
            Err(error) => Err(StarmineAdStatus::from_truehd_error(&error)),
        }
    }
}

#[derive(Default)]
pub struct StarmineAdTrueHdRenderer714Handle {
    decoder: TrueHdDecoder,
    renderer: Renderer714,
    last_rendered: Option<Render714Frame>,
}

impl StarmineAdTrueHdRenderer714Handle {
    fn reset(&mut self) {
        self.decoder.reset();
        self.renderer.reset();
        self.clear_last_rendered();
    }

    fn clear_last_rendered(&mut self) {
        self.last_rendered = None;
    }

    fn push_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<StarmineAdTrueHdAccessUnitInfo, StarmineAdStatus> {
        self.clear_last_rendered();
        validate_truehd_access_unit(access_unit)?;

        match self.decoder.push_access_unit(access_unit) {
            Ok(Some(result)) => {
                let info = StarmineAdTrueHdAccessUnitInfo::from_result(&result);
                let input = result.pcm.into_render_input();
                self.last_rendered = Some(
                    self.renderer
                        .push_frame(&input)
                        .map_err(StarmineAdStatus::from_render_error)?,
                );
                Ok(info)
            }
            Ok(None) => Ok(StarmineAdTrueHdAccessUnitInfo::from_decoder_no_frame(
                &self.decoder,
            )),
            Err(error) => Err(StarmineAdStatus::from_truehd_error(&error)),
        }
    }

    fn flush(&mut self) {
        self.clear_last_rendered();
        self.last_rendered = self.renderer.flush();
    }
}

fn slice_ptr<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        ptr::null()
    } else {
        slice.as_ptr()
    }
}

fn validate_truehd_access_unit(access_unit: &[u8]) -> Result<(), StarmineAdStatus> {
    if access_unit.len() < 2 {
        return Err(StarmineAdStatus::ShortPacket);
    }

    let expected = ((u16::from_be_bytes([access_unit[0], access_unit[1]]) & 0x0FFF) << 1) as usize;
    if access_unit.len() < expected {
        return Err(StarmineAdStatus::TruncatedFrame);
    }
    if access_unit.len() != expected {
        return Err(StarmineAdStatus::TrailingData);
    }

    Ok(())
}

#[unsafe(no_mangle)]
/// Create a new E-AC-3 decoder handle.
pub extern "C" fn starmine_ad_eac3_decoder_new() -> *mut Eac3Decoder {
    Box::into_raw(Box::new(Eac3Decoder::new()))
}

#[unsafe(no_mangle)]
/// Destroy a decoder handle previously returned by [`starmine_ad_eac3_decoder_new`].
pub unsafe extern "C" fn starmine_ad_eac3_decoder_free(decoder: *mut Eac3Decoder) {
    if !decoder.is_null() {
        unsafe {
            drop(Box::from_raw(decoder));
        }
    }
}

#[unsafe(no_mangle)]
/// Reset an E-AC-3 decoder handle after a seek or discontinuity.
pub unsafe extern "C" fn starmine_ad_eac3_decoder_reset(
    decoder: *mut Eac3Decoder,
) -> StarmineAdStatus {
    let Some(decoder) = (unsafe { decoder.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    decoder.reset();
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Parse one complete E-AC-3 access unit through the C ABI.
pub unsafe extern "C" fn starmine_ad_eac3_decoder_push_access_unit(
    decoder: *mut Eac3Decoder,
    data: *const u8,
    len: usize,
    out_info: *mut StarmineAdEac3AccessUnitInfo,
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
                *out_info = StarmineAdEac3AccessUnitInfo::from(&result);
            }
            StarmineAdStatus::Ok
        }
        Err(error) => StarmineAdStatus::from_parse_error(error),
    }
}

#[unsafe(no_mangle)]
/// Create a stateful E-AC-3 7.1.4 renderer handle.
pub extern "C" fn starmine_ad_eac3_renderer_714_new() -> *mut StarmineAdEac3Renderer714Handle {
    Box::into_raw(Box::new(StarmineAdEac3Renderer714Handle::default()))
}

#[unsafe(no_mangle)]
/// Destroy a renderer handle created by [`starmine_ad_eac3_renderer_714_new`].
pub unsafe extern "C" fn starmine_ad_eac3_renderer_714_free(
    renderer: *mut StarmineAdEac3Renderer714Handle,
) {
    if !renderer.is_null() {
        unsafe {
            drop(Box::from_raw(renderer));
        }
    }
}

#[unsafe(no_mangle)]
/// Reset an E-AC-3 renderer handle after a seek or discontinuity.
pub unsafe extern "C" fn starmine_ad_eac3_renderer_714_reset(
    renderer: *mut StarmineAdEac3Renderer714Handle,
) -> StarmineAdStatus {
    let Some(renderer) = (unsafe { renderer.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    renderer.reset();
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Decode one E-AC-3 access unit and, when possible, render it to 7.1.4 float PCM.
pub unsafe extern "C" fn starmine_ad_eac3_renderer_714_push_access_unit(
    renderer: *mut StarmineAdEac3Renderer714Handle,
    data: *const u8,
    len: usize,
    out_info: *mut StarmineAdEac3AccessUnitInfo,
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
                *out_info = StarmineAdEac3AccessUnitInfo::from_parts(&info, renderer.frames_seen);
            }
            if let Some(out_frame) = unsafe { out_frame.as_mut() }
                && let Some(frame) = renderer.last_rendered.as_ref()
            {
                *out_frame = StarmineAdRender714Frame::from(frame);
            }
            StarmineAdStatus::Ok
        }
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
/// Emit the final short limiter block after the last E-AC-3 input frame.
pub unsafe extern "C" fn starmine_ad_eac3_renderer_714_flush(
    renderer: *mut StarmineAdEac3Renderer714Handle,
    out_frame: *mut StarmineAdRender714Frame,
) -> StarmineAdStatus {
    let Some(renderer) = (unsafe { renderer.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    let Some(out_frame) = (unsafe { out_frame.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };

    *out_frame = StarmineAdRender714Frame::empty();
    renderer.flush();
    if let Some(frame) = renderer.last_rendered.as_ref() {
        *out_frame = StarmineAdRender714Frame::from(frame);
    }
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Create a new TrueHD object decoder handle.
pub extern "C" fn starmine_ad_truehd_decoder_new() -> *mut StarmineAdTrueHdDecoderHandle {
    Box::into_raw(Box::new(StarmineAdTrueHdDecoderHandle::default()))
}

#[unsafe(no_mangle)]
/// Destroy a decoder handle created by [`starmine_ad_truehd_decoder_new`].
pub unsafe extern "C" fn starmine_ad_truehd_decoder_free(
    decoder: *mut StarmineAdTrueHdDecoderHandle,
) {
    if !decoder.is_null() {
        unsafe {
            drop(Box::from_raw(decoder));
        }
    }
}

#[unsafe(no_mangle)]
/// Reset a TrueHD decoder handle after a seek or discontinuity.
pub unsafe extern "C" fn starmine_ad_truehd_decoder_reset(
    decoder: *mut StarmineAdTrueHdDecoderHandle,
) -> StarmineAdStatus {
    let Some(decoder) = (unsafe { decoder.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    decoder.reset();
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Decode one complete TrueHD access unit into bed / object PCM.
pub unsafe extern "C" fn starmine_ad_truehd_decoder_push_access_unit(
    decoder: *mut StarmineAdTrueHdDecoderHandle,
    data: *const u8,
    len: usize,
    out_info: *mut StarmineAdTrueHdAccessUnitInfo,
    out_frame: *mut StarmineAdObjectPcmFrame,
) -> StarmineAdStatus {
    let Some(decoder) = (unsafe { decoder.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    if data.is_null() {
        return StarmineAdStatus::NullPointer;
    }

    if let Some(out_frame) = unsafe { out_frame.as_mut() } {
        *out_frame = StarmineAdObjectPcmFrame::empty();
    }

    let access_unit = unsafe { slice::from_raw_parts(data, len) };
    match decoder.push_access_unit(access_unit) {
        Ok(info) => {
            if let Some(out_info) = unsafe { out_info.as_mut() } {
                *out_info = info;
            }
            if let Some(out_frame) = unsafe { out_frame.as_mut() }
                && decoder.last_pcm.is_some()
            {
                *out_frame = StarmineAdObjectPcmFrame::from_handle(decoder);
            }
            StarmineAdStatus::Ok
        }
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
/// Create a stateful TrueHD 7.1.4 renderer handle.
pub extern "C" fn starmine_ad_truehd_renderer_714_new() -> *mut StarmineAdTrueHdRenderer714Handle {
    Box::into_raw(Box::new(StarmineAdTrueHdRenderer714Handle::default()))
}

#[unsafe(no_mangle)]
/// Destroy a renderer handle created by [`starmine_ad_truehd_renderer_714_new`].
pub unsafe extern "C" fn starmine_ad_truehd_renderer_714_free(
    renderer: *mut StarmineAdTrueHdRenderer714Handle,
) {
    if !renderer.is_null() {
        unsafe {
            drop(Box::from_raw(renderer));
        }
    }
}

#[unsafe(no_mangle)]
/// Reset a TrueHD renderer handle after a seek or discontinuity.
pub unsafe extern "C" fn starmine_ad_truehd_renderer_714_reset(
    renderer: *mut StarmineAdTrueHdRenderer714Handle,
) -> StarmineAdStatus {
    let Some(renderer) = (unsafe { renderer.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    renderer.reset();
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Decode one TrueHD access unit and, when possible, render it to 7.1.4 float PCM.
pub unsafe extern "C" fn starmine_ad_truehd_renderer_714_push_access_unit(
    renderer: *mut StarmineAdTrueHdRenderer714Handle,
    data: *const u8,
    len: usize,
    out_info: *mut StarmineAdTrueHdAccessUnitInfo,
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
                *out_info = info;
            }
            if let Some(out_frame) = unsafe { out_frame.as_mut() }
                && let Some(frame) = renderer.last_rendered.as_ref()
            {
                *out_frame = StarmineAdRender714Frame::from(frame);
            }
            StarmineAdStatus::Ok
        }
        Err(status) => status,
    }
}

#[unsafe(no_mangle)]
/// Emit the final short limiter block after the last TrueHD input frame.
pub unsafe extern "C" fn starmine_ad_truehd_renderer_714_flush(
    renderer: *mut StarmineAdTrueHdRenderer714Handle,
    out_frame: *mut StarmineAdRender714Frame,
) -> StarmineAdStatus {
    let Some(renderer) = (unsafe { renderer.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };
    let Some(out_frame) = (unsafe { out_frame.as_mut() }) else {
        return StarmineAdStatus::NullPointer;
    };

    *out_frame = StarmineAdRender714Frame::empty();
    renderer.flush();
    if let Some(frame) = renderer.last_rendered.as_ref() {
        *out_frame = StarmineAdRender714Frame::from(frame);
    }
    StarmineAdStatus::Ok
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
        StarmineAdStatus::BedChannelCountMismatch => STATUS_BED_CHANNEL_COUNT_MISMATCH.as_ptr(),
        StarmineAdStatus::TrueHdParse => STATUS_TRUEHD_PARSE.as_ptr(),
        StarmineAdStatus::TrueHdDecode => STATUS_TRUEHD_DECODE.as_ptr(),
        StarmineAdStatus::TrueHdUnsupportedLayout => STATUS_TRUEHD_UNSUPPORTED_LAYOUT.as_ptr(),
        StarmineAdStatus::TrueHdInvalidMetadata => STATUS_TRUEHD_INVALID_METADATA.as_ptr(),
    }
    .cast::<c_char>()
}

#[unsafe(no_mangle)]
/// Initialize an E-AC-3 info struct to zero / empty defaults.
pub extern "C" fn starmine_ad_eac3_access_unit_info_init(
    out_info: *mut StarmineAdEac3AccessUnitInfo,
) -> StarmineAdStatus {
    let Some(out_info) = ptr::NonNull::new(out_info) else {
        return StarmineAdStatus::NullPointer;
    };
    unsafe {
        out_info.as_ptr().write(StarmineAdEac3AccessUnitInfo {
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
/// Initialize a TrueHD info struct to zero / empty defaults.
pub extern "C" fn starmine_ad_truehd_access_unit_info_init(
    out_info: *mut StarmineAdTrueHdAccessUnitInfo,
) -> StarmineAdStatus {
    let Some(out_info) = ptr::NonNull::new(out_info) else {
        return StarmineAdStatus::NullPointer;
    };
    unsafe {
        out_info
            .as_ptr()
            .write(StarmineAdTrueHdAccessUnitInfo::empty());
    }
    StarmineAdStatus::Ok
}

#[unsafe(no_mangle)]
/// Initialize an object-PCM frame struct to the empty / no-output state.
pub extern "C" fn starmine_ad_object_pcm_frame_init(
    out_frame: *mut StarmineAdObjectPcmFrame,
) -> StarmineAdStatus {
    let Some(out_frame) = ptr::NonNull::new(out_frame) else {
        return StarmineAdStatus::NullPointer;
    };
    unsafe {
        out_frame.as_ptr().write(StarmineAdObjectPcmFrame::empty());
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
