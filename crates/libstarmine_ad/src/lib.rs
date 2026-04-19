//! Stateful E-AC-3 object-audio decoding and 7.1.4 rendering.
//!
//! The crate is organized as a small pipeline. Pick the highest layer you need:
//!
//! - [`inspect_access_unit`] validates a single complete access unit and returns parsed
//!   syncframe / EMDF / payload metadata.
//! - [`Decoder`] does the same thing statefully across frames and keeps cross-frame metadata
//!   state in sync.
//! - [`PcmDecoder`] adds decoded core PCM channels.
//! - [`ObjectPcmDecoder`] adds decoded object PCM plus the parsed object metadata payloads.
//! - [`Renderer714`] turns [`ObjectPcmFrame`] values into 7.1.4 float PCM.
//!
//! All decoders are stateful. Feed complete access units in stream order and call `reset()` after
//! seeks, discontinuities, or when you intentionally drop intermediate packets.
//!
//! # Rust Usage
//!
//! ```no_run
//! use std::fs;
//! use starmine_ad::{ObjectPcmDecoder, Renderer714};
//!
//! let access_unit = fs::read("frame.eac3")?;
//! let mut decoder = ObjectPcmDecoder::new();
//! let mut renderer = Renderer714::new();
//!
//! if let Some(result) = decoder.push_access_unit(&access_unit)? {
//!     let rendered = renderer.push_frame(&result.pcm)?;
//!     assert_eq!(rendered.channel_count(), 12);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Command-Line Tool
//!
//! The crate also ships with an integrated CLI target:
//!
//! ```text
//! cargo run -p libstarmine_ad --bin starmine-ad-cli -- <input.eac3> --render-714-check
//! ```
//!
//! # C Integration
//!
//! A C ABI is provided through [starmine_ad.h](../../include/starmine_ad.h). The header exposes
//! both the low-level [`Decoder`] entry point and a stateful 7.1.4 rendering path: create either
//! a decoder handle or a renderer handle, push one complete access unit at a time, read a copied
//! [`AccessUnitInfo`]-style summary, and reset the handle when the stream position jumps. The
//! render path exports borrowed planar `float` pointers whose lifetime is tied to the renderer
//! handle. A libav-based end-to-end C example lives under `Starmine_ad/examples/`.

mod allocation;
mod bitstream;
mod decoder;
mod ffi;
mod imdct;
mod joc;
mod metadata;
mod pcm;
mod qmf;
mod render;
mod syncframe;

pub use decoder::{Decoder, PushResult};
pub use joc::{JocObjectMatrices, JocSubbandMatrix, JocTimeslotMatrices};
pub use metadata::{
    BedChannel, JocObject, JocObjectData, JocPayload, OamdBlockUpdate, OamdElement,
    OamdElementKind, OamdObjectBlock, OamdObjectElement, OamdPayload, ObjectAnchor,
    ParsedEmdfPayloadData, ParsedEmdfPayloadKind, Vec3,
};
pub use pcm::{
    CorePcmFrame, ObjectPcmDecoder, ObjectPcmFrame, ObjectPcmPushResult, PcmDecoder, PcmPushResult,
};
pub use render::{
    RENDER_714_CHANNEL_ORDER, Render714Error, Render714Frame, Render714SourceDebug,
    Render714TimeslotDebug, Renderer714,
};
pub use syncframe::{
    AccessUnitInfo, AuxParseStatus, EmdfBlockInfo, EmdfPayloadInfo, FrameType, ParseError,
    PayloadInfo, SkipFieldInfo, inspect_access_unit,
};
