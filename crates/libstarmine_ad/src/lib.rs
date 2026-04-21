//! Stateful E-AC-3 object-audio decoding and 7.1.4 rendering.
//!
//! The crate is split into two main namespaces:
//!
//! - [`eac3dec`] contains the E-AC-3/JOC/OAMD parser and decoder pipeline.
//! - [`truehddec`] contains the Dolby TrueHD/Atmos parser and decoder bridge.
//! - [`renderer`] contains the codec-neutral render contract and the stateful 7.1.4 renderer.
//!
//! The explicit seam between them is [`renderer::RenderInputFrame`] /
//! [`renderer::RenderFrameSource`].
//!
//! All decoders are stateful. Feed complete access units in stream order and call `reset()` after
//! seeks, discontinuities, or when you intentionally drop intermediate packets.
//!
//! # Rust Usage
//!
//! ```no_run
//! use std::fs;
//! use starmine_ad::{eac3dec::ObjectPcmDecoder, renderer::Renderer714};
//!
//! let access_unit = fs::read("frame.eac3")?;
//! let mut decoder = ObjectPcmDecoder::new();
//! let mut renderer = Renderer714::new();
//!
//! if let Some(result) = decoder.push_access_unit(&access_unit)? {
//!     let input = result.pcm.to_render_input();
//!     let rendered = renderer.push_frame(&input)?;
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
//! explicit codec-specific entry points:
//!
//! - `starmine_ad_eac3_*` for E-AC-3/JOC inspection and 7.1.4 rendering.
//! - `starmine_ad_truehd_*` for TrueHD/Atmos access-unit decoding and 7.1.4 rendering.
//!
//! The render paths export borrowed planar `float` pointers whose lifetime is tied to the owning
//! handle. A libav-based end-to-end C example lives under `Starmine_ad/examples/`.

pub mod eac3dec;
mod ffi;
pub mod renderer;
pub mod truehddec;
