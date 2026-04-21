// SPDX-License-Identifier: Apache-2.0

use super::render::{Render714Error, Render714Frame, Render714TimeslotDebug, Renderer714};
use super::render_input::RenderInputFrame;

/// Bridge from a codec-specific decoded frame into the renderer's codec-neutral IR.
///
/// New codecs should expose their own decoded frame type and implement this trait so they can be
/// routed into [`Renderer714`] without teaching the renderer about codec internals.
pub trait RenderFrameSource {
    fn to_render_input(&self) -> RenderInputFrame;
}

impl Renderer714 {
    /// Render any decoded frame that implements the shared decoder-to-renderer bridge.
    pub fn push_source_frame(
        &mut self,
        source: &impl RenderFrameSource,
    ) -> Result<Render714Frame, Render714Error> {
        let input = source.to_render_input();
        self.push_frame(&input)
    }

    /// Like [`Self::push_source_frame`], but also returns per-timeslot debug state.
    pub fn push_source_frame_with_debug(
        &mut self,
        source: &impl RenderFrameSource,
    ) -> Result<(Render714Frame, Vec<Render714TimeslotDebug>), Render714Error> {
        let input = source.to_render_input();
        self.push_frame_with_debug(&input)
    }
}
