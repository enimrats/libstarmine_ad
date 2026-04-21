// SPDX-License-Identifier: Apache-2.0
// Derived from `truehdd`; modified for `libstarmine_ad`.

use super::decoder::ObjectPcmFrame;
use crate::renderer::{RenderFrameSource, RenderInputChannel, RenderInputFrame};

impl ObjectPcmFrame {
    /// Convert the decoded TrueHD/Atmos frame into the renderer's codec-neutral IR.
    pub fn to_render_input(&self) -> RenderInputFrame {
        self.clone().into_render_input()
    }

    /// Convert the decoded frame into the renderer IR without cloning channel buffers.
    pub fn into_render_input(self) -> RenderInputFrame {
        let ObjectPcmFrame {
            sample_rate,
            bed_channel_order,
            bed_channels,
            object_channels,
            metadata_updates,
        } = self;

        let bed_channels = bed_channel_order
            .into_iter()
            .zip(bed_channels)
            .map(|(channel, samples)| RenderInputChannel { channel, samples })
            .collect();

        RenderInputFrame {
            sample_rate,
            bed_channels,
            object_channels,
            metadata_updates,
        }
    }
}

impl RenderFrameSource for ObjectPcmFrame {
    fn to_render_input(&self) -> RenderInputFrame {
        ObjectPcmFrame::to_render_input(self)
    }
}
