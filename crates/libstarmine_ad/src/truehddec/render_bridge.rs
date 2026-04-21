use super::decoder::ObjectPcmFrame;
use crate::renderer::{RenderFrameSource, RenderInputChannel, RenderInputFrame};

impl ObjectPcmFrame {
    /// Convert the decoded TrueHD/Atmos frame into the renderer's codec-neutral IR.
    pub fn to_render_input(&self) -> RenderInputFrame {
        let bed_channels = self
            .bed_channel_order
            .iter()
            .copied()
            .zip(self.bed_channels.iter())
            .map(|(channel, samples)| RenderInputChannel {
                channel,
                samples: samples.clone(),
            })
            .collect();

        RenderInputFrame {
            sample_rate: self.sample_rate,
            bed_channels,
            object_channels: self.object_channels.clone(),
            metadata_updates: self.metadata_updates.clone(),
        }
    }
}

impl RenderFrameSource for ObjectPcmFrame {
    fn to_render_input(&self) -> RenderInputFrame {
        ObjectPcmFrame::to_render_input(self)
    }
}
