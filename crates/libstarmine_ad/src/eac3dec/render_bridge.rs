use super::metadata::{OamdElementKind, OamdObjectBlock, OamdPayload};
use super::pcm::{CorePcmFrame, ObjectPcmFrame};
use crate::renderer::{
    BedChannel, RenderFrameSource, RenderInputChannel, RenderInputFrame, RenderMetadata,
    RenderMetadataBlockUpdate, RenderMetadataElement, RenderMetadataObject, RenderMetadataUpdate,
};

pub(crate) fn render_input_from_eac3_parts(
    core: &CorePcmFrame,
    object_channels: &[Vec<f32>],
    oamd_payloads: &[(&OamdPayload, Option<u16>)],
) -> RenderInputFrame {
    let mut bed_channels =
        Vec::with_capacity(core.fullband_channels.len() + usize::from(core.lfe_channel.is_some()));
    for (channel, samples) in core
        .fullband_channel_order
        .iter()
        .copied()
        .zip(core.fullband_channels.iter())
    {
        bed_channels.push(RenderInputChannel {
            channel,
            samples: samples.clone(),
        });
    }
    if let Some(samples) = core.lfe_channel.as_ref() {
        bed_channels.push(RenderInputChannel {
            channel: BedChannel::LowFrequencyEffects,
            samples: samples.clone(),
        });
    }

    let metadata_updates = oamd_payloads
        .iter()
        .map(|(payload, sample_offset)| {
            RenderMetadataUpdate::from_oamd_payload(payload, *sample_offset)
        })
        .collect();

    RenderInputFrame {
        sample_rate: core.sample_rate,
        bed_channels,
        object_channels: object_channels.to_vec(),
        metadata_updates,
    }
}

impl ObjectPcmFrame {
    /// Convert the E-AC-3/JOC decoded frame into the renderer's codec-neutral IR.
    pub fn to_render_input(&self) -> RenderInputFrame {
        let oamd_payloads = self
            .oamd_payloads
            .iter()
            .map(|(payload, sample_offset)| (payload, *sample_offset))
            .collect::<Vec<_>>();
        render_input_from_eac3_parts(&self.core, &self.object_channels, &oamd_payloads)
    }
}

impl RenderFrameSource for ObjectPcmFrame {
    fn to_render_input(&self) -> RenderInputFrame {
        ObjectPcmFrame::to_render_input(self)
    }
}

impl RenderMetadataUpdate {
    pub fn from_oamd_payload(payload: &OamdPayload, sample_offset: Option<u16>) -> Self {
        Self {
            sample_offset: sample_offset.unwrap_or_default(),
            metadata: RenderMetadata::from(payload),
        }
    }
}

impl From<&OamdObjectBlock> for RenderMetadataObject {
    fn from(block: &OamdObjectBlock) -> Self {
        Self {
            gain: block.gain,
            anchor: block.anchor,
            position_valid: block.valid_position,
            differential_position: block.differential_position,
            position: block.position,
            distance: block.distance,
            size: block.size,
            screen_factor: block.screen_factor.unwrap_or(1.0),
            depth_factor: block.depth_factor.unwrap_or(1.0),
        }
    }
}

impl From<&OamdPayload> for RenderMetadata {
    fn from(payload: &OamdPayload) -> Self {
        let bed_channels = payload
            .bed_assignment
            .iter()
            .flat_map(|instance| instance.iter().copied())
            .collect();
        let elements = payload
            .elements
            .iter()
            .filter_map(|element| {
                let OamdElementKind::Object(object_element) = &element.kind else {
                    return None;
                };

                Some(RenderMetadataElement {
                    block_updates: object_element
                        .block_updates
                        .iter()
                        .map(|update| RenderMetadataBlockUpdate {
                            offset: i64::from(update.offset),
                            ramp_duration: i64::from(update.ramp_duration),
                        })
                        .collect(),
                    object_blocks: object_element
                        .object_blocks
                        .iter()
                        .map(|blocks| blocks.iter().map(RenderMetadataObject::from).collect())
                        .collect(),
                })
            })
            .collect();

        Self {
            object_count: payload.object_count,
            bed_or_isf_objects: payload.bed_or_isf_objects,
            bed_channels,
            elements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::render_input_from_eac3_parts;
    use crate::eac3dec::{CorePcmFrame, JocPayload, OamdPayload, ObjectPcmFrame};
    use crate::renderer::{BedChannel, RenderFrameSource};

    fn bed_payload(channel: BedChannel) -> OamdPayload {
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
            bed_assignment: vec![vec![channel]],
            elements: Vec::new(),
        }
    }

    #[test]
    fn object_pcm_frame_bridges_to_render_ir() {
        let frame = ObjectPcmFrame {
            core: CorePcmFrame {
                sample_rate: 48_000,
                fullband_channel_order: vec![BedChannel::FrontLeft],
                fullband_channels: vec![vec![0.25; 4]],
                lfe_channel: None,
            },
            object_channels: vec![vec![1.0; 4]],
            object_active: vec![true],
            joc: JocPayload {
                downmix_config: 0,
                channel_count: 0,
                object_count: 0,
                gain: 1.0,
                sequence_counter: 0,
                objects: Vec::new(),
            },
            oamd_payloads: Vec::new(),
        };

        let input = frame.to_render_input();
        assert_eq!(input.sample_rate, 48_000);
        assert_eq!(input.bed_channels.len(), 1);
        assert_eq!(input.object_channels.len(), 1);
        assert!(input.metadata_updates.is_empty());

        let trait_input = RenderFrameSource::to_render_input(&frame);
        assert_eq!(trait_input, input);
    }

    #[test]
    fn eac3_parts_bridge_keeps_lfe_channel() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft],
            fullband_channels: vec![vec![0.25; 2]],
            lfe_channel: Some(vec![0.5; 2]),
        };

        let input = render_input_from_eac3_parts(&core, &[], &[]);
        assert_eq!(input.bed_channels.len(), 2);
        assert_eq!(
            input.bed_channels[1].channel,
            BedChannel::LowFrequencyEffects
        );
    }

    #[test]
    fn eac3_parts_bridge_keeps_all_oamd_payloads() {
        let core = CorePcmFrame {
            sample_rate: 48_000,
            fullband_channel_order: vec![BedChannel::FrontLeft],
            fullband_channels: vec![vec![0.25; 128]],
            lfe_channel: None,
        };
        let first = bed_payload(BedChannel::FrontLeft);
        let second = bed_payload(BedChannel::FrontRight);
        let oamd_payloads = [(&first, Some(8)), (&second, Some(64))];

        let input = render_input_from_eac3_parts(&core, &[], &oamd_payloads);
        assert_eq!(input.metadata_updates.len(), 2);
        assert_eq!(input.metadata_updates[0].sample_offset, 8);
        assert_eq!(
            input.metadata_updates[0].metadata.bed_channels,
            vec![BedChannel::FrontLeft]
        );
        assert_eq!(input.metadata_updates[1].sample_offset, 64);
        assert_eq!(
            input.metadata_updates[1].metadata.bed_channels,
            vec![BedChannel::FrontRight]
        );
    }
}
