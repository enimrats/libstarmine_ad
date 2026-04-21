use crate::metadata::{
    BedChannel, OamdElementKind, OamdObjectBlock, OamdPayload, ObjectAnchor, Vec3,
};

#[derive(Debug, Clone, PartialEq)]
/// One labeled bed channel carried by a [`RenderInputFrame`].
pub struct RenderInputChannel {
    pub channel: BedChannel,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
/// One block update inside a renderer metadata element.
pub struct RenderMetadataBlockUpdate {
    pub offset: i64,
    pub ramp_duration: i64,
}

#[derive(Debug, Clone, PartialEq)]
/// One fully-resolved object state carried by renderer metadata.
pub struct RenderMetadataObject {
    pub gain: Option<f32>,
    pub anchor: ObjectAnchor,
    pub position_valid: bool,
    pub differential_position: bool,
    pub position: Option<Vec3>,
    pub distance: Option<f32>,
    pub size: Option<f32>,
    pub screen_factor: f32,
    pub depth_factor: f32,
}

#[derive(Debug, Clone, PartialEq)]
/// One metadata element consumed by [`crate::Renderer714`].
pub struct RenderMetadataElement {
    pub block_updates: Vec<RenderMetadataBlockUpdate>,
    pub object_blocks: Vec<Vec<RenderMetadataObject>>,
}

#[derive(Debug, Clone, PartialEq)]
/// Codec-neutral object metadata consumed by [`crate::Renderer714`].
pub struct RenderMetadata {
    pub object_count: usize,
    pub bed_or_isf_objects: usize,
    pub bed_channels: Vec<BedChannel>,
    pub elements: Vec<RenderMetadataElement>,
}

#[derive(Debug, Clone, PartialEq)]
/// One metadata payload update carried by a [`RenderInputFrame`].
pub struct RenderMetadataUpdate {
    pub sample_offset: u16,
    pub metadata: RenderMetadata,
}

#[derive(Debug, Clone, PartialEq)]
/// Codec-agnostic render contract shared between decoders and [`crate::Renderer714`].
///
/// Decoders fill this structure with labeled bed PCM, dynamic object PCM, and any metadata
/// updates that become active during the same frame. The renderer only depends on this IR and
/// does not need to know which codec produced it.
pub struct RenderInputFrame {
    pub sample_rate: u32,
    pub bed_channels: Vec<RenderInputChannel>,
    pub object_channels: Vec<Vec<f32>>,
    pub metadata_updates: Vec<RenderMetadataUpdate>,
}

impl RenderInputFrame {
    /// Number of samples carried by each input channel.
    pub fn samples_per_channel(&self) -> usize {
        self.bed_channels
            .first()
            .map(|channel| channel.samples.len())
            .or_else(|| self.object_channels.first().map(Vec::len))
            .unwrap_or(0)
    }

    /// Number of bed channels in this frame.
    pub fn bed_channel_count(&self) -> usize {
        self.bed_channels.len()
    }

    /// Number of dynamic object channels in this frame.
    pub fn object_count(&self) -> usize {
        self.object_channels.len()
    }

    /// Number of metadata payload updates carried by this frame.
    pub fn metadata_update_count(&self) -> usize {
        self.metadata_updates.len()
    }
}

impl RenderMetadata {
    pub(crate) fn dynamic_object_count(&self) -> usize {
        self.object_count.saturating_sub(self.bed_or_isf_objects)
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
