#pragma once

#include <stddef.h>
#include <stdint.h>

#define STARMINE_AD_RENDER_714_CHANNEL_COUNT 12u

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque stateful decoder handle.
 *
 * The handle preserves cross-frame metadata state, so callers must push access
 * units in stream order and call `starmine_ad_decoder_reset()` after seeks,
 * packet loss, or any discontinuity.
 */
typedef struct starmine_ad_decoder starmine_ad_decoder;

/**
 * Opaque stateful 7.1.4 renderer handle.
 *
 * The handle owns both the object-PCM decoder state and the 7.1.4 renderer
 * state, so callers must push access units in stream order and call
 * `starmine_ad_renderer_714_reset()` after seeks, packet loss, or any
 * discontinuity.
 */
typedef struct starmine_ad_renderer_714 starmine_ad_renderer_714;

/**
 * Parsed summary for one complete access unit.
 *
 * This struct is a copied snapshot owned by the caller. It stays valid after
 * the decode call returns and does not borrow from any library handle.
 */
typedef struct starmine_ad_access_unit_info {
    /** Total frame size in bytes. */
    uint32_t frame_size;
    /** E-AC-3 bitstream identifier (`bsid`). */
    uint8_t bitstream_id;
    /**
   * Frame coding mode:
   * - `0`: independent
   * - `1`: dependent
   * - `2`: AC-3 convert
   */
    uint8_t frame_type;
    /** E-AC-3 substream id. */
    uint8_t substreamid;
    /** Sample rate in Hz. */
    uint32_t sample_rate;
    /** Audio block count in the access unit. */
    uint8_t num_blocks;
    /** AC-3 `acmod` channel mode. */
    uint8_t channel_mode;
    /** Total core channel count including LFE if present. */
    uint8_t channels;
    /** `1` when the core frame contains an LFE channel. */
    uint8_t lfe_on;
    /** `1` when an `addbsi` section is present. */
    uint8_t addbsi_present;
    /** `1` when addbsi advertises extension type A. */
    uint8_t extension_type_a;
    /** Complexity index from extension type A, or `0` when absent. */
    uint8_t complexity_index_type_a;
    /** Number of recovered EMDF blocks in this access unit. */
    uint32_t emdf_block_count;
    /** Total number of recovered EMDF payloads across all blocks. */
    uint32_t payload_count;
    /** Number of JOC payloads in this access unit. */
    uint32_t joc_payload_count;
    /** Number of OAMD payloads in this access unit. */
    uint32_t oamd_payload_count;
    /** `1` when `first_emdf_sync_offset` contains a valid byte offset. */
    uint8_t has_first_emdf_sync_offset;
    /** Byte offset of the first EMDF sync marker when one was found. */
    uint32_t first_emdf_sync_offset;
    /** Count of accepted access units since the last reset. */
    uint64_t frames_seen;
} starmine_ad_access_unit_info;

/**
 * Stable speaker / bed-channel identifiers used by the C ABI.
 */
typedef enum starmine_ad_bed_channel {
    STARMINE_AD_BED_CHANNEL_UNKNOWN = -1,
    STARMINE_AD_BED_CHANNEL_FRONT_LEFT = 0,
    STARMINE_AD_BED_CHANNEL_FRONT_RIGHT = 1,
    STARMINE_AD_BED_CHANNEL_CENTER = 2,
    STARMINE_AD_BED_CHANNEL_LOW_FREQUENCY_EFFECTS = 3,
    STARMINE_AD_BED_CHANNEL_SURROUND_LEFT = 4,
    STARMINE_AD_BED_CHANNEL_SURROUND_RIGHT = 5,
    STARMINE_AD_BED_CHANNEL_REAR_LEFT = 6,
    STARMINE_AD_BED_CHANNEL_REAR_RIGHT = 7,
    STARMINE_AD_BED_CHANNEL_TOP_FRONT_LEFT = 8,
    STARMINE_AD_BED_CHANNEL_TOP_FRONT_RIGHT = 9,
    STARMINE_AD_BED_CHANNEL_TOP_SURROUND_LEFT = 10,
    STARMINE_AD_BED_CHANNEL_TOP_SURROUND_RIGHT = 11,
    STARMINE_AD_BED_CHANNEL_TOP_REAR_LEFT = 12,
    STARMINE_AD_BED_CHANNEL_TOP_REAR_RIGHT = 13,
    STARMINE_AD_BED_CHANNEL_WIDE_LEFT = 14,
    STARMINE_AD_BED_CHANNEL_WIDE_RIGHT = 15,
    STARMINE_AD_BED_CHANNEL_LOW_FREQUENCY_EFFECTS2 = 16,
} starmine_ad_bed_channel;

/**
 * Borrowed view of one rendered 7.1.4 PCM frame.
 *
 * When `has_frame` is `1`, `channels[i]` points to `samples_per_channel` planar
 * `float` samples. The pointers are owned by `starmine_ad_renderer_714` and
 * remain valid only until the next
 * `starmine_ad_renderer_714_push_access_unit()`,
 * `starmine_ad_renderer_714_reset()`, or `starmine_ad_renderer_714_free()` on
 * the same handle.
 *
 * `channel_order[i]` names the speaker carried by `channels[i]`. For the
 * current renderer the fixed order is: `FL, FR, C, LFE, RL, RR, SL, SR, TFL,
 * TFR, TRL, TRR`.
 *
 * When `has_frame` is `0`, the access unit parsed successfully but did not
 * yield a rendered 7.1.4 output frame, typically because it did not carry the
 * required JOC payload.
 */
typedef struct starmine_ad_render_714_frame {
    /** `1` when this struct contains a rendered frame, otherwise `0`. */
    uint8_t has_frame;
    /** Output sample rate in Hz when `has_frame` is `1`, otherwise `0`. */
    uint32_t sample_rate;
    /** Number of float samples available through each channel pointer. */
    size_t samples_per_channel;
    /** Number of valid entries in `channels[]` and `channel_order[]`. */
    size_t channel_count;
    /** Planar 7.1.4 channel pointers owned by the renderer handle. */
    const float* channels[STARMINE_AD_RENDER_714_CHANNEL_COUNT];
    /** Speaker mapping for each exported channel pointer. */
    starmine_ad_bed_channel channel_order[STARMINE_AD_RENDER_714_CHANNEL_COUNT];
} starmine_ad_render_714_frame;

/**
 * Status code returned by every C API function.
 */
typedef enum starmine_ad_status {
    STARMINE_AD_STATUS_OK = 0,
    STARMINE_AD_STATUS_NULL_POINTER = -1,
    STARMINE_AD_STATUS_SHORT_PACKET = -2,
    STARMINE_AD_STATUS_BAD_SYNCWORD = -3,
    STARMINE_AD_STATUS_NOT_EAC3 = -4,
    STARMINE_AD_STATUS_INVALID_HEADER = -5,
    STARMINE_AD_STATUS_TRUNCATED_FRAME = -6,
    STARMINE_AD_STATUS_TRAILING_DATA = -7,
    STARMINE_AD_STATUS_UNSUPPORTED_FEATURE = -8,
    STARMINE_AD_STATUS_MISSING_OAMD = -9,
    STARMINE_AD_STATUS_OAMD_STATE_UNINITIALIZED = -10,
    STARMINE_AD_STATUS_OBJECT_COUNT_MISMATCH = -11,
    STARMINE_AD_STATUS_UNSUPPORTED_SAMPLE_COUNT = -12,
    STARMINE_AD_STATUS_UNSUPPORTED_BED_CHANNEL = -13,
    STARMINE_AD_STATUS_SAMPLE_RATE_CHANGED = -14,
} starmine_ad_status;

/**
 * Create a new decoder handle.
 *
 * Returns `NULL` only if allocation fails.
 */
starmine_ad_decoder* starmine_ad_decoder_new(void);

/**
 * Destroy a decoder handle created by `starmine_ad_decoder_new()`.
 *
 * Passing `NULL` is allowed.
 */
void starmine_ad_decoder_free(starmine_ad_decoder* decoder);

/**
 * Clear all decoder state.
 *
 * Call this after a seek or any other discontinuity before pushing more access
 * units.
 */
starmine_ad_status starmine_ad_decoder_reset(starmine_ad_decoder* decoder);

/**
 * Parse one complete E-AC-3 access unit.
 *
 * `data` must point to exactly one full access unit. The function reports both
 * short buffers and trailing bytes so the caller can keep frame boundaries
 * explicit.
 *
 * `out_info` is optional. Pass `NULL` if you only need the status code.
 */
starmine_ad_status
starmine_ad_decoder_push_access_unit(starmine_ad_decoder* decoder,
                                     const uint8_t* data, size_t len,
                                     starmine_ad_access_unit_info* out_info);

/**
 * Create a new 7.1.4 renderer handle.
 *
 * Returns `NULL` only if allocation fails.
 */
starmine_ad_renderer_714* starmine_ad_renderer_714_new(void);

/**
 * Destroy a renderer handle created by `starmine_ad_renderer_714_new()`.
 *
 * Passing `NULL` is allowed.
 */
void starmine_ad_renderer_714_free(starmine_ad_renderer_714* renderer);

/**
 * Clear all renderer state.
 *
 * Call this after a seek or any other discontinuity before pushing more access
 * units.
 */
starmine_ad_status
starmine_ad_renderer_714_reset(starmine_ad_renderer_714* renderer);

/**
 * Decode one complete E-AC-3 access unit and, when possible, render it to 7.1.4
 * float PCM.
 *
 * `data` must point to exactly one full access unit.
 *
 * `out_info` is optional. Pass `NULL` if you only need the status code.
 * `out_frame` is optional. Pass `NULL` if you only want the decoder / renderer
 * state to advance.
 *
 * On success, `out_frame->has_frame` reports whether this access unit produced
 * a rendered frame.
 */
starmine_ad_status starmine_ad_renderer_714_push_access_unit(
    starmine_ad_renderer_714* renderer, const uint8_t* data, size_t len,
    starmine_ad_access_unit_info* out_info,
    starmine_ad_render_714_frame* out_frame);

/**
 * Convert a status code to a stable ASCII string.
 *
 * The returned pointer is owned by the library and must not be freed.
 */
const char* starmine_ad_status_string(starmine_ad_status status);

/**
 * Initialize an info struct to zero / empty defaults.
 *
 * This is useful for callers that want a predictable value before the first
 * decode call.
 */
starmine_ad_status
starmine_ad_access_unit_info_init(starmine_ad_access_unit_info* out_info);

/**
 * Initialize a render frame struct to the empty / no-output state.
 */
starmine_ad_status
starmine_ad_render_714_frame_init(starmine_ad_render_714_frame* out_frame);

#ifdef __cplusplus
}
#endif
