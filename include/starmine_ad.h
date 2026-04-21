#pragma once

#include <stddef.h>
#include <stdint.h>

#define STARMINE_AD_RENDER_714_CHANNEL_COUNT 12u

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque stateful E-AC-3 access-unit inspector.
 *
 * The handle preserves cross-frame metadata state, so callers must push access
 * units in stream order and call `starmine_ad_eac3_decoder_reset()` after
 * seeks, packet loss, or any discontinuity.
 */
typedef struct starmine_ad_eac3_decoder starmine_ad_eac3_decoder;

/**
 * Opaque stateful E-AC-3 7.1.4 renderer.
 *
 * The handle owns both the object-PCM decoder state and the 7.1.4 renderer
 * state, so callers must push access units in stream order and call
 * `starmine_ad_eac3_renderer_714_reset()` after seeks, packet loss, or any
 * discontinuity.
 */
typedef struct starmine_ad_eac3_renderer_714 starmine_ad_eac3_renderer_714;

/**
 * Opaque stateful TrueHD object decoder for presentation 3 Atmos streams.
 *
 * The handle preserves parser and decode state across access units. Push
 * complete access units in stream order and call
 * `starmine_ad_truehd_decoder_reset()` after seeks, packet loss, or any
 * discontinuity.
 */
typedef struct starmine_ad_truehd_decoder starmine_ad_truehd_decoder;

/**
 * Opaque stateful TrueHD 7.1.4 renderer for presentation 3 Atmos streams.
 *
 * The handle owns both the TrueHD object decoder state and the 7.1.4 renderer
 * state, so callers must push access units in stream order and call
 * `starmine_ad_truehd_renderer_714_reset()` after seeks, packet loss, or any
 * discontinuity.
 */
typedef struct starmine_ad_truehd_renderer_714 starmine_ad_truehd_renderer_714;

/**
 * Parsed summary for one complete E-AC-3 access unit.
 *
 * This struct is a copied snapshot owned by the caller. It stays valid after
 * the decode call returns and does not borrow from any library handle.
 */
typedef struct starmine_ad_eac3_access_unit_info {
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
} starmine_ad_eac3_access_unit_info;

/**
 * Decoded summary for one complete TrueHD access unit.
 *
 * This struct is a copied snapshot owned by the caller. It stays valid after
 * the decode call returns and does not borrow from any library handle.
 */
typedef struct starmine_ad_truehd_access_unit_info {
    /**
   * `1` when this access unit produced an object frame for presentation 3,
   * otherwise `0`.
   */
    uint8_t has_frame;
    /** `1` when substream info changed on this access unit. */
    uint8_t substream_info_changed;
    /** Decoded sample rate in Hz when `has_frame` is `1`, otherwise `0`. */
    uint32_t sample_rate;
    /** Samples carried by each bed/object channel when `has_frame` is `1`. */
    uint32_t samples_per_channel;
    /** Decoded bed-channel count when `has_frame` is `1`. */
    uint32_t bed_channel_count;
    /** Decoded dynamic-object count when `has_frame` is `1`. */
    uint32_t object_count;
    /** Number of renderer metadata updates carried by this frame. */
    uint32_t metadata_update_count;
    /** Count of accepted access units since the last reset. */
    uint64_t access_units_seen;
    /** Count of object frames emitted since the last reset. */
    uint64_t frames_seen;
} starmine_ad_truehd_access_unit_info;

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
 * Borrowed view of one decoded object-PCM frame.
 *
 * When `has_frame` is `1`, `bed_channel_order[i]` names the speaker carried by
 * `bed_channels[i]`. Each `bed_channels[i]` and `object_channels[i]` pointer
 * addresses `samples_per_channel` planar `float` samples.
 *
 * The arrays and sample pointers are owned by `starmine_ad_truehd_decoder` and
 * remain valid only until the next
 * `starmine_ad_truehd_decoder_push_access_unit()`,
 * `starmine_ad_truehd_decoder_reset()`, or
 * `starmine_ad_truehd_decoder_free()` on the same handle.
 *
 * When `has_frame` is `0`, the access unit decoded successfully but did not
 * expose any dynamic objects for presentation 3.
 */
typedef struct starmine_ad_object_pcm_frame {
    /** `1` when this struct contains decoded PCM, otherwise `0`. */
    uint8_t has_frame;
    /** Decoded sample rate in Hz when `has_frame` is `1`, otherwise `0`. */
    uint32_t sample_rate;
    /** Number of float samples available through each channel pointer. */
    size_t samples_per_channel;
    /** Number of valid entries in `bed_channel_order[]` / `bed_channels[]`. */
    size_t bed_channel_count;
    /** Number of valid entries in `object_channels[]`. */
    size_t object_count;
    /** Speaker mapping for each decoded bed channel. */
    const starmine_ad_bed_channel* bed_channel_order;
    /** Planar bed-channel pointers owned by the decoder handle. */
    const float* const* bed_channels;
    /** Planar dynamic-object pointers owned by the decoder handle. */
    const float* const* object_channels;
} starmine_ad_object_pcm_frame;

/**
 * Borrowed view of one rendered 7.1.4 PCM frame.
 *
 * When `has_frame` is `1`, `channels[i]` points to `samples_per_channel` planar
 * `float` samples. The pointers are owned by the corresponding renderer handle
 * and remain valid only until the next push, reset, or free call on that same
 * handle.
 *
 * `channel_order[i]` names the speaker carried by `channels[i]`. The current
 * renderer order is fixed to `FL, FR, C, LFE, RL, RR, SL, SR, TFL, TFR, TRL,
 * TRR`.
 *
 * When `has_frame` is `0`, the decode call succeeded but did not emit a frame.
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
    STARMINE_AD_STATUS_BED_CHANNEL_COUNT_MISMATCH = -15,
    STARMINE_AD_STATUS_TRUEHD_PARSE = -16,
    STARMINE_AD_STATUS_TRUEHD_DECODE = -17,
    STARMINE_AD_STATUS_TRUEHD_UNSUPPORTED_LAYOUT = -18,
    STARMINE_AD_STATUS_TRUEHD_INVALID_METADATA = -19,
} starmine_ad_status;

/**
 * Create a new E-AC-3 decoder handle.
 *
 * Returns `NULL` only if allocation fails.
 */
starmine_ad_eac3_decoder* starmine_ad_eac3_decoder_new(void);

/**
 * Destroy a decoder handle created by `starmine_ad_eac3_decoder_new()`.
 *
 * Passing `NULL` is allowed.
 */
void starmine_ad_eac3_decoder_free(starmine_ad_eac3_decoder* decoder);

/**
 * Clear all E-AC-3 decoder state.
 *
 * Call this after a seek or any other discontinuity before pushing more access
 * units.
 */
starmine_ad_status
starmine_ad_eac3_decoder_reset(starmine_ad_eac3_decoder* decoder);

/**
 * Parse one complete E-AC-3 access unit.
 *
 * `data` must point to exactly one full access unit. The function reports both
 * short buffers and trailing bytes so the caller can keep frame boundaries
 * explicit.
 *
 * `out_info` is optional. Pass `NULL` if you only need the status code.
 */
starmine_ad_status starmine_ad_eac3_decoder_push_access_unit(
    starmine_ad_eac3_decoder* decoder, const uint8_t* data, size_t len,
    starmine_ad_eac3_access_unit_info* out_info);

/**
 * Create a new E-AC-3 7.1.4 renderer handle.
 *
 * Returns `NULL` only if allocation fails.
 */
starmine_ad_eac3_renderer_714* starmine_ad_eac3_renderer_714_new(void);

/**
 * Destroy a renderer handle created by `starmine_ad_eac3_renderer_714_new()`.
 *
 * Passing `NULL` is allowed.
 */
void starmine_ad_eac3_renderer_714_free(
    starmine_ad_eac3_renderer_714* renderer);

/**
 * Clear all E-AC-3 renderer state.
 *
 * Call this after a seek or any other discontinuity before pushing more access
 * units.
 */
starmine_ad_status
starmine_ad_eac3_renderer_714_reset(starmine_ad_eac3_renderer_714* renderer);

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
starmine_ad_status starmine_ad_eac3_renderer_714_push_access_unit(
    starmine_ad_eac3_renderer_714* renderer, const uint8_t* data, size_t len,
    starmine_ad_eac3_access_unit_info* out_info,
    starmine_ad_render_714_frame* out_frame);

/**
 * Emit the final short limiter block after the last E-AC-3 input frame.
 *
 * `out_frame` is required and receives either a rendered tail block or an
 * empty frame with `has_frame == 0`.
 */
starmine_ad_status
starmine_ad_eac3_renderer_714_flush(starmine_ad_eac3_renderer_714* renderer,
                                    starmine_ad_render_714_frame* out_frame);

/**
 * Create a new TrueHD object decoder handle.
 *
 * Returns `NULL` only if allocation fails.
 */
starmine_ad_truehd_decoder* starmine_ad_truehd_decoder_new(void);

/**
 * Destroy a decoder handle created by `starmine_ad_truehd_decoder_new()`.
 *
 * Passing `NULL` is allowed.
 */
void starmine_ad_truehd_decoder_free(starmine_ad_truehd_decoder* decoder);

/**
 * Clear all TrueHD decoder state.
 *
 * Call this after a seek or any other discontinuity before pushing more access
 * units.
 */
starmine_ad_status
starmine_ad_truehd_decoder_reset(starmine_ad_truehd_decoder* decoder);

/**
 * Decode one complete TrueHD access unit into bed / object PCM.
 *
 * `data` must point to exactly one full access unit whose byte length matches
 * the access-unit length declared in the header.
 *
 * `out_info` is optional. Pass `NULL` if you only need the status code.
 * `out_frame` is optional. Pass `NULL` if you only want the decoder state to
 * advance.
 *
 * On success, `out_frame->has_frame` reports whether this access unit produced
 * a presentation-3 object frame.
 */
starmine_ad_status starmine_ad_truehd_decoder_push_access_unit(
    starmine_ad_truehd_decoder* decoder, const uint8_t* data, size_t len,
    starmine_ad_truehd_access_unit_info* out_info,
    starmine_ad_object_pcm_frame* out_frame);

/**
 * Create a new TrueHD 7.1.4 renderer handle.
 *
 * Returns `NULL` only if allocation fails.
 */
starmine_ad_truehd_renderer_714* starmine_ad_truehd_renderer_714_new(void);

/**
 * Destroy a renderer handle created by `starmine_ad_truehd_renderer_714_new()`.
 *
 * Passing `NULL` is allowed.
 */
void starmine_ad_truehd_renderer_714_free(
    starmine_ad_truehd_renderer_714* renderer);

/**
 * Clear all TrueHD renderer state.
 *
 * Call this after a seek or any other discontinuity before pushing more access
 * units.
 */
starmine_ad_status starmine_ad_truehd_renderer_714_reset(
    starmine_ad_truehd_renderer_714* renderer);

/**
 * Decode one complete TrueHD access unit and, when possible, render it to 7.1.4
 * float PCM.
 *
 * `data` must point to exactly one full access unit whose byte length matches
 * the access-unit length declared in the header.
 *
 * `out_info` is optional. Pass `NULL` if you only need the status code.
 * `out_frame` is optional. Pass `NULL` if you only want the decoder / renderer
 * state to advance.
 *
 * On success, `out_frame->has_frame` reports whether this access unit produced
 * a rendered frame.
 */
starmine_ad_status starmine_ad_truehd_renderer_714_push_access_unit(
    starmine_ad_truehd_renderer_714* renderer, const uint8_t* data, size_t len,
    starmine_ad_truehd_access_unit_info* out_info,
    starmine_ad_render_714_frame* out_frame);

/**
 * Emit the final short limiter block after the last TrueHD input frame.
 *
 * `out_frame` is required and receives either a rendered tail block or an
 * empty frame with `has_frame == 0`.
 */
starmine_ad_status
starmine_ad_truehd_renderer_714_flush(starmine_ad_truehd_renderer_714* renderer,
                                      starmine_ad_render_714_frame* out_frame);

/**
 * Convert a status code to a stable ASCII string.
 *
 * The returned pointer is owned by the library and must not be freed.
 */
const char* starmine_ad_status_string(starmine_ad_status status);

/**
 * Initialize an E-AC-3 info struct to zero / empty defaults.
 */
starmine_ad_status starmine_ad_eac3_access_unit_info_init(
    starmine_ad_eac3_access_unit_info* out_info);

/**
 * Initialize a TrueHD info struct to zero / empty defaults.
 */
starmine_ad_status starmine_ad_truehd_access_unit_info_init(
    starmine_ad_truehd_access_unit_info* out_info);

/**
 * Initialize an object-PCM frame struct to the empty / no-output state.
 */
starmine_ad_status
starmine_ad_object_pcm_frame_init(starmine_ad_object_pcm_frame* out_frame);

/**
 * Initialize a render frame struct to the empty / no-output state.
 */
starmine_ad_status
starmine_ad_render_714_frame_init(starmine_ad_render_714_frame* out_frame);

#ifdef __cplusplus
}
#endif
