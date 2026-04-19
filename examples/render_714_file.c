#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <libavcodec/avcodec.h>
#include <libavcodec/codec_id.h>
#include <libavformat/avformat.h>

#include "starmine_ad.h"

enum {
    WAV_HEADER_SIZE = 68,
    WAV_CHANNEL_MASK_714 = 0x02d63f,
};

struct wav_writer {
    const char *path;
    FILE *file;
    uint32_t sample_rate;
    size_t channel_count;
    uint64_t data_bytes;
    bool initialized;
};

struct progress_stats {
    double start_monotonic_seconds;
    double processed_audio_seconds;
};

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <input> <output.wav> [--stream-index N] [--limit N]\n",
            argv0);
}

static bool parse_int_arg(const char *flag, const char *value, int *out) {
    char *end = NULL;
    long parsed = 0;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno != 0 || !end || *end != '\0') {
        fprintf(stderr, "%s expects an integer, got '%s'\n", flag, value);
        return false;
    }

    *out = (int)parsed;
    return true;
}

static double ts_to_seconds(int64_t ts, AVRational time_base) {
    if (ts == AV_NOPTS_VALUE)
        return -1.0;
    return ts * av_q2d(time_base);
}

static double monotonic_seconds(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void progress_stats_init(struct progress_stats *stats) {
    stats->start_monotonic_seconds = monotonic_seconds();
    stats->processed_audio_seconds = 0.0;
}

static void progress_stats_add_access_unit(
    struct progress_stats *stats, const starmine_ad_access_unit_info *info) {
    if (!stats || !info || info->sample_rate == 0 || info->num_blocks == 0)
        return;

    stats->processed_audio_seconds +=
        (256.0 * (double)info->num_blocks) / (double)info->sample_rate;
}

static void format_media_time(double seconds, char *buffer, size_t buffer_len) {
    unsigned long long total_centiseconds = 0;
    unsigned long long hours = 0;
    unsigned long long minutes = 0;
    unsigned long long secs = 0;
    unsigned long long centiseconds = 0;

    if (seconds > 0.0)
        total_centiseconds = (unsigned long long)llround(seconds * 100.0);

    hours = total_centiseconds / 360000ull;
    minutes = (total_centiseconds / 6000ull) % 60ull;
    secs = (total_centiseconds / 100ull) % 60ull;
    centiseconds = total_centiseconds % 100ull;

    snprintf(buffer, buffer_len, "%02llu:%02llu:%02llu.%02llu", hours, minutes,
             secs, centiseconds);
}

static void format_speed(const struct progress_stats *stats, char *buffer,
                         size_t buffer_len) {
    double elapsed = 0.0;
    double speed = 0.0;

    if (!stats) {
        snprintf(buffer, buffer_len, "N/A");
        return;
    }

    elapsed = monotonic_seconds() - stats->start_monotonic_seconds;
    if (elapsed <= 0.0 || stats->processed_audio_seconds <= 0.0) {
        snprintf(buffer, buffer_len, "N/A");
        return;
    }

    speed = stats->processed_audio_seconds / elapsed;
    if (!isfinite(speed)) {
        snprintf(buffer, buffer_len, "N/A");
        return;
    }

    snprintf(buffer, buffer_len, "%.2fx", speed);
}

static const char *bed_channel_name(starmine_ad_bed_channel channel) {
    switch (channel) {
    case STARMINE_AD_BED_CHANNEL_FRONT_LEFT:
        return "FL";
    case STARMINE_AD_BED_CHANNEL_FRONT_RIGHT:
        return "FR";
    case STARMINE_AD_BED_CHANNEL_CENTER:
        return "C";
    case STARMINE_AD_BED_CHANNEL_LOW_FREQUENCY_EFFECTS:
        return "LFE";
    case STARMINE_AD_BED_CHANNEL_SURROUND_LEFT:
        return "SL";
    case STARMINE_AD_BED_CHANNEL_SURROUND_RIGHT:
        return "SR";
    case STARMINE_AD_BED_CHANNEL_REAR_LEFT:
        return "RL";
    case STARMINE_AD_BED_CHANNEL_REAR_RIGHT:
        return "RR";
    case STARMINE_AD_BED_CHANNEL_TOP_FRONT_LEFT:
        return "TFL";
    case STARMINE_AD_BED_CHANNEL_TOP_FRONT_RIGHT:
        return "TFR";
    case STARMINE_AD_BED_CHANNEL_TOP_SURROUND_LEFT:
        return "TSL";
    case STARMINE_AD_BED_CHANNEL_TOP_SURROUND_RIGHT:
        return "TSR";
    case STARMINE_AD_BED_CHANNEL_TOP_REAR_LEFT:
        return "TRL";
    case STARMINE_AD_BED_CHANNEL_TOP_REAR_RIGHT:
        return "TRR";
    case STARMINE_AD_BED_CHANNEL_WIDE_LEFT:
        return "WL";
    case STARMINE_AD_BED_CHANNEL_WIDE_RIGHT:
        return "WR";
    case STARMINE_AD_BED_CHANNEL_LOW_FREQUENCY_EFFECTS2:
        return "LFE2";
    case STARMINE_AD_BED_CHANNEL_UNKNOWN:
    default:
        return "?";
    }
}

static void print_channel_order(const starmine_ad_render_714_frame *frame) {
    printf("render714_layout=");
    for (size_t i = 0; i < frame->channel_count; i++) {
        printf("%s%s", i == 0 ? "" : ",",
               bed_channel_name(frame->channel_order[i]));
    }
    printf("\n");
}

static bool write_bytes(FILE *file, const void *data, size_t len) {
    return fwrite(data, 1, len, file) == len;
}

static bool write_u16_le(FILE *file, uint16_t value) {
    unsigned char bytes[2] = {
        (unsigned char)(value & 0xff),
        (unsigned char)((value >> 8) & 0xff),
    };
    return write_bytes(file, bytes, sizeof(bytes));
}

static bool write_u32_le(FILE *file, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xff),
        (unsigned char)((value >> 8) & 0xff),
        (unsigned char)((value >> 16) & 0xff),
        (unsigned char)((value >> 24) & 0xff),
    };
    return write_bytes(file, bytes, sizeof(bytes));
}

static bool write_f32_le(FILE *file, float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return write_u32_le(file, bits);
}

static bool wav_writer_open(struct wav_writer *writer, const char *path) {
    memset(writer, 0, sizeof(*writer));
    writer->path = path;
    writer->file = fopen(path, "wb");
    if (!writer->file) {
        fprintf(stderr, "failed to open output '%s'\n", path);
        return false;
    }
    return true;
}

static bool wav_writer_write_frame(struct wav_writer *writer,
                                   const starmine_ad_render_714_frame *frame) {
    if (!frame->has_frame)
        return true;

    if (frame->channel_count != STARMINE_AD_RENDER_714_CHANNEL_COUNT) {
        fprintf(stderr,
                "unexpected render channel count: expected %u got %zu\n",
                STARMINE_AD_RENDER_714_CHANNEL_COUNT, frame->channel_count);
        return false;
    }

    if (!writer->initialized) {
        writer->sample_rate = frame->sample_rate;
        writer->channel_count = frame->channel_count;
        if (fseek(writer->file, WAV_HEADER_SIZE, SEEK_SET) != 0) {
            fprintf(stderr, "failed to seek '%s'\n", writer->path);
            return false;
        }
        writer->initialized = true;
    } else if (writer->sample_rate != frame->sample_rate ||
               writer->channel_count != frame->channel_count) {
        fprintf(
            stderr,
            "render format changed mid-stream in '%s': expected %u Hz / %zu "
            "ch, got %u Hz / %zu ch\n",
            writer->path, writer->sample_rate, writer->channel_count,
            frame->sample_rate, frame->channel_count);
        return false;
    }

    for (size_t sample_index = 0; sample_index < frame->samples_per_channel;
         sample_index++) {
        for (size_t channel_index = 0; channel_index < frame->channel_count;
             channel_index++) {
            if (!write_f32_le(writer->file,
                              frame->channels[channel_index][sample_index])) {
                fprintf(stderr, "failed to write '%s'\n", writer->path);
                return false;
            }
            writer->data_bytes += sizeof(float);
        }
    }

    return true;
}

static bool wav_writer_finalize(struct wav_writer *writer) {
    static const unsigned char ieee_float_subformat[16] = {
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
    };
    uint16_t channel_count = 0;
    uint16_t bits_per_sample = 32;
    uint16_t block_align = 0;
    uint32_t byte_rate = 0;
    uint32_t riff_size = 0;
    uint32_t data_size = 0;

    if (!writer->file)
        return true;
    if (!writer->initialized)
        return true;
    if (writer->channel_count > UINT16_MAX) {
        fprintf(stderr, "channel count is too large for WAV: %zu\n",
                writer->channel_count);
        return false;
    }
    if (writer->data_bytes > UINT32_MAX) {
        fprintf(stderr, "WAV data chunk is too large for '%s'\n", writer->path);
        return false;
    }

    channel_count = (uint16_t)writer->channel_count;
    block_align = (uint16_t)(channel_count * sizeof(float));
    byte_rate = writer->sample_rate * (uint32_t)block_align;
    data_size = (uint32_t)writer->data_bytes;
    if (writer->data_bytes > UINT32_MAX - (WAV_HEADER_SIZE - 8)) {
        fprintf(stderr, "RIFF chunk is too large for '%s'\n", writer->path);
        return false;
    }
    riff_size = (uint32_t)(writer->data_bytes + (WAV_HEADER_SIZE - 8));

    if (fseek(writer->file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "failed to seek '%s'\n", writer->path);
        return false;
    }

    if (!write_bytes(writer->file, "RIFF", 4) ||
        !write_u32_le(writer->file, riff_size) ||
        !write_bytes(writer->file, "WAVE", 4) ||
        !write_bytes(writer->file, "fmt ", 4) ||
        !write_u32_le(writer->file, 40) ||
        !write_u16_le(writer->file, 0xfffe) ||
        !write_u16_le(writer->file, channel_count) ||
        !write_u32_le(writer->file, writer->sample_rate) ||
        !write_u32_le(writer->file, byte_rate) ||
        !write_u16_le(writer->file, block_align) ||
        !write_u16_le(writer->file, bits_per_sample) ||
        !write_u16_le(writer->file, 22) ||
        !write_u16_le(writer->file, bits_per_sample) ||
        !write_u32_le(writer->file, WAV_CHANNEL_MASK_714) ||
        !write_bytes(writer->file, ieee_float_subformat,
                     sizeof(ieee_float_subformat)) ||
        !write_bytes(writer->file, "data", 4) ||
        !write_u32_le(writer->file, data_size)) {
        fprintf(stderr, "failed to write WAV header '%s'\n", writer->path);
        return false;
    }

    return true;
}

static void wav_writer_close(struct wav_writer *writer) {
    if (!writer->file)
        return;
    fclose(writer->file);
    writer->file = NULL;
}

static bool process_access_unit(starmine_ad_renderer_714 *renderer,
                                struct wav_writer *writer, const uint8_t *data,
                                size_t len, int packet_index,
                                int access_unit_index, int64_t pts,
                                AVRational time_base, int *rendered_frames,
                                bool *printed_layout,
                                struct progress_stats *progress) {
    starmine_ad_access_unit_info info;
    starmine_ad_render_714_frame frame;
    starmine_ad_status status;
    char time_buf[32];
    char speed_buf[32];

    if (starmine_ad_access_unit_info_init(&info) != STARMINE_AD_STATUS_OK ||
        starmine_ad_render_714_frame_init(&frame) != STARMINE_AD_STATUS_OK) {
        fprintf(stderr, "failed to initialize render structs\n");
        return false;
    }

    status = starmine_ad_renderer_714_push_access_unit(renderer, data, len,
                                                       &info, &frame);
    if (status != STARMINE_AD_STATUS_OK) {
        fprintf(stderr,
                "render failed: packet=%d au=%d pts=%.6f size=%zu status=%s\n",
                packet_index, access_unit_index, ts_to_seconds(pts, time_base),
                len, starmine_ad_status_string(status));
        return false;
    }

    progress_stats_add_access_unit(progress, &info);
    format_media_time(progress->processed_audio_seconds, time_buf,
                      sizeof(time_buf));
    format_speed(progress, speed_buf, sizeof(speed_buf));

    printf("packet=%d au=%d pts=%.6f time=%s speed=%s frame_size=%u sr=%u "
           "joc=%u oamd=%u rendered=%u samples=%zu\n",
           packet_index, access_unit_index, ts_to_seconds(pts, time_base),
           time_buf, speed_buf, info.frame_size, info.sample_rate,
           info.joc_payload_count, info.oamd_payload_count, frame.has_frame,
           frame.samples_per_channel);

    if (frame.has_frame && !*printed_layout) {
        print_channel_order(&frame);
        *printed_layout = true;
    }

    if (frame.has_frame) {
        if (!wav_writer_write_frame(writer, &frame))
            return false;
        (*rendered_frames)++;
    }

    return true;
}

int main(int argc, char **argv) {
    const char *input = NULL;
    const char *output = NULL;
    AVFormatContext *fmt = NULL;
    AVPacket *pkt = NULL;
    AVCodecParserContext *parser = NULL;
    AVCodecContext *codec_ctx = NULL;
    const AVCodec *codec = NULL;
    starmine_ad_renderer_714 *renderer = NULL;
    struct wav_writer writer;
    struct progress_stats progress;
    int stream_index = -1;
    int limit = -1;
    int packet_index = 0;
    int access_unit_index = 0;
    int rendered_frames = 0;
    int result = 1;
    bool printed_layout = false;

    memset(&writer, 0, sizeof(writer));
    progress_stats_init(&progress);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--stream-index") == 0) {
            if (i + 1 >= argc ||
                !parse_int_arg(argv[i], argv[i + 1], &stream_index))
                goto done;
            i++;
        } else if (strcmp(argv[i], "--limit") == 0) {
            if (i + 1 >= argc || !parse_int_arg(argv[i], argv[i + 1], &limit))
                goto done;
            i++;
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            goto done;
        } else if (!input) {
            input = argv[i];
        } else if (!output) {
            output = argv[i];
        } else {
            usage(argv[0]);
            goto done;
        }
    }

    if (!input || !output) {
        usage(argv[0]);
        goto done;
    }

    av_log_set_level(AV_LOG_ERROR);

    if (!wav_writer_open(&writer, output))
        goto done;

    if (avformat_open_input(&fmt, input, NULL, NULL) < 0) {
        fprintf(stderr, "failed to open input: %s\n", input);
        goto done;
    }
    if (avformat_find_stream_info(fmt, NULL) < 0) {
        fprintf(stderr, "failed to read stream info\n");
        goto done;
    }

    if (stream_index < 0) {
        stream_index =
            av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
        if (stream_index < 0) {
            fprintf(stderr, "failed to find an audio stream\n");
            goto done;
        }
    }

    if (stream_index >= (int)fmt->nb_streams) {
        fprintf(stderr, "stream index %d is out of range\n", stream_index);
        goto done;
    }
    if (fmt->streams[stream_index]->codecpar->codec_id != AV_CODEC_ID_EAC3) {
        fprintf(stderr, "stream %d is not E-AC-3 (codec_id=%d)\n", stream_index,
                fmt->streams[stream_index]->codecpar->codec_id);
        goto done;
    }

    codec = avcodec_find_decoder(AV_CODEC_ID_EAC3);
    if (!codec) {
        fprintf(stderr,
                "failed to find libavcodec E-AC-3 parser/decoder metadata\n");
        goto done;
    }

    parser = av_parser_init(AV_CODEC_ID_EAC3);
    if (!parser) {
        fprintf(stderr, "failed to create E-AC-3 parser\n");
        goto done;
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        fprintf(stderr, "failed to allocate codec context\n");
        goto done;
    }
    if (avcodec_parameters_to_context(
            codec_ctx, fmt->streams[stream_index]->codecpar) < 0) {
        fprintf(stderr, "failed to copy codec parameters\n");
        goto done;
    }

    renderer = starmine_ad_renderer_714_new();
    if (!renderer) {
        fprintf(stderr, "failed to allocate starmine_ad renderer\n");
        goto done;
    }

    pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "failed to allocate packet\n");
        goto done;
    }

    printf("input=%s output=%s stream=%d codec=eac3 limit=%d\n", input, output,
           stream_index, limit);

    while (av_read_frame(fmt, pkt) >= 0) {
        AVStream *stream = NULL;
        const uint8_t *packet_data = NULL;
        int packet_size = 0;

        if (pkt->stream_index != stream_index) {
            av_packet_unref(pkt);
            continue;
        }

        stream = fmt->streams[stream_index];
        packet_data = pkt->data;
        packet_size = pkt->size;

        while (packet_size > 0) {
            uint8_t *access_unit = NULL;
            int access_unit_size = 0;
            int consumed = av_parser_parse2(
                parser, codec_ctx, &access_unit, &access_unit_size, packet_data,
                packet_size, pkt->pts, pkt->dts, pkt->pos);
            if (consumed < 0) {
                fprintf(stderr, "libav parser failed on packet %d\n",
                        packet_index);
                av_packet_unref(pkt);
                goto done;
            }

            packet_data += consumed;
            packet_size -= consumed;

            if (access_unit_size > 0) {
                if (!process_access_unit(renderer, &writer, access_unit,
                                         (size_t)access_unit_size, packet_index,
                                         access_unit_index, pkt->pts,
                                         stream->time_base, &rendered_frames,
                                         &printed_layout, &progress)) {
                    av_packet_unref(pkt);
                    goto done;
                }
                access_unit_index++;
                if (limit >= 0 && access_unit_index >= limit) {
                    av_packet_unref(pkt);
                    goto finalize;
                }
            }

            if (consumed == 0 && access_unit_size == 0) {
                fprintf(stderr, "libav parser made no progress on packet %d\n",
                        packet_index);
                av_packet_unref(pkt);
                goto done;
            }
        }

        packet_index++;
        av_packet_unref(pkt);
    }

    while (1) {
        uint8_t *access_unit = NULL;
        int access_unit_size = 0;
        int consumed =
            av_parser_parse2(parser, codec_ctx, &access_unit, &access_unit_size,
                             NULL, 0, AV_NOPTS_VALUE, AV_NOPTS_VALUE, -1);
        if (consumed < 0) {
            fprintf(stderr, "libav parser flush failed\n");
            goto done;
        }
        if (access_unit_size <= 0)
            break;

        if (!process_access_unit(renderer, &writer, access_unit,
                                 (size_t)access_unit_size, packet_index,
                                 access_unit_index, AV_NOPTS_VALUE,
                                 fmt->streams[stream_index]->time_base,
                                 &rendered_frames, &printed_layout,
                                 &progress)) {
            goto done;
        }
        access_unit_index++;
        if (limit >= 0 && access_unit_index >= limit)
            break;
    }

finalize:
    if (!wav_writer_finalize(&writer))
        goto done;

    {
        char time_buf[32];
        char speed_buf[32];
        format_media_time(progress.processed_audio_seconds, time_buf,
                          sizeof(time_buf));
        format_speed(&progress, speed_buf, sizeof(speed_buf));
        printf("access_units=%d rendered_frames=%d time=%s speed=%s output=%s\n",
               access_unit_index, rendered_frames, time_buf, speed_buf, output);
    }
    if (rendered_frames == 0) {
        printf("no 7.1.4 frames were produced\n");
    }
    result = 0;

done:
    if (pkt)
        av_packet_free(&pkt);
    if (renderer)
        starmine_ad_renderer_714_free(renderer);
    if (codec_ctx)
        avcodec_free_context(&codec_ctx);
    if (parser)
        av_parser_close(parser);
    if (fmt)
        avformat_close_input(&fmt);
    wav_writer_close(&writer);
    return result;
}
