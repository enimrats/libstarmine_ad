mod raw_eac3;

use std::env;
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use raw_eac3::RawEac3FrameIter;
use starmine_ad::{
    BedChannel, CorePcmFrame, Decoder, JocObjectMatrices, ObjectPcmDecoder, ObjectPcmFrame, PcmDecoder,
    RENDER_714_CHANNEL_ORDER, Render714Frame, Render714TimeslotDebug, Renderer714,
};

#[derive(Debug, Clone)]
struct RunOptions {
    input: PathBuf,
    limit: Option<usize>,
    dump_frame_dir: Option<PathBuf>,
    dump_aux_dir: Option<PathBuf>,
    decode_core_check: bool,
    dump_core_f32: Option<PathBuf>,
    decode_objects_check: bool,
    dump_objects_f32: Option<PathBuf>,
    dump_joc_matrices_f32: Option<PathBuf>,
    render_714_check: bool,
    dump_render_714_wav: Option<PathBuf>,
    dump_render_positions_csv: Option<PathBuf>,
}

struct CoreDumpWriter {
    path: PathBuf,
    file: File,
    sample_rate: Option<u32>,
    channel_count: Option<usize>,
    channel_layout: Option<String>,
}

struct ObjectDumpWriter {
    path: PathBuf,
    files: Vec<File>,
    sample_rate: Option<u32>,
    object_count: Option<usize>,
}

struct JocMatrixDumpWriter {
    path: PathBuf,
    files: Vec<File>,
    sample_rate: Option<u32>,
    object_count: Option<usize>,
    channel_count: Option<usize>,
    timeslots_per_frame: Option<usize>,
}

struct Render714WavWriter {
    path: PathBuf,
    file: File,
    sample_rate: Option<u32>,
    data_bytes: u64,
}

struct RenderPositionsCsvWriter {
    path: PathBuf,
    file: File,
    timeslots_written: usize,
}

impl CoreDumpWriter {
    fn create(path: &Path) -> Result<Self, String> {
        let file = File::create(path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            sample_rate: None,
            channel_count: None,
            channel_layout: None,
        })
    }

    fn write_frame(&mut self, frame: &CorePcmFrame) -> Result<Option<String>, String> {
        let channel_layout = format_channel_order(frame);
        let channel_count = frame.total_channels();
        let first_layout = if self.sample_rate.is_none() {
            self.sample_rate = Some(frame.sample_rate);
            self.channel_count = Some(channel_count);
            self.channel_layout = Some(channel_layout.clone());
            Some(channel_layout)
        } else {
            if self.sample_rate != Some(frame.sample_rate) {
                return Err(format!(
                    "core PCM sample rate changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.sample_rate.unwrap_or_default(),
                    frame.sample_rate,
                ));
            }
            if self.channel_count != Some(channel_count) {
                return Err(format!(
                    "core PCM channel count changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.channel_count.unwrap_or_default(),
                    channel_count,
                ));
            }
            if self.channel_layout.as_deref() != Some(channel_layout.as_str()) {
                return Err(format!(
                    "core PCM channel layout changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.channel_layout.as_deref().unwrap_or("-"),
                    channel_layout,
                ));
            }
            None
        };

        for sample_index in 0..frame.samples_per_channel() {
            for channel in &frame.fullband_channels {
                self.file
                    .write_all(&channel[sample_index].to_le_bytes())
                    .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
            }
            if let Some(lfe) = &frame.lfe_channel {
                self.file
                    .write_all(&lfe[sample_index].to_le_bytes())
                    .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
            }
        }

        Ok(first_layout)
    }
}

impl ObjectDumpWriter {
    fn create(path: &Path) -> Result<Self, String> {
        fs::create_dir_all(path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            files: Vec::new(),
            sample_rate: None,
            object_count: None,
        })
    }

    fn write_frame(&mut self, frame: &ObjectPcmFrame) -> Result<Option<String>, String> {
        let object_count = frame.object_count();
        let first_summary = if self.sample_rate.is_none() {
            self.sample_rate = Some(frame.core.sample_rate);
            self.object_count = Some(object_count);
            self.ensure_files(object_count)?;
            Some(format!(
                "objects={} active={} sparse_active={}",
                object_count,
                frame.object_active.iter().filter(|active| **active).count(),
                frame
                    .joc
                    .objects
                    .iter()
                    .filter(|object| object.active && object.sparse_coded)
                    .count(),
            ))
        } else {
            if self.sample_rate != Some(frame.core.sample_rate) {
                return Err(format!(
                    "object PCM sample rate changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.sample_rate.unwrap_or_default(),
                    frame.core.sample_rate,
                ));
            }
            if self.object_count != Some(object_count) {
                return Err(format!(
                    "object PCM object count changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.object_count.unwrap_or_default(),
                    object_count,
                ));
            }
            None
        };

        for (index, channel) in frame.object_channels.iter().enumerate() {
            let mut bytes = Vec::with_capacity(channel.len() * std::mem::size_of::<f32>());
            for sample in channel {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            self.files[index]
                .write_all(&bytes)
                .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        }

        Ok(first_summary)
    }

    fn ensure_files(&mut self, object_count: usize) -> Result<(), String> {
        while self.files.len() < object_count {
            let path = self.path.join(format!("obj{:03}.f32", self.files.len()));
            let file = File::create(&path)
                .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
            self.files.push(file);
        }
        Ok(())
    }
}

impl JocMatrixDumpWriter {
    fn create(path: &Path) -> Result<Self, String> {
        fs::create_dir_all(path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            files: Vec::new(),
            sample_rate: None,
            object_count: None,
            channel_count: None,
            timeslots_per_frame: None,
        })
    }

    fn write_frame(
        &mut self,
        frame: &ObjectPcmFrame,
        matrices: &JocObjectMatrices,
    ) -> Result<Option<String>, String> {
        let object_count = matrices.len();
        let channel_count = frame.joc.channel_count;
        let timeslots_per_frame = frame.samples_per_channel() / 64;
        if object_count != frame.object_count() {
            return Err(format!(
                "JOC matrix object count changed in {}: expected {} got {}",
                self.path.display(),
                frame.object_count(),
                object_count,
            ));
        }

        for (object_index, object) in matrices.iter().enumerate() {
            if object.len() != timeslots_per_frame {
                return Err(format!(
                    "JOC matrix timeslot count mismatch in {} for object {}: expected {} got {}",
                    self.path.display(),
                    object_index,
                    timeslots_per_frame,
                    object.len(),
                ));
            }
            for (timeslot_index, matrix) in object.iter().enumerate() {
                if matrix.len() != channel_count {
                    return Err(format!(
                        "JOC matrix channel count mismatch in {} for object {} timeslot {}: expected {} got {}",
                        self.path.display(),
                        object_index,
                        timeslot_index,
                        channel_count,
                        matrix.len(),
                    ));
                }
            }
        }

        let first_summary = if self.sample_rate.is_none() {
            self.sample_rate = Some(frame.core.sample_rate);
            self.object_count = Some(object_count);
            self.channel_count = Some(channel_count);
            self.timeslots_per_frame = Some(timeslots_per_frame);
            self.ensure_files(object_count)?;
            fs::write(
                self.path.join("meta.txt"),
                format!(
                    "objects={object_count}\nchannels={channel_count}\ntimeslots_per_frame={timeslots_per_frame}\nsubbands=64\nsample_rate={}\n",
                    frame.core.sample_rate
                ),
            )
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
            Some(format!(
                "objects={object_count} channels={channel_count} timeslots={timeslots_per_frame}"
            ))
        } else {
            if self.sample_rate != Some(frame.core.sample_rate) {
                return Err(format!(
                    "JOC matrix sample rate changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.sample_rate.unwrap_or_default(),
                    frame.core.sample_rate,
                ));
            }
            if self.object_count != Some(object_count) {
                return Err(format!(
                    "JOC matrix object count changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.object_count.unwrap_or_default(),
                    object_count,
                ));
            }
            if self.channel_count != Some(channel_count) {
                return Err(format!(
                    "JOC matrix channel count changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.channel_count.unwrap_or_default(),
                    channel_count,
                ));
            }
            if self.timeslots_per_frame != Some(timeslots_per_frame) {
                return Err(format!(
                    "JOC matrix timeslot count changed mid-stream in {}: expected {} got {}",
                    self.path.display(),
                    self.timeslots_per_frame.unwrap_or_default(),
                    timeslots_per_frame,
                ));
            }
            None
        };

        for (index, object) in matrices.iter().enumerate() {
            let mut bytes =
                Vec::with_capacity(object.len() * channel_count * 64 * std::mem::size_of::<f32>());
            for matrix in object {
                for channel in matrix {
                    for sample in channel {
                        bytes.extend_from_slice(&sample.to_le_bytes());
                    }
                }
            }
            self.files[index]
                .write_all(&bytes)
                .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        }

        Ok(first_summary)
    }

    fn ensure_files(&mut self, object_count: usize) -> Result<(), String> {
        while self.files.len() < object_count {
            let path = self.path.join(format!("obj{:03}.mix.f32", self.files.len()));
            let file = File::create(&path)
                .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
            self.files.push(file);
        }
        Ok(())
    }
}

impl Render714WavWriter {
    const HEADER_SIZE: u64 = 68;
    const CHANNEL_MASK_714: u32 = 0x02d63f;

    fn create(path: &Path) -> Result<Self, String> {
        let file = File::create(path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            sample_rate: None,
            data_bytes: 0,
        })
    }

    fn write_frame(&mut self, frame: &Render714Frame) -> Result<Option<String>, String> {
        if self.sample_rate.is_none() {
            self.sample_rate = Some(frame.sample_rate);
            self.file
                .seek(SeekFrom::Start(Self::HEADER_SIZE))
                .map_err(|err| format!("failed to seek {}: {err}", self.path.display()))?;
        } else if self.sample_rate != Some(frame.sample_rate) {
            return Err(format!(
                "render 7.1.4 sample rate changed mid-stream in {}: expected {} got {}",
                self.path.display(),
                self.sample_rate.unwrap_or_default(),
                frame.sample_rate,
            ));
        }

        for sample_index in 0..frame.samples_per_channel() {
            for channel in &frame.channels {
                self.file
                    .write_all(&channel[sample_index].to_le_bytes())
                    .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
                self.data_bytes += std::mem::size_of::<f32>() as u64;
            }
        }

        Ok(
            if self.data_bytes
                == (frame.samples_per_channel()
                    * frame.channel_count()
                    * std::mem::size_of::<f32>()) as u64
            {
                Some(format_channel_names(&frame.channel_order))
            } else {
                None
            },
        )
    }

    fn finalize(&mut self) -> Result<(), String> {
        let Some(sample_rate) = self.sample_rate else {
            return Ok(());
        };

        let channel_count = RENDER_714_CHANNEL_ORDER.len() as u16;
        let bits_per_sample = 32u16;
        let block_align = channel_count * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let riff_size = self
            .data_bytes
            .checked_add(Self::HEADER_SIZE - 8)
            .ok_or_else(|| format!("wav size overflow for {}", self.path.display()))?;
        let data_size = u32::try_from(self.data_bytes)
            .map_err(|_| format!("data chunk too large for {}", self.path.display()))?;
        let riff_size = u32::try_from(riff_size)
            .map_err(|_| format!("RIFF too large for {}", self.path.display()))?;

        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|err| format!("failed to seek {}: {err}", self.path.display()))?;
        self.file
            .write_all(b"RIFF")
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&riff_size.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(b"WAVE")
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;

        self.file
            .write_all(b"fmt ")
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&40u32.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&0xfffeu16.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&channel_count.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&sample_rate.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&byte_rate.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&block_align.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&bits_per_sample.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&22u16.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&32u16.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&Self::CHANNEL_MASK_714.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&[
                0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38,
                0x9b, 0x71,
            ])
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;

        self.file
            .write_all(b"data")
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .write_all(&data_size.to_le_bytes())
            .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|err| format!("failed to seek {}: {err}", self.path.display()))?;
        Ok(())
    }
}

impl RenderPositionsCsvWriter {
    fn create(path: &Path) -> Result<Self, String> {
        let mut file = File::create(path)
            .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
        file.write_all(
            b"timeslot,sample_offset,object,static_channel,position_x,position_y,position_z,volume,size,lfe,position_valid\n",
        )
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            timeslots_written: 0,
        })
    }

    fn write_timeslots(
        &mut self,
        frame_sample_offset: usize,
        timeslots: &[Render714TimeslotDebug],
    ) -> Result<Option<String>, String> {
        let first_summary = if self.timeslots_written == 0 {
            let object_count = timeslots.first().map(|timeslot| timeslot.sources.len()).unwrap_or(0);
            Some(format!("timeslots={} objects={object_count}", timeslots.len()))
        } else {
            None
        };

        for timeslot in timeslots {
            let global_timeslot = self.timeslots_written;
            let global_sample_offset = frame_sample_offset + timeslot.sample_offset;
            for source in &timeslot.sources {
                writeln!(
                    self.file,
                    "{global_timeslot},{global_sample_offset},{},{},{},{},{},{},{},{},{}",
                    source.object_index,
                    source
                        .static_channel
                        .map(bed_channel_name)
                        .unwrap_or(""),
                    source.position.x,
                    source.position.y,
                    source.position.z,
                    source.gain,
                    source.size,
                    if source.lfe { 1 } else { 0 },
                    if source.position_valid { 1 } else { 0 },
                )
                .map_err(|err| format!("failed to write {}: {err}", self.path.display()))?;
            }
            self.timeslots_written += 1;
        }

        Ok(first_summary)
    }
}

fn usage(program: &str) {
    eprintln!(
        "usage: {program} <input.eac3|frame-dir> [--limit N] [--dump-frame-dir DIR] [--dump-aux-dir DIR] [--decode-core-check] [--dump-core-f32 PATH] [--decode-objects-check] [--dump-objects-f32 DIR] [--dump-joc-matrices-f32 DIR] [--render-714-check] [--dump-render-714-wav PATH] [--dump-render-positions-csv PATH]"
    );
}

fn parse_args() -> Result<RunOptions, String> {
    let mut args = env::args().skip(1);
    let mut input = None;
    let mut limit = None;
    let mut dump_frame_dir = None;
    let mut dump_aux_dir = None;
    let mut decode_core_check = false;
    let mut dump_core_f32 = None;
    let mut decode_objects_check = false;
    let mut dump_objects_f32 = None;
    let mut dump_joc_matrices_f32 = None;
    let mut render_714_check = false;
    let mut dump_render_714_wav = None;
    let mut dump_render_positions_csv = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let value = args.next().ok_or("--limit expects a number")?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --limit value: {value}"))?;
                limit = Some(parsed);
            }
            "--dump-frame-dir" | "--frame-dir" => {
                let value = args.next().ok_or("--dump-frame-dir expects a directory")?;
                dump_frame_dir = Some(PathBuf::from(value));
            }
            "--dump-aux-dir" => {
                let value = args.next().ok_or("--dump-aux-dir expects a directory")?;
                dump_aux_dir = Some(PathBuf::from(value));
            }
            "--decode-core-check" => {
                decode_core_check = true;
            }
            "--dump-core-f32" => {
                let value = args.next().ok_or("--dump-core-f32 expects a file path")?;
                dump_core_f32 = Some(PathBuf::from(value));
            }
            "--decode-objects-check" => {
                decode_objects_check = true;
            }
            "--dump-objects-f32" => {
                let value = args
                    .next()
                    .ok_or("--dump-objects-f32 expects a directory")?;
                dump_objects_f32 = Some(PathBuf::from(value));
            }
            "--dump-joc-matrices-f32" => {
                let value = args
                    .next()
                    .ok_or("--dump-joc-matrices-f32 expects a directory")?;
                dump_joc_matrices_f32 = Some(PathBuf::from(value));
            }
            "--render-714-check" => {
                render_714_check = true;
            }
            "--dump-render-714-wav" => {
                let value = args
                    .next()
                    .ok_or("--dump-render-714-wav expects a file path")?;
                dump_render_714_wav = Some(PathBuf::from(value));
            }
            "--dump-render-positions-csv" => {
                let value = args
                    .next()
                    .ok_or("--dump-render-positions-csv expects a file path")?;
                dump_render_positions_csv = Some(PathBuf::from(value));
            }
            _ if arg.starts_with('-') => return Err(format!("unknown flag: {arg}")),
            _ if input.is_none() => input = Some(PathBuf::from(arg)),
            _ => return Err("only one input path is supported".to_string()),
        }
    }

    input
        .map(|input| RunOptions {
            input,
            limit,
            dump_frame_dir,
            dump_aux_dir,
            decode_core_check,
            dump_core_f32,
            decode_objects_check,
            dump_objects_f32,
            dump_joc_matrices_f32,
            render_714_check,
            dump_render_714_wav,
            dump_render_positions_csv,
        })
        .ok_or("missing input.eac3 or frame-dir".to_string())
}

fn write_frame(output_dir: &Path, index: usize, bytes: &[u8]) -> Result<PathBuf, String> {
    let path = output_dir.join(format!("{index:06}.eac3"));
    fs::write(&path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    Ok(path)
}

fn write_aux(output_dir: &Path, index: usize, bytes: &[u8]) -> Result<PathBuf, String> {
    let path = output_dir.join(format!("{index:06}.aux.bin"));
    fs::write(&path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    Ok(path)
}

fn collect_frame_files(input_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    for entry in fs::read_dir(input_dir)
        .map_err(|err| format!("failed to read {}: {err}", input_dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read dir entry: {err}"))?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("eac3"))
        {
            files.push(path);
        }
    }

    files.sort();
    Ok(files)
}

fn format_channel_order(frame: &CorePcmFrame) -> String {
    let mut channels = frame
        .fullband_channel_order
        .iter()
        .map(|channel| bed_channel_name(*channel))
        .collect::<Vec<_>>();
    if frame.lfe_channel.is_some() {
        channels.push("LFE");
    }
    channels.join(",")
}

fn format_channel_names(channels: &[BedChannel]) -> String {
    channels
        .iter()
        .copied()
        .map(bed_channel_name)
        .collect::<Vec<_>>()
        .join(",")
}

fn bed_channel_name(channel: BedChannel) -> &'static str {
    match channel {
        BedChannel::FrontLeft => "FL",
        BedChannel::FrontRight => "FR",
        BedChannel::Center => "FC",
        BedChannel::LowFrequencyEffects => "LFE",
        BedChannel::SurroundLeft => "SL",
        BedChannel::SurroundRight => "SR",
        BedChannel::RearLeft => "RL",
        BedChannel::RearRight => "RR",
        BedChannel::TopFrontLeft => "TFL",
        BedChannel::TopFrontRight => "TFR",
        BedChannel::TopSurroundLeft => "TSL",
        BedChannel::TopSurroundRight => "TSR",
        BedChannel::TopRearLeft => "TRL",
        BedChannel::TopRearRight => "TRR",
        BedChannel::WideLeft => "WL",
        BedChannel::WideRight => "WR",
        BedChannel::LowFrequencyEffects2 => "LFE2",
    }
}

fn process_raw_eac3(input: &Path, bytes: &[u8], options: &RunOptions) -> ExitCode {
    let use_render_714 = options.render_714_check
        || options.dump_render_714_wav.is_some()
        || options.dump_render_positions_csv.is_some();
    let use_object_pcm =
        options.decode_objects_check
            || options.dump_objects_f32.is_some()
            || options.dump_joc_matrices_f32.is_some()
            || use_render_714;
    let use_core_pcm = options.decode_core_check || options.dump_core_f32.is_some();
    let mut summary_decoder = Decoder::new();
    let mut pcm_decoder = PcmDecoder::new();
    let mut object_decoder = ObjectPcmDecoder::new();
    let mut renderer_714 = Renderer714::new();
    let mut dump_writer = match options.dump_core_f32.as_deref() {
        Some(path) => match CoreDumpWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut object_dump_writer = match options.dump_objects_f32.as_deref() {
        Some(path) => match ObjectDumpWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut joc_matrix_dump_writer = match options.dump_joc_matrices_f32.as_deref() {
        Some(path) => match JocMatrixDumpWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut render_714_writer = match options.dump_render_714_wav.as_deref() {
        Some(path) => match Render714WavWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut render_positions_writer = match options.dump_render_positions_csv.as_deref() {
        Some(path) => match RenderPositionsCsvWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut frames = 0usize;

    println!(
        "input={} mode=raw-eac3 bytes={} limit={} dump_frame_dir={} dump_aux_dir={} decode_core={} dump_core_f32={} decode_objects={} dump_objects_f32={} dump_joc_matrices_f32={} render_714={} dump_render_714_wav={} dump_render_positions_csv={}",
        input.display(),
        bytes.len(),
        options
            .limit
            .map(|value| value.to_string())
            .unwrap_or_else(|| "all".to_string()),
        options
            .dump_frame_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        options
            .dump_aux_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        if use_core_pcm { 1 } else { 0 },
        options
            .dump_core_f32
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        if use_object_pcm { 1 } else { 0 },
        options
            .dump_objects_f32
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        options
            .dump_joc_matrices_f32
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        if use_render_714 { 1 } else { 0 },
        options
            .dump_render_714_wav
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        options
            .dump_render_positions_csv
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
    );

    for frame in RawEac3FrameIter::new(bytes) {
        let frame = match frame {
            Ok(frame) => frame,
            Err(err) => {
                eprintln!("split error after {} frames: {err}", frames);
                return ExitCode::FAILURE;
            }
        };

        if use_object_pcm {
            let result = match object_decoder.push_access_unit(frame.bytes) {
                Ok(Some(result)) => result,
                Ok(None) => {
                    eprintln!("object decoder found no JOC payload on frame {}", frames);
                    return ExitCode::FAILURE;
                }
                Err(err) => {
                    eprintln!("object decoder error on frame {}: {err}", frames);
                    return ExitCode::FAILURE;
                }
            };

            let (render_714, render_debug) = if use_render_714 {
                if render_positions_writer.is_some() {
                    match renderer_714.push_frame_with_debug(&result.pcm) {
                        Ok((rendered, debug)) => (Some(rendered), Some(debug)),
                        Err(err) => {
                            eprintln!("render 7.1.4 error on frame {}: {err}", frames);
                            return ExitCode::FAILURE;
                        }
                    }
                } else {
                    match renderer_714.push_frame(&result.pcm) {
                        Ok(rendered) => (Some(rendered), None),
                        Err(err) => {
                            eprintln!("render 7.1.4 error on frame {}: {err}", frames);
                            return ExitCode::FAILURE;
                        }
                    }
                }
            } else {
                (None, None)
            };

            println!(
                "frame={} offset={} frames_seen={} {} corepcm={}s/{}ch objectpcm={}obj/{}active/{}sparse{}",
                frames,
                frame.offset,
                result.frames_seen,
                result.info.summary(),
                result.pcm.core.samples_per_channel(),
                result.pcm.core.total_channels(),
                result.pcm.object_count(),
                result
                    .pcm
                    .object_active
                    .iter()
                    .filter(|active| **active)
                    .count(),
                result
                    .pcm
                    .joc
                    .objects
                    .iter()
                    .filter(|object| object.active && object.sparse_coded)
                    .count(),
                render_714
                    .as_ref()
                    .map(|rendered| format!(
                        " render714={}s/{}ch",
                        rendered.samples_per_channel(),
                        rendered.channel_count()
                    ))
                    .unwrap_or_default(),
            );

            if let Some(writer) = dump_writer.as_mut() {
                match writer.write_frame(&result.pcm.core) {
                    Ok(Some(layout)) => println!(
                        "corepcm_layout={} sample_rate={} path={}",
                        layout,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let Some(writer) = object_dump_writer.as_mut() {
                match writer.write_frame(&result.pcm) {
                    Ok(Some(summary)) => println!(
                        "objectpcm_summary={} sample_rate={} path={}",
                        summary,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let Some(writer) = joc_matrix_dump_writer.as_mut() {
                match writer.write_frame(&result.pcm, object_decoder.last_joc_matrices()) {
                    Ok(Some(summary)) => println!(
                        "joc_matrix_summary={} sample_rate={} path={}",
                        summary,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let (Some(writer), Some(rendered)) =
                (render_714_writer.as_mut(), render_714.as_ref())
            {
                match writer.write_frame(rendered) {
                    Ok(Some(layout)) => println!(
                        "render714_layout={} sample_rate={} path={}",
                        layout,
                        rendered.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let (Some(writer), Some(debug)) =
                (render_positions_writer.as_mut(), render_debug.as_deref())
            {
                let frame_sample_offset = frames * result.pcm.core.samples_per_channel();
                match writer.write_timeslots(frame_sample_offset, debug) {
                    Ok(Some(summary)) => println!(
                        "render714_positions_summary={} sample_rate={} path={}",
                        summary,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }

            if let Some(dir) = options.dump_frame_dir.as_deref() {
                if let Err(err) = write_frame(dir, frames, frame.bytes) {
                    eprintln!("{err}");
                    return ExitCode::FAILURE;
                }
            }
            if let Some(dir) = options.dump_aux_dir.as_deref() {
                if !result.info.aux_data.is_empty() {
                    if let Err(err) = write_aux(dir, frames, &result.info.aux_data) {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        } else if use_core_pcm {
            let result = match pcm_decoder.push_access_unit(frame.bytes) {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("PCM decoder error on frame {}: {err}", frames);
                    return ExitCode::FAILURE;
                }
            };

            println!(
                "frame={} offset={} frames_seen={} {} corepcm={}s/{}ch",
                frames,
                frame.offset,
                result.frames_seen,
                result.info.summary(),
                result.pcm.samples_per_channel(),
                result.pcm.total_channels(),
            );

            if let Some(writer) = dump_writer.as_mut() {
                match writer.write_frame(&result.pcm) {
                    Ok(Some(layout)) => println!(
                        "corepcm_layout={} sample_rate={} path={}",
                        layout,
                        result.pcm.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }

            if let Some(dir) = options.dump_frame_dir.as_deref() {
                if let Err(err) = write_frame(dir, frames, frame.bytes) {
                    eprintln!("{err}");
                    return ExitCode::FAILURE;
                }
            }
            if let Some(dir) = options.dump_aux_dir.as_deref() {
                if !result.info.aux_data.is_empty() {
                    if let Err(err) = write_aux(dir, frames, &result.info.aux_data) {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        } else {
            let result = match summary_decoder.push_access_unit(frame.bytes) {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("decoder error on frame {}: {err}", frames);
                    return ExitCode::FAILURE;
                }
            };

            println!(
                "frame={} offset={} frames_seen={} {}",
                frames,
                frame.offset,
                result.frames_seen,
                result.info.summary(),
            );

            if let Some(dir) = options.dump_frame_dir.as_deref() {
                if let Err(err) = write_frame(dir, frames, frame.bytes) {
                    eprintln!("{err}");
                    return ExitCode::FAILURE;
                }
            }
            if let Some(dir) = options.dump_aux_dir.as_deref() {
                if !result.info.aux_data.is_empty() {
                    if let Err(err) = write_aux(dir, frames, &result.info.aux_data) {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        }

        frames += 1;
        if options.limit.is_some_and(|limit| frames >= limit) {
            break;
        }
    }

    if let Some(writer) = render_714_writer.as_mut() {
        if let Err(err) = writer.finalize() {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    }

    println!("frames_dumped={frames}");
    ExitCode::SUCCESS
}

fn process_frame_dir(input_dir: &Path, options: &RunOptions) -> ExitCode {
    let files = match collect_frame_files(input_dir) {
        Ok(files) => files,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };
    let use_render_714 = options.render_714_check || options.dump_render_714_wav.is_some();
    let use_object_pcm =
        options.decode_objects_check
            || options.dump_objects_f32.is_some()
            || options.dump_joc_matrices_f32.is_some()
            || use_render_714;
    let use_core_pcm = options.decode_core_check || options.dump_core_f32.is_some();
    let mut summary_decoder = Decoder::new();
    let mut pcm_decoder = PcmDecoder::new();
    let mut object_decoder = ObjectPcmDecoder::new();
    let mut renderer_714 = Renderer714::new();
    let mut dump_writer = match options.dump_core_f32.as_deref() {
        Some(path) => match CoreDumpWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut object_dump_writer = match options.dump_objects_f32.as_deref() {
        Some(path) => match ObjectDumpWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut joc_matrix_dump_writer = match options.dump_joc_matrices_f32.as_deref() {
        Some(path) => match JocMatrixDumpWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut render_714_writer = match options.dump_render_714_wav.as_deref() {
        Some(path) => match Render714WavWriter::create(path) {
            Ok(writer) => Some(writer),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let mut frames = 0usize;

    println!(
        "input={} mode=frame-dir files={} limit={} dump_aux_dir={} decode_core={} dump_core_f32={} decode_objects={} dump_objects_f32={} dump_joc_matrices_f32={} render_714={} dump_render_714_wav={}",
        input_dir.display(),
        files.len(),
        options
            .limit
            .map(|value| value.to_string())
            .unwrap_or_else(|| "all".to_string()),
        options
            .dump_aux_dir
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        if use_core_pcm { 1 } else { 0 },
        options
            .dump_core_f32
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        if use_object_pcm { 1 } else { 0 },
        options
            .dump_objects_f32
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        options
            .dump_joc_matrices_f32
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
        if use_render_714 { 1 } else { 0 },
        options
            .dump_render_714_wav
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string()),
    );

    for path in files {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                eprintln!("failed to read {}: {err}", path.display());
                return ExitCode::FAILURE;
            }
        };

        if use_object_pcm {
            let result = match object_decoder.push_access_unit(&bytes) {
                Ok(Some(result)) => result,
                Ok(None) => {
                    eprintln!("object decoder found no JOC payload on {}", path.display());
                    return ExitCode::FAILURE;
                }
                Err(err) => {
                    eprintln!("object decoder error on {}: {err}", path.display());
                    return ExitCode::FAILURE;
                }
            };

            let render_714 = if use_render_714 {
                match renderer_714.push_frame(&result.pcm) {
                    Ok(rendered) => Some(rendered),
                    Err(err) => {
                        eprintln!("render 7.1.4 error on {}: {err}", path.display());
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                None
            };

            println!(
                "frame={} source={} frames_seen={} {} corepcm={}s/{}ch objectpcm={}obj/{}active/{}sparse{}",
                frames,
                path.display(),
                result.frames_seen,
                result.info.summary(),
                result.pcm.core.samples_per_channel(),
                result.pcm.core.total_channels(),
                result.pcm.object_count(),
                result
                    .pcm
                    .object_active
                    .iter()
                    .filter(|active| **active)
                    .count(),
                result
                    .pcm
                    .joc
                    .objects
                    .iter()
                    .filter(|object| object.active && object.sparse_coded)
                    .count(),
                render_714
                    .as_ref()
                    .map(|rendered| format!(
                        " render714={}s/{}ch",
                        rendered.samples_per_channel(),
                        rendered.channel_count()
                    ))
                    .unwrap_or_default(),
            );

            if let Some(writer) = dump_writer.as_mut() {
                match writer.write_frame(&result.pcm.core) {
                    Ok(Some(layout)) => println!(
                        "corepcm_layout={} sample_rate={} path={}",
                        layout,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let Some(writer) = object_dump_writer.as_mut() {
                match writer.write_frame(&result.pcm) {
                    Ok(Some(summary)) => println!(
                        "objectpcm_summary={} sample_rate={} path={}",
                        summary,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let Some(writer) = joc_matrix_dump_writer.as_mut() {
                match writer.write_frame(&result.pcm, object_decoder.last_joc_matrices()) {
                    Ok(Some(summary)) => println!(
                        "joc_matrix_summary={} sample_rate={} path={}",
                        summary,
                        result.pcm.core.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            if let (Some(writer), Some(rendered)) =
                (render_714_writer.as_mut(), render_714.as_ref())
            {
                match writer.write_frame(rendered) {
                    Ok(Some(layout)) => println!(
                        "render714_layout={} sample_rate={} path={}",
                        layout,
                        rendered.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }

            if let Some(dir) = options.dump_aux_dir.as_deref() {
                if !result.info.aux_data.is_empty() {
                    if let Err(err) = write_aux(dir, frames, &result.info.aux_data) {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        } else if use_core_pcm {
            let result = match pcm_decoder.push_access_unit(&bytes) {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("PCM decoder error on {}: {err}", path.display());
                    return ExitCode::FAILURE;
                }
            };

            println!(
                "frame={} source={} frames_seen={} {} corepcm={}s/{}ch",
                frames,
                path.display(),
                result.frames_seen,
                result.info.summary(),
                result.pcm.samples_per_channel(),
                result.pcm.total_channels(),
            );

            if let Some(writer) = dump_writer.as_mut() {
                match writer.write_frame(&result.pcm) {
                    Ok(Some(layout)) => println!(
                        "corepcm_layout={} sample_rate={} path={}",
                        layout,
                        result.pcm.sample_rate,
                        writer.path.display(),
                    ),
                    Ok(None) => {}
                    Err(err) => {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }

            if let Some(dir) = options.dump_aux_dir.as_deref() {
                if !result.info.aux_data.is_empty() {
                    if let Err(err) = write_aux(dir, frames, &result.info.aux_data) {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        } else {
            let result = match summary_decoder.push_access_unit(&bytes) {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("decoder error on {}: {err}", path.display());
                    return ExitCode::FAILURE;
                }
            };

            println!(
                "frame={} source={} frames_seen={} {}",
                frames,
                path.display(),
                result.frames_seen,
                result.info.summary(),
            );

            if let Some(dir) = options.dump_aux_dir.as_deref() {
                if !result.info.aux_data.is_empty() {
                    if let Err(err) = write_aux(dir, frames, &result.info.aux_data) {
                        eprintln!("{err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        }

        frames += 1;
        if options.limit.is_some_and(|limit| frames >= limit) {
            break;
        }
    }

    if let Some(writer) = render_714_writer.as_mut() {
        if let Err(err) = writer.finalize() {
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    }

    println!("frames_dumped={frames}");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let program = env::args()
        .next()
        .unwrap_or_else(|| "starmine-ad-cli".to_string());

    let options = match parse_args() {
        Ok(options) => options,
        Err(err) => {
            usage(&program);
            eprintln!("{err}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(dir) = &options.dump_frame_dir {
        if let Err(err) = fs::create_dir_all(dir) {
            eprintln!("failed to create {}: {err}", dir.display());
            return ExitCode::FAILURE;
        }
    }
    if let Some(dir) = &options.dump_aux_dir {
        if let Err(err) = fs::create_dir_all(dir) {
            eprintln!("failed to create {}: {err}", dir.display());
            return ExitCode::FAILURE;
        }
    }
    if let Some(path) = &options.dump_core_f32 {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if let Err(err) = fs::create_dir_all(parent) {
                eprintln!("failed to create {}: {err}", parent.display());
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(path) = &options.dump_objects_f32 {
        if let Err(err) = fs::create_dir_all(path) {
            eprintln!("failed to create {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    }
    if let Some(path) = &options.dump_joc_matrices_f32 {
        if let Err(err) = fs::create_dir_all(path) {
            eprintln!("failed to create {}: {err}", path.display());
            return ExitCode::FAILURE;
        }
    }
    if let Some(path) = &options.dump_render_714_wav {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            if let Err(err) = fs::create_dir_all(parent) {
                eprintln!("failed to create {}: {err}", parent.display());
                return ExitCode::FAILURE;
            }
        }
    }

    if options.input.is_dir() {
        if options.dump_frame_dir.is_some() {
            eprintln!("--dump-frame-dir is only valid for raw .eac3 input");
            return ExitCode::FAILURE;
        }
        return process_frame_dir(&options.input, &options);
    }

    let bytes = match fs::read(&options.input) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("failed to read {}: {err}", options.input.display());
            return ExitCode::FAILURE;
        }
    };

    process_raw_eac3(&options.input, &bytes, &options)
}
