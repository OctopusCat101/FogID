//! Audio watermark embedding / extraction via external FFmpeg.
//!
//! Uses the FFT spectral-band engine (fft.rs) for robust watermarking
//! that survives lossy codecs like AAC.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use bytemuck::{cast_slice, cast_slice_mut};

use super::core::{self, WatermarkParams};
use super::fft::{self, FRAME_SIZE};

/// Windows: hide console window for child processes.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Apply platform-specific flags to hide console windows.
#[cfg(windows)]
fn no_window(cmd: &mut Command) -> &mut Command {
    cmd.creation_flags(CREATE_NO_WINDOW)
}

#[cfg(not(windows))]
fn no_window(cmd: &mut Command) -> &mut Command {
    cmd
}

pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f64,
    pub codec_name: String,
    pub bit_rate: Option<u32>,
}

/// Determine FFmpeg encoder arguments based on input codec and file extension.
/// Returns (codec_args, output_extension).
fn encoder_args_for(info: &AudioInfo, input_path: &Path) -> (Vec<String>, String) {
    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Choose codec + default bitrate based on input format
    let (codec, bitrate, out_ext) = match ext.as_str() {
        "mp3" => ("libmp3lame", info.bit_rate.unwrap_or(192_000), "mp3"),
        "ogg" => ("libvorbis", info.bit_rate.unwrap_or(192_000), "ogg"),
        "flac" => ("flac", 0, "flac"),
        "wav" => ("pcm_s16le", 0, "wav"),
        "aac" | "m4a" => ("aac", info.bit_rate.unwrap_or(192_000), "m4a"),
        // Fallback: use AAC for unknown formats
        _ => ("aac", info.bit_rate.unwrap_or(192_000), "m4a"),
    };

    let mut args: Vec<String> = vec!["-c:a".into(), codec.into()];
    // Only add bitrate for lossy codecs
    if bitrate > 0 {
        let br_str = format!("{}k", bitrate / 1000);
        args.push("-b:a".into());
        args.push(br_str);
    }
    (args, out_ext.into())
}

pub fn probe_audio(path: &Path) -> Result<AudioInfo, String> {
    let out = no_window(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels,codec_name,bit_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(path),
    )
    .output()
    .map_err(|e| crate::i18n::tf("ffprobe_not_found", &e))?;

    if !out.status.success() {
        return Err(crate::i18n::tf("ffprobe_failed", &String::from_utf8_lossy(&out.stderr)));
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut sample_rate: Option<u32> = None;
    let mut channels: Option<u16> = None;
    let mut duration_secs: Option<f64> = None;
    let mut codec_name: Option<String> = None;
    let mut bit_rate: Option<u32> = None;

    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "sample_rate" => {
                if let Ok(v) = v.trim().parse::<u32>() {
                    sample_rate = Some(v);
                }
            }
            "channels" => {
                if let Ok(v) = v.trim().parse::<u16>() {
                    channels = Some(v);
                }
            }
            "duration" => {
                if let Ok(v) = v.trim().parse::<f64>() {
                    duration_secs = Some(v);
                }
            }
            "codec_name" => codec_name = Some(v.trim().to_owned()),
            "bit_rate" => {
                if bit_rate.is_none() {
                    bit_rate = v.trim().parse::<u32>().ok();
                }
            }
            _ => {}
        }
    }

    let sample_rate = sample_rate.ok_or_else(|| crate::i18n::t("parse_sample_rate").to_string())?;
    let channels = channels.ok_or_else(|| crate::i18n::t("parse_channels").to_string())?;
    let duration_secs = duration_secs.ok_or_else(|| crate::i18n::t("parse_duration").to_string())?;

    Ok(AudioInfo {
        sample_rate,
        channels,
        duration_secs,
        codec_name: codec_name.unwrap_or_else(|| "unknown".to_string()),
        bit_rate,
    })
}

fn format_audio_meta(info: &AudioInfo) -> String {
    let bit_rate = info
        .bit_rate
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!(
        "codec={}, bit_rate={}, duration={:.2}s",
        info.codec_name, bit_rate, info.duration_secs
    )
}

/// Read exactly `len` bytes. Returns `Ok(false)` on clean EOF before any data.
fn read_exact_or_eof(reader: &mut impl Read, buf: &mut [u8]) -> Result<bool, String> {
    let mut offset = 0usize;
    while offset < buf.len() {
        match reader.read(&mut buf[offset..]) {
            Ok(0) => {
                if offset == 0 {
                    return Ok(false);
                }
                return Ok(false);
            }
            Ok(n) => offset += n,
            Err(e) => return Err(crate::i18n::tf("read_audio_failed", &e)),
        }
    }
    Ok(true)
}

/// Downmix stereo interleaved samples to mono (average L+R).
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return interleaved.to_vec();
    }
    let mono_len = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(mono_len);
    for i in 0..mono_len {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += interleaved[i * channels + ch];
        }
        mono.push(sum / channels as f32);
    }
    mono
}

/// Embed watermark bits into an audio file and output AAC/M4A.
///
/// Uses FFT spectral-band modulation with overlap-add synthesis.
pub fn embed_audio(
    input: &Path,
    output: &Path,
    wm_bits: &[u8],
    params: &WatermarkParams,
    mut on_progress: impl FnMut(f32),
) -> Result<(), String> {
    if wm_bits.len() != super::text::FIXED_WM_BITS {
        return Err(crate::i18n::tf2("wm_bits_invalid", &wm_bits.len(), &super::text::FIXED_WM_BITS));
    }

    let info = probe_audio(input)?;
    if !(info.channels == 1 || info.channels == 2) {
        return Err(crate::i18n::tf2("unsupported_channels", &info.channels, &format_audio_meta(&info)));
    }

    let channels = info.channels as usize;
    let sample_rate = info.sample_rate;

    let shuffled = core::prepare_bits(wm_bits, params.seed_wm);

    // Decode all audio into memory (needed for overlap-add)
    let mut decoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(input)
            .args([
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                &channels.to_string(),
                "-ar",
                &sample_rate.to_string(),
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .spawn()
    .map_err(|e| crate::i18n::tf("ffmpeg_audio_decoder_start", &e))?;

    let mut dec_out = BufReader::new(decoder.stdout.take().unwrap());
    let mut all_interleaved: Vec<f32> = Vec::new();
    let chunk_size = FRAME_SIZE * channels;
    let mut chunk = vec![0.0f32; chunk_size];

    while read_exact_or_eof(&mut dec_out, cast_slice_mut::<f32, u8>(&mut chunk))? {
        all_interleaved.extend_from_slice(&chunk);
    }

    let dec_status = decoder.wait().map_err(|e| crate::i18n::tf("wait_decoder_failed", &e))?;
    if !dec_status.success() {
        return Err(crate::i18n::t("ffmpeg_audio_decode_failed").into());
    }

    if all_interleaved.len() < chunk_size {
        return Err(crate::i18n::t("audio_too_short").into());
    }

    on_progress(0.05);

    let mono_samples = downmix_to_mono(&all_interleaved, channels);

    // Embed via overlap-add FFT engine
    let watermarked_mono = fft::embed_overlap_add(&mono_samples, &shuffled, params.seed_wm);

    on_progress(0.70);

    // Compute delta (watermarked - original) and apply to all channels
    let mut output_interleaved = all_interleaved.clone();
    let mono_len = mono_samples.len();
    for i in 0..mono_len {
        let delta = watermarked_mono[i] - mono_samples[i];
        for ch in 0..channels {
            let idx = i * channels + ch;
            if idx < output_interleaved.len() {
                output_interleaved[idx] =
                    (output_interleaved[idx] + delta).clamp(-1.0, 1.0);
            }
        }
    }

    on_progress(0.80);

    // Encode output – match input codec/format
    let (codec_args, _) = encoder_args_for(&info, input);
    let mut encoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y"])
            .args([
                "-f",
                "f32le",
                "-ar",
                &sample_rate.to_string(),
                "-ac",
                &channels.to_string(),
                "-i",
                "pipe:0",
            ])
            .args(&codec_args)
            .arg(output),
    )
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .spawn()
    .map_err(|e| crate::i18n::tf("ffmpeg_audio_encoder_start", &e))?;

    let mut enc_in = BufWriter::new(encoder.stdin.take().unwrap());

    // Write in chunks for progress tracking
    let total_bytes = output_interleaved.len() * std::mem::size_of::<f32>();
    let output_bytes = cast_slice::<f32, u8>(&output_interleaved);
    let write_chunk_size = FRAME_SIZE * channels * std::mem::size_of::<f32>();
    let mut written = 0usize;

    while written < total_bytes {
        let end = (written + write_chunk_size).min(total_bytes);
        enc_in
            .write_all(&output_bytes[written..end])
            .map_err(|e| crate::i18n::tf("write_audio_encoder", &e))?;
        written = end;
        let frac = 0.80 + 0.15 * (written as f32 / total_bytes as f32);
        on_progress(frac.min(0.95));
    }

    enc_in
        .flush()
        .map_err(|e| crate::i18n::tf("flush_audio_encoder", &e))?;
    drop(enc_in);

    let enc_status = encoder.wait().map_err(|e| crate::i18n::tf("wait_encoder_failed", &e))?;
    if !enc_status.success() {
        return Err(crate::i18n::t("ffmpeg_audio_encode_failed").into());
    }

    let _ = quick_verify_embedded_audio(output, params);
    on_progress(1.0);
    Ok(())
}

/// Extract watermark text from an audio file.
///
/// Buffers the entire decoded audio, then extracts soft decisions
/// using the FFT spectral-band engine. Searches multiple sample
/// offsets to handle trim-induced misalignment.
pub fn extract_audio(
    input: &Path,
    params: &WatermarkParams,
    mut on_progress: impl FnMut(f32),
) -> Result<String, String> {
    let info = probe_audio(input)?;
    if !(info.channels == 1 || info.channels == 2) {
        return Err(crate::i18n::tf2("unsupported_channels", &info.channels, &format_audio_meta(&info)));
    }

    let channels = info.channels as usize;
    let sample_rate = info.sample_rate;

    // Decode all audio into memory
    let mut decoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(input)
            .args([
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                &channels.to_string(),
                "-ar",
                &sample_rate.to_string(),
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .spawn()
    .map_err(|e| crate::i18n::tf("ffmpeg_audio_decoder_start", &e))?;

    let mut dec_out = BufReader::new(decoder.stdout.take().unwrap());
    let mut all_interleaved: Vec<f32> = Vec::new();
    let chunk_size = FRAME_SIZE * channels;
    let mut chunk = vec![0.0f32; chunk_size];

    while read_exact_or_eof(&mut dec_out, cast_slice_mut::<f32, u8>(&mut chunk))? {
        all_interleaved.extend_from_slice(&chunk);
    }

    let dec_status = decoder.wait().map_err(|e| crate::i18n::tf("wait_decoder_failed", &e))?;
    if !dec_status.success() {
        return Err(crate::i18n::t("ffmpeg_audio_decode_failed").into());
    }

    if all_interleaved.len() < chunk_size {
        return Err(crate::i18n::t("audio_too_short").into());
    }

    on_progress(0.05);

    let mono_samples = downmix_to_mono(&all_interleaved, channels);

    let wm_bit_count = super::text::FIXED_WM_BITS;

    // Fast path: exact alignment (offset=0, frame_offset=0)
    if let Some(text) = try_extract_at_offset(&mono_samples, 0, wm_bit_count, params) {
        on_progress(1.0);
        return Ok(text);
    }

    on_progress(0.08);

    // Comprehensive alignment search:
    //   sample_offset: handles sub-frame misalignment from codec padding / non-aligned trim
    //   frame_offset:  handles cyclic bit-mapping shift when whole frames are lost
    //
    // For each sample_offset we precompute FFT spectra once, then sweep all
    // frame_offsets cheaply (no extra FFTs).
    let bands = fft::precompute_bands(wm_bit_count, params.seed_wm);
    let sample_step = 64;
    let num_sample_offsets = FRAME_SIZE / sample_step; // 16
    let total_iters = num_sample_offsets * wm_bit_count;
    let mut iter_count = 0usize;

    for so_idx in 0..num_sample_offsets {
        let sample_offset = so_idx * sample_step;
        if sample_offset >= mono_samples.len() {
            break;
        }
        let samples = &mono_samples[sample_offset..];
        if samples.len() < FRAME_SIZE {
            break;
        }

        let spectra = fft::precompute_frame_spectra(samples);

        for frame_offset in 0..wm_bit_count {
            let soft = fft::soft_decisions_from_spectra(&spectra, &bands, frame_offset);
            let unshuffled = core::unshuffle_f64(&soft, params.seed_wm);
            let candidate = fft::soft_to_hard_bits(&unshuffled);

            if super::text::check_watermark_magic(&candidate) {
                if let Ok(text) = super::text::bits_to_text(&candidate) {
                    on_progress(1.0);
                    return Ok(text);
                }
            }

            iter_count += 1;
            if iter_count % 256 == 0 {
                on_progress(0.08 + 0.87 * (iter_count as f32 / total_iters as f32));
            }
        }
    }

    on_progress(1.0);
    Err(crate::i18n::t("audio_no_watermark").into())
}

/// Try extracting watermark at a specific sample offset.
fn try_extract_at_offset(
    mono_samples: &[f32],
    offset: usize,
    wm_bit_count: usize,
    params: &WatermarkParams,
) -> Option<String> {
    if offset >= mono_samples.len() {
        return None;
    }

    let samples = &mono_samples[offset..];
    if samples.len() < FRAME_SIZE {
        return None;
    }

    let soft = fft::extract_soft_decisions(samples, wm_bit_count, params.seed_wm);

    // Unshuffle to recover original bit order
    let unshuffled = core::unshuffle_f64(&soft, params.seed_wm);
    let candidate = fft::soft_to_hard_bits(&unshuffled);

    if super::text::check_watermark_magic(&candidate) {
        if let Ok(text) = super::text::bits_to_text(&candidate) {
            return Some(text);
        }
    }

    None
}

fn quick_verify_embedded_audio(output: &Path, params: &WatermarkParams) -> Result<(), String> {
    match extract_audio(output, params, |_| {}) {
        Ok(text) => {
            eprintln!("{}", crate::i18n::tf("audio_verify_ok", &text));
            Ok(())
        }
        Err(e) => {
            eprintln!("{}", crate::i18n::tf("audio_verify_fail", &e));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg").arg("-version").output().is_ok()
            && Command::new("ffprobe").arg("-version").output().is_ok()
    }

    #[test]
    fn test_downmix_to_mono() {
        let stereo = vec![0.5f32, -0.3, 0.2, 0.4, -0.1, 0.6];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.1).abs() < 1e-6);
        assert!((mono[1] - 0.3).abs() < 1e-6);
        assert!((mono[2] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_mono_passthrough() {
        let mono = vec![0.1f32, 0.2, 0.3];
        let result = downmix_to_mono(&mono, 1);
        assert_eq!(result, mono);
    }

    #[test]
    #[ignore]
    fn ffmpeg_roundtrip_embed_extract_wav() {
        if !ffmpeg_available() {
            return;
        }

        let base = std::env::temp_dir().join(format!(
            "fogid_audio_fft_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let input = base.join("input.wav");
        let output = base.join("output.m4a");
        let text = "FogID-FFT-Test";
        let params = WatermarkParams::from_password("test-pass");
        let wm_bits = super::super::text::text_to_bits(text).unwrap();

        let synth_status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=8",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(synth_status.success());

        embed_audio(&input, &output, &wm_bits, &params, |_| {}).unwrap();
        let decoded = extract_audio(&output, &params, |_| {}).unwrap();
        assert_eq!(decoded, text);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    #[ignore]
    fn ffmpeg_extract_with_wrong_password_fails() {
        if !ffmpeg_available() {
            return;
        }

        let base = std::env::temp_dir().join(format!(
            "fogid_audio_fft_test_wrong_pw_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let input = base.join("input.wav");
        let output = base.join("output.m4a");
        let wm_bits = super::super::text::text_to_bits("WrongPassCase").unwrap();
        let params_ok = WatermarkParams::from_password("correct");
        let params_bad = WatermarkParams::from_password("wrong");

        let synth_status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=8",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&input)
            .status()
            .unwrap();
        assert!(synth_status.success());

        embed_audio(&input, &output, &wm_bits, &params_ok, |_| {}).unwrap();
        assert!(extract_audio(&output, &params_bad, |_| {}).is_err());

        let _ = std::fs::remove_dir_all(&base);
    }
}
