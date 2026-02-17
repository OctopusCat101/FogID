//! Video watermark embedding / extraction via external FFmpeg.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use image::RgbImage;

use super::core::{self, WatermarkParams};

const PROGRESS_UPDATE_INTERVAL: u64 = 10;
const EARLY_DECODE_INTERVAL: u64 = 5;

/// Determine FFmpeg video encoder arguments based on output file extension.
/// Returns (video_codec_args, audio_codec_args).
fn video_encoder_args(output: &Path) -> (Vec<&'static str>, Vec<&'static str>) {
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "webm" => (
            vec!["-c:v", "libvpx-vp9", "-crf", "18", "-b:v", "0", "-pix_fmt", "yuv420p"],
            vec!["-c:a", "libopus", "-b:a", "128k"],
        ),
        // mp4, mkv, avi, mov, flv and others: use H.264
        _ => (
            vec!["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-x264-params", "aq-mode=0"],
            vec!["-c:a", "copy"],
        ),
    }
}

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

// ---------------------------------------------------------------------------
// Video probing
// ---------------------------------------------------------------------------

pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps: String, // e.g. "30/1"
}

pub fn probe(path: &Path) -> Result<VideoInfo, String> {
    let out = no_window(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(path),
    )
    .output()
    .map_err(|e| format!("无法运行 ffprobe: {e}\n请确保已安装 FFmpeg 并加入系统 PATH"))?;

    if !out.status.success() {
        return Err(format!(
            "ffprobe 失败: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.trim().lines().collect();
    if lines.len() < 4 {
        return Err(format!("ffprobe 输出格式异常: {text}"));
    }

    let width: u32 = lines[0].trim().parse().map_err(|_| "无法解析视频宽度")?;
    let height: u32 = lines[1].trim().parse().map_err(|_| "无法解析视频高度")?;
    let fps = lines[2].trim().to_string();
    let frame_count = lines[3]
        .trim()
        .parse::<u64>()
        .unwrap_or_else(|_| estimate_frames(path, &fps).unwrap_or(0));

    Ok(VideoInfo {
        width,
        height,
        frame_count,
        fps,
    })
}

fn estimate_frames(path: &Path, fps_str: &str) -> Option<u64> {
    let out = no_window(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(path),
    )
    .output()
    .ok()?;
    let dur: f64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    let fps = parse_fps(fps_str)?;
    Some((dur * fps).ceil() as u64)
}

fn parse_fps(s: &str) -> Option<f64> {
    if let Some((n, d)) = s.split_once('/') {
        let num: f64 = n.trim().parse().ok()?;
        let den: f64 = d.trim().parse().ok()?;
        if den > 0.0 { Some(num / den) } else { None }
    } else {
        s.trim().parse().ok()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read exactly `buf.len()` bytes. Returns `Ok(false)` on clean EOF (0 bytes).
fn read_frame(r: &mut impl Read, buf: &mut [u8]) -> Result<bool, String> {
    let mut pos = 0;
    while pos < buf.len() {
        match r.read(&mut buf[pos..]) {
            Ok(0) if pos == 0 => return Ok(false),
            Ok(0) => return Err("不完整的帧数据".into()),
            Ok(n) => pos += n,
            Err(e) => return Err(format!("读取帧失败: {e}")),
        }
    }
    Ok(true)
}

/// Quick post-embed verification: decode a few frames and attempt CRC-validated recovery.
fn quick_verify_embedded_video(
    output: &Path,
    wm_bit_count: usize,
    params: &WatermarkParams,
) -> Result<(), String> {
    let info = probe(output)?;
    let frame_size = (info.width * info.height * 3) as usize;

    let mut decoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(output)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .spawn()
    .map_err(|e| format!("FFmpeg 自校验解码器启动失败: {e}"))?;

    let mut dec_out = BufReader::new(decoder.stdout.take().unwrap());
    let mut buf = vec![0u8; frame_size];
    let mut accumulated = vec![0.0f64; wm_bit_count];
    let mut accumulated_weight = vec![0.0f64; wm_bit_count];
    let mut used = 0u64;

    const MAX_VERIFY_FRAMES: u64 = 30;
    while used < MAX_VERIFY_FRAMES {
        if !read_frame(&mut dec_out, &mut buf)? {
            break;
        }

        let frame_buf = std::mem::take(&mut buf);
        let img = RgbImage::from_raw(info.width, info.height, frame_buf)
            .ok_or("自校验帧数据大小不匹配")?;
        let (extracted, confidence) =
            core::extract_video_frame_with_confidence(&img, wm_bit_count, params)?;
        for i in 0..wm_bit_count {
            let w = confidence[i];
            accumulated[i] += extracted[i] as f64 * w;
            accumulated_weight[i] += w;
        }
        buf = img.into_raw();
        used += 1;

        if used >= 3 {
            let avg_bits: Vec<u8> = accumulated
                .iter()
                .zip(accumulated_weight.iter())
                .map(|(&v, &w)| {
                    if w > 0.0 {
                        (v / w).round().clamp(0.0, 255.0) as u8
                    } else {
                        0u8
                    }
                })
                .collect();
            if super::text::bits_to_text(&avg_bits).is_ok() {
                let _ = decoder.kill();
                let _ = decoder.wait();
                return Ok(());
            }
        }
    }

    let _ = decoder.kill();
    let _ = decoder.wait();

    // Final decode attempt after all accumulated frames
    if used > 0 {
        let avg_bits: Vec<u8> = accumulated
            .iter()
            .zip(accumulated_weight.iter())
            .map(|(&v, &w)| {
                if w > 0.0 {
                    (v / w).round().clamp(0.0, 255.0) as u8
                } else {
                    0u8
                }
            })
            .collect();
        if super::text::bits_to_text(&avg_bits).is_ok() {
            return Ok(());
        }
    }

    Err("视频已导出，但自校验未通过（建议提高码率或重试）".into())
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

/// Embed a watermark into every frame of a video file.
///
/// Uses two FFmpeg child processes (decoder → Rust → encoder).
/// Audio is copied from the original. `on_progress` receives values in `[0, 1]`.
pub fn embed_video(
    input: &Path,
    output: &Path,
    wm_bits: &[u8],
    params: &WatermarkParams,
    mut on_progress: impl FnMut(f32),
) -> Result<(), String> {
    let info = probe(input)?;
    let frame_size = (info.width * info.height * 3) as usize;

    // Decoder: video → raw RGB frames on stdout
    let mut decoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(input)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .spawn()
    .map_err(|e| format!("FFmpeg 解码器启动失败: {e}"))?;

    // Encoder: raw RGB on stdin → output file (+ audio from original)
    let (v_args, a_args) = video_encoder_args(output);
    let mut encoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{}x{}", info.width, info.height)])
            .args(["-r", &info.fps])
            .args(["-i", "pipe:0"])
            .arg("-i")
            .arg(input)
            .args(["-map", "0:v:0", "-map", "1:a?"])
            .args(&v_args)
            .args(&a_args)
            .args(["-shortest"])
            .arg(output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null()),
    )
    .spawn()
    .map_err(|e| format!("FFmpeg 编码器启动失败: {e}"))?;

    let mut dec_out = BufReader::new(decoder.stdout.take().unwrap());
    let mut enc_in = BufWriter::new(encoder.stdin.take().unwrap());
    let mut buf = vec![0u8; frame_size];
    let total_frames = info.frame_count.max(1);
    let total = total_frames as f32;
    let mut idx = 0u64;

    loop {
        if !read_frame(&mut dec_out, &mut buf)? {
            break;
        }

        let frame_buf = std::mem::take(&mut buf);
        let mut carrier =
            RgbImage::from_raw(info.width, info.height, frame_buf).ok_or("帧数据大小不匹配")?;
        core::embed_video_frame_in_place(&mut carrier, wm_bits, params)?;
        enc_in
            .write_all(carrier.as_raw())
            .map_err(|e| format!("写入帧失败: {e}"))?;
        buf = carrier.into_raw();

        idx += 1;
        if idx % PROGRESS_UPDATE_INTERVAL == 0 || idx >= total_frames {
            on_progress((idx as f32 / total).min(0.99));
        }
    }

    enc_in
        .flush()
        .map_err(|e| format!("编码器写入刷新失败: {e}"))?;
    drop(enc_in);
    let _ = decoder.wait();
    let enc_status = encoder.wait().map_err(|e| format!("编码器异常: {e}"))?;
    if !enc_status.success() {
        return Err("FFmpeg 编码失败".into());
    }

    // Self-verification is best-effort; failure does not invalidate the export.
    let _ = quick_verify_embedded_video(output, wm_bits.len(), params);
    on_progress(1.0);

    Ok(())
}

// ---------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------

/// Extract a watermark from a video fully automatically.
///
/// No prior knowledge of watermark content or length is needed.
/// Uses fixed watermark size (`FIXED_WM_BITS`) and accumulates extracted bits
/// from each frame; periodically tries full decode (CRC) for early exit.
pub fn extract_video(
    input: &Path,
    params: &WatermarkParams,
    mut on_progress: impl FnMut(f32),
) -> Result<String, String> {
    let info = probe(input)?;
    let frame_size = (info.width * info.height * 3) as usize;
    let total_frames = info.frame_count.max(1);
    let wm_bit_count = super::text::FIXED_WM_BITS;

    // Decode frames sequentially and accumulate extracted bits.
    let mut decoder = no_window(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(input)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null()),
    )
    .spawn()
    .map_err(|e| format!("FFmpeg 解码器启动失败: {e}"))?;

    let mut dec_out = BufReader::new(decoder.stdout.take().unwrap());
    let mut buf = vec![0u8; frame_size];
    let mut accumulated = vec![0.0f64; wm_bit_count];
    let mut accumulated_weight = vec![0.0f64; wm_bit_count];
    let mut wm_count = 0u64;
    let mut frame_idx = 0u64;

    loop {
        if !read_frame(&mut dec_out, &mut buf)? {
            break;
        }

        let frame_buf = std::mem::take(&mut buf);
        let img =
            RgbImage::from_raw(info.width, info.height, frame_buf).ok_or("帧数据大小不匹配")?;
        let (extracted, confidence) =
            core::extract_video_frame_with_confidence(&img, wm_bit_count, params)?;
        for i in 0..wm_bit_count {
            let w = confidence[i];
            accumulated[i] += extracted[i] as f64 * w;
            accumulated_weight[i] += w;
        }
        buf = img.into_raw();
        wm_count += 1;

        // Periodically try full decode for early exit
        if wm_count >= 3 && wm_count % EARLY_DECODE_INTERVAL == 0 {
            let avg_bits: Vec<u8> = accumulated
                .iter()
                .zip(accumulated_weight.iter())
                .map(|(&v, &w)| {
                    if w > 0.0 {
                        (v / w).round().clamp(0.0, 255.0) as u8
                    } else {
                        0u8
                    }
                })
                .collect();
            if let Ok(text) = super::text::bits_to_text(&avg_bits) {
                let _ = decoder.kill();
                let _ = decoder.wait();
                on_progress(1.0);
                return Ok(text);
            }
        }

        frame_idx += 1;
        if frame_idx % PROGRESS_UPDATE_INTERVAL == 0 || frame_idx >= total_frames {
            on_progress((frame_idx as f32 / total_frames as f32).min(0.99));
        }
    }

    let _ = decoder.wait();

    if wm_count == 0 {
        return Err("未能从视频中检测到水印".into());
    }

    // Final decode attempt on accumulated average
    let avg_bits: Vec<u8> = accumulated
        .iter()
        .zip(accumulated_weight.iter())
        .map(|(&v, &w)| {
            if w > 0.0 {
                (v / w).round().clamp(0.0, 255.0) as u8
            } else {
                0u8
            }
        })
        .collect();
    on_progress(1.0);

    super::text::bits_to_text(&avg_bits)
}
