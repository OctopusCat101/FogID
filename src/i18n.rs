use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq)]
pub enum Lang {
    Zh,
    En,
}

static LANG: OnceLock<Lang> = OnceLock::new();

pub fn init() {
    LANG.set(detect()).ok();
}

pub fn lang() -> Lang {
    *LANG.get().unwrap_or(&Lang::Zh)
}

fn is_zh() -> bool {
    lang() == Lang::Zh
}

fn detect() -> Lang {
    if let Some(locale) = sys_locale::get_locale() {
        if locale.starts_with("zh") { Lang::Zh } else { Lang::En }
    } else {
        Lang::Zh
    }
}

/// Translate a key to the current language.
#[allow(clippy::too_many_lines)]
pub fn t(key: &str) -> &str {
    let zh = is_zh();
    match key {
        // ── Status ──
        "ready" => if zh { "就绪" } else { "Ready" },
        "extracting" => if zh { "提取中…" } else { "Extracting…" },
        "embedding_image" => if zh { "图片嵌入中…" } else { "Embedding image…" },
        "embedding_video" => if zh { "视频嵌入中…" } else { "Embedding video…" },
        "embedding_audio" => if zh { "音频嵌入中…" } else { "Embedding audio…" },
        "extracting_video" => if zh { "视频提取中…" } else { "Extracting video…" },
        "extracting_audio" => if zh { "音频提取中…" } else { "Extracting audio…" },
        "extract_done" => if zh { "✓ 提取完成" } else { "✓ Extraction complete" },
        // ── Validation errors ──
        "err_enter_password" => if zh { "请输入密码" } else { "Please enter password" },
        "err_select_carrier_image" => if zh { "请选择载体图片" } else { "Please select carrier image" },
        "err_enter_watermark" => if zh { "请输入水印文字" } else { "Please enter watermark text" },
        "err_select_video" => if zh { "请选择视频文件" } else { "Please select video file" },
        "err_select_audio" => if zh { "请选择音频文件" } else { "Please select audio file" },
        "err_select_wm_image" => if zh { "请选择含水印图片" } else { "Please select watermarked image" },
        "err_select_wm_video" => if zh { "请选择含水印视频" } else { "Please select watermarked video" },
        "err_select_wm_audio" => if zh { "请选择含水印音频" } else { "Please select watermarked audio" },
        // ── Format templates ──
        "done_image_embed" => if zh { "✓ 图片嵌入完成 → {}" } else { "✓ Image embedded → {}" },
        "done_video_embed" => if zh { "✓ 视频嵌入完成 → {}" } else { "✓ Video embedded → {}" },
        "done_audio_embed" => if zh { "✓ 音频嵌入完成 → {}" } else { "✓ Audio embedded → {}" },
        "err_carrier_open" => if zh { "载体图片打开失败：{}" } else { "Failed to open carrier image: {}" },
        "err_save" => if zh { "保存失败：{}" } else { "Save failed: {}" },
        "err_image_open" => if zh { "图片打开失败：{}" } else { "Failed to open image: {}" },
        // ── UI labels ──
        "mode_embed" => if zh { "  嵌入水印  " } else { "  Embed  " },
        "mode_extract" => if zh { "  提取水印  " } else { "  Extract  " },
        "media_image" => if zh { "图片" } else { "Image" },
        "media_video" => if zh { "视频" } else { "Video" },
        "media_audio" => if zh { "音频" } else { "Audio" },
        "not_selected" => if zh { "未选择" } else { "Not selected" },
        "btn_select" => if zh { "选择" } else { "Select" },
        "carrier_image" => if zh { "载体图片" } else { "Carrier Image" },
        "carrier_video" => if zh { "载体视频" } else { "Carrier Video" },
        "carrier_audio" => if zh { "载体音频" } else { "Carrier Audio" },
        "wm_image" => if zh { "含水印图片" } else { "Watermarked Image" },
        "wm_video" => if zh { "含水印视频" } else { "Watermarked Video" },
        "wm_audio" => if zh { "含水印音频" } else { "Watermarked Audio" },
        "watermark_text" => if zh { "水印文字" } else { "Watermark Text" },
        "hint_watermark" => if zh { "输入水印内容…" } else { "Enter watermark…" },
        "no_password" => if zh { "无密码" } else { "No Password" },
        "use_password" => if zh { "使用密码" } else { "Use Password" },
        "hint_password" => if zh { "输入密码…" } else { "Enter password…" },
        "btn_embed" => if zh { "嵌入水印" } else { "Embed Watermark" },
        "btn_extract" => if zh { "提取水印" } else { "Extract Watermark" },
        "subtitle" => if zh { "隐形水印工具" } else { "Invisible Watermark Tool" },
        "section_files" => if zh { "文  件" } else { "Files" },
        "selected" => if zh { "已选择" } else { "Selected" },
        "embed_result" => if zh { "嵌入结果" } else { "Embed Result" },
        "extract_result" => if zh { "提取结果" } else { "Extract Result" },
        "ph_select_carrier" => if zh { "选择载体图片" } else { "Select carrier image" },
        "ph_select_wm_image" => if zh { "选择含水印的图片" } else { "Select watermarked image" },
        "ph_no_video" => if zh { "未选择视频文件" } else { "No video selected" },
        "ph_no_audio" => if zh { "未选择音频文件" } else { "No audio selected" },
        "ph_embed_image" => if zh { "嵌入后的图片路径将显示在此处" } else { "Embedded image path will appear here" },
        "ph_embed_video" => if zh { "嵌入后的视频路径将显示在此处" } else { "Embedded video path will appear here" },
        "ph_embed_audio" => if zh { "嵌入后的音频路径将显示在此处" } else { "Embedded audio path will appear here" },
        "ph_extract_result" => if zh { "提取的水印文字将显示在此处" } else { "Extracted watermark will appear here" },
        // ── File dialog filters ──
        "filter_jpeg" => if zh { "JPEG 图片" } else { "JPEG Image" },
        "filter_bmp" => if zh { "BMP 图片" } else { "BMP Image" },
        "filter_png" => if zh { "PNG 图片" } else { "PNG Image" },
        "filter_image" => if zh { "图片" } else { "Images" },
        "filter_video" => if zh { "视频" } else { "Videos" },
        "filter_audio" => if zh { "音频" } else { "Audio" },
        "filter_mkv" => if zh { "MKV 视频" } else { "MKV Video" },
        "filter_avi" => if zh { "AVI 视频" } else { "AVI Video" },
        "filter_mov" => if zh { "MOV 视频" } else { "MOV Video" },
        "filter_webm" => if zh { "WebM 视频" } else { "WebM Video" },
        "filter_flv" => if zh { "FLV 视频" } else { "FLV Video" },
        "filter_mp4" => if zh { "MP4 视频" } else { "MP4 Video" },
        "filter_mp3" => if zh { "MP3 音频" } else { "MP3 Audio" },
        "filter_wav" => if zh { "WAV 音频" } else { "WAV Audio" },
        "filter_flac" => if zh { "FLAC 音频" } else { "FLAC Audio" },
        "filter_aac" => if zh { "AAC 音频" } else { "AAC Audio" },
        "filter_ogg" => if zh { "OGG 音频" } else { "OGG Audio" },
        "filter_m4a" => if zh { "M4A 音频" } else { "M4A Audio" },
        // ── Watermark text errors ──
        "wm_text_empty" => if zh { "水印文字不能为空" } else { "Watermark text cannot be empty" },
        "wm_text_too_long" => if zh { "水印文字过长（最多 {} 字节 UTF-8）" } else { "Watermark text too long (max {} bytes UTF-8)" },
        "wm_extract_insufficient" => if zh { "提取数据不足" } else { "Insufficient extraction data" },
        "wm_magic_mismatch" => if zh { "未检测到水印（魔数不匹配）" } else { "No watermark detected (magic mismatch)" },
        "wm_data_incomplete" => if zh { "提取数据不完整" } else { "Extraction data incomplete" },
        "wm_crc_mismatch" => if zh { "水印校验失败（CRC 不匹配）" } else { "Watermark verification failed (CRC mismatch)" },
        "wm_utf8_failed" => if zh { "水印文字解码失败（无效 UTF-8）" } else { "Watermark text decode failed (invalid UTF-8)" },
        // ── Core watermark errors ──
        "wm_too_large" => if zh {
            "水印太大：需要 {} 个块，但载体只有 {} 个块的容量。\n请减小水印尺寸或增大载体图片。"
        } else {
            "Watermark too large: needs {} blocks, but carrier only has {} blocks.\nReduce watermark size or use a larger carrier image."
        },
        "wm_data_empty" => if zh { "水印数据为空" } else { "Watermark data is empty" },
        "wm_frame_too_small_embed" => if zh { "视频帧尺寸过小，无法嵌入水印" } else { "Video frame too small to embed watermark" },
        "wm_bit_count_invalid" => if zh { "水印位数无效" } else { "Invalid watermark bit count" },
        "wm_frame_too_small_extract" => if zh { "视频帧尺寸过小，无法提取水印" } else { "Video frame too small to extract watermark" },
        "wm_not_detected_align" => if zh { "未检测到水印（已尝试多种对齐方式）" } else { "No watermark detected (tried multiple alignments)" },
        // ── Video errors ──
        "ffprobe_not_found" => if zh { "无法运行 ffprobe: {}\n请确保已安装 FFmpeg 并加入系统 PATH" } else { "Cannot run ffprobe: {}\nPlease ensure FFmpeg is installed and added to system PATH" },
        "ffprobe_failed" => if zh { "ffprobe 失败: {}" } else { "ffprobe failed: {}" },
        "ffprobe_bad_output" => if zh { "ffprobe 输出格式异常: {}" } else { "ffprobe output format error: {}" },
        "parse_video_width" => if zh { "无法解析视频宽度" } else { "Cannot parse video width" },
        "parse_video_height" => if zh { "无法解析视频高度" } else { "Cannot parse video height" },
        "incomplete_frame" => if zh { "不完整的帧数据" } else { "Incomplete frame data" },
        "read_frame_failed" => if zh { "读取帧失败: {}" } else { "Read frame failed: {}" },
        "ffmpeg_verify_start" => if zh { "FFmpeg 自校验解码器启动失败: {}" } else { "FFmpeg verify decoder failed to start: {}" },
        "verify_frame_mismatch" => if zh { "自校验帧数据大小不匹配" } else { "Verify frame data size mismatch" },
        "verify_failed" => if zh { "视频已导出，但自校验未通过（建议提高码率或重试）" } else { "Video exported, but verification failed (try increasing bitrate or retry)" },
        "ffmpeg_decoder_start" => if zh { "FFmpeg 解码器启动失败: {}" } else { "FFmpeg decoder failed to start: {}" },
        "ffmpeg_encoder_start" => if zh { "FFmpeg 编码器启动失败: {}" } else { "FFmpeg encoder failed to start: {}" },
        "frame_size_mismatch" => if zh { "帧数据大小不匹配" } else { "Frame data size mismatch" },
        "write_frame_failed" => if zh { "写入帧失败: {}" } else { "Write frame failed: {}" },
        "encoder_flush_failed" => if zh { "编码器写入刷新失败: {}" } else { "Encoder flush failed: {}" },
        "encoder_error" => if zh { "编码器异常: {}" } else { "Encoder error: {}" },
        "ffmpeg_encode_failed" => if zh { "FFmpeg 编码失败" } else { "FFmpeg encoding failed" },
        "video_no_watermark" => if zh { "未能从视频中检测到水印" } else { "No watermark detected in video" },
        // ── Audio errors ──
        "parse_sample_rate" => if zh { "无法解析音频采样率" } else { "Cannot parse audio sample rate" },
        "parse_channels" => if zh { "无法解析音频声道数" } else { "Cannot parse audio channel count" },
        "parse_duration" => if zh { "无法解析音频时长" } else { "Cannot parse audio duration" },
        "read_audio_failed" => if zh { "读取音频帧失败: {}" } else { "Read audio frame failed: {}" },
        "wm_bits_invalid" => if zh { "水印位数无效: {} != {}" } else { "Invalid watermark bit count: {} != {}" },
        "unsupported_channels" => if zh { "暂不支持该声道数: {}（仅支持单声道或双声道）。{}" } else { "Unsupported channel count: {} (only mono or stereo supported). {}" },
        "ffmpeg_audio_decoder_start" => if zh { "FFmpeg 音频解码器启动失败: {}" } else { "FFmpeg audio decoder failed to start: {}" },
        "wait_decoder_failed" => if zh { "等待解码器失败: {}" } else { "Wait for decoder failed: {}" },
        "ffmpeg_audio_decode_failed" => if zh { "FFmpeg 音频解码失败" } else { "FFmpeg audio decoding failed" },
        "audio_too_short" => if zh { "音频过短，无法完成至少一个处理帧" } else { "Audio too short for processing" },
        "ffmpeg_audio_encoder_start" => if zh { "FFmpeg 音频编码器启动失败: {}" } else { "FFmpeg audio encoder failed to start: {}" },
        "write_audio_encoder" => if zh { "写入音频编码器失败: {}" } else { "Write to audio encoder failed: {}" },
        "flush_audio_encoder" => if zh { "刷新音频编码器失败: {}" } else { "Flush audio encoder failed: {}" },
        "wait_encoder_failed" => if zh { "等待编码器失败: {}" } else { "Wait for encoder failed: {}" },
        "ffmpeg_audio_encode_failed" => if zh { "FFmpeg 音频编码失败" } else { "FFmpeg audio encoding failed" },
        "audio_no_watermark" => if zh { "未检测到水印（所有偏移均未匹配）" } else { "No watermark detected (all offsets failed)" },
        // ── Debug / log ──
        "audio_verify_ok" => if zh { "[音频] 自校验通过: {}" } else { "[audio] Verification passed: {}" },
        "audio_verify_fail" => if zh { "[音频] 自校验失败 (best-effort): {}" } else { "[audio] Verification failed (best-effort): {}" },

        _ => key,
    }
}

/// Format a translated template with one argument.
pub fn tf(key: &str, arg: &dyn std::fmt::Display) -> String {
    t(key).replacen("{}", &arg.to_string(), 1)
}

/// Format a translated template with two arguments.
pub fn tf2(key: &str, arg1: &dyn std::fmt::Display, arg2: &dyn std::fmt::Display) -> String {
    let s = t(key).replacen("{}", &arg1.to_string(), 1);
    s.replacen("{}", &arg2.to_string(), 1)
}
