use eframe::egui::{self, Color32, CornerRadius, Margin, RichText, Stroke, TextureHandle};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use crate::i18n::{t, tf};
use crate::theme::{
    self, BG_BASE, BG_SURFACE, ERROR, SUCCESS, TEXT_DIM, TEXT_PRIMARY, TEXT_SECONDARY,
};
use crate::watermark::WatermarkParams;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Embed,
    Extract,
}

#[derive(PartialEq, Clone, Copy)]
enum MediaType {
    Image,
    Video,
    Audio,
}

enum ProcessResult {
    EmbedImageDone(PathBuf),
    ExtractDone(String),
    EmbedVideoDone(PathBuf),
    EmbedAudioDone(PathBuf),
    Error(String),
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

pub struct App {
    mode: Mode,
    media_type: MediaType,
    theme_applied: bool,

    // Password mode
    use_password: bool,
    password: String,

    // Text watermark
    watermark_text: String,

    // Embed mode
    carrier_path: Option<PathBuf>,
    carrier_texture: Option<TextureHandle>,
    carrier_dims: Option<(u32, u32)>,

    // Extract mode
    input_path: Option<PathBuf>,
    input_texture: Option<TextureHandle>,
    input_dims: Option<(u32, u32)>,

    // Result
    result_text: Option<String>,
    result_image_path: Option<PathBuf>,
    result_video_path: Option<PathBuf>,
    result_audio_path: Option<PathBuf>,

    // Processing
    status: String,
    processing: bool,
    receiver: Option<mpsc::Receiver<ProcessResult>>,
    progress: Arc<Mutex<f32>>,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            mode: Mode::Embed,
            media_type: MediaType::Image,
            theme_applied: false,
            use_password: false,
            password: String::new(),
            watermark_text: String::new(),
            carrier_path: None,
            carrier_texture: None,
            carrier_dims: None,
            input_path: None,
            input_texture: None,
            input_dims: None,
            result_text: None,
            result_image_path: None,
            result_video_path: None,
            result_audio_path: None,
            status: t("ready").into(),
            processing: false,
            receiver: None,
            progress: Arc::new(Mutex::new(0.0)),
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn parse_params(&self) -> Result<WatermarkParams, String> {
        if self.use_password {
            let pw = self.password.trim();
            if pw.is_empty() {
                return Err(t("err_enter_password").into());
            }
            Ok(WatermarkParams::from_password(pw))
        } else {
            Ok(WatermarkParams::default())
        }
    }

    /// White frosted-glass card frame.
    fn glass_frame() -> egui::Frame {
        egui::Frame {
            inner_margin: Margin::same(16),
            corner_radius: CornerRadius::same(10),
            fill: Color32::from_white_alpha(10),
            stroke: Stroke::new(1.0, Color32::from_white_alpha(22)),
            ..Default::default()
        }
    }

    fn section_label(ui: &mut egui::Ui, text: &str) {
        ui.add_space(4.0);
        ui.label(
            RichText::new(text)
                .size(11.0)
                .color(Color32::from_white_alpha(160))
                .strong(),
        );
        ui.add_space(6.0);
    }

    fn ellipsize_middle(text: &str, max_chars: usize) -> String {
        let count = text.chars().count();
        if count <= max_chars {
            return text.to_string();
        }
        if max_chars <= 3 {
            return "...".to_string();
        }
        let keep = max_chars - 3;
        let head = keep / 2;
        let tail = keep - head;
        let head_str: String = text.chars().take(head).collect();
        let tail_str: String = text
            .chars()
            .rev()
            .take(tail)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        format!("{head_str}...{tail_str}")
    }

    fn load_preview(
        ctx: &egui::Context,
        path: &PathBuf,
        name: &str,
    ) -> Option<(TextureHandle, (u32, u32))> {
        let img = image::open(path).ok()?;
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();

        let max_dim = 512u32;
        let (tw, th) = if w.max(h) > max_dim {
            let s = max_dim as f64 / w.max(h) as f64;
            ((w as f64 * s) as u32, (h as f64 * s) as u32)
        } else {
            (w, h)
        };

        let thumb = if (tw, th) != (w, h) {
            image::imageops::resize(&rgba, tw, th, image::imageops::FilterType::Triangle)
        } else {
            rgba
        };

        let ci =
            egui::ColorImage::from_rgba_unmultiplied([tw as usize, th as usize], thumb.as_raw());
        Some((
            ctx.load_texture(name, ci, egui::TextureOptions::default()),
            (w, h),
        ))
    }

    fn show_preview(ui: &mut egui::Ui, tex: &TextureHandle, max: egui::Vec2) {
        let ts = tex.size_vec2();
        let scale = (max.x / ts.x).min(max.y / ts.y).min(1.0);
        let size = ts * scale;
        ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(
            tex.id(),
            size,
        )));
    }

    // ------------------------------------------------------------------
    // Processing
    // ------------------------------------------------------------------

    fn start_embed_image(&mut self) {
        let params = match self.parse_params() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("✗ {e}");
                return;
            }
        };
        let cp = match &self.carrier_path {
            Some(p) => p.clone(),
            None => {
                self.status = format!("✗ {}", t("err_select_carrier_image"));
                return;
            }
        };
        let wm_text = self.watermark_text.clone();
        if wm_text.trim().is_empty() {
            self.status = format!("✗ {}", t("err_enter_watermark"));
            return;
        }

        // Ask for output path
        let input_ext = cp
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("png")
            .to_ascii_lowercase();
        let (filter_label, filter_ext) = match input_ext.as_str() {
            "jpg" | "jpeg" => (t("filter_jpeg"), "jpg"),
            "bmp" => (t("filter_bmp"), "bmp"),
            _ => (t("filter_png"), "png"),
        };
        let stem = cp
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".to_string());
        let output = match rfd::FileDialog::new()
            .add_filter(filter_label, &[filter_ext])
            .set_file_name(&format!("{stem}_watermarked.{filter_ext}"))
            .save_file()
        {
            Some(p) => p,
            None => return,
        };

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.processing = true;
        self.status = t("embedding_image").into();
        self.result_image_path = None;
        self.result_video_path = None;
        self.result_audio_path = None;
        let progress = self.progress.clone();
        *progress.lock().unwrap() = 0.0;

        thread::spawn(move || {
            let wm_bits = match crate::watermark::text_to_bits(&wm_text) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                    return;
                }
            };
            let carrier = match image::open(&cp) {
                Ok(i) => i,
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(tf("err_carrier_open", &e)));
                    return;
                }
            };
            match crate::watermark::embed(&carrier.to_rgb8(), &wm_bits, &params, |p| {
                *progress.lock().unwrap() = p.clamp(0.0, 1.0);
            }) {
                Ok(img) => match img.save(&output) {
                    Ok(_) => {
                        let _ = tx.send(ProcessResult::EmbedImageDone(output));
                    }
                    Err(e) => {
                        let _ = tx.send(ProcessResult::Error(tf("err_save", &e)));
                    }
                },
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                }
            }
        });
    }

    fn start_embed_video(&mut self) {
        let params = match self.parse_params() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("✗ {e}");
                return;
            }
        };
        let cp = match &self.carrier_path {
            Some(p) => p.clone(),
            None => {
                self.status = format!("✗ {}", t("err_select_video"));
                return;
            }
        };
        let wm_text = self.watermark_text.clone();
        if wm_text.trim().is_empty() {
            self.status = format!("✗ {}", t("err_enter_watermark"));
            return;
        }

        // Ask for output path – default to same format as input
        let input_ext = cp
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp4")
            .to_ascii_lowercase();
        let (filter_label, filter_ext) = match input_ext.as_str() {
            "mkv" => (t("filter_mkv"), "mkv"),
            "avi" => (t("filter_avi"), "avi"),
            "mov" => (t("filter_mov"), "mov"),
            "webm" => (t("filter_webm"), "webm"),
            "flv" => (t("filter_flv"), "flv"),
            _ => (t("filter_mp4"), "mp4"),
        };
        let stem = cp
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".to_string());
        let output = match rfd::FileDialog::new()
            .add_filter(filter_label, &[filter_ext])
            .set_file_name(&format!("{stem}_watermarked.{filter_ext}"))
            .save_file()
        {
            Some(p) => p,
            None => return,
        };

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.processing = true;
        self.status = t("embedding_video").into();
        self.result_image_path = None;
        self.result_video_path = None;
        self.result_audio_path = None;
        let progress = self.progress.clone();
        *progress.lock().unwrap() = 0.0;

        thread::spawn(move || {
            let wm_bits = match crate::watermark::text_to_bits(&wm_text) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                    return;
                }
            };
            match crate::watermark::video::embed_video(&cp, &output, &wm_bits, &params, |p| {
                *progress.lock().unwrap() = p.clamp(0.0, 1.0);
            }) {
                Ok(()) => {
                    let _ = tx.send(ProcessResult::EmbedVideoDone(output));
                }
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                }
            }
        });
    }

    fn start_embed_audio(&mut self) {
        let params = match self.parse_params() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("✗ {e}");
                return;
            }
        };
        let cp = match &self.carrier_path {
            Some(p) => p.clone(),
            None => {
                self.status = format!("✗ {}", t("err_select_audio"));
                return;
            }
        };
        let wm_text = self.watermark_text.clone();
        if wm_text.trim().is_empty() {
            self.status = format!("✗ {}", t("err_enter_watermark"));
            return;
        }

        // Ask for output path
        let input_ext = cp
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("m4a")
            .to_ascii_lowercase();
        let (filter_label, filter_ext) = match input_ext.as_str() {
            "mp3" => (t("filter_mp3"), "mp3"),
            "wav" => (t("filter_wav"), "wav"),
            "flac" => (t("filter_flac"), "flac"),
            "aac" => (t("filter_aac"), "aac"),
            "ogg" => (t("filter_ogg"), "ogg"),
            _ => (t("filter_m4a"), "m4a"),
        };
        let stem = cp
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "audio".to_string());
        let output = match rfd::FileDialog::new()
            .add_filter(filter_label, &[filter_ext])
            .set_file_name(&format!("{stem}_watermarked.{filter_ext}"))
            .save_file()
        {
            Some(p) => p,
            None => return,
        };

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.processing = true;
        self.status = t("embedding_audio").into();
        self.result_image_path = None;
        self.result_video_path = None;
        self.result_audio_path = None;
        let progress = self.progress.clone();
        *progress.lock().unwrap() = 0.0;

        thread::spawn(move || {
            let wm_bits = match crate::watermark::text_to_bits(&wm_text) {
                Ok(b) => b,
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                    return;
                }
            };

            match crate::watermark::audio::embed_audio(&cp, &output, &wm_bits, &params, |p| {
                *progress.lock().unwrap() = p.clamp(0.0, 1.0);
            }) {
                Ok(()) => {
                    let _ = tx.send(ProcessResult::EmbedAudioDone(output));
                }
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                }
            }
        });
    }

    fn start_extract_image(&mut self) {
        let params = match self.parse_params() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("✗ {e}");
                return;
            }
        };
        let ip = match &self.input_path {
            Some(p) => p.clone(),
            None => {
                self.status = format!("✗ {}", t("err_select_wm_image"));
                return;
            }
        };

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.processing = true;
        self.status = t("extracting").into();
        self.result_image_path = None;
        self.result_text = None;
        self.result_video_path = None;
        self.result_audio_path = None;
        let progress = self.progress.clone();
        *progress.lock().unwrap() = 0.0;

        thread::spawn(move || {
            let img = match image::open(&ip) {
                Ok(i) => i,
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(tf("err_image_open", &e)));
                    return;
                }
            };
            match crate::watermark::extract_robust(&img.to_rgb8(), &params, |p| {
                *progress.lock().unwrap() = p.clamp(0.0, 1.0);
            }) {
                Ok(text) => {
                    let _ = tx.send(ProcessResult::ExtractDone(text));
                }
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                }
            }
        });
    }

    fn start_extract_video(&mut self) {
        let params = match self.parse_params() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("✗ {e}");
                return;
            }
        };
        let ip = match &self.input_path {
            Some(p) => p.clone(),
            None => {
                self.status = format!("✗ {}", t("err_select_wm_video"));
                return;
            }
        };

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.processing = true;
        self.status = t("extracting_video").into();
        self.result_image_path = None;
        self.result_text = None;
        self.result_audio_path = None;
        let progress = self.progress.clone();
        *progress.lock().unwrap() = 0.0;

        thread::spawn(move || {
            match crate::watermark::video::extract_video(&ip, &params, |p| {
                *progress.lock().unwrap() = p.clamp(0.0, 1.0);
            }) {
                Ok(text) => {
                    let _ = tx.send(ProcessResult::ExtractDone(text));
                }
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                }
            }
        });
    }

    fn start_extract_audio(&mut self) {
        let params = match self.parse_params() {
            Ok(p) => p,
            Err(e) => {
                self.status = format!("✗ {e}");
                return;
            }
        };
        let ip = match &self.input_path {
            Some(p) => p.clone(),
            None => {
                self.status = format!("✗ {}", t("err_select_wm_audio"));
                return;
            }
        };

        let (tx, rx) = mpsc::channel();
        self.receiver = Some(rx);
        self.processing = true;
        self.status = t("extracting_audio").into();
        self.result_image_path = None;
        self.result_text = None;
        self.result_video_path = None;
        self.result_audio_path = None;
        let progress = self.progress.clone();
        *progress.lock().unwrap() = 0.0;

        thread::spawn(move || {
            match crate::watermark::audio::extract_audio(&ip, &params, |p| {
                *progress.lock().unwrap() = p.clamp(0.0, 1.0);
            }) {
                Ok(text) => {
                    let _ = tx.send(ProcessResult::ExtractDone(text));
                }
                Err(e) => {
                    let _ = tx.send(ProcessResult::Error(e));
                }
            }
        });
    }

    fn poll_result(&mut self, _ctx: &egui::Context) {
        if let Some(rx) = &self.receiver {
            if let Ok(result) = rx.try_recv() {
                self.processing = false;
                self.receiver = None;
                match result {
                    ProcessResult::EmbedImageDone(path) => {
                        self.result_image_path = Some(path.clone());
                        self.result_video_path = None;
                        self.result_audio_path = None;
                        self.status = tf("done_image_embed", &path.display());
                    }
                    ProcessResult::ExtractDone(text) => {
                        self.result_text = Some(text);
                        self.status = t("extract_done").into();
                    }
                    ProcessResult::EmbedVideoDone(path) => {
                        self.result_image_path = None;
                        self.result_video_path = Some(path.clone());
                        self.result_audio_path = None;
                        self.status = tf("done_video_embed", &path.display());
                    }
                    ProcessResult::EmbedAudioDone(path) => {
                        self.result_image_path = None;
                        self.result_audio_path = Some(path.clone());
                        self.result_video_path = None;
                        self.status = tf("done_audio_embed", &path.display());
                    }
                    ProcessResult::Error(e) => {
                        self.status = format!("✗ {e}");
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // UI sections
    // ------------------------------------------------------------------

    fn ui_mode_toggle(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (mode, label) in [
                (Mode::Embed, t("mode_embed")),
                (Mode::Extract, t("mode_extract")),
            ] {
                let active = self.mode == mode;
                let btn = egui::Button::new(RichText::new(label).size(13.0).color(if active {
                    Color32::WHITE
                } else {
                    Color32::from_white_alpha(120)
                }))
                .fill(if active {
                    Color32::from_white_alpha(25)
                } else {
                    Color32::TRANSPARENT
                })
                .stroke(if active {
                    Stroke::new(1.0, Color32::from_white_alpha(50))
                } else {
                    Stroke::new(1.0, Color32::from_white_alpha(18))
                })
                .corner_radius(CornerRadius::same(8));

                if ui.add(btn).clicked() && !self.processing && self.mode != mode {
                    self.mode = mode;
                    // Clear results when switching modes
                    self.result_text = None;
                    self.result_image_path = None;
                    self.result_video_path = None;
                    self.result_audio_path = None;
                }
                ui.add_space(2.0);
            }
        });
    }

    fn ui_media_toggle(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (mt, label) in [
                (MediaType::Image, t("media_image")),
                (MediaType::Video, t("media_video")),
                (MediaType::Audio, t("media_audio")),
            ] {
                let active = self.media_type == mt;
                let btn = egui::Button::new(RichText::new(label).size(12.0).color(if active {
                    Color32::WHITE
                } else {
                    Color32::from_white_alpha(100)
                }))
                .fill(if active {
                    Color32::from_white_alpha(20)
                } else {
                    Color32::TRANSPARENT
                })
                .stroke(if active {
                    Stroke::new(1.0, Color32::from_white_alpha(40))
                } else {
                    Stroke::new(1.0, Color32::from_white_alpha(14))
                })
                .corner_radius(CornerRadius::same(6));

                if ui.add(btn).clicked() && !self.processing {
                    self.media_type = mt;
                    // Reset file selections on media type change
                    self.carrier_path = None;
                    self.carrier_texture = None;
                    self.carrier_dims = None;
                    self.input_path = None;
                    self.input_texture = None;
                    self.input_dims = None;
                    self.result_text = None;
                    self.result_image_path = None;
                    self.result_video_path = None;
                    self.result_audio_path = None;
                }
                ui.add_space(2.0);
            }
        });
    }

    fn ui_file_row(ui: &mut egui::Ui, label: &str, path: &Option<PathBuf>, enabled: bool) -> bool {
        let mut clicked = false;
        ui.label(RichText::new(label).size(12.0).color(TEXT_SECONDARY));
        ui.horizontal(|ui| {
            let txt = path
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| t("not_selected").into());

            let txt_color = if path.is_some() {
                TEXT_PRIMARY
            } else {
                TEXT_DIM
            };

            let display = Self::ellipsize_middle(&txt, 26);
            let resp = ui.label(RichText::new(&display).size(12.0).color(txt_color));
            if display != txt {
                resp.on_hover_text(txt);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        enabled,
                        egui::Button::new(t("btn_select")).corner_radius(CornerRadius::same(6)),
                    )
                    .clicked()
                {
                    clicked = true;
                }
            });
        });
        clicked
    }

    fn file_filters_for_media(&self) -> (&str, Vec<&str>) {
        match self.media_type {
            MediaType::Image => (t("filter_image"), vec!["png", "jpg", "jpeg", "bmp"]),
            MediaType::Video => (
                t("filter_video"),
                vec!["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"],
            ),
            MediaType::Audio => (t("filter_audio"), vec!["mp3", "wav", "flac", "aac", "m4a", "ogg"]),
        }
    }

    fn ui_files_embed(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let carrier_label = match self.media_type {
            MediaType::Image => t("carrier_image"),
            MediaType::Video => t("carrier_video"),
            MediaType::Audio => t("carrier_audio"),
        };
        if Self::ui_file_row(ui, carrier_label, &self.carrier_path, !self.processing) {
            let (filter_name, exts) = self.file_filters_for_media();
            if let Some(p) = rfd::FileDialog::new()
                .add_filter(filter_name, &exts)
                .pick_file()
            {
                if self.media_type == MediaType::Image {
                    if let Some((tex, dims)) = Self::load_preview(ctx, &p, "carrier") {
                        self.carrier_texture = Some(tex);
                        self.carrier_dims = Some(dims);
                    }
                } else {
                    self.carrier_texture = None;
                    self.carrier_dims = None;
                }
                self.carrier_path = Some(p);
            }
        }
    }

    fn ui_files_extract(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let input_label = match self.media_type {
            MediaType::Image => t("wm_image"),
            MediaType::Video => t("wm_video"),
            MediaType::Audio => t("wm_audio"),
        };
        if Self::ui_file_row(ui, input_label, &self.input_path, !self.processing) {
            let (filter_name, exts) = self.file_filters_for_media();
            if let Some(p) = rfd::FileDialog::new()
                .add_filter(filter_name, &exts)
                .pick_file()
            {
                if self.media_type == MediaType::Image {
                    if let Some((tex, dims)) = Self::load_preview(ctx, &p, "input") {
                        self.input_texture = Some(tex);
                        self.input_dims = Some(dims);
                    }
                } else {
                    self.input_texture = None;
                    self.input_dims = None;
                }
                self.input_path = Some(p);
            }
        }
    }

    fn ui_watermark_text(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.label(RichText::new(t("watermark_text")).size(12.0).color(TEXT_SECONDARY));
        ui.add(
            egui::TextEdit::singleline(&mut self.watermark_text)
                .hint_text(t("hint_watermark"))
                .desired_width(ui.available_width())
                .font(egui::TextStyle::Monospace),
        );
    }

    fn ui_password_toggle(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (use_pw, label) in [(false, t("no_password")), (true, t("use_password"))] {
                let active = self.use_password == use_pw;
                let btn = egui::Button::new(RichText::new(label).size(12.0).color(if active {
                    Color32::WHITE
                } else {
                    Color32::from_white_alpha(100)
                }))
                .fill(if active {
                    Color32::from_white_alpha(20)
                } else {
                    Color32::TRANSPARENT
                })
                .stroke(if active {
                    Stroke::new(1.0, Color32::from_white_alpha(40))
                } else {
                    Stroke::new(1.0, Color32::from_white_alpha(14))
                })
                .corner_radius(CornerRadius::same(6));

                if ui.add(btn).clicked() && !self.processing {
                    self.use_password = use_pw;
                }
                ui.add_space(2.0);
            }
        });
        if self.use_password {
            ui.add_space(6.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.password)
                    .hint_text(t("hint_password"))
                    .password(true)
                    .desired_width(ui.available_width())
                    .font(egui::TextStyle::Monospace),
            );
        }
    }

    fn ui_exec_button(&mut self, ui: &mut egui::Ui) {
        let label = match self.mode {
            Mode::Embed => t("btn_embed"),
            Mode::Extract => t("btn_extract"),
        };
        let btn = egui::Button::new(
            RichText::new(label)
                .size(14.0)
                .strong()
                .color(Color32::BLACK),
        )
        .fill(Color32::from_gray(230))
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(10))
        .min_size(egui::vec2(ui.available_width(), 40.0));

        if ui.add_enabled(!self.processing, btn).clicked() {
            match (self.mode, self.media_type) {
                (Mode::Embed, MediaType::Image) => self.start_embed_image(),
                (Mode::Embed, MediaType::Video) => self.start_embed_video(),
                (Mode::Embed, MediaType::Audio) => self.start_embed_audio(),
                (Mode::Extract, MediaType::Image) => self.start_extract_image(),
                (Mode::Extract, MediaType::Video) => self.start_extract_video(),
                (Mode::Extract, MediaType::Audio) => self.start_extract_audio(),
            }
        }
    }

    fn ui_status(&self, ui: &mut egui::Ui) {
        ui.add_space(12.0);
        if self.processing {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(&self.status).size(12.0).color(TEXT_SECONDARY));
            });
            let p = *self.progress.lock().unwrap();
            ui.add_space(6.0);
            let bar = egui::ProgressBar::new(p)
                .text(format!("{:.0}%", p * 100.0))
                .animate(true)
                .desired_width(ui.available_width());
            ui.add(bar);
        } else {
            let color = if self.status.starts_with('✓') {
                SUCCESS
            } else if self.status.starts_with('✗') {
                ERROR
            } else {
                TEXT_DIM
            };
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new(&self.status).size(12.0).color(color));
            });
        }
    }

    // ------------------------------------------------------------------
    // Preview area
    // ------------------------------------------------------------------

    fn ui_preview_card(
        ui: &mut egui::Ui,
        title: &str,
        tex: &Option<TextureHandle>,
        dims: &Option<(u32, u32)>,
        placeholder: &str,
        max_img: egui::Vec2,
    ) {
        Self::glass_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                RichText::new(title)
                    .size(11.0)
                    .color(Color32::from_white_alpha(120)),
            );
            ui.add_space(4.0);
            if let Some(t) = tex {
                ui.vertical_centered(|ui| {
                    Self::show_preview(ui, t, max_img);
                });
                if let Some(d) = dims {
                    ui.label(
                        RichText::new(format!("{}×{}", d.0, d.1))
                            .size(10.0)
                            .color(TEXT_DIM),
                    );
                }
            } else {
                let (rect, _) = ui.allocate_exact_size(max_img, egui::Sense::hover());
                ui.put(
                    rect,
                    egui::Label::new(
                        RichText::new(placeholder)
                            .size(13.0)
                            .color(Color32::from_white_alpha(40)),
                    ),
                );
            }
        });
    }

    fn ui_path_placeholder(
        ui: &mut egui::Ui,
        title: &str,
        path: &Option<PathBuf>,
        max_img: egui::Vec2,
        empty_text: &str,
    ) {
        Self::glass_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                RichText::new(title)
                    .size(11.0)
                    .color(Color32::from_white_alpha(120)),
            );
            ui.add_space(4.0);
            let (rect, _) = ui.allocate_exact_size(max_img, egui::Sense::hover());
            let text = if let Some(p) = path {
                p.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| t("selected").into())
            } else {
                empty_text.into()
            };
            ui.put(
                rect,
                egui::Label::new(
                    RichText::new(text)
                        .size(13.0)
                        .color(Color32::from_white_alpha(if path.is_some() {
                            120
                        } else {
                            40
                        })),
                ),
            );
        });
    }

    fn ui_previews_embed(&self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let gap = 8.0;
        let card_overhead = 60.0;
        let img_h = ((avail.y - gap - card_overhead * 2.0) / 2.0).max(100.0);
        let img_w = ui.available_width() - 32.0;

        match self.media_type {
            MediaType::Image => {
                // Carrier preview (full width)
                Self::ui_preview_card(
                    ui,
                    t("carrier_image"),
                    &self.carrier_texture,
                    &self.carrier_dims,
                    t("ph_select_carrier"),
                    egui::vec2(img_w, img_h),
                );
                ui.add_space(gap);
                // 嵌入结果输出路径
                Self::glass_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(t("embed_result"))
                            .size(11.0)
                            .color(Color32::from_white_alpha(120)),
                    );
                    ui.add_space(4.0);
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(img_w, img_h), egui::Sense::hover());
                    let text = if let Some(p) = &self.result_image_path {
                        format!("✓ {}", p.display())
                    } else {
                        t("ph_embed_image").into()
                    };
                    let alpha = if self.result_image_path.is_some() {
                        180
                    } else {
                        40
                    };
                    ui.put(
                        rect,
                        egui::Label::new(
                            RichText::new(text)
                                .size(13.0)
                                .color(Color32::from_white_alpha(alpha)),
                        ),
                    );
                });
            }
            MediaType::Video => {
                Self::ui_path_placeholder(
                    ui,
                    t("carrier_video"),
                    &self.carrier_path,
                    egui::vec2(img_w, img_h),
                    t("ph_no_video"),
                );
                ui.add_space(gap);
                Self::glass_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(t("embed_result"))
                            .size(11.0)
                            .color(Color32::from_white_alpha(120)),
                    );
                    ui.add_space(4.0);
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(img_w, img_h), egui::Sense::hover());
                    let text = if let Some(p) = &self.result_video_path {
                        format!("✓ {}", p.display())
                    } else {
                        t("ph_embed_video").into()
                    };
                    let alpha = if self.result_video_path.is_some() {
                        180
                    } else {
                        40
                    };
                    ui.put(
                        rect,
                        egui::Label::new(
                            RichText::new(text)
                                .size(13.0)
                                .color(Color32::from_white_alpha(alpha)),
                        ),
                    );
                });
            }
            MediaType::Audio => {
                Self::ui_path_placeholder(
                    ui,
                    t("carrier_audio"),
                    &self.carrier_path,
                    egui::vec2(img_w, img_h),
                    t("ph_no_audio"),
                );
                ui.add_space(gap);
                Self::glass_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(
                        RichText::new(t("embed_result"))
                            .size(11.0)
                            .color(Color32::from_white_alpha(120)),
                    );
                    ui.add_space(4.0);
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(img_w, img_h), egui::Sense::hover());
                    let text = if let Some(p) = &self.result_audio_path {
                        format!("✓ {}", p.display())
                    } else {
                        t("ph_embed_audio").into()
                    };
                    let alpha = if self.result_audio_path.is_some() {
                        180
                    } else {
                        40
                    };
                    ui.put(
                        rect,
                        egui::Label::new(
                            RichText::new(text)
                                .size(13.0)
                                .color(Color32::from_white_alpha(alpha)),
                        ),
                    );
                });
            }
        }
    }

    fn ui_previews_extract(&self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let gap = 8.0;
        let card_overhead = 60.0;
        let img_h = ((avail.y - gap - card_overhead * 2.0) / 2.0).max(100.0);
        let img_w = ui.available_width() - 32.0;

        match self.media_type {
            MediaType::Image => {
                Self::ui_preview_card(
                    ui,
                    t("wm_image"),
                    &self.input_texture,
                    &self.input_dims,
                    t("ph_select_wm_image"),
                    egui::vec2(img_w, img_h),
                );
            }
            MediaType::Video => {
                Self::ui_path_placeholder(
                    ui,
                    t("wm_video"),
                    &self.input_path,
                    egui::vec2(img_w, img_h),
                    t("ph_no_video"),
                );
            }
            MediaType::Audio => {
                Self::ui_path_placeholder(
                    ui,
                    t("wm_audio"),
                    &self.input_path,
                    egui::vec2(img_w, img_h),
                    t("ph_no_audio"),
                );
            }
        }

        ui.add_space(gap);

        // Extracted text result card
        Self::glass_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                RichText::new(t("extract_result"))
                    .size(11.0)
                    .color(Color32::from_white_alpha(120)),
            );
            ui.add_space(4.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(img_w, img_h), egui::Sense::hover());
            if let Some(ref text) = self.result_text {
                ui.put(
                    rect,
                    egui::Label::new(RichText::new(text).size(20.0).strong().color(SUCCESS)),
                );
            } else {
                ui.put(
                    rect,
                    egui::Label::new(
                        RichText::new(t("ph_extract_result"))
                            .size(13.0)
                            .color(Color32::from_white_alpha(40)),
                    ),
                );
            }
        });
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            theme::apply(ctx);
            self.theme_applied = true;
        }

        self.poll_result(ctx);
        if self.processing {
            ctx.request_repaint();
        }

        // ── Left panel ──
        let side_frame = egui::Frame {
            inner_margin: Margin::same(20),
            fill: BG_SURFACE,
            stroke: Stroke::new(1.0, Color32::from_white_alpha(10)),
            ..Default::default()
        };

        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(300.0)
            .frame(side_frame)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(4.0);

                    ui.label(
                        RichText::new("FogID")
                            .size(22.0)
                            .strong()
                            .color(TEXT_PRIMARY),
                    );
                    ui.label(RichText::new(t("subtitle")).size(12.0).color(TEXT_DIM));
                    ui.add_space(14.0);

                    self.ui_mode_toggle(ui);
                    ui.add_space(6.0);

                    self.ui_media_toggle(ui);
                    ui.add_space(14.0);

                    // 文件选择 + 嵌入模式下的水印文字输入
                    Self::glass_frame().show(ui, |ui| {
                        Self::section_label(ui, t("section_files"));
                        match self.mode {
                            Mode::Embed => {
                                self.ui_files_embed(ui, ctx);
                                self.ui_watermark_text(ui);
                            }
                            Mode::Extract => self.ui_files_extract(ui, ctx),
                        }
                    });
                    ui.add_space(10.0);

                    self.ui_password_toggle(ui);
                    ui.add_space(14.0);

                    self.ui_exec_button(ui);

                    self.ui_status(ui);
                });
            });

        // ── Central panel (previews) ──
        let central_frame = egui::Frame {
            inner_margin: Margin::same(24),
            fill: BG_BASE,
            ..Default::default()
        };

        egui::CentralPanel::default()
            .frame(central_frame)
            .show(ctx, |ui| match self.mode {
                Mode::Embed => self.ui_previews_embed(ui),
                Mode::Extract => self.ui_previews_extract(ui),
            });
    }
}
