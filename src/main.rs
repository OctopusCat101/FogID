#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod i18n;
mod theme;
mod watermark;

use std::fs;

use eframe::egui::{self, FontData, FontDefinitions, FontFamily};

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();

    // Common CJK fonts on Windows. First one found will be used.
    let candidates = [
        ("cjk", "C:\\Windows\\Fonts\\msyh.ttc"),   // Microsoft YaHei
        ("cjk", "C:\\Windows\\Fonts\\msyhbd.ttc"), // Microsoft YaHei Bold
        ("cjk", "C:\\Windows\\Fonts\\simhei.ttf"), // SimHei
        ("cjk", "C:\\Windows\\Fonts\\simsun.ttc"), // SimSun
        ("cjk", "C:\\Windows\\Fonts\\Deng.ttf"),   // DengXian
    ];

    let mut loaded_name: Option<&'static str> = None;
    for (name, path) in candidates {
        if let Ok(bytes) = fs::read(path) {
            fonts
                .font_data
                .insert(name.to_owned(), FontData::from_owned(bytes).into());
            loaded_name = Some(name);
            break;
        }
    }

    if let Some(name) = loaded_name {
        if let Some(family) = fonts.families.get_mut(&FontFamily::Proportional) {
            family.insert(0, name.to_owned());
        }
        if let Some(family) = fonts.families.get_mut(&FontFamily::Monospace) {
            family.insert(0, name.to_owned());
        }
    }

    ctx.set_fonts(fonts);
}

fn load_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!("../assets/icon.ico");
    let image = image::load_from_memory(bytes).ok()?;
    let image = image.to_rgba8();
    let (width, height) = image.dimensions();
    Some(egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    })
}

fn main() -> eframe::Result {
    i18n::init();
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1100.0, 750.0])
        .with_min_inner_size([900.0, 600.0]);

    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "FogID",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Ok(Box::new(app::App::new(cc)))
        }),
    )
}
