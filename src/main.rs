#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
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

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 750.0])
            .with_min_inner_size([900.0, 600.0]),
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
