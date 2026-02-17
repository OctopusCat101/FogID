use eframe::egui::{self, Color32, CornerRadius, Stroke, Visuals};

// ─── Black + White Palette ───────────────────────────────────────────────────

pub const BG_BASE: Color32 = Color32::from_rgb(8, 8, 8);
/// Slightly elevated surface for side-panel contrast.
pub const BG_SURFACE: Color32 = Color32::from_rgb(16, 16, 16);
pub const TEXT_PRIMARY: Color32 = Color32::from_gray(240);
pub const TEXT_SECONDARY: Color32 = Color32::from_gray(140);
/// Used for placeholders and disabled elements.
pub const TEXT_DIM: Color32 = Color32::from_gray(60);
pub const SUCCESS: Color32 = Color32::from_rgb(100, 230, 150);
pub const ERROR: Color32 = Color32::from_rgb(240, 100, 100);

pub fn apply(ctx: &egui::Context) {
    let mut v = Visuals::dark();

    // ── 1. Backgrounds ──
    v.window_fill = BG_BASE;
    v.panel_fill = BG_BASE;
    v.faint_bg_color = Color32::from_white_alpha(4);
    v.extreme_bg_color = Color32::from_rgb(5, 5, 5);

    // ── 2. Geometry ──
    let r = CornerRadius::same(10);
    v.window_corner_radius = r;
    v.menu_corner_radius = r;
    v.widgets.noninteractive.corner_radius = r;
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.corner_radius = r;
    v.widgets.open.corner_radius = r;

    // ── 3. Widgets ──

    // NON-INTERACTIVE
    v.widgets.noninteractive.weak_bg_fill = Color32::TRANSPARENT;
    v.widgets.noninteractive.bg_fill = Color32::TRANSPARENT;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(12));
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);

    // INACTIVE – frosted glass look
    v.widgets.inactive.weak_bg_fill = Color32::from_white_alpha(8);
    v.widgets.inactive.bg_fill = Color32::from_white_alpha(6);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(14));
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);

    // HOVERED
    v.widgets.hovered.weak_bg_fill = Color32::from_white_alpha(18);
    v.widgets.hovered.bg_fill = Color32::from_white_alpha(14);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(40));
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.widgets.hovered.expansion = 1.0;

    // ACTIVE
    v.widgets.active.weak_bg_fill = Color32::from_white_alpha(28);
    v.widgets.active.bg_fill = Color32::from_white_alpha(28);
    v.widgets.active.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(70));
    v.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.widgets.active.expansion = 1.0;

    // OPEN
    v.widgets.open.weak_bg_fill = Color32::from_white_alpha(15);
    v.widgets.open.bg_fill = Color32::from_rgb(24, 24, 24);
    v.widgets.open.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(22));
    v.widgets.open.fg_stroke = Stroke::new(1.0, Color32::WHITE);

    // ── 4. Selection & Window ──
    v.selection.bg_fill = Color32::from_white_alpha(30);
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.window_stroke = Stroke::new(1.0, Color32::from_white_alpha(10));

    ctx.set_visuals(v);
}
