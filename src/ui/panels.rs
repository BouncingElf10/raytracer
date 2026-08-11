//! Every control surface in the studio.
//!
//! Panels only read and write `Studio` fields; anything that needs the canvas or
//! blocks is pushed onto `studio.actions` and applied by the caller once the UI is
//! done. That keeps the borrow situation simple and means a slow operation can
//! never run in the middle of building a frame.

use std::borrow::Cow;

use imgui::{Condition, StyleColor, TreeNodeFlags, Ui};

use crate::bvh::SplitHeuristic;
use crate::gpu_types::DisplayMode;
use crate::viz::Palette;

use super::{Action, Studio};

/// Ramps offered for the live cost views. The order matches the `palette_sample`
/// switch in `shaders/palette.wgsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteChoice {
    Inferno = 0,
    Viridis = 1,
    Turbo = 2,
    Grayscale = 3,
}

impl PaletteChoice {
    pub const ALL: [PaletteChoice; 4] = [
        PaletteChoice::Inferno,
        PaletteChoice::Viridis,
        PaletteChoice::Turbo,
        PaletteChoice::Grayscale,
    ];

    pub fn label(self) -> &'static str {
        match self {
            PaletteChoice::Inferno => "Inferno",
            PaletteChoice::Viridis => "Viridis",
            PaletteChoice::Turbo => "Turbo",
            PaletteChoice::Grayscale => "Grayscale",
        }
    }

    /// Nearest equivalent for the raster exporter, which has no grayscale ramp.
    pub fn to_viz(self) -> Palette {
        match self {
            PaletteChoice::Viridis => Palette::Viridis,
            PaletteChoice::Turbo => Palette::Turbo,
            _ => Palette::Inferno,
        }
    }
}

const ACCENT: [f32; 4] = [0.21, 0.81, 0.78, 1.0];
const MUTED: [f32; 4] = [0.60, 0.58, 0.66, 1.0];
const WARN: [f32; 4] = [0.95, 0.55, 0.30, 1.0];

pub fn apply_theme(imgui: &mut imgui::Context) {
    let style = imgui.style_mut();
    style.window_rounding = 6.0;
    style.frame_rounding = 4.0;
    style.grab_rounding = 4.0;
    style.scrollbar_rounding = 4.0;
    style.window_padding = [12.0, 12.0];
    style.frame_padding = [7.0, 4.0];
    style.item_spacing = [8.0, 7.0];
    style.window_border_size = 1.0;
    style.window_title_align = [0.02, 0.5];

    let colors = &mut style.colors;
    colors[StyleColor::WindowBg as usize] = [0.07, 0.06, 0.09, 0.94];
    colors[StyleColor::TitleBg as usize] = [0.10, 0.09, 0.13, 1.0];
    colors[StyleColor::TitleBgActive as usize] = [0.13, 0.12, 0.17, 1.0];
    colors[StyleColor::FrameBg as usize] = [0.15, 0.14, 0.19, 1.0];
    colors[StyleColor::FrameBgHovered as usize] = [0.20, 0.19, 0.26, 1.0];
    colors[StyleColor::FrameBgActive as usize] = [0.24, 0.23, 0.31, 1.0];
    colors[StyleColor::Button as usize] = [0.17, 0.30, 0.32, 1.0];
    colors[StyleColor::ButtonHovered as usize] = [0.13, 0.48, 0.47, 1.0];
    colors[StyleColor::ButtonActive as usize] = [0.16, 0.62, 0.60, 1.0];
    colors[StyleColor::SliderGrab as usize] = ACCENT;
    colors[StyleColor::SliderGrabActive as usize] = [0.35, 0.92, 0.88, 1.0];
    colors[StyleColor::CheckMark as usize] = ACCENT;
    colors[StyleColor::Header as usize] = [0.16, 0.24, 0.26, 1.0];
    colors[StyleColor::HeaderHovered as usize] = [0.15, 0.36, 0.36, 1.0];
    colors[StyleColor::HeaderActive as usize] = [0.16, 0.45, 0.44, 1.0];
    colors[StyleColor::Border as usize] = [0.22, 0.21, 0.28, 1.0];
    colors[StyleColor::Separator as usize] = [0.22, 0.21, 0.28, 1.0];
    colors[StyleColor::PlotHistogram as usize] = ACCENT;
    colors[StyleColor::PlotHistogramHovered as usize] = [0.35, 0.92, 0.88, 1.0];
    colors[StyleColor::Text as usize] = [0.90, 0.89, 0.94, 1.0];
    colors[StyleColor::TextDisabled as usize] = MUTED;
}

pub fn draw(ui: &Ui, studio: &mut Studio) {
    view_panel(ui, studio);
    bvh_panel(ui, studio);
    overlay_panel(ui, studio);
    camera_panel(ui, studio);
    export_panel(ui, studio);

    if studio.show_help {
        help_window(ui, studio);
    }
}

// ---------------------------------------------------------------------------

/// imgui puts a widget's label to its right, so every window reserves a fixed
/// gutter for labels; without this the longer ones are clipped by the window edge.
const LABEL_GUTTER: f32 = -140.0;

fn view_panel(ui: &Ui, studio: &mut Studio) {
    ui.window("View")
        .position([16.0, 16.0], Condition::FirstUseEver)
        .size([360.0, 400.0], Condition::FirstUseEver)
        .build(|| {
            let _width = ui.push_item_width(LABEL_GUTTER);
            ui.text_colored(MUTED, "WHAT THE VIEWPORT SHOWS");

            let mut mode_index = DisplayMode::ALL
                .iter()
                .position(|m| *m == studio.live.display_mode)
                .unwrap_or(0);

            if ui.combo("Mode", &mut mode_index, &DisplayMode::ALL, |mode| {
                Cow::Borrowed(mode.label())
            }) {
                studio.live.display_mode = DisplayMode::ALL[mode_index];
                // Each metric lives on a different order of magnitude, so the
                // ramp ceiling follows the mode unless it has been pinned.
                if studio.auto_scale {
                    studio.live.heat_scale = studio.live.display_mode.default_scale();
                }
                studio.actions.push(Action::ResetAccumulation);
            }

            if studio.live.display_mode != DisplayMode::Beauty {
                let mut palette_index = PaletteChoice::ALL
                    .iter()
                    .position(|p| *p == studio.palette_choice)
                    .unwrap_or(0);

                if ui.combo("Palette", &mut palette_index, &PaletteChoice::ALL, |p| {
                    Cow::Borrowed(p.label())
                }) {
                    studio.palette_choice = PaletteChoice::ALL[palette_index];
                    studio.actions.push(Action::ResetAccumulation);
                }

                if ui.slider("Ramp max", 1.0, 200.0, &mut studio.live.heat_scale) {
                    studio.actions.push(Action::ResetAccumulation);
                }
                tooltip(ui, "Value that maps to the top of the colour ramp. Lower it to bring out detail in cheap regions.");

                if ui.checkbox("Follow mode default", &mut studio.auto_scale) && studio.auto_scale {
                    studio.live.heat_scale = studio.live.display_mode.default_scale();
                    studio.actions.push(Action::ResetAccumulation);
                }

                if ui.slider("Blend with render", 0.0, 1.0, &mut studio.live.heat_mix) {
                    studio.actions.push(Action::ResetAccumulation);
                }
                tooltip(ui, "Cross-fades the cost view back toward the rendered image, so you can tell which surface a hot region belongs to.");

                colour_ramp_preview(ui, studio);
            }

            ui.separator();
            ui.text_colored(MUTED, "SAMPLING");

            if ui.slider("Samples / frame", 1, 16, &mut studio.live.samples) {
                studio.actions.push(Action::ResetAccumulation);
            }
            tooltip(ui, "More samples per dispatch converges faster but drops the frame rate. Total quality comes from accumulation either way.");

            if ui.slider("Max bounces", 1, 24, &mut studio.live.max_bounces) {
                studio.actions.push(Action::ResetAccumulation);
            }
            tooltip(ui, "Path depth. Raising it means more secondary rays, which raises traversal cost across the board.");

            let mut seed = studio.live.rng_seed as i32;
            if ui.input_int("RNG seed", &mut seed).build() {
                studio.live.rng_seed = seed.max(0) as u32;
                studio.actions.push(Action::ResetAccumulation);
            }

            if ui.button("Reset accumulation") {
                studio.actions.push(Action::ResetAccumulation);
            }

            ui.separator();
            ui.text_colored(MUTED, "FRAME BUDGET");
            ui.text(format!("{:.1} ms   {:.0} fps", studio.frame_ms, 1000.0 / studio.frame_ms.max(0.001)));

            let trace_ms = crate::profiler::frame_section_ms("render gpu");
            let accum_ms = crate::profiler::frame_section_ms("cpu accumulation");
            let overlay_ms = crate::profiler::frame_section_ms("overlay");
            // The panels are built before the frame is presented, so whatever is
            // left over is the UI and present cost carried from last frame.
            let rest_ms = (studio.frame_ms as f64 - trace_ms - accum_ms - overlay_ms).max(0.0);

            stat_row(ui, "path trace", &format!("{trace_ms:.1} ms"));
            stat_row(ui, "accumulate", &format!("{accum_ms:.1} ms"));
            stat_row(ui, "wireframe", &format!("{overlay_ms:.1} ms"));
            stat_row(ui, "ui + present", &format!("{rest_ms:.1} ms"));
            if studio.view.enabled && studio.view.depth_focus.is_none() && studio.stats.node_count > 800 {
                ui.text_colored(WARN, "overlay is drawing every box");
                tooltip(ui, "The wireframe is rasterised on the CPU. Pick a single depth level in the overlay panel to get the frame rate back.");
            }
        });
}

/// Draws the active ramp as a strip of coloured bars, so the mapping from colour
/// to cost is visible without leaving the app.
fn colour_ramp_preview(ui: &Ui, studio: &Studio) {
    let palette = studio.palette_choice;
    let draw_list = ui.get_window_draw_list();
    let [x, y] = ui.cursor_screen_pos();
    let width = ui.content_region_avail()[0].max(40.0);
    let height = 14.0;
    let steps = 48;

    for step in 0..steps {
        let t = step as f32 / (steps - 1) as f32;
        let colour = match palette {
            PaletteChoice::Grayscale => [t, t, t],
            other => {
                let [r, g, b] = other.to_viz().sample(t);
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
            }
        };
        let x0 = x + width * (step as f32 / steps as f32);
        let x1 = x + width * ((step + 1) as f32 / steps as f32);
        draw_list
            .add_rect([x0, y], [x1, y + height], colour)
            .filled(true)
            .build();
    }

    ui.dummy([width, height + 2.0]);
    ui.text_colored(MUTED, format!("0 .. {:.0}", studio.live.heat_scale));
}

// ---------------------------------------------------------------------------

fn bvh_panel(ui: &Ui, studio: &mut Studio) {
    ui.window("BVH")
        .position([16.0, 428.0], Condition::FirstUseEver)
        .size([360.0, 456.0], Condition::FirstUseEver)
        .build(|| {
            let _width = ui.push_item_width(LABEL_GUTTER);
            ui.text_colored(MUTED, "CONSTRUCTION");

            if ui.combo("Heuristic", &mut studio.heuristic_index, &SplitHeuristic::ALL, |h| {
                Cow::Borrowed(h.name())
            }) {
                studio.build.heuristic = SplitHeuristic::ALL[studio.heuristic_index];
                studio.actions.push(Action::RebuildBvh);
            }

            let mut cutoff = studio.build.leaf_cutoff as i32;
            if ui.slider("Leaf cutoff", 1, 64, &mut cutoff) {
                studio.build.leaf_cutoff = cutoff.max(1) as usize;
                studio.actions.push(Action::RebuildBvh);
            }
            tooltip(ui, "Maximum primitives in a leaf. Larger leaves mean a smaller tree and more triangle tests per hit.");

            let mut seed = studio.build.seed as i32;
            if ui.input_int("Split seed", &mut seed).build() {
                studio.build.seed = seed.max(0) as u64;
                if studio.build.heuristic == SplitHeuristic::Random {
                    studio.actions.push(Action::RebuildBvh);
                }
            }
            tooltip(ui, "Only affects the Random heuristic. Same seed rebuilds the same tree.");

            if ui.button("Rebuild") {
                studio.actions.push(Action::RebuildBvh);
            }
            ui.same_line();
            if ui.button("Cycle heuristic") {
                studio.heuristic_index = (studio.heuristic_index + 1) % SplitHeuristic::ALL.len();
                studio.build.heuristic = SplitHeuristic::ALL[studio.heuristic_index];
                studio.actions.push(Action::RebuildBvh);
            }

            ui.separator();
            ui.text_colored(MUTED, "TREE QUALITY");

            let stats = &studio.stats;
            stat_row(ui, "Primitives", &format!("{}", studio.prim_count));
            stat_row(ui, "Nodes", &format!("{}", stats.node_count));
            stat_row(ui, "Leaves", &format!("{}", stats.leaf_count));
            stat_row(ui, "Max depth", &format!("{}", stats.max_depth));
            stat_row(ui, "Avg leaf depth", &format!("{:.2}", stats.avg_depth));
            stat_row(ui, "Avg leaf prims", &format!("{:.2}", stats.avg_leaf_prims));
            stat_row(ui, "SAH cost", &format!("{:.2}", stats.total_sah_cost));
            stat_row(ui, "Build time", &format!("{:.2} ms", stats.build_time_ms));

            if stats.max_depth >= 64 {
                ui.text_colored(WARN, "depth meets the 64-entry GPU stack limit");
                tooltip(ui, "Traversals deeper than the shader stack bail out early, so counts for this tree are a lower bound.");
            }

            if !studio.depth_profile.is_empty() {
                ui.separator();
                ui.text_colored(MUTED, "NODES PER DEPTH");
                ui.plot_histogram("##depth", &studio.depth_profile)
                    .graph_size([ui.content_region_avail()[0], 70.0])
                    .build();
                ui.text_colored(MUTED, format!("depth 0 .. {}", studio.depth_profile.len().saturating_sub(1)));
            }
        });
}

fn stat_row(ui: &Ui, label: &str, value: &str) {
    ui.text_colored(MUTED, label);
    ui.same_line_with_pos(150.0);
    ui.text(value);
}

// ---------------------------------------------------------------------------

fn overlay_panel(ui: &Ui, studio: &mut Studio) {
    ui.window("Wireframe overlay")
        .position([388.0, 16.0], Condition::FirstUseEver)
        .size([360.0, 400.0], Condition::FirstUseEver)
        .build(|| {
            let _width = ui.push_item_width(LABEL_GUTTER);
            ui.checkbox("Draw overlay", &mut studio.view.enabled);
            ui.checkbox("Leaves only", &mut studio.view.leaves_only);
            tooltip(ui, "Hides interior boxes. Useful once you want to see only where geometry actually sits.");

            ui.separator();
            ui.text_colored(MUTED, "DEPTH SLICE");

            let mut all_depths = studio.view.depth_focus.is_none();
            if ui.checkbox("All levels", &mut all_depths) {
                studio.view.depth_focus = if all_depths { None } else { Some(0) };
            }

            if let Some(depth) = studio.view.depth_focus {
                let mut value = depth as i32;
                let max = studio.view.max_depth as i32;
                ui.slider("Level", 0, max.max(0), &mut value);
                studio.view.depth_focus = Some(value.clamp(0, max.max(0)) as usize);

                if ui.button("<") && value > 0 {
                    studio.view.depth_focus = Some((value - 1) as usize);
                }
                ui.same_line();
                if ui.button(">") && value < max {
                    studio.view.depth_focus = Some((value + 1) as usize);
                }
                ui.same_line();
                ui.text_colored(MUTED, format!("of {}", studio.view.max_depth));
            } else {
                ui.text_colored(MUTED, format!("drawing all {} levels", studio.view.max_depth + 1));
            }

            ui.separator();
            ui.text_colored(MUTED, "APPEARANCE");

            let ramps = [Palette::Turbo, Palette::Inferno, Palette::Viridis];
            let names = ["Turbo", "Inferno", "Viridis"];
            let mut ramp_index = ramps.iter().position(|p| *p == studio.view.palette).unwrap_or(0);
            if ui.combo("Depth ramp", &mut ramp_index, &names, |name| Cow::Borrowed(*name)) {
                studio.view.palette = ramps[ramp_index];
            }

            ui.slider("Interior dim", 0.0, 1.0, &mut studio.view.interior_dim);
            tooltip(ui, "How bright interior boxes are relative to leaves. Drop it to near zero to let the leaves dominate.");

            ui.checkbox("Brighten full leaves", &mut studio.view.highlight_full_leaves);
            tooltip(ui, "Scales each leaf's brightness by how close it is to the primitive cutoff, so the boxes that cost the most stand out.");

            ui.separator();
            if ui.button("Export one image per level") {
                studio.actions.push(Action::ExportDepthSweep);
            }
            tooltip(ui, "Writes the current accumulated frame once per tree level, with only that level's boxes drawn.");
        });
}

// ---------------------------------------------------------------------------

fn camera_panel(ui: &Ui, studio: &mut Studio) {
    ui.window("Camera")
        .position([388.0, 428.0], Condition::FirstUseEver)
        .size([360.0, 300.0], Condition::FirstUseEver)
        .build(|| {
            let _width = ui.push_item_width(LABEL_GUTTER);
            let ray = studio.camera.ray();
            let mut origin = [ray.origin().x, ray.origin().y, ray.origin().z];

            if ui.input_float3("Position", &mut origin).build() {
                studio.camera.set_ray(crate::ray::Ray::new(
                    glam::Vec3::new(origin[0], origin[1], origin[2]),
                    ray.direction(),
                ));
                studio.actions.push(Action::ResetAccumulation);
            }

            if ui.slider("Yaw", -180.0, 180.0, &mut studio.yaw) {
                studio.actions.push(Action::ResetAccumulation);
            }
            if ui.slider("Pitch", -89.0, 89.0, &mut studio.pitch) {
                studio.actions.push(Action::ResetAccumulation);
            }

            ui.separator();
            ui.slider("Move speed", 0.2, 20.0, &mut studio.move_speed);
            ui.slider("Look sensitivity", 0.02, 0.5, &mut studio.look_sensitivity);

            ui.separator();
            if ui.button("Frame the model") {
                studio.actions.push(Action::FrameModel);
            }
            tooltip(ui, "Backs the camera off until the mesh fills the frame.");

            ui.separator();
            ui.text_colored(MUTED, "Hold right mouse to look, WASD to move,");
            ui.text_colored(MUTED, "space / shift for up and down.");
        });
}

// ---------------------------------------------------------------------------

fn export_panel(ui: &Ui, studio: &mut Studio) {
    // Anchored to the right edge of whatever window size the studio opened at,
    // so it is not clipped on a smaller display.
    let display_width = ui.io().display_size[0];
    let panel_width = 360.0;
    let x = (display_width - panel_width - 16.0).max(760.0);

    ui.window("Export")
        .position([x, 16.0], Condition::FirstUseEver)
        .size([panel_width, 560.0], Condition::FirstUseEver)
        .build(|| {
            let _width = ui.push_item_width(LABEL_GUTTER);
            ui.input_text("Output folder", &mut studio.output_dir).build();

            ui.separator();
            if ui.collapsing_header("Viewport capture", TreeNodeFlags::DEFAULT_OPEN) {
                ui.text_colored(MUTED, "Saves exactly what is on screen now,");
                ui.text_colored(MUTED, "overlay included, at the window size.");
                if ui.button("Save viewport PNG") {
                    studio.actions.push(Action::SaveViewport);
                }
                ui.same_line();
                if ui.button("Save structure SVGs") {
                    studio.actions.push(Action::ExportStructureDiagrams);
                }
                tooltip(ui, "Writes the icicle plot and depth histogram for the tree currently loaded.");
            }

            ui.separator();
            if ui.collapsing_header("Full figure set", TreeNodeFlags::DEFAULT_OPEN) {
                ui.text_colored(MUTED, "Runs all four heuristics from the current");
                ui.text_colored(MUTED, "camera and writes heatmaps, contact");
                ui.text_colored(MUTED, "sheets, difference maps and diagrams.");

                let mut width = studio.export_width as i32;
                if ui.input_int("Width", &mut width).build() {
                    studio.export_width = width.clamp(64, 4096) as u32;
                }
                let mut height = studio.export_height as i32;
                if ui.input_int("Height", &mut height).build() {
                    studio.export_height = height.clamp(64, 4096) as u32;
                }

                let mut samples = studio.export_samples as i32;
                if ui.slider("Samples", 1, 256, &mut samples) {
                    studio.export_samples = samples.max(1) as u32;
                }
                tooltip(ui, "Figure quality scales with this. Below about 16 the heatmaps carry visible sampling noise.");

                let mut warmup = studio.export_warmup as i32;
                if ui.slider("Warm-up runs", 0, 10, &mut warmup) {
                    studio.export_warmup = warmup.max(0) as u32;
                }
                let mut runs = studio.export_runs as i32;
                if ui.slider("Timed runs", 1, 20, &mut runs) {
                    studio.export_runs = runs.max(1) as u32;
                }

                ui.separator();

                if studio.harness_running {
                    ui.text_colored(ACCENT, "running...");
                    ui.text_colored(MUTED, "the viewport stays live while it works");
                } else if ui.button_with_size("Generate figure set", [-1.0, 30.0]) {
                    studio.actions.push(Action::ExportFigureSet);
                }
            }

            ui.separator();
            ui.text_colored(MUTED, "STATUS");
            ui.text_wrapped(&studio.status);

            ui.separator();
            if ui.button("Controls") {
                studio.show_help = !studio.show_help;
            }
        });
}

// ---------------------------------------------------------------------------

fn help_window(ui: &Ui, studio: &mut Studio) {
    let mut open = studio.show_help;
    ui.window("Controls")
        .opened(&mut open)
        .position([400.0, 200.0], Condition::FirstUseEver)
        .size([420.0, 320.0], Condition::FirstUseEver)
        .build(|| {
            ui.text_colored(ACCENT, "CAMERA");
            ui.bullet_text("Right mouse drag - look around");
            ui.bullet_text("W A S D - move, Space / Shift - up and down");
            ui.bullet_text("Camera panel - type an exact position, or frame the model");

            ui.separator();
            ui.text_colored(ACCENT, "READING A COST VIEW");
            ui.text_wrapped(
                "Node visits counts every bounding box tested; triangle tests counts every \
                 ray-triangle intersection attempted. Both are divided by the number of traversals \
                 the pixel issued, so deep paths are not penalised. Neither depends on the GPU.",
            );

            ui.separator();
            ui.text_colored(ACCENT, "COMPARING HEURISTICS");
            ui.text_wrapped(
                "Pin the ramp maximum before switching heuristics, otherwise each view rescales \
                 and they all look the same. Turn off 'Follow mode default' to keep it fixed.",
            );

            ui.separator();
            ui.text_colored(ACCENT, "EXPORTING");
            ui.text_wrapped(
                "'Save viewport' captures the window as-is. 'Generate figure set' re-runs all four \
                 heuristics offscreen at the chosen resolution and writes the full set of figures \
                 with a shared colour scale, which is what makes panels comparable.",
            );
        });
    studio.show_help = open;
}

fn tooltip(ui: &Ui, text: &str) {
    if ui.is_item_hovered() {
        ui.tooltip(|| {
            let wrap = ui.push_text_wrap_pos_with_pos(320.0);
            ui.text(text);
            wrap.end();
        });
    }
}
