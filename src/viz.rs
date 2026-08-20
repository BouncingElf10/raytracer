//! Raster figure generation for the BVH study.
//!
//! Turns the per-pixel counters collected in Pass A into publication-style
//! heatmaps, and composes them into labelled contact sheets so heuristics can be
//! compared side by side.
//!
//! The one rule that matters for honest figures: heatmaps that will be compared
//! must share a colour scale. `Scale::shared_over` computes one scale across a
//! whole group and every image in the group is rendered against it, so a lighter
//! image always means genuinely less work -- never just a different normalisation.
//!
//! The figures are drawn light-on-white throughout, because they are read in a
//! document rather than on a screen. `BG`/`FG`/`MUTED` and the ramps below are
//! the single place that decision lives.

use std::path::Path;

// ---- Image buffer ----------------------------------------------------------

#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGB8.
    pixels: Vec<u8>,
}

// Light theme throughout: these figures are read on a white page (the study
// write-up is a document, not a terminal), so the panel ground is the page and
// ink is dark. Everything downstream -- ramps, wireframe hues, silhouettes --
// is anchored to that choice.
pub const BG: [u8; 3] = [255, 255, 255];
pub const FG: [u8; 3] = [26, 26, 32];
pub const MUTED: [u8; 3] = [104, 104, 116];
/// The page a contact sheet lays its panels on, a shade off white so the panels
/// read as separate objects rather than as one flood of white.
pub const SHEET_BG: [u8; 3] = [243, 243, 247];
/// Hairline around a panel. Without it a heatmap whose empty space is white has
/// no edge at all once it is pasted into a document.
pub const PANEL_BORDER: [u8; 3] = [205, 205, 214];

impl Image {
    pub fn new(width: u32, height: u32, fill: [u8; 3]) -> Self {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&fill);
        }
        Self { width, height, pixels }
    }

    /// Wraps an existing tightly-packed RGB8 buffer.
    pub fn from_rgb(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), (width * height * 3) as usize, "buffer must be tightly packed RGB8");
        Self { width, height, pixels }
    }

    pub fn set(&mut self, x: u32, y: u32, color: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let offset = ((y * self.width + x) * 3) as usize;
        self.pixels[offset..offset + 3].copy_from_slice(&color);
    }

    pub fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: [u8; 3]) {
        for dy in 0..h {
            for dx in 0..w {
                self.set(x + dx, y + dy, color);
            }
        }
    }

    pub fn stroke_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: [u8; 3]) {
        if w == 0 || h == 0 {
            return;
        }
        for dx in 0..w {
            self.set(x + dx, y, color);
            self.set(x + dx, y + h - 1, color);
        }
        for dy in 0..h {
            self.set(x, y + dy, color);
            self.set(x + w - 1, y + dy, color);
        }
    }

    /// Bresenham line between two integer screen points. Coordinates are signed
    /// because projected geometry routinely lands outside the frame; `set`
    /// discards anything off-canvas.
    pub fn draw_line(&mut self, a: (i32, i32), b: (i32, i32), color: [u8; 3]) {
        let (mut x, mut y) = a;
        let (x1, y1) = b;

        let dx = (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let step_x = if x < x1 { 1 } else { -1 };
        let step_y = if y < y1 { 1 } else { -1 };
        let mut error = dx + dy;

        // A line whose endpoints are both far off-canvas can iterate for a very
        // long time; the span of a projected box is bounded in practice, but cap
        // it so a degenerate projection cannot hang the figure pass.
        let budget = (dx - dy) as u32 + 2;
        for _ in 0..budget.min(1 << 16) {
            if x >= 0 && y >= 0 {
                self.set(x as u32, y as u32, color);
            }
            if x == x1 && y == y1 {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x += step_x;
            }
            if doubled <= dx {
                error += dx;
                y += step_y;
            }
        }
    }

    pub fn blit(&mut self, source: &Image, x: u32, y: u32) {
        for sy in 0..source.height {
            for sx in 0..source.width {
                let offset = ((sy * source.width + sx) * 3) as usize;
                let color = [
                    source.pixels[offset],
                    source.pixels[offset + 1],
                    source.pixels[offset + 2],
                ];
                self.set(x + sx, y + sy, color);
            }
        }
    }

    /// Draws `text` with the built-in 5x7 font. Returns the width consumed.
    pub fn draw_text(&mut self, x: u32, y: u32, text: &str, scale: u32, color: [u8; 3]) -> u32 {
        let mut cursor = x;
        for character in text.chars() {
            let rows = glyph(character);
            for (row_index, bits) in rows.iter().enumerate() {
                for column in 0..GLYPH_WIDTH {
                    // Bit 4 is the leftmost column.
                    if bits & (1 << (GLYPH_WIDTH - 1 - column)) != 0 {
                        self.fill_rect(
                            cursor + column as u32 * scale,
                            y + row_index as u32 * scale,
                            scale,
                            scale,
                            color,
                        );
                    }
                }
            }
            cursor += (GLYPH_WIDTH as u32 + 1) * scale;
        }
        cursor - x
    }

    pub fn text_width(text: &str, scale: u32) -> u32 {
        (text.chars().count() as u32) * (GLYPH_WIDTH as u32 + 1) * scale
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        image::save_buffer(
            path,
            &self.pixels,
            self.width,
            self.height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(std::io::Error::other)
    }
}

// ---- Colour maps -----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    /// Perceptually uniform, page-to-dark. Good default for cost heatmaps.
    Inferno,
    /// Perceptually uniform, colour-blind safe.
    Viridis,
    /// High contrast rainbow. Reads well for "how deep did this ray go".
    Turbo,
}

// Inferno, run backwards from its bright end and anchored on the page colour, so
// that low cost fades into white and high cost is the darkest ink on the sheet.
// Reversing a perceptually uniform ramp keeps it perceptually uniform; what it
// changes is which end of the scale disappears into the paper, and on a white
// page that has to be the cheap end.
//
// The pale end is given two extra stops that inferno does not have. Under a log
// ramp the bottom of the scale covers most of the frame, and going straight from
// white to inferno's full yellow floods the panel with colour at values that are
// barely above nothing; easing through cream keeps "almost free" looking almost
// blank.
const INFERNO: &[[u8; 3]] = &[
    [255, 255, 255], [255, 251, 214], [253, 246, 150], [250, 213, 95], [246, 186, 39],
    [243, 128, 26], [221, 81, 58], [188, 55, 84], [147, 38, 103], [106, 23, 110],
    [66, 10, 104], [22, 11, 57], [0, 0, 4],
];

const VIRIDIS: &[[u8; 3]] = &[
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];

const TURBO: &[[u8; 3]] = &[
    [48, 18, 59], [70, 107, 227], [54, 166, 249], [25, 214, 203], [60, 234, 141],
    [129, 248, 75], [191, 240, 45], [238, 206, 50], [253, 152, 39], [238, 86, 17],
    [165, 17, 2],
];

/// Cool -> page -> warm, for signed difference maps.
///
/// The neutral midpoint is the page colour, so "no difference" is literally
/// blank and only the pixels where two heuristics disagree carry ink. Both arms
/// darken away from the centre, which is what makes the magnitude readable in
/// print and in greyscale.
const DIVERGENT: &[[u8; 3]] = &[
    [16, 60, 122], [43, 110, 190], [138, 184, 236], [255, 255, 255],
    [244, 166, 138], [201, 72, 46], [122, 24, 16],
];

fn ramp(stops: &[[u8; 3]], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let scaled = t * (stops.len() - 1) as f32;
    let index = (scaled.floor() as usize).min(stops.len() - 2);
    let frac = scaled - index as f32;

    let (a, b) = (stops[index], stops[index + 1]);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * frac) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * frac) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * frac) as u8,
    ]
}

impl Palette {
    pub fn sample(self, t: f32) -> [u8; 3] {
        match self {
            Palette::Inferno => ramp(INFERNO, t),
            Palette::Viridis => ramp(VIRIDIS, t),
            Palette::Turbo => ramp(TURBO, t),
        }
    }
}

/// `t` in [-1, 1]; 0 maps to neutral.
pub fn divergent(t: f32) -> [u8; 3] {
    ramp(DIVERGENT, (t.clamp(-1.0, 1.0) + 1.0) * 0.5)
}

// ---- Scaling ---------------------------------------------------------------

/// How a value in [0, `Scale::max`] is mapped onto the colour ramp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleKind {
    Linear,
    /// `ln(1 + v) / ln(1 + max)`. Used when the dynamic range is so wide that a
    /// linear ramp would push the bulk of the image into the first few percent.
    /// The `1 +` keeps zero at zero, so empty pixels stay empty.
    Log,
}

/// The value range a group of heatmaps is rendered against.
#[derive(Debug, Clone, Copy)]
pub struct Scale {
    pub max: f32,
    pub kind: ScaleKind,
}

impl Scale {
    /// One scale covering every field in `fields`, so images rendered against it
    /// are directly comparable.
    ///
    /// The maximum is the 99.5th percentile rather than the true maximum: a
    /// handful of pathological pixels (grazing rays down the length of a thin
    /// box) would otherwise compress the entire rest of the image into the first
    /// few percent of the ramp. Values above it are clamped, which the colourbar
    /// marks with a `+`.
    pub fn shared_over<'a>(fields: impl IntoIterator<Item = &'a [f32]>) -> Self {
        let mut all: Vec<f32> = Vec::new();
        for field in fields {
            all.extend(field.iter().copied().filter(|v| *v > 0.0));
        }

        if all.is_empty() {
            return Scale { max: 1.0, kind: ScaleKind::Linear };
        }

        all.sort_by(f32::total_cmp);
        let rank = ((0.995 * all.len() as f32).ceil() as usize).clamp(1, all.len());
        Scale { max: all[rank - 1].max(1e-6), kind: ScaleKind::Linear }
    }

    /// True maximum over every field, with no percentile clipping, and a ramp
    /// chosen from the observed dynamic range.
    ///
    /// This is the scale for figures whose caption claims a *shared absolute*
    /// colour scale: nothing is clipped, so a pixel's colour maps back to a real
    /// number on the colourbar. When the top of the range is more than 20x the
    /// median of the non-empty pixels a linear ramp would leave every typical
    /// pixel all but blank -- traversal fields routinely run 40:1 or worse, because a
    /// handful of grazing rays cost far more than any typical one -- so the ramp
    /// switches to logarithmic, which the caller states in the caption via
    /// `Scale::kind_label`.
    pub fn absolute_over<'a>(fields: impl IntoIterator<Item = &'a [f32]>) -> Self {
        let mut all: Vec<f32> = Vec::new();
        for field in fields {
            all.extend(field.iter().copied().filter(|v| *v > 0.0));
        }

        if all.is_empty() {
            return Scale { max: 1.0, kind: ScaleKind::Linear };
        }

        all.sort_by(f32::total_cmp);
        let max = all[all.len() - 1].max(1e-6);
        let median = all[all.len() / 2].max(1e-6);
        let kind = if max / median > 20.0 { ScaleKind::Log } else { ScaleKind::Linear };

        Scale { max, kind }
    }

    /// Maps a value onto [0, 1] for the colour ramp.
    pub fn normalize(&self, value: f32) -> f32 {
        match self.kind {
            ScaleKind::Linear => value / self.max,
            ScaleKind::Log => {
                let denominator = (1.0 + self.max).ln();
                if denominator <= 0.0 { 0.0 } else { (1.0 + value.max(0.0)).ln() / denominator }
            }
        }
    }

    /// Tick values for the colourbar, in ramp order. Linear ticks are evenly
    /// spaced in value; log ticks are evenly spaced in ramp *position*, which is
    /// what makes a log colourbar readable.
    pub fn ticks(&self, count: usize) -> Vec<f32> {
        let count = count.max(2);
        (0..count)
            .map(|index| {
                let t = index as f32 / (count - 1) as f32;
                match self.kind {
                    ScaleKind::Linear => t * self.max,
                    ScaleKind::Log => ((1.0 + self.max).ln() * t).exp() - 1.0,
                }
            })
            .collect()
    }

    pub fn kind_label(&self) -> &'static str {
        match self.kind {
            ScaleKind::Linear => "linear",
            ScaleKind::Log => "log",
        }
    }
}

// ---- Heatmaps --------------------------------------------------------------

pub struct HeatmapSpec<'a> {
    pub values: &'a [f32],
    pub width: u32,
    pub height: u32,
    pub scale: Scale,
    pub palette: Palette,
    /// Large label across the top, e.g. the heuristic name.
    pub title: &'a str,
    /// Smaller line under the title, e.g. "4.52 tests/ray".
    pub subtitle: &'a str,
    /// Colourbar caption, e.g. "node visits per ray".
    pub legend: &'a str,
}

const HEADER_HEIGHT: u32 = 34;
const BAR_HEIGHT: u32 = 40;
const PAD: u32 = 8;

/// Renders one field as a titled heatmap with a colourbar underneath.
pub fn render_heatmap(spec: &HeatmapSpec) -> Image {
    let total_height = HEADER_HEIGHT + spec.height + BAR_HEIGHT;
    let mut canvas = Image::new(spec.width, total_height, BG);

    canvas.draw_text(PAD, 6, &spec.title.to_uppercase(), 2, FG);
    canvas.draw_text(PAD, 22, &spec.subtitle.to_uppercase(), 1, MUTED);

    for y in 0..spec.height {
        for x in 0..spec.width {
            let value = spec.values[(y * spec.width + x) as usize];
            let color = if value <= 0.0 {
                // Zero is "no traversal happened here", and it stays the page
                // colour. On the light ramp that sits just beyond the cheapest
                // value rather than opposite it, so empty space and near-free
                // space no longer read as two unrelated things.
                BG
            } else {
                spec.palette.sample(spec.scale.normalize(value))
            };
            canvas.set(x, HEADER_HEIGHT + y, color);
        }
    }

    frame_field(&mut canvas, HEADER_HEIGHT, spec.width, spec.height);
    draw_colorbar(&mut canvas, HEADER_HEIGHT + spec.height, spec);
    canvas
}

/// Outlines the image area of a panel.
///
/// On the dark theme the field was bounded by its own darkness against the page;
/// white-on-white has no such edge, and a heatmap pasted into a document needs
/// one or it bleeds into the paragraph around it.
fn frame_field(canvas: &mut Image, top: u32, width: u32, height: u32) {
    canvas.stroke_rect(0, top, width, height, PANEL_BORDER);
}

fn draw_colorbar(canvas: &mut Image, top: u32, spec: &HeatmapSpec) {
    let bar_width = canvas.width.saturating_sub(PAD * 2);
    let bar_top = top + 6;
    let bar_height = 8;

    for x in 0..bar_width {
        let color = spec.palette.sample(x as f32 / bar_width.max(1) as f32);
        for y in 0..bar_height {
            canvas.set(PAD + x, bar_top + y, color);
        }
    }
    canvas.stroke_rect(PAD, bar_top, bar_width, bar_height, MUTED);

    // Numbered ticks along the bar, so a colour can be read back as a value
    // rather than only as "more" or "less". Five is as many as the 5x7 font fits
    // across a 640px panel without labels colliding.
    let label_y = bar_top + bar_height + 4;
    let ticks = spec.scale.ticks(5);
    let tick_count = ticks.len();
    for (index, value) in ticks.iter().enumerate() {
        let t = index as f32 / (tick_count - 1) as f32;
        let x = PAD + (t * bar_width.saturating_sub(1) as f32) as u32;

        for y in 0..3 {
            canvas.set(x, bar_top + bar_height + y, MUTED);
        }

        let label = format_number(*value);
        let label_width = Image::text_width(&label, 1);
        // Nudge the end labels inward so neither runs off the panel.
        let label_x = (x + label_width / 2)
            .saturating_sub(label_width)
            .min(PAD + bar_width - label_width);
        canvas.draw_text(label_x.max(PAD), label_y, &label, 1, MUTED);
    }

    let legend = format!("{}  ({} scale)", spec.legend, spec.scale.kind_label()).to_uppercase();
    let legend_width = Image::text_width(&legend, 1);
    canvas.draw_text(
        PAD + bar_width / 2 - (legend_width / 2).min(bar_width / 2),
        label_y + 9,
        &legend,
        1,
        MUTED,
    );
}

/// Signed difference between two fields, rendered against a divergent ramp.
/// Blue = `a` cheaper than `b`, red = `a` more expensive.
pub fn render_difference(
    a: &[f32],
    b: &[f32],
    width: u32,
    height: u32,
    title: &str,
    subtitle: &str,
) -> Image {
    let peak = difference_peak([(a, b)]);
    render_difference_scaled(a, b, peak, width, height, title, subtitle)
}

/// The symmetric half-range a set of difference maps should share.
///
/// Taken from the 99th percentile of the magnitudes that actually differ rather
/// than the raw maximum: a handful of extreme pixels would otherwise push every
/// typical difference into the neutral band and the figures would read blank.
/// Computed across *all* pairs at once so a group of difference panels can be
/// compared against each other, not just each against itself.
pub fn difference_peak<'a>(pairs: impl IntoIterator<Item = (&'a [f32], &'a [f32])>) -> f32 {
    let mut magnitudes: Vec<f32> = Vec::new();
    for (a, b) in pairs {
        magnitudes.extend(
            a.iter()
                .zip(b)
                .map(|(left, right)| (left - right).abs())
                .filter(|d| *d > 0.0),
        );
    }
    if magnitudes.is_empty() {
        return 1.0;
    }
    magnitudes.sort_by(f32::total_cmp);
    let rank = ((0.99 * magnitudes.len() as f32).ceil() as usize).clamp(1, magnitudes.len());
    magnitudes[rank - 1].max(1e-6)
}

/// As `render_difference`, but against a caller-supplied symmetric half-range so
/// several panels can share one scale.
pub fn render_difference_scaled(
    a: &[f32],
    b: &[f32],
    peak: f32,
    width: u32,
    height: u32,
    title: &str,
    subtitle: &str,
) -> Image {
    let peak = peak.max(1e-6);
    let mut canvas = Image::new(width, HEADER_HEIGHT + height + BAR_HEIGHT, BG);
    canvas.draw_text(PAD, 6, &title.to_uppercase(), 2, FG);
    canvas.draw_text(PAD, 22, &subtitle.to_uppercase(), 1, MUTED);

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let delta = a[index] - b[index];
            let color = if a[index] <= 0.0 && b[index] <= 0.0 {
                BG
            } else {
                divergent(delta / peak)
            };
            canvas.set(x, HEADER_HEIGHT + y, color);
        }
    }

    frame_field(&mut canvas, HEADER_HEIGHT, width, height);

    let bar_width = width.saturating_sub(PAD * 2);
    let bar_top = HEADER_HEIGHT + height + 6;
    for x in 0..bar_width {
        let t = (x as f32 / bar_width.max(1) as f32) * 2.0 - 1.0;
        for y in 0..8 {
            canvas.set(PAD + x, bar_top + y, divergent(t));
        }
    }
    canvas.stroke_rect(PAD, bar_top, bar_width, 8, MUTED);

    let label_y = bar_top + 12;
    canvas.draw_text(PAD, label_y, &format!("-{}", format_number(peak)), 1, MUTED);
    let right = format!("+{}", format_number(peak));
    canvas.draw_text(
        PAD + bar_width - Image::text_width(&right, 1),
        label_y,
        &right,
        1,
        MUTED,
    );
    // The centre tick is the point of a diverging map: it marks where the two
    // heuristics cost exactly the same.
    let zero_width = Image::text_width("0", 1);
    canvas.draw_text(PAD + bar_width / 2 - zero_width / 2, label_y, "0", 1, MUTED);

    canvas
}

/// Tone-maps a linear HDR colour buffer into a displayable image.
pub fn render_beauty(
    colors: &[[f32; 3]],
    width: u32,
    height: u32,
    title: &str,
    subtitle: &str,
) -> Image {
    let mut canvas = Image::new(width, HEADER_HEIGHT + height, BG);
    canvas.draw_text(PAD, 6, &title.to_uppercase(), 2, FG);
    canvas.draw_text(PAD, 22, &subtitle.to_uppercase(), 1, MUTED);

    for y in 0..height {
        for x in 0..width {
            let rgb = colors[(y * width + x) as usize];
            let encode = |channel: f32| -> u8 {
                // Standard 2.2 gamma; the interactive renderer uses the same curve.
                (channel.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
            };
            canvas.set(x, HEADER_HEIGHT + y, [encode(rgb[0]), encode(rgb[1]), encode(rgb[2])]);
        }
    }

    frame_field(&mut canvas, HEADER_HEIGHT, width, height);
    canvas
}

// ---- Composition -----------------------------------------------------------

/// Lays panels out in a grid under a banner. Used for the per-scene comparison
/// sheets, where seeing all four heuristics at once is the whole point.
pub fn contact_sheet(panels: &[Image], columns: u32, banner: &str, caption: &str) -> Image {
    if panels.is_empty() {
        return Image::new(1, 1, BG);
    }

    let columns = columns.max(1);
    let rows = (panels.len() as u32).div_ceil(columns);
    let panel_width = panels.iter().map(|p| p.width).max().unwrap_or(1);
    let panel_height = panels.iter().map(|p| p.height).max().unwrap_or(1);

    let gap = 10;
    let banner_height = 46;
    let width = columns * panel_width + (columns + 1) * gap;
    let height = banner_height + rows * panel_height + (rows + 1) * gap;

    let mut sheet = Image::new(width, height, SHEET_BG);
    sheet.draw_text(gap, 8, &banner.to_uppercase(), 3, FG);
    sheet.draw_text(gap, 30, &caption.to_uppercase(), 1, MUTED);

    for (index, panel) in panels.iter().enumerate() {
        let column = index as u32 % columns;
        let row = index as u32 / columns;
        let x = gap + column * (panel_width + gap);
        let y = banner_height + gap + row * (panel_height + gap);
        sheet.blit(panel, x, y);
        sheet.stroke_rect(x, y, panel.width, panel.height, PANEL_BORDER);
    }

    sheet
}

/// Compact human-readable number: 1234 -> "1.2K", 0.42 -> "0.42".
pub fn format_number(value: f32) -> String {
    let magnitude = value.abs();
    if magnitude >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if magnitude >= 1_000.0 {
        format!("{:.1}K", value / 1_000.0)
    } else if magnitude >= 100.0 {
        format!("{value:.0}")
    } else if magnitude >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

// ---- 5x7 bitmap font -------------------------------------------------------

const GLYPH_WIDTH: usize = 5;

/// Rows top-to-bottom; bit 4 is the leftmost column.
fn glyph(character: char) -> [u8; 7] {
    match character.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '%' => [0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        _ => [0x00; 7],
    }
}
