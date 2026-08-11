//! Raster figure generation for the BVH study.
//!
//! Turns the per-pixel counters collected in Pass A into publication-style
//! heatmaps, and composes them into labelled contact sheets so heuristics can be
//! compared side by side.
//!
//! The one rule that matters for honest figures: heatmaps that will be compared
//! must share a colour scale. `Scale::shared_over` computes one scale across a
//! whole group and every image in the group is rendered against it, so a darker
//! image always means genuinely less work -- never just a different normalisation.

use std::path::Path;

// ---- Image buffer ----------------------------------------------------------

#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGB8.
    pixels: Vec<u8>,
}

pub const BG: [u8; 3] = [18, 18, 22];
pub const FG: [u8; 3] = [235, 235, 240];
pub const MUTED: [u8; 3] = [150, 150, 160];

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
    /// Perceptually uniform, dark-to-bright. Good default for cost heatmaps.
    Inferno,
    /// Perceptually uniform, colour-blind safe.
    Viridis,
    /// High contrast rainbow. Reads well for "how deep did this ray go".
    Turbo,
}

const INFERNO: &[[u8; 3]] = &[
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
    [188, 55, 84], [221, 81, 58], [243, 128, 26], [246, 186, 39], [252, 255, 164],
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

/// Cool -> dark -> warm, for signed difference maps.
///
/// The neutral midpoint is dark rather than white so "no difference" recedes
/// instead of dominating, and so the figure sits alongside the heatmaps.
const DIVERGENT: &[[u8; 3]] = &[
    [120, 190, 255], [60, 120, 200], [28, 48, 90], [20, 20, 26],
    [95, 35, 40], [200, 70, 50], [255, 150, 100],
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

/// The value range a group of heatmaps is rendered against.
#[derive(Debug, Clone, Copy)]
pub struct Scale {
    pub max: f32,
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
            return Scale { max: 1.0 };
        }

        all.sort_by(f32::total_cmp);
        let rank = ((0.995 * all.len() as f32).ceil() as usize).clamp(1, all.len());
        Scale { max: all[rank - 1].max(1e-6) }
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
const BAR_HEIGHT: u32 = 30;
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
                // Zero is "no traversal happened here" -- keep it flat background
                // rather than the palette's darkest colour, so empty space reads
                // as empty rather than as cheap.
                BG
            } else {
                spec.palette.sample(value / spec.scale.max)
            };
            canvas.set(x, HEADER_HEIGHT + y, color);
        }
    }

    draw_colorbar(&mut canvas, HEADER_HEIGHT + spec.height, spec);
    canvas
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

    let label_y = bar_top + bar_height + 4;
    canvas.draw_text(PAD, label_y, "0", 1, MUTED);

    let max_label = format!("{}+", format_number(spec.scale.max));
    let max_width = Image::text_width(&max_label, 1);
    canvas.draw_text(PAD + bar_width - max_width, label_y, &max_label, 1, MUTED);

    let legend = spec.legend.to_uppercase();
    let legend_width = Image::text_width(&legend, 1);
    canvas.draw_text(
        PAD + bar_width / 2 - legend_width / 2,
        label_y,
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
    // Scale to the 99th percentile of the magnitudes that actually differ, not
    // the raw maximum: a handful of extreme pixels would otherwise push every
    // typical difference into the neutral band and the figure would read blank.
    let mut magnitudes: Vec<f32> = a
        .iter()
        .zip(b)
        .map(|(left, right)| (left - right).abs())
        .filter(|d| *d > 0.0)
        .collect();
    magnitudes.sort_by(f32::total_cmp);

    let peak = if magnitudes.is_empty() {
        1.0
    } else {
        let rank = ((0.99 * magnitudes.len() as f32).ceil() as usize).clamp(1, magnitudes.len());
        magnitudes[rank - 1].max(1e-6)
    };

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

    let mut sheet = Image::new(width, height, [10, 10, 13]);
    sheet.draw_text(gap, 8, &banner.to_uppercase(), 3, FG);
    sheet.draw_text(gap, 30, &caption.to_uppercase(), 1, MUTED);

    for (index, panel) in panels.iter().enumerate() {
        let column = index as u32 % columns;
        let row = index as u32 / columns;
        let x = gap + column * (panel_width + gap);
        let y = banner_height + gap + row * (panel_height + gap);
        sheet.blit(panel, x, y);
        sheet.stroke_rect(x, y, panel.width, panel.height, [45, 45, 52]);
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
