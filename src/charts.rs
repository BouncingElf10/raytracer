//! SVG chart primitives for the study figures.
//!
//! Separate from `diagrams` (which draws tree *structure*) and `viz` (which
//! rasterises per-pixel fields): this module draws the aggregate numbers -- bars,
//! lines and scatters over the study's tables.
//!
//! SVG rather than raster because these end up in a document at whatever size the
//! page wants, and because axis labels drawn with real text stay legible in a way
//! the 5x7 bitmap font in `viz` does not.
//!
//! Every chart carries its caption inside the file. A figure that travels without
//! its caption is a figure someone will misread.

use std::fmt::Write as _;

// Light theme, matched to the raster figures in `viz`: white ground, dark ink.
// These land in a document next to body text, and a dark plate in the middle of
// a page reads as a screenshot rather than as a figure.
const BG: &str = "#ffffff";
const FG: &str = "#1a1a20";
const MUTED: &str = "#5f5f6b";
const GRID: &str = "#e3e3ea";
const AXIS: &str = "#7a7a86";

/// One colour per heuristic, used by every figure so a colour means the same
/// thing across the whole set.
///
/// Darkened for a white ground: the pastel versions these came from were pitched
/// against a near-black panel, and mint and amber in particular carry almost no
/// contrast on paper. Hue order is unchanged, so a colour still means the same
/// heuristic it always did.
pub fn heuristic_color(heuristic: &str) -> &'static str {
    match heuristic {
        "LongestAxisCentroid" => "#1f6fd0",
        "Median" => "#0f8266",
        "Sah" => "#a8701a",
        "Random" => "#c93b3b",
        _ => "#6a4bc4",
    }
}

/// Escapes the five characters that would otherwise break an SVG document.
fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

// ---- Axis scaling ----------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AxisScale {
    Linear,
    Log10,
}

/// Maps data values onto pixel positions along one axis.
#[derive(Clone, Copy)]
struct Mapping {
    lo: f64,
    hi: f64,
    scale: AxisScale,
    pixel_lo: f64,
    pixel_hi: f64,
}

impl Mapping {
    fn new(lo: f64, hi: f64, scale: AxisScale, pixel_lo: f64, pixel_hi: f64) -> Self {
        // A degenerate range would divide by zero; widen it so a single-valued
        // series still renders (centred) instead of vanishing.
        let (lo, hi) = if (hi - lo).abs() < f64::EPSILON {
            (lo - 0.5, hi + 0.5)
        } else {
            (lo, hi)
        };
        Self { lo, hi, scale, pixel_lo, pixel_hi }
    }

    fn project(&self, value: f64) -> f64 {
        let t = match self.scale {
            AxisScale::Linear => (value - self.lo) / (self.hi - self.lo),
            AxisScale::Log10 => {
                let (lo, hi, v) = (self.lo.max(1e-9), self.hi.max(1e-9), value.max(1e-9));
                (v.log10() - lo.log10()) / (hi.log10() - lo.log10())
            }
        };
        self.pixel_lo + t * (self.pixel_hi - self.pixel_lo)
    }
}

/// Round numbers covering [lo, hi], at a 1/2/5 x 10^k step.
fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    if !(hi > lo) || target == 0 {
        return vec![lo];
    }
    let raw_step = (hi - lo) / target as f64;
    let magnitude = 10f64.powf(raw_step.log10().floor());
    let normalised = raw_step / magnitude;
    let step = magnitude
        * if normalised <= 1.0 {
            1.0
        } else if normalised <= 2.0 {
            2.0
        } else if normalised <= 5.0 {
            5.0
        } else {
            10.0
        };

    let mut ticks = Vec::new();
    let mut value = (lo / step).floor() * step;
    while value <= hi + step * 0.5 && ticks.len() < 64 {
        if value >= lo - step * 0.5 {
            // Snap values that are a hair off a round number by accumulated
            // floating point error, so labels read "0.3" and not "0.30000000004".
            ticks.push((value / step).round() * step);
        }
        value += step;
    }
    ticks
}

/// Decade ticks spanning [lo, hi], plus the decade either side so the data is
/// never flush against the frame.
fn log_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let start = lo.max(1e-9).log10().floor() as i32;
    let end = hi.max(1e-9).log10().ceil() as i32;
    (start..=end).map(|exponent| 10f64.powi(exponent)).collect()
}

/// Axis label with a sensible number of decimals for its magnitude.
fn tick_label(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if magnitude >= 1_000.0 {
        format!("{:.0}k", value / 1_000.0)
    } else if magnitude >= 100.0 {
        format!("{value:.0}")
    } else if magnitude >= 10.0 {
        format!("{value:.1}")
    } else if magnitude >= 1.0 {
        format!("{value:.2}")
    } else if magnitude > 0.0 {
        format!("{value:.3}")
    } else {
        "0".to_string()
    }
}

// ---- Shared frame ----------------------------------------------------------

struct Frame {
    svg: String,
    width: f64,
    height: f64,
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

impl Frame {
    /// The caption is wrapped rather than truncated, and the plot area is pushed
    /// down to make room for however many lines it takes. A caption that runs off
    /// the right edge of the viewBox is invisible in most SVG viewers, which
    /// would quietly strip the figure of the text it needs to be read.
    fn new(width: f64, height: f64, title: &str, caption: &str) -> Self {
        let (left, right, bottom) = (78.0, 22.0, 62.0);
        // 6.15 px per character at font-size 10.5 in the monospace stack above.
        let caption_lines = wrap_label(caption, (((width - left - right) / 6.15) as usize).max(20));
        let extra = (caption_lines.len().saturating_sub(1) as f64) * 13.0;
        let top = 74.0 + extra;
        // Grow the canvas rather than the plot area, so a long caption never
        // costs the chart vertical resolution.
        let height = height + extra;

        let mut svg = String::new();
        let _ = write!(
            svg,
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">"##
        );
        let _ = write!(svg, r##"<rect width="100%" height="100%" fill="{BG}"/>"##);
        let _ = write!(
            svg,
            r##"<text x="{left}" y="28" fill="{FG}" font-size="16" font-weight="600">{}</text>"##,
            esc(title)
        );
        for (index, line) in caption_lines.iter().enumerate() {
            let _ = write!(
                svg,
                r##"<text x="{left}" y="{:.1}" fill="{MUTED}" font-size="10.5">{}</text>"##,
                46.0 + index as f64 * 13.0,
                esc(line)
            );
        }
        Self { svg, width, height, left, right, top, bottom }
    }

    fn plot_width(&self) -> f64 {
        self.width - self.left - self.right
    }

    fn plot_height(&self) -> f64 {
        self.height - self.top - self.bottom
    }

    fn axis_titles(&mut self, x_label: &str, y_label: &str) {
        let _ = write!(
            self.svg,
            r##"<text x="{:.1}" y="{:.1}" fill="{MUTED}" font-size="10.5" text-anchor="middle">{}</text>"##,
            self.left + self.plot_width() / 2.0,
            self.height - 14.0,
            esc(x_label)
        );
        let _ = write!(
            self.svg,
            r##"<text transform="translate(16,{:.1}) rotate(-90)" fill="{MUTED}" font-size="10.5" text-anchor="middle">{}</text>"##,
            self.top + self.plot_height() / 2.0,
            esc(y_label)
        );
    }

    /// Horizontal gridlines plus their y-axis labels.
    fn y_grid(&mut self, mapping: &Mapping, ticks: &[f64]) {
        for value in ticks {
            let y = mapping.project(*value);
            if y < self.top - 1.0 || y > self.top + self.plot_height() + 1.0 {
                continue;
            }
            let _ = write!(
                self.svg,
                r##"<line x1="{:.1}" y1="{y:.2}" x2="{:.1}" y2="{y:.2}" stroke="{GRID}"/>"##,
                self.left,
                self.width - self.right
            );
            let _ = write!(
                self.svg,
                r##"<text x="{:.1}" y="{:.2}" fill="{AXIS}" font-size="9.5" text-anchor="end">{}</text>"##,
                self.left - 7.0,
                y + 3.4,
                tick_label(*value)
            );
        }
    }

    fn legend(&mut self, entries: &[(String, &'static str)]) {
        // Right-aligned along the top of the plot area, laid out right to left so
        // the first series ends up leftmost.
        let mut x = self.width - self.right;
        for (name, color) in entries.iter().rev() {
            let entry_width = name.chars().count() as f64 * 6.1 + 20.0;
            x -= entry_width;
            let _ = write!(
                self.svg,
                r##"<rect x="{x:.1}" y="{:.1}" width="9" height="9" fill="{color}"/>"##,
                self.top - 17.0
            );
            let _ = write!(
                self.svg,
                r##"<text x="{:.1}" y="{:.1}" fill="{MUTED}" font-size="10">{}</text>"##,
                x + 13.0,
                self.top - 9.0,
                esc(name)
            );
        }
    }

    fn finish(mut self) -> String {
        self.svg.push_str("</svg>");
        self.svg
    }
}

// ---- Grouped bar chart -----------------------------------------------------

/// One heuristic's value in every group, with an optional +/- error bar.
pub struct BarSeries {
    pub name: String,
    /// One entry per group, in group order. `None` leaves a gap rather than a
    /// zero bar -- a missing measurement is not a measurement of zero.
    pub values: Vec<Option<f64>>,
    pub errors: Vec<Option<f64>>,
}

pub struct BarSpec<'a> {
    pub title: &'a str,
    pub caption: &'a str,
    pub y_label: &'a str,
    pub groups: Vec<String>,
    pub series: Vec<BarSeries>,
}

pub fn grouped_bars(spec: &BarSpec) -> String {
    let width = 900.0;
    let height = 440.0;
    let mut frame = Frame::new(width, height, spec.title, spec.caption);

    // The top of the axis has to clear the error bars, not just the bars.
    let peak = spec
        .series
        .iter()
        .flat_map(|series| {
            series.values.iter().zip(&series.errors).filter_map(|(value, error)| {
                value.map(|v| v + error.unwrap_or(0.0))
            })
        })
        .fold(0.0f64, f64::max);

    let y_ticks = nice_ticks(0.0, peak.max(1e-9), 5);
    let y_top = y_ticks.last().copied().unwrap_or(peak).max(peak);
    let mapping = Mapping::new(
        0.0,
        y_top,
        AxisScale::Linear,
        frame.top + frame.plot_height(),
        frame.top,
    );

    frame.y_grid(&mapping, &y_ticks);

    let group_count = spec.groups.len().max(1) as f64;
    let group_width = frame.plot_width() / group_count;
    let series_count = spec.series.len().max(1) as f64;
    // 0.78 leaves a visible gutter between groups; without it neighbouring
    // groups run together and the grouping stops doing its job.
    let slot = (group_width * 0.78) / series_count;

    for (series_index, series) in spec.series.iter().enumerate() {
        let color = heuristic_color(&series.name);
        for (group_index, value) in series.values.iter().enumerate() {
            let Some(value) = value else { continue };
            let x = frame.left
                + group_index as f64 * group_width
                + group_width * 0.11
                + series_index as f64 * slot;
            let y = mapping.project(*value);
            let bar_height = (frame.top + frame.plot_height() - y).max(0.0);

            let _ = write!(
                frame.svg,
                r##"<rect x="{x:.2}" y="{y:.2}" width="{:.2}" height="{bar_height:.2}" fill="{color}"><title>{} &#183; {} &#183; {:.4}</title></rect>"##,
                slot * 0.86,
                esc(&series.name),
                esc(&spec.groups[group_index]),
                value
            );

            if let Some(error) = series.errors.get(group_index).copied().flatten() {
                if error > 0.0 {
                    let centre = x + slot * 0.43;
                    let top = mapping.project(value + error);
                    let bottom = mapping.project((value - error).max(0.0));
                    let cap = (slot * 0.28).min(7.0);
                    let _ = write!(
                        frame.svg,
                        r##"<path d="M{centre:.2} {top:.2} L{centre:.2} {bottom:.2} M{:.2} {top:.2} L{:.2} {top:.2} M{:.2} {bottom:.2} L{:.2} {bottom:.2}" stroke="{FG}" stroke-width="1" fill="none" opacity="0.75"/>"##,
                        centre - cap, centre + cap, centre - cap, centre + cap
                    );
                }
            }
        }
    }

    // Group labels, split onto two lines when a scene name is too long to fit.
    for (index, group) in spec.groups.iter().enumerate() {
        let centre = frame.left + (index as f64 + 0.5) * group_width;
        let baseline = frame.top + frame.plot_height() + 15.0;
        for (line_index, line) in wrap_label(group, (group_width / 6.1) as usize).iter().enumerate() {
            let _ = write!(
                frame.svg,
                r##"<text x="{centre:.1}" y="{:.1}" fill="{MUTED}" font-size="10" text-anchor="middle">{}</text>"##,
                baseline + line_index as f64 * 11.0,
                esc(line)
            );
        }
    }

    let _ = write!(
        frame.svg,
        r##"<line x1="{:.1}" y1="{:.2}" x2="{:.1}" y2="{:.2}" stroke="{AXIS}"/>"##,
        frame.left,
        frame.top + frame.plot_height(),
        frame.width - frame.right,
        frame.top + frame.plot_height()
    );

    let entries: Vec<(String, &'static str)> = spec
        .series
        .iter()
        .map(|series| (series.name.clone(), heuristic_color(&series.name)))
        .collect();
    frame.legend(&entries);
    frame.axis_titles("", spec.y_label);
    frame.finish()
}

/// Greedy word wrap, so a long scene name stacks instead of overlapping its
/// neighbour.
fn wrap_label(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(6);
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.chars().count() + 1 + word.chars().count() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

// ---- Line chart ------------------------------------------------------------

pub struct LineSeries {
    pub name: String,
    /// (x, y, +/- error). Plotted in the order given.
    pub points: Vec<(f64, f64, Option<f64>)>,
}

pub struct LineSpec<'a> {
    pub title: &'a str,
    pub caption: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub x_scale: AxisScale,
    pub y_scale: AxisScale,
    pub series: Vec<LineSeries>,
    /// Explicit x tick positions. Used for the primitive-count axis, where the
    /// meaningful ticks are the scenes actually measured, not round decades.
    pub x_ticks: Option<Vec<f64>>,
}

pub fn line_chart(spec: &LineSpec) -> String {
    let width = 900.0;
    let height = 460.0;
    let mut frame = Frame::new(width, height, spec.title, spec.caption);

    let all: Vec<(f64, f64, Option<f64>)> =
        spec.series.iter().flat_map(|series| series.points.iter().copied()).collect();
    if all.is_empty() {
        return frame.finish();
    }

    let x_min = all.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = all.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min_raw = all
        .iter()
        .map(|p| p.1 - p.2.unwrap_or(0.0))
        .fold(f64::INFINITY, f64::min);
    let y_max_raw = all
        .iter()
        .map(|p| p.1 + p.2.unwrap_or(0.0))
        .fold(f64::NEG_INFINITY, f64::max);

    // A log axis cannot start at zero, and a linear axis reads better anchored
    // there -- the reader should not have to check whether the baseline is real.
    let (y_lo, y_hi) = match spec.y_scale {
        AxisScale::Linear => (0.0, y_max_raw * 1.08),
        AxisScale::Log10 => (
            10f64.powf(y_min_raw.max(1e-9).log10().floor()),
            10f64.powf(y_max_raw.max(1e-9).log10().ceil()),
        ),
    };
    let (x_lo, x_hi) = match spec.x_scale {
        AxisScale::Linear => (x_min, x_max),
        // A tenth of a decade of padding either side keeps the end markers off
        // the frame without pretending the series extends further than it does.
        AxisScale::Log10 => (x_min * 0.82, x_max * 1.22),
    };

    let x_map = Mapping::new(x_lo, x_hi, spec.x_scale, frame.left, frame.width - frame.right);
    let y_map = Mapping::new(
        y_lo,
        y_hi,
        spec.y_scale,
        frame.top + frame.plot_height(),
        frame.top,
    );

    let y_ticks = match spec.y_scale {
        AxisScale::Linear => nice_ticks(y_lo, y_hi, 5),
        AxisScale::Log10 => log_ticks(y_lo, y_hi),
    };
    frame.y_grid(&y_map, &y_ticks);

    let x_ticks = spec.x_ticks.clone().unwrap_or_else(|| match spec.x_scale {
        AxisScale::Linear => nice_ticks(x_lo, x_hi, 6),
        AxisScale::Log10 => log_ticks(x_lo, x_hi),
    });
    for value in &x_ticks {
        let x = x_map.project(*value);
        let _ = write!(
            frame.svg,
            r##"<line x1="{x:.2}" y1="{:.1}" x2="{x:.2}" y2="{:.1}" stroke="{GRID}"/>"##,
            frame.top,
            frame.top + frame.plot_height()
        );
        let _ = write!(
            frame.svg,
            r##"<text x="{x:.2}" y="{:.1}" fill="{AXIS}" font-size="9.5" text-anchor="middle">{}</text>"##,
            frame.top + frame.plot_height() + 15.0,
            tick_label(*value)
        );
    }

    for series in &spec.series {
        let color = heuristic_color(&series.name);
        let path: String = series
            .points
            .iter()
            .enumerate()
            .map(|(index, (x, y, _))| {
                format!(
                    "{}{:.2} {:.2}",
                    if index == 0 { "M" } else { " L" },
                    x_map.project(*x),
                    y_map.project(*y)
                )
            })
            .collect();
        let _ = write!(
            frame.svg,
            r##"<path d="{path}" stroke="{color}" stroke-width="2" fill="none"/>"##
        );

        for (x, y, error) in &series.points {
            let px = x_map.project(*x);
            let py = y_map.project(*y);
            if let Some(error) = error {
                if *error > 0.0 {
                    let top = y_map.project(y + error);
                    let bottom = y_map.project((y - error).max(y_lo));
                    let _ = write!(
                        frame.svg,
                        r##"<path d="M{px:.2} {top:.2} L{px:.2} {bottom:.2} M{:.2} {top:.2} L{:.2} {top:.2} M{:.2} {bottom:.2} L{:.2} {bottom:.2}" stroke="{color}" stroke-width="1" fill="none" opacity="0.8"/>"##,
                        px - 4.0, px + 4.0, px - 4.0, px + 4.0
                    );
                }
            }
            let _ = write!(
                frame.svg,
                r##"<circle cx="{px:.2}" cy="{py:.2}" r="3.4" fill="{color}" stroke="{BG}" stroke-width="1"><title>{} &#183; x={} &#183; y={:.4}</title></circle>"##,
                esc(&series.name),
                tick_label(*x),
                y
            );
        }
    }

    let entries: Vec<(String, &'static str)> = spec
        .series
        .iter()
        .map(|series| (series.name.clone(), heuristic_color(&series.name)))
        .collect();
    frame.legend(&entries);
    frame.axis_titles(spec.x_label, spec.y_label);
    frame.finish()
}

// ---- Scatter ---------------------------------------------------------------

pub struct ScatterPoint {
    pub x: f64,
    pub y: f64,
    /// Decides the marker colour, and groups the legend.
    pub group: String,
    /// Hover text; also drawn next to the marker when `label_points` is set.
    pub label: String,
}

pub struct ScatterSpec<'a> {
    pub title: &'a str,
    pub caption: &'a str,
    pub x_label: &'a str,
    pub y_label: &'a str,
    pub points: Vec<ScatterPoint>,
    /// Draws y = x. Only meaningful when both axes are in the same units, which
    /// for this study means the normalised-ratio scatter.
    pub unit_diagonal: bool,
    /// Draws the least-squares fit and prints Pearson's r in the corner.
    pub fit_line: bool,
}

pub fn scatter(spec: &ScatterSpec) -> String {
    let width = 900.0;
    let height = 520.0;
    let mut frame = Frame::new(width, height, spec.title, spec.caption);

    if spec.points.is_empty() {
        return frame.finish();
    }

    let xs: Vec<f64> = spec.points.iter().map(|p| p.x).collect();
    let ys: Vec<f64> = spec.points.iter().map(|p| p.y).collect();
    let x_min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let x_pad = ((x_max - x_min) * 0.08).max(1e-9);
    let y_pad = ((y_max - y_min) * 0.08).max(1e-9);
    let (mut x_lo, mut x_hi) = (x_min - x_pad, x_max + x_pad);
    let (mut y_lo, mut y_hi) = (y_min - y_pad, y_max + y_pad);

    if spec.unit_diagonal {
        // A y = x reference line is only honest when both axes cover the same
        // interval; otherwise the diagonal's slope on the page is meaningless.
        let lo = x_lo.min(y_lo);
        let hi = x_hi.max(y_hi);
        x_lo = lo;
        y_lo = lo;
        x_hi = hi;
        y_hi = hi;
    }

    let x_map = Mapping::new(x_lo, x_hi, AxisScale::Linear, frame.left, frame.width - frame.right);
    let y_map = Mapping::new(
        y_lo,
        y_hi,
        AxisScale::Linear,
        frame.top + frame.plot_height(),
        frame.top,
    );

    frame.y_grid(&y_map, &nice_ticks(y_lo, y_hi, 5));
    for value in nice_ticks(x_lo, x_hi, 6) {
        let x = x_map.project(value);
        if x < frame.left - 1.0 || x > frame.width - frame.right + 1.0 {
            continue;
        }
        let _ = write!(
            frame.svg,
            r##"<line x1="{x:.2}" y1="{:.1}" x2="{x:.2}" y2="{:.1}" stroke="{GRID}"/>"##,
            frame.top,
            frame.top + frame.plot_height()
        );
        let _ = write!(
            frame.svg,
            r##"<text x="{x:.2}" y="{:.1}" fill="{AXIS}" font-size="9.5" text-anchor="middle">{}</text>"##,
            frame.top + frame.plot_height() + 15.0,
            tick_label(value)
        );
    }

    if spec.unit_diagonal {
        let _ = write!(
            frame.svg,
            r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{AXIS}" stroke-width="1" stroke-dasharray="5 4"/>"##,
            x_map.project(x_lo.max(y_lo)),
            y_map.project(x_lo.max(y_lo)),
            x_map.project(x_hi.min(y_hi)),
            y_map.project(x_hi.min(y_hi))
        );
        let _ = write!(
            frame.svg,
            r##"<text x="{:.1}" y="{:.1}" fill="{AXIS}" font-size="9.5">y = x</text>"##,
            x_map.project(x_hi.min(y_hi)) - 42.0,
            y_map.project(x_hi.min(y_hi)) - 6.0
        );
    }

    if spec.fit_line {
        if let Some((slope, intercept, r)) = least_squares(&xs, &ys) {
            let _ = write!(
                frame.svg,
                r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{FG}" stroke-width="1" stroke-dasharray="6 4" opacity="0.5"/>"##,
                x_map.project(x_lo),
                y_map.project(slope * x_lo + intercept),
                x_map.project(x_hi),
                y_map.project(slope * x_hi + intercept)
            );
            let _ = write!(
                frame.svg,
                r##"<text x="{:.1}" y="{:.1}" fill="{MUTED}" font-size="10.5">least squares fit &#183; r = {r:.3} &#183; r&#178; = {:.3}</text>"##,
                frame.left + 8.0,
                frame.top + 14.0,
                r * r
            );
        }
    }

    for point in &spec.points {
        let color = heuristic_color(&point.group);
        let _ = write!(
            frame.svg,
            r##"<circle cx="{:.2}" cy="{:.2}" r="4.6" fill="{color}" fill-opacity="0.85" stroke="{BG}" stroke-width="1"><title>{}</title></circle>"##,
            x_map.project(point.x),
            y_map.project(point.y),
            esc(&point.label)
        );
    }

    let mut groups: Vec<String> = Vec::new();
    for point in &spec.points {
        if !groups.contains(&point.group) {
            groups.push(point.group.clone());
        }
    }
    let entries: Vec<(String, &'static str)> = groups
        .iter()
        .map(|group| (group.clone(), heuristic_color(group)))
        .collect();
    frame.legend(&entries);
    frame.axis_titles(spec.x_label, spec.y_label);
    frame.finish()
}

/// Ordinary least squares plus Pearson's r. `None` when x has no variance, in
/// which case a fit line would be a vertical asymptote rather than a trend.
fn least_squares(xs: &[f64], ys: &[f64]) -> Option<(f64, f64, f64)> {
    let n = xs.len() as f64;
    if xs.len() < 3 {
        return None;
    }
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        sxx += (x - mean_x).powi(2);
        syy += (y - mean_y).powi(2);
        sxy += (x - mean_x) * (y - mean_y);
    }
    if sxx <= f64::EPSILON || syy <= f64::EPSILON {
        return None;
    }

    let slope = sxy / sxx;
    Some((slope, mean_y - slope * mean_x, sxy / (sxx.sqrt() * syy.sqrt())))
}
