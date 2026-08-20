//! Headless AABB wireframe rendering for the BVH structure figure.
//!
//! The interactive overlay in `renderer` draws the same boxes, but it needs a
//! window and a live canvas. The study runs offscreen, so this draws the boxes
//! straight into a `viz::Image` using the same camera projection the measurement
//! rays were generated from -- which is what makes the wireframe panel and the
//! heatmap panels line up pixel for pixel.


use crate::bvh::AABB;
use crate::camera::Camera;
use crate::viz::{Image, BG, FG, MUTED};

const HEADER_HEIGHT: u32 = 34;
const PAD: u32 = 8;

/// Draws one heuristic's boxes at a fixed tree level over a dim silhouette of
/// the geometry.
///
/// The silhouette is there because an empty box only means something next to the
/// geometry it was supposed to bound: overlap and enclosed empty space are the
/// two things this figure exists to show, and neither is visible against a blank
/// page.
/// It comes from the traversal field rather than the path-traced image because a
/// single-dispatch render is mostly Monte Carlo noise, and noise behind a
/// wireframe is worse than no backdrop at all.
pub fn render_boxes(
    boxes: &[(AABB, bool)],
    silhouette: Option<&[f32]>,
    camera: &Camera,
    width: u32,
    height: u32,
    title: &str,
    subtitle: &str,
) -> Image {
    let mut canvas = Image::new(width, HEADER_HEIGHT + height, BG);

    canvas.draw_text(PAD, 6, &title.to_uppercase(), 2, FG);
    canvas.draw_text(PAD, 22, &subtitle.to_uppercase(), 1, MUTED);

    if let Some(field) = silhouette.filter(|field| field.len() >= (width * height) as usize) {
        // Thresholded, not just dimmed. Secondary bounces put a nonzero count on
        // almost every pixel in the frame, so "value > 0" is not a silhouette --
        // it is the whole image. Anchoring on a high percentile and cutting well
        // below it leaves the model shaded and the scattered background blank.
        let mut positive: Vec<f32> = field.iter().copied().filter(|v| *v > 0.0).collect();
        positive.sort_by(f32::total_cmp);
        let reference = positive
            .get(positive.len() * 3 / 4)
            .copied()
            .unwrap_or(1.0)
            .max(1e-6);

        const FLOOR: f32 = 0.35;
        const CEILING: f32 = 1.2;

        for y in 0..height {
            for x in 0..width {
                let ratio = field[(y * width + x) as usize] / reference;
                if ratio <= FLOOR {
                    continue;
                }
                let t = ((ratio - FLOOR) / (CEILING - FLOOR)).clamp(0.0, 1.0);
                // Away from the page rather than towards it: on white the
                // silhouette has to darken to appear at all. The swing is wider
                // than the dark theme used because the box edges over it are
                // themselves darkened, and a pale grey would vanish under them.
                let level = (BG[0] as f32 - 96.0 * t) as u8;
                canvas.set(x, HEADER_HEIGHT + y, [level, level, level.saturating_add(6)]);
            }
        }
    }

    // Painted back to front so nearer boxes sit on top; without it the figure is
    // a flat tangle and the depth ordering is unreadable.
    let mut ordered: Vec<&(AABB, bool)> = boxes.iter().collect();
    let eye = camera.ray().origin();
    ordered.sort_by(|a, b| {
        let key = |item: &(AABB, bool)| ((item.0.min + item.0.max) * 0.5 - eye).length_squared();
        key(b).total_cmp(&key(a))
    });

    let count = ordered.len().max(1);
    for (index, (aabb, is_leaf)) in ordered.iter().enumerate() {
        // Hue by index, so adjacent boxes at the same level stay distinguishable
        // where they overlap; leaves darker than interiors.
        let t = index as f32 / count as f32;
        let color = box_color(t, *is_leaf);

        for (a, b) in aabb.edges() {
            if let Some((start, end)) = camera.project_segment(a, b) {
                canvas.draw_line(
                    (start.0, start.1 + HEADER_HEIGHT as i32),
                    (end.0, end.1 + HEADER_HEIGHT as i32),
                    color,
                );
            }
        }
    }

    canvas
}

/// A rotating hue so overlapping boxes stay separable, at two weights for leaf
/// and interior.
fn box_color(t: f32, is_leaf: bool) -> [u8; 3] {
    let hue = t * 6.0;
    let sector = hue.floor() as u32 % 6;
    let fraction = hue - hue.floor();
    let (r, g, b) = match sector {
        0 => (1.0, fraction, 0.0),
        1 => (1.0 - fraction, 1.0, 0.0),
        2 => (0.0, 1.0, fraction),
        3 => (0.0, 1.0 - fraction, 1.0),
        4 => (fraction, 0.0, 1.0),
        _ => (1.0, 0.0, 1.0 - fraction),
    };
    // Two corrections for a white ground. First a luminance ceiling: at full
    // brightness the warm half of the wheel -- yellow above all -- is nearly the
    // page, so every hue is darkened until it carries the same weight of ink.
    // Then interiors wash back towards the page, which is the recessive role
    // that plain dimming used to fill when the ground was black.
    const INK: f32 = 0.34;
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let gain = (INK / luma.max(1e-3)).min(1.0);
    let wash = if is_leaf { 0.0 } else { 0.42 };
    let channel = |value: f32| {
        let value = value * gain;
        ((value + (1.0 - value) * wash) * 255.0) as u8
    };
    [channel(r), channel(g), channel(b)]
}

