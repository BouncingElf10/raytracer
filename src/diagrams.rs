//! Vector diagrams of BVH *structure*, as opposed to the raster heatmaps in
//! `viz`, which show traversal *cost*.
//!
//! Two figures, both SVG so they stay crisp at any print size:
//!
//!   * an icicle plot -- one row per depth level, each node's width proportional
//!     to the primitives beneath it. A balanced build fills every row edge to
//!     edge; an unbalanced one visibly frays into a staircase.
//!   * a depth histogram comparing all four heuristics on one axis.

use std::fmt::Write as _;

use crate::bvh::BVHNode;
use crate::viz::Palette;

/// Flattened description of one node, ready to lay out.
struct Span {
    depth: usize,
    /// Horizontal extent in [0, 1], proportional to primitives contained.
    start: f64,
    width: f64,
    prims: usize,
    is_leaf: bool,
}

fn subtree_prims(node: &BVHNode) -> usize {
    match node {
        BVHNode::LeafNode { objects, .. } => objects.get_triangles().len(),
        BVHNode::BVHNode { left, right, .. } => subtree_prims(left) + subtree_prims(right),
    }
}

fn collect_spans(node: &BVHNode, depth: usize, start: f64, width: f64, out: &mut Vec<Span>, min_width: f64) {
    let prims = subtree_prims(node);
    let is_leaf = matches!(node, BVHNode::LeafNode { .. });
    out.push(Span { depth, start, width, prims, is_leaf });

    // Below roughly a third of a pixel a node cannot be seen; recursing further
    // only inflates the file. Deep trees still show their depth through the rows
    // that remain wide enough to draw.
    if width < min_width {
        return;
    }

    if let BVHNode::BVHNode { left, right, .. } = node {
        let left_prims = subtree_prims(left) as f64;
        let right_prims = subtree_prims(right) as f64;
        let total = (left_prims + right_prims).max(1.0);

        let left_width = width * (left_prims / total);
        collect_spans(left, depth + 1, start, left_width, out, min_width);
        collect_spans(right, depth + 1, start + left_width, width - left_width, out, min_width);
    }
}

/// Icicle plot of how one tree partitions its primitives.
pub fn bvh_icicle_svg(root: &BVHNode, heuristic: &str, scene: &str) -> String {
    const WIDTH: f64 = 900.0;
    const ROW: f64 = 13.0;
    const MARGIN_TOP: f64 = 54.0;
    const MARGIN_SIDE: f64 = 16.0;

    let plot_width = WIDTH - MARGIN_SIDE * 2.0;
    let mut spans = Vec::new();
    collect_spans(root, 0, 0.0, 1.0, &mut spans, 0.35 / plot_width);

    let max_depth = spans.iter().map(|s| s.depth).max().unwrap_or(0);
    let height = MARGIN_TOP + (max_depth as f64 + 1.0) * ROW + 44.0;

    let mut svg = String::new();
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0}" width="{WIDTH}" height="{height:.0}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">"##
    );
    let _ = write!(svg, r##"<rect width="100%" height="100%" fill="#111116"/>"##);
    let _ = write!(
        svg,
        r##"<text x="{MARGIN_SIDE}" y="26" fill="#ebebf0" font-size="17" font-weight="600">{heuristic}</text>"##
    );
    let _ = write!(
        svg,
        r##"<text x="{MARGIN_SIDE}" y="43" fill="#9696a0" font-size="11">{scene} &#183; {} nodes &#183; max depth {max_depth} &#183; width = primitives contained</text>"##,
        spans.len()
    );

    for span in &spans {
        let x = MARGIN_SIDE + span.start * plot_width;
        let w = (span.width * plot_width).max(0.35);
        let y = MARGIN_TOP + span.depth as f64 * ROW;

        let t = if max_depth == 0 { 0.0 } else { span.depth as f32 / max_depth as f32 };
        let [r, g, b] = Palette::Viridis.sample(t);
        // Leaves are drawn solid and interiors translucent, so the boundary
        // between "still subdividing" and "done" is readable at a glance.
        let opacity = if span.is_leaf { 1.0 } else { 0.55 };

        let _ = write!(
            svg,
            r##"<rect x="{x:.2}" y="{y:.1}" width="{w:.2}" height="{:.1}" fill="rgb({r},{g},{b})" opacity="{opacity}"><title>depth {} &#183; {} prims{}</title></rect>"##,
            ROW - 1.5,
            span.depth,
            span.prims,
            if span.is_leaf { " (leaf)" } else { "" },
        );
    }

    // Depth axis ticks every 4 levels.
    let mut depth = 0;
    while depth <= max_depth {
        let y = MARGIN_TOP + depth as f64 * ROW + ROW - 3.0;
        let _ = write!(
            svg,
            r##"<text x="{:.0}" y="{y:.1}" fill="#5a5a66" font-size="8">{depth}</text>"##,
            MARGIN_SIDE - 12.0
        );
        depth += 4;
    }

    let _ = write!(
        svg,
        r##"<text x="{MARGIN_SIDE}" y="{:.0}" fill="#9696a0" font-size="10">SOLID = LEAF &#183; TRANSLUCENT = INTERIOR &#183; ROW = TREE DEPTH</text>"##,
        height - 14.0
    );
    svg.push_str("</svg>");
    svg
}

/// Nodes per depth level for every heuristic, on one shared axis.
pub fn depth_histogram_svg(scene: &str, series: &[(String, Vec<usize>)]) -> String {
    const WIDTH: f64 = 900.0;
    const HEIGHT: f64 = 340.0;
    const LEFT: f64 = 46.0;
    const RIGHT: f64 = 16.0;
    const TOP: f64 = 56.0;
    const BOTTOM: f64 = 46.0;

    let max_depth = series.iter().map(|(_, counts)| counts.len()).max().unwrap_or(1);
    let max_count = series
        .iter()
        .flat_map(|(_, counts)| counts.iter().copied())
        .max()
        .unwrap_or(1)
        .max(1);

    let plot_width = WIDTH - LEFT - RIGHT;
    let plot_height = HEIGHT - TOP - BOTTOM;
    let group_width = plot_width / max_depth as f64;
    let bar_width = (group_width / series.len() as f64) * 0.82;

    let colors = ["#4f9dff", "#59d9a4", "#ffc857", "#ff6b6b"];

    let mut svg = String::new();
    let _ = write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">"##
    );
    let _ = write!(svg, r##"<rect width="100%" height="100%" fill="#111116"/>"##);
    let _ = write!(
        svg,
        r##"<text x="{LEFT}" y="26" fill="#ebebf0" font-size="17" font-weight="600">NODES PER DEPTH &#8212; {scene}</text>"##
    );
    let _ = write!(
        svg,
        r##"<text x="{LEFT}" y="43" fill="#9696a0" font-size="11">a balanced build stops early; a long tail means deep, lopsided subtrees</text>"##
    );

    // Horizontal gridlines with counts.
    for step in 0..=4 {
        let fraction = step as f64 / 4.0;
        let y = TOP + plot_height * (1.0 - fraction);
        let _ = write!(
            svg,
            r##"<line x1="{LEFT}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#2a2a33"/>"##,
            WIDTH - RIGHT
        );
        let _ = write!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" fill="#5a5a66" font-size="9" text-anchor="end">{}</text>"##,
            LEFT - 6.0,
            y + 3.0,
            (max_count as f64 * fraction).round() as usize
        );
    }

    for (index, (name, counts)) in series.iter().enumerate() {
        let color = colors[index % colors.len()];
        for (depth, count) in counts.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let height = (*count as f64 / max_count as f64) * plot_height;
            let x = LEFT + depth as f64 * group_width + index as f64 * (group_width / series.len() as f64);
            let _ = write!(
                svg,
                r##"<rect x="{x:.2}" y="{:.2}" width="{bar_width:.2}" height="{height:.2}" fill="{color}"><title>{name} &#183; depth {depth} &#183; {count} nodes</title></rect>"##,
                TOP + plot_height - height,
            );
        }
    }

    // Depth axis.
    let tick_step = (max_depth / 12).max(1);
    let mut depth = 0;
    while depth < max_depth {
        let _ = write!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" fill="#5a5a66" font-size="9" text-anchor="middle">{depth}</text>"##,
            LEFT + (depth as f64 + 0.5) * group_width,
            TOP + plot_height + 14.0
        );
        depth += tick_step;
    }
    let _ = write!(
        svg,
        r##"<text x="{:.1}" y="{:.1}" fill="#9696a0" font-size="10" text-anchor="middle">TREE DEPTH</text>"##,
        LEFT + plot_width / 2.0,
        HEIGHT - 12.0
    );

    // Legend.
    let mut legend_x = LEFT + plot_width;
    for (index, (name, _)) in series.iter().enumerate().rev() {
        let label_width = name.len() as f64 * 6.0 + 18.0;
        legend_x -= label_width;
        let _ = write!(
            svg,
            r##"<rect x="{legend_x:.1}" y="{:.1}" width="9" height="9" fill="{}"/>"##,
            TOP - 14.0,
            colors[index % colors.len()]
        );
        let _ = write!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" fill="#9696a0" font-size="10">{name}</text>"##,
            legend_x + 13.0,
            TOP - 6.0
        );
    }

    svg.push_str("</svg>");
    svg
}

/// Node count at each depth level, index = depth.
pub fn depth_profile(root: &BVHNode) -> Vec<usize> {
    let mut counts = Vec::new();
    fn walk(node: &BVHNode, depth: usize, counts: &mut Vec<usize>) {
        if counts.len() <= depth {
            counts.resize(depth + 1, 0);
        }
        counts[depth] += 1;
        if let BVHNode::BVHNode { left, right, .. } = node {
            walk(left, depth + 1, counts);
            walk(right, depth + 1, counts);
        }
    }
    walk(root, 0, &mut counts);
    counts
}
