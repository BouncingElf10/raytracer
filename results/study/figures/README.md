# Figures

Each figure answers one question. Captions carry the descriptive load; interpretation belongs in the analysis.

## Set A -- distribution

- **Figure 1** (`fig01_setA_node_visits.svg`) -- Node visits per ray, grouped by scene, one bar per heuristic. Does heuristic choice matter more as the distribution becomes uneven?
- **Figure 2** (`fig02_setA_render_time.svg`) -- Render time, same grouping, +/-1 SD. Does the count advantage survive into wall-clock?

## Set B -- primitive count

- **Figure 3** (`fig03_setB_node_visits.svg`) -- Node visits per ray vs. primitive count, one line per heuristic, log x.
- **Figure 4** (`fig04_setB_build_time.svg`) -- Build time vs. primitive count, log-log, so the full-sweep SAH's steeper scaling is visible as a steeper slope.
- **Figure 5** (`fig05_setB_render_time.svg`) -- Render time vs. primitive count, log x. Where does the traversal saving outgrow the build cost?

## Cross-cutting

- **Figure 6** (`fig06_sah_cost_vs_visits.svg`) -- Total SAH cost against measured node visits per ray, every condition as one point, with a least-squares fit and r. Does the cost model predict measured traversal work?
- **Figure 7** (`fig07_normalised_visits_vs_render.svg`) -- Normalised node visits against normalised render time. Points off the diagonal are conditions where structural work and wall-clock disagree.

## Per-pixel and structure figures

### Icosphere

- `fig_heatmap_icosphere_prim_tests.png` -- Primitive tests per ray, 2x2 across the four heuristics. All four panels share one absolute, perceptually uniform colour scale; the colourbar caption states whether that scale is linear or log.
- `fig_heatmap_icosphere_node_visits.png` -- The same, for node visits per ray.
- `fig_difference_icosphere_prim_tests.png` -- Difference image, (heuristic - SAH), diverging colormap centred on zero and shared across the three panels. Isolates where the excess cost lives.
- `fig_structure_icosphere_depth06.png` -- AABB wireframes at tree depth 06, same camera, 2x2 across heuristics.
- `fig_depth_profile_icosphere.svg` -- Nodes per depth level for all four heuristics on one axis.
- `fig_render_icosphere.png` -- The rendered image from each of the four trees. An integrity check rather than a result: they must be identical.

### Utah Teapot

- `fig_heatmap_utah_teapot_prim_tests.png` -- Primitive tests per ray, 2x2 across the four heuristics. All four panels share one absolute, perceptually uniform colour scale; the colourbar caption states whether that scale is linear or log.
- `fig_heatmap_utah_teapot_node_visits.png` -- The same, for node visits per ray.
- `fig_difference_utah_teapot_prim_tests.png` -- Difference image, (heuristic - SAH), diverging colormap centred on zero and shared across the three panels. Isolates where the excess cost lives.
- `fig_structure_utah_teapot_depth06.png` -- AABB wireframes at tree depth 06, same camera, 2x2 across heuristics.
- `fig_depth_profile_utah_teapot.svg` -- Nodes per depth level for all four heuristics on one axis.
- `fig_render_utah_teapot.png` -- The rendered image from each of the four trees. An integrity check rather than a result: they must be identical.

### Coral

- `fig_heatmap_coral_prim_tests.png` -- Primitive tests per ray, 2x2 across the four heuristics. All four panels share one absolute, perceptually uniform colour scale; the colourbar caption states whether that scale is linear or log.
- `fig_heatmap_coral_node_visits.png` -- The same, for node visits per ray.
- `fig_difference_coral_prim_tests.png` -- Difference image, (heuristic - SAH), diverging colormap centred on zero and shared across the three panels. Isolates where the excess cost lives.
- `fig_structure_coral_depth06.png` -- AABB wireframes at tree depth 06, same camera, 2x2 across heuristics.
- `fig_depth_profile_coral.svg` -- Nodes per depth level for all four heuristics on one axis.
- `fig_render_coral.png` -- The rendered image from each of the four trees. An integrity check rather than a result: they must be identical.

