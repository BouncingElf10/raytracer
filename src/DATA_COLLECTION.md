# Implementation spec: BVH data-collection harness (Rust + WGSL)

## Context (for the LLM)

I'm building a GPU path tracer to compare BVH construction heuristics for a Computer Science
research project. Architecture: the BVH is **built CPU-side in Rust**, then flattened, uploaded, and
**traversed GPU-side in a WGSL shader** (assume `wgpu`). Bounding volumes are AABBs; construction is
top-down.

I am comparing four construction heuristics as the independent variable:
`LongestAxisCentroid`, `Median` (equal-counts), `Sah` (exact full-sweep), and `Random` (a control).

I need a **data-collection harness**: for every (heuristic × scene) combination it must produce a set
of metrics, reproducibly, and write them to a machine-readable file for later analysis. Below is
everything the harness must do. Implement the parts marked **[NEW]**; the parts marked **[ASSUMED]**
already exist per my current code — verify they meet the stated requirements but don't rebuild them.

## Objective

For each scene `.obj` and each heuristic, produce one record containing:
- **Primary, hardware-independent metrics:** mean node visits per ray, mean ray–primitive tests per ray.
- **Secondary, hardware-dependent metrics:** CPU build time, GPU render (wall-clock) time.
- **Tree-quality stats:** node count, max depth, average depth, average leaf primitive count, total SAH cost.

All records go to one CSV (or JSON) file whose columns match the schema at the bottom.

---

## Requirements

### 1. Selectable, reproducible BVH construction  [ASSUMED — verify]
- [ ] `SplitHeuristic` enum with all four variants; construction dispatches on it.
- [ ] Leaf cutoff is a single configurable constant (≈ ≤ 20 primitives), identical for every heuristic.
- [ ] **Empty-partition fallback** exists: if `Random` or `LongestAxisCentroid` puts all primitives in
  one child, fall back to a median split or force a leaf. Without this, clustered scenes recurse
  infinitely. Confirm this is in place.
- [ ] `Random` split uses a **seeded** RNG so runs are deterministic and reproducible.

### 2. Tree-quality statistics  [NEW]
- [ ] After each build, walk the tree once and compute: node count, max depth, average depth,
  average leaf primitive count, and **total SAH cost**
  (Σ over leaves of `SA(leaf_bounds) / SA(root_bounds) × leaf_prim_count`, plus the interior-node
  term if you want the full model — state which you used).
- [ ] Time the construction call itself (CPU build time) with a monotonic clock, around construction
  **only** — not loading or upload.
- [ ] Store all of the above in a `BuildStats` struct.

### 3. Flatten + upload  [ASSUMED — verify]
- [ ] BVH flattened to a linear node array (AABB + child/primitive indices + leaf flag) and uploaded
  to a GPU storage buffer. Confirm the layout the WGSL shader expects matches.

### 4. Instrumented WGSL traversal  [NEW]
- [ ] Traversal shader can be built in **two variants**: instrumented and clean. Prefer a single
  source with an override/`const` flag or a preprocessor define, so the two variants can't drift.
- [ ] **Instrumented variant:** increment per-ray counters for (a) interior-node visits and
  (b) ray–primitive tests, and write them out (see §5 for where).
- [ ] **Clean variant:** the counter writes are **compiled out entirely** — no atomics, no extra buffer
  writes — so it measures pure traversal time.

### 5. Counter storage + aggregation  [NEW]
Pick one and note the choice:
- [ ] **Option A (means only, simplest):** a few global `atomic<u32>` accumulators (total node visits,
  total prim tests, total rays); mean = total / rays. Cheap, but no distribution data.
- [ ] **Option B (per-ray buffer, richer):** a storage buffer sized to the ray count; each ray writes
  its own counts. Enables mean **and** max/percentiles/variance for stronger analysis. Watch buffer
  size on the 500k-prim / high-res scenes.
- [ ] Read the buffer(s) back to the CPU and reduce to per-ray means (and, for Option B, whatever
  distribution stats you want to report).

### 6. Two-pass metric collection  [NEW]
- [ ] **Pass A (instrumented):** dispatch the instrumented variant, read back counters, keep the
  **counts**, discard its timing (atomics serialise threads and corrupt timing).
- [ ] **Pass B (timed):** dispatch the clean variant with identical camera/resolution/samples/seed,
  keep the **wall-clock**, discard counts.
- [ ] Never collect counts and timing in the same pass.

### 7. GPU timing via timestamp queries  [NEW]
- [ ] Enable the `TIMESTAMP_QUERY` feature; create a `QuerySet`; write timestamps before/after the
  traversal dispatch; resolve to a buffer; read back; convert ticks → ms using the queue's
  timestamp period. Do **not** wrap the dispatch in a CPU timer (GPU is async; that measures queue
  latency, not shader time).
- [ ] Discard warm-up dispatches (the first includes shader compile / pipeline warm-up).
- [ ] Average the remaining timed runs (`run_count`, configurable).

### 8. Controlled-variable config  [NEW]
- [ ] One `ExperimentConfig` struct holding: camera pose, resolution, sample count, RNG seed, leaf
  cutoff, warm-up count, run count. Held **identical** across all heuristics and scenes — only the
  heuristic and the scene `.obj` change between records.

### 9. Experiment driver  [NEW]
- [ ] Loop: for each scene `.obj` → load geometry once → for each heuristic → build BVH (§2, timed) →
  flatten + upload (§3) → Pass A (§6) → Pass B (§6, warm-up discard + average) → assemble one record
  → append to output.

### 10. Output  [NEW]
- [ ] Write one row per (scene, heuristic) to CSV (or JSON), matching the schema below. Machine-readable
  and deterministic so it can be plotted/analysed later. This schema **is** the logging contract —
  the harness must populate every column.

---

## Constraints & gotchas (do not skip)

- **Empty-partition fallback is mandatory** — random/centroid on clustered geometry recurse infinitely otherwise.
- **NaN in the slab test:** `1.0 / direction` gives `0 × ∞ = NaN` when a ray origin lies exactly on a
  slab plane. The uniform-grid scene generates the axis-parallel rays that trigger this, which can
  corrupt intersection results and therefore the counts on that scene. Either use a robust slab
  traversal (Ize-style signed-zero / min-max handling) or, if left as-is, ensure NaN can't silently
  distort the counts and record it as a known limitation for that scene.
- **Counts must stay hardware-independent:** they must depend only on tree structure and rays, never on
  timing or hardware. This is what lets the same counts validate across machines.
- **Determinism:** same seed + same config ⇒ identical rays and identical random splits ⇒ counts
  identical run-to-run on the same input.

## Output schema (CSV columns)

```
scene, heuristic,
node_visits_per_ray, prim_tests_per_ray,      # primary, hardware-independent
build_time_ms, render_time_ms,                # secondary, hardware-dependent
node_count, max_depth, avg_depth, avg_leaf_prims, total_sah_cost   # tree quality
```
*(If using Option B in §5, optionally add `node_visits_max`, `prim_tests_p95`, etc.)*

## Acceptance criteria

- [ ] Running the harness over all scenes × 4 heuristics emits one CSV row per combination, every
  column populated.
- [ ] Re-running on the same input reproduces the count columns **exactly** and the timing columns
  within run-to-run noise.
- [ ] The two passes use identical camera/resolution/samples/seed; only the instrumentation differs.
- [ ] Counter aggregation yields the same per-ray means if re-run (Option A) or reduced from the same
  buffer (Option B).