# Provenance and uncertainty

## Apparatus

- Adapter: AMD Radeon RX 9070 XT (Vulkan)

### Controlled variables

| Variable | Value |
|---|---|
| Resolution | 800x600 |
| Sample count | 150 samples per dispatch |
| Max bounces | 5 |
| RNG seed | 0 |
| Scene | Standard Cornell box (white floor/ceiling/back, red left wall, green right wall, ceiling light) with one 3D model |
| Camera | 90 degree vertical FOV, origin [0.0, 0.0, 2.3] looking [0.0, 0.0, -1.0], identical every run |
| Leaf node primitive cutoff | 20 |

- Scene normalisation: every mesh centred and scaled so its longest extent is 3, inside a fixed Cornell-style room whose planes are intersected outside the BVH and so contribute nothing to the traversal counts.

## How the spread was obtained

Build time: 2 warm-up builds discarded, then 5 timed builds. The clock is a monotonic CPU timer around construction alone -- the triangle list is cloned and the finished tree is dropped outside the timed region, because neither is construction work.

Render time: 2 warm-up dispatches discarded, then 5 timed dispatches, measured on the GPU timeline with timestamp queries around the compute pass. Counting and timing never share a dispatch: the instrumented shader variant produces the counts, the clean variant produces the times.

Repeats are interleaved rather than blocked. Round r measures every variant once before round r+1 begins, so a thermal or clock excursion partway through a scene is spread across all heuristics instead of landing on whichever one happened to be running. Quoted SD is the sample standard deviation (n-1) over the 5 timed runs.

The random baseline is different in kind: it is built 5 times from 5 fixed seeds, and each of its five reported runs is one seed's own mean. Its SD therefore carries seed-to-seed *structural* variance, which is larger than the pure timing noise in the other rows and is the honest uncertainty for a control meant to stand for "an arbitrary split".

## Timestamp resolution floor

The adapter reports a timestamp period of 10.0000 ns, so the smallest interval it can distinguish is 0.000010 ms. Render times are quoted against that floor.

Every measured render time is more than a thousand ticks above the floor, so quantisation contributes less than 0.1% to any of them.

Run-to-run spread above 5% of the mean:

- Coral / SAH: SD is 10.4% of the mean
- Stanford Dragon 10k / SAH: SD is 9.6% of the mean
- Stanford Dragon 150k / SAH: SD is 9.3% of the mean
- Utah Teapot / SAH: SD is 6.5% of the mean

## Integrity checks

- Primitives lost on the way to the GPU: 0 (must be 0; anything else means the counts describe different geometry than the tree)
- Traversals that hit the 64-entry stack guard: 0 (must be 0; anything else makes the affected counts a lower bound)
- Rows whose repeat counting pass disagreed: 0 (must be 0)

### Run log

- image agreement vs SAH on Icosphere: max channel difference 6.109e-6
- image agreement vs SAH on Utah Teapot: max channel difference 0.000e0
- image agreement vs SAH on Coral: max channel difference 4.582e-3
- image agreement vs SAH on Stanford Dragon: max channel difference 0.000e0
- image agreement vs SAH on Stanford Dragon: max channel difference 8.941e-8
- image agreement vs SAH on Stanford Dragon: max channel difference 3.682e-3
- image agreement vs SAH on Stanford Dragon: max channel difference 3.155e-3
- image agreement vs SAH on Stanford Dragon: max channel difference 3.400e-3

## Scenes

| Set | Scene | Nominal | Actual prims | Source |
|---|---|---|---:|---|
| A | Icosphere | 150k | 150000 | `src/models/uv_sphere_150k.obj` |
| A | Utah Teapot | 150k | 150000 | `src/models/teapot_150k.obj` |
| A | Coral | 150k | 150000 | `src/models/coral_150k.obj` |
| B | Stanford Dragon | 10k | 10000 | `src/models/standford_dragon_10k.obj` |
| B | Stanford Dragon | 50k | 50000 | `src/models/standford_dragon_50k.obj` |
| B | Stanford Dragon | 150k | 150000 | `src/models/standford_dragon_150k.obj` |
| B | Stanford Dragon | 400k | 400000 | `src/models/standford_dragon_400k.obj` |
| B | Stanford Dragon | 1000k | 1000000 | `src/models/standford_dragon_1000k.obj` |
