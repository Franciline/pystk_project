# Multi-map experiment - first 5k-step check

The old Abyss results and configuration files remain separate. Shared overrides
are in `multimap_smoke.yaml`:

| Split | STK map IDs |
| --- | --- |
| Train | `abyss`, `volcano_island`, `sandtrack` |
| Unseen | `snowtuxpeak`, `xr591`, `zengarden` |

Both methods load the same Team4 offline dataset. Collection uses epsilons
`[0, 0.1, 0.3]` and 3 episodes per epsilon per training map: 27 episodes before
whole-episode deduplication. No unseen-map data enters training. Duplicate removal
is scoped to the map. Longer episodes still contribute more transitions to the
uniform offline sampling/training dataset; per-map sizes are saved.

DAgger keeps fitted normalization, trained only on initial training-map data.
RLPD uses raw encoded observations. It saves identity normalization statistics
(zeros/ones) solely to preserve the existing checkpoint/loading format.
Both keep the 182-path encoder and force executed rescue actions to False.

## Commands

From the repository root, using the existing Python 3.12 environment:

```bash
../.venv/bin/python src/main/build_camera_binding.py
export PYTHONPATH=/tmp/pystk-camera-build/build312
```

Builds an isolated PySTK with camera and rescue-state bindings; it does not replace
the installed package. Build requirements are described in `patches/README.md`.
To reuse the asset cache used during development, optionally set
`export XDG_CACHE_HOME=/tmp/pystk-font-cache`.

Collect the shared offline data once (existing episode files are skipped):

```bash
caffeinate -i ../.venv/bin/python src/main/collect_multitrack_offline.py \
  --experiment src/main/multimap_smoke.yaml
```

Run seed 1000 for each method, with fresh output directories:

```bash
caffeinate -i ../.venv/bin/python src/main/DAgger.py \
  --experiment src/main/multimap_smoke.yaml \
  output_dir=datasets/multimap/smoke_dagger_seed1000

caffeinate -i ../.venv/bin/python src/main/RLPD.py \
  --experiment src/main/multimap_smoke.yaml \
  output_dir=datasets/multimap/smoke_rlpd_seed1000
```

The **5k budget is online interaction across all three training maps combined**.
Each new episode runs on the least-collected map. Episodes finish or time out
before switching, so collection can exceed 5k by less than one episode. It is
approximate balance by steps, not a fixed number of laps per map.

DAgger fits/evaluates offline first, then collects at least 5k online steps and
trains/evaluates again (5 epochs per update). RLPD updates online and evaluates
at the episode boundary after reaching 5k. Offline collection, expert benchmarks
and evaluation steps are excluded from the online budget.

The smoke config uses one training seed (1000) and preserves **five evaluation
seeds per map**, `[10000,10001,10002,10003,10004]`, so mean and std are available.
Expert benchmarks use those same five seeds. For a faster plumbing-only check,
override `"evaluation.seeds=[10000]" expert_benchmark.runs=1` on both training
commands; this does not provide an evaluation std. Both train and unseen maps
are evaluated. Best/last models
are kept; best selection uses only the training maps (macro completion first,
then macro mean finishing steps if available on every training map). Unseen
results cannot select a checkpoint. Std is retained per map, not mixed across
tracks of different lengths.

After both runs, create final-use PDF/SVG comparison plots:

```bash
../.venv/bin/python src/main/plot_multimap_comparison.py \
  --dagger datasets/multimap/smoke_dagger_seed1000 \
  --rlpd datasets/multimap/smoke_rlpd_seed1000 \
  --output-dir datasets/plots/multimap_smoke
```

Plots show the **latest** evaluation for each method: completion, finishing steps
relative to a common per-map expert mean (from the DAgger run's benchmark), confirmed rescues and time stuck. A missing
finish/baseline has no ratio. Relative-step std divides the actor's std by the
saved expert mean; the denominator is treated as a fixed reference. The expert
benchmark runs on all six maps and is saved per run.

## What is saved

Existing episode fields remain: T+1 raw observations and T executed actions,
expert actions, rewards, terminated and truncated flags. RLPD online/evaluation
rollouts do not query an expert: their `expert_actions` entries are explicitly
`None`, and metadata says labels are unavailable. DAgger and offline collection
retain actual expert labels.

New per-state `diagnostics` (T+1 entries) contain world position, elapsed episode
time, simulator race time, track progress, rescue status, racing/stuck status.
`diagnostic_summary` contains `rescue_status_available`, `rescue_count`,
`stuck_count`, and `stuck_duration_s`.

- Rescue: count entries into the native rescue animation, not its frames. This
  confirms rescues, but does not identify their cause as a fall. With an old
  binding, status/count are unknown (`None`), never silently reported as zero.
- Stuck: position stays within 1 metre of its anchor **and** progress changes by
  no more than 1 metre for at least 5 seconds. Countdown and rescue animation
  time are excluded. One continuous qualifying interval counts once; duration
  includes its initial 5 seconds. Thresholds are configurable under `diagnostics`.

Outputs within each run directory:

- Original metrics/loss/checkpoint files, plus `offline_summary.csv` with map sizes.
- `evaluation_by_track.csv`: train/unseen metrics per map and evaluation.
- `evaluation_episodes.csv`: individual evaluation laps with diagnostic counts.
- `evaluation_episodes/<round>/<track>/seed_<seed>.pkl.gz`: detailed evaluation laps.
- DAgger `iteration_*/*.pkl.gz`: aggregated training episodes, as before.
- RLPD `online_trajectories/*.pkl.gz`: newly persisted online episodes.
- `expert_benchmark/`: baseline metrics, detailed laps and official track lengths.

Historical offline episodes without diagnostic fields are preserved; missing
world positions or rescue events cannot be reconstructed from them reliably.
