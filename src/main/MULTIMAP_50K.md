# Full multi-map experiment - 50k online target

Use `multimap_50k.yaml` with either algorithm. It is separate from the 5k smoke
config and pins the existing network and training hyperparameters.

| Setting | Value |
| --- | --- |
| Training maps | `abyss`, `volcano_island`, `sandtrack` |
| Unseen evaluation maps | `snowtuxpeak`, `xr591`, `zengarden` |
| Independent training seeds | 1000, 2000, 3000 |
| Evaluation seeds on every map | 10000, 10001, 10002, 10003, 10004 |
| Task | One kart, one lap, step size 0.1 s |
| Online target | 50,000 steps across the three training maps combined |
| Evaluation frequency | Near each 5k online boundary, between episodes |
| Episode limits | Shared frozen expert maximum finishing steps + 400, per map |
| Manual rescue | Always False |
| Encoder / networks | 182 paths, 10 items; hidden layers [256, 256] |
| DAgger | Gaussian exploration on continuous actions (probability 1, accel std 0.1, steer std 0.05); 5 supervised epochs per update; batch 256 |
| RLPD | Stochastic actor during training; 50/50 offline/online minibatches of 256; UTD 10; 5 critics, subset 2 |
| Normalization | DAgger fitted on initial training data; RLPD raw encoded inputs |

## Confirmed offline data

**Reuse `datasets/multimap/offline/`. No collection command is needed.**

It contains 27 unique Team4 episodes and 19,493 transitions: nine episodes per
training map (three each at epsilon 0, 0.1 and 0.3, using seeds 0, 1 and 2).
DAgger's `initialization.dataset` resolves to the same `offline.dataset` path
that RLPD loads. Both apply the same exact episode deduplication. Existing
observations, executed actions and expert labels stay unchanged. The historical
offline episodes keep their original 1,000-step timeout.

## 1. Prepare the environment

Run these commands from the repository root in the same terminal:

```bash
../.venv/bin/python src/main/build_camera_binding.py
export PYTHONPATH=/tmp/pystk-camera-build/build312
export XDG_CACHE_HOME=/tmp/pystk-font-cache

../.venv/bin/python -c 'import pystk2; print(pystk2.__file__); assert hasattr(pystk2.Kart, "rescue_active")'
```

The build helper reuses/rebuilds the isolated camera/rescue-enabled PySTK copy.
It does not replace the installed package. The cache setting reuses the local
game assets; missing assets are downloaded automatically. See `patches/README.md`
for native build prerequisites. The temporary build/cache can be recreated if
`/tmp` is cleared.

## 2. Prepare the shared expert reference once

```bash
caffeinate -i ../.venv/bin/python src/main/benchmark_expert.py \
  --experiment src/main/multimap_50k.yaml \
  --freeze-timeouts
```

This runs/reuses Team4 on all six maps with the fixed five benchmark seeds and
a generous 5,000-step cap. It freezes per-map limits and baseline metrics under
`datasets/multimap/expert_reference/`. All six training runs reuse this exact
reference. If the expert never finishes a map, inspect that result before
starting training; no timeout is invented for it.

The fixed expert reference is independent of each algorithm's training seed.
Unseen-map expert results define task limits/reference metrics, and do not enter
policy training or best-checkpoint selection.

## 3. Train the three DAgger seeds

```bash
for s in 1000 2000 3000; do
  echo "DAgger - training seed $s"
  caffeinate -i ../.venv/bin/python src/main/DAgger.py \
    --experiment src/main/multimap_50k.yaml \
    seed=$s online_seed_start=$s \
    output_dir=datasets/multimap/full50k/dagger_seed$s || break
done
```

`N=11`: fit/evaluate on offline data first, then ten collect/train/evaluate
rounds. Each online round collects at least 5k steps and trains for five epochs.
The zero-online-step evaluation is preserved. Per-map episode seed counters
start at the selected training seed and persist across rounds.

## 4. Train the three RLPD seeds

```bash
for s in 1000 2000 3000; do
  echo "RLPD - training seed $s"
  caffeinate -i ../.venv/bin/python src/main/RLPD.py \
    --experiment src/main/multimap_50k.yaml \
    seed=$s online_seed_start=$s \
    output_dir=datasets/multimap/full50k/rlpd_seed$s || break
done
```

RLPD updates during online collection using the shared offline data and its own
online data. Evaluation uses deterministic actions. The numerically stable tanh
log-probability correction is included. Identity mean/scale arrays in saved
checkpoints provide loader compatibility without normalizing observations.

Run the two training loops sequentially to avoid sharing GPU/memory resources.
Each run needs a fresh output directory; a failure stops its loop. Existing smoke
and single-map results are preserved.

## 5. Plot each training seed's comparison

After both algorithms finish for a seed:

```bash
for s in 1000 2000 3000; do
  ../.venv/bin/python src/main/plot_multimap_comparison.py \
    --dagger datasets/multimap/full50k/dagger_seed$s \
    --rlpd datasets/multimap/full50k/rlpd_seed$s \
    --output-dir datasets/plots/multimap_50k/seed$s || break
done
```

SVG only. Each seed gets final-evaluation comparisons and per-map learning curves
for completion and expert-relative finishing steps, plus rescue/stuck summaries.
Standard-deviation bands describe variation over the five evaluation episodes.
Plots and CSVs use actual cumulative online steps. DAgger keeps its zero-step
post-offline point; no zero-step RLPD point is added.

## Budgets and saved results

50k is an online target, excluding the 19,493 offline transitions, expert
benchmarks and evaluation. Complete episodes can overshoot each collection
boundary. DAgger's overshoot accumulates across its ten minimum-5k rounds; RLPD
finishes its final episode after reaching the overall 50k target. Read actual
steps from the logs when comparing data efficiency. A fixed number of laps per
map is not imposed; collection chooses the least-collected map by steps.

Best checkpoints use training-map completion, then macro finishing steps when
available on every training map. Unseen results do not select the model. Final
comparison plots show the final evaluation and full histories; the saved best
checkpoint may come from an earlier evaluation.

Each directory under `datasets/multimap/full50k/` includes:

- `config.json`, offline-size summary and the frozen expert reference.
- `collection_by_track.csv`: interval/cumulative trajectories, steps and fractions.
- `evaluation_by_track.csv`, `evaluation_episodes.csv`, aggregate `evaluation.csv`.
- Detailed online and evaluation episodes with original fields plus world-position,
  time, rescue and stuck diagnostics; see `MULTIMAP.md` for the exact schema.
- DAgger loss CSV, iteration checkpoints, `actor_best.pt` and `actor_last.pt`.
- RLPD online episode/loss CSV, `agent_best.pt` and `agent_last.pt`.
