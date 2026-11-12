# Running experiments

Commands below start from the repository root (`pystk_project/`), with dependencies installed in the environment used by `uv run`. DAgger additionally needs PyTorch (`uv pip install torch`).

## DAgger training — `DAgger.py`

One kart, one lap, one map. Iteration 1 initializes the dataset and trains the actor; later iterations collect whole episodes until at least `steps_per_iteration` new steps, aggregate, train, and evaluate. The last episode can overshoot the budget. Actor has seven outputs; rescue is always forced false when executed. Expert labels are recorded every step, including during random exploration.

```bash
uv run src/main/DAgger.py
uv run src/main/DAgger.py --config src/main/dagger_config.yaml seed=2000 online_seed_start=2000 output_dir=datasets/protocol/dagger_seed2000
```

`key=value` overrides YAML for one invocation, including nested settings such as `actor.epochs=5`.

```bash
for s in 1000 2000 3000; do
  caffeinate -i ../.venv/bin/python src/main/DAgger.py seed=$s online_seed_start=$s output_dir=datasets/protocol/dagger_seed$s
done
```

The Python path above assumes the environment is in the parent directory; use your environment's Python path if different.

| Parameter | Meaning |
|---|---|
| `N` | Total rounds including initialization/training round 1; default 11 |
| `steps_per_iteration` | Minimum new steps in rounds 2…N; default 5000 |
| `max_steps` | Timeout per collection/evaluation episode; default 1000 |
| `track`, `step_size` | Fixed map and simulation duration per step |
| `seed` | Network initialization and training shuffle seed |
| `online_seed_start` | First collection episode seed; increments each episode |
| `online_epsilon` | One fixed random-action replacement probability for learner collection; default 0.1 |
| `initialization.mode` | `generate_offline`, `single_expert`, or `load_offline` |
| `initialization.epsilons` | Expert/random replacement probabilities for initialization; `[0.0, 0.1, 0.3]` |
| `initialization.episodes_per_epsilon` | Default 3: nine initialization episodes; duplicates retained |
| `initialization.dataset` | Source folder for trusted offline episodes when loading; existing expert labels used, missing labels reconstructed sequentially with Team4 |
| `actor.num_paths`, `actor.num_items` | Encoder limits; 182 paths and 10 items |
| `actor.hidden_sizes` | MLP widths; `[256, 256]` |
| `actor.learning_rate`, `actor.batch_size` | Adam learning rate and examples per update |
| `actor.epochs` | Fixed passes over all aggregate data per round; default 5 |
| `actor.continuous_weight`, `actor.boolean_weight` | MSE/BCE weights |
| `actor.device` | `auto` prefers MPS, otherwise CPU; `cpu` forces CPU |
| `evaluation.seeds` | Fixed evaluation seeds; defaults 10000–10004 |
| `evaluation.min_steps_improvement` | Minimum decrease in successful finishing steps at tied completion |
| `expert_benchmark.runs`, `expert_benchmark.seed_start` | Expert baseline settings; set `expert_benchmark: null` to skip |
| `output_dir` | Fresh experiment directory, relative to repository root |

`single_expert` collects one clean episode. `generate_offline` samples fresh whole random actions with the selected epsilon each step. Later rounds use only `online_epsilon`, not the initialization epsilon list. Rescue is disabled for all executed actions, expert labels, and evaluation. Raw seven-output architecture is preserved. Normalization fits initialization data once and freezes; minibatches sample uniformly over aggregated transitions.

Outputs: `iteration_*/episode_*.pkl.gz`, `episodes.json`, `config.json`, `metrics.csv`, `losses.csv`, `normalizer.npz`, `evaluation_episodes.csv`, `evaluation.csv`, `best_evaluation.json`, and checkpoints `actor_iteration_*.pt`, `actor_best.pt`, `actor_last.pt`. Best model prioritizes completion then mean successful finishing steps. All N rounds run. Initialization and online steps are recorded separately; evaluation and expert benchmark steps are not charged to online collection. Reused offline historical actions remain unchanged; their expert labels have rescue disabled. Existing experiment output is not resumed or overwritten.

## One displayed demonstration — `single_track_race_display.py`

Run Team4 alone on `abyss`, one lap, at most 1,000 steps; save a training episode.

```bash
uv run src/main/single_track_race_display.py --output datasets/episode_1.pkl.gz
```

- `--output`: optional recording file path. Parent directories are created; existing files are refused.
- Track, agent, and step limit are currently edited in the script, not via CLI.

## Multiple displayed demonstrations — `collect_runs.py`

Launch the previous script repeatedly, print run progress, and pass numbered output paths.

```bash
uv run src/main/collect_runs.py --runs 50 --start 1 --output-dir datasets
```

- `--runs`: episode count, default **5**.
- `--start`: first episode number, default **1**.
- `--output-dir`: destination, default repository `datasets/`.

Writes `episode_1.pkl.gz`, etc. Refuses existing destination filenames and stops if a subprocess fails. To continue numbering, pass a new `--start` explicitly.

## Offline RLPD data — `collect_rlpd.py`

Collect solo team-agent episodes with occasional random executed actions, retaining expert labels. This collects data; it does not train RLPD.

```bash
uv run src/main/collect_rlpd.py
```

No CLI parameters. Edit constants in the file: `TRACK`, `AGENTS`, `EPSILONS` (random-action probabilities), `SEEDS`, `MAX_STEPS`, `STEP_SIZE`, and `OUTPUT_DIR`.

Current defaults: `abyss`, teams 1–5, probabilities `[0.0, 0.1, 0.3]`, seeds 0–9: **150 episodes**. Writes `datasets/rlpd/<track>/<team>_eps<epsilon>_seed<seed>.pkl.gz`. Skips existing files, reports agent errors, and continues with remaining episodes.

## Legacy team comparisons

These compare teams 1–5 and write aggregate HTML results, rather than training datasets. Run from `src/main` because output paths are relative to the working directory:

```bash
mkdir -p docs
cd src/main
uv run multi_track_race_display.py
# Or:
uv run full_no_display.py
```

- `multi_track_race_display.py`: displayed races; default 10 races on random tracks. If `NB_RACES=1`, uses `xr591`. Edit `NB_RACES`, `MAX_STEPS`, and track selection in the file.
- `full_no_display.py`: headless comparison over `MAPS`, repeated `NB_REPEAT` times (currently 21 tracks × 4 repeats). Edit those constants and `MAX_STEPS` in the file.

Both overwrite repository `docs/index.html`. Neither accepts CLI parameters. `scores.py` and `__init__.py` are support modules, not experiment entry points.

## Saved episode format

Each trusted `.pkl.gz` file contains one episode dictionary:

- `observations`: `steps + 1` raw observation dictionaries.
- `actions`, `expert_actions`, `rewards`, `terminated`, `truncated`: `steps` entries each.
- `metadata`: task and collection settings.

Transition `t` uses `observations[t]` and `observations[t + 1]`. During expert-only driving, executed and expert actions match. Pickle files should only be loaded from trusted sources.

## Watch a saved DAgger actor

```bash
uv run src/main/single_track_race_display.py --checkpoint datasets/dagger/actor_best.pt --seed 10000
```

Loads network weights and normalization from the checkpoint. Reads the first training track, step duration, and step limit from adjacent `config.json`; use `--track` to watch another map. Older single-track configs remain supported. Keep the experiment config beside the checkpoint. Drives one kart for one lap with graphics.

- `--checkpoint`: actor checkpoint; omit to drive with Team4.
- `--seed`: optional simulator seed.
- `--device`: `auto` (MPS if available, otherwise CPU), `cpu`, or `mps`.
- `--output`: optional episode recording. When recording actor driving, Team4 supplies separate expert labels; it does not control the kart.
- `--track`: choose track.

## Expert performance baseline — `benchmark_expert.py`

Run Team4 alone on the configured `track`, with a fresh expert per episode. Older multi-track configurations are also supported. Task settings (`max_steps`, `step_size`, one kart, one lap) match DAgger. Episodes are evaluated only and never added to training.

```bash
uv run src/main/benchmark_expert.py --runs 5
uv run src/main/benchmark_expert.py --config src/main/dagger_config.yaml --runs 5 --seed-start 10000 --output-dir datasets/expert_benchmark_2
```

Options: `--config` selects YAML; `--runs` defaults to 5 episodes per map; `--seed-start` defaults to 10000 and seeds increment per episode; `--output-dir` defaults to `datasets/expert_benchmark/`. Use a fresh directory.

Saves `episodes.csv`, `summary.csv`, and `config.json`. Per-map completion rate and mean successful finishing steps can be plotted as horizontal expert baselines, with standard-deviation bands when available.

Override the display track with `--track minigolf`, including when loading a checkpoint:

```bash
uv run src/main/single_track_race_display.py --checkpoint datasets/dagger/actor_best.pt --track minigolf --seed 10000
```

## Plot DAgger results — `plot_dagger.py`

```bash
uv run src/main/plot_dagger.py --experiment datasets/dagger
```

- `--experiment`: required experiment directory containing CSV logs.
- `--output-dir`: optional destination; defaults to `<experiment>/plots/`.

Writes `losses.svg` (MSE/BCE/total by logged epoch), `evaluation_by_track.svg` (completion and all/successful episode steps, with expert reference lines), `relative_steps.svg` (actor/expert finishing-step ratio), and `training_maps_aggregate.svg` (aggregate training-map performance versus collected steps). The last two require the corresponding newer log columns/files. Standard-deviation bands show variation across episodes, not uncertainty across independent training runs. Unavailable statistics remain blank; successful-only speed must be read alongside completion rate. Run again after training to refresh plots.

## Track segment strips — `plot_track_segments.py`

```bash
uv run src/main/plot_track_segments.py --experiment datasets/dagger
```

One horizontal row per map, measured in meters, with vertical ticks separating segments. Reads `expert_benchmark/track_geometry/*_segments.csv`. Saves a zoomable `plots/track_segments.svg`. Optional `--output path/to/figure.svg` overrides destination. Segments are concatenated in segment-ID order; branched tracks need not sum to official track length, which is also shown in the labels.

## Watch expert/random mixtures

From repository root, watch Team4 mixed with random actions in five-step policy blocks:

```bash
uv run src/main/single_track_race_display.py --track abyss --seed 10000 --epsilon 0.2 --hold-steps 5
```

Every five steps, choose random policy with probability 0.2, otherwise Team4. During random blocks, sample a new action each step; `rescue` always stays false. Change `--hold-steps` to 1 to choose the policy every step. No `--random-mode` argument is needed.

```bash
# Choose policy every step.
uv run src/main/single_track_race_display.py --track abyss --seed 10000 --epsilon 0.2 --hold-steps 1
# Hold random/expert policy selection for 5 steps; sample fresh random controls each step.
uv run src/main/single_track_race_display.py --track abyss --seed 10000 --epsilon 0.2 --hold-steps 5
```

`--epsilon` defaults to 0 and `--hold-steps` to 1. Random actions are freshly sampled every step within random-policy blocks. At block boundaries, random is chosen with probability epsilon; consecutive random blocks are possible. Expert recomputes its suggestion every step, even during random blocks. Random rescue is always disabled. Mixing is unavailable with `--checkpoint`. Optional `--output` records executed actions and separate expert labels. No mode overlay or per-step prints are enabled; uncomment the action print in the runner if needed.

## Compare DAgger initializations: relative finishing steps

```bash
uv run src/main/plot_dagger.py --compare-protocol datasets/protocol --seeds 1000 2000 3000
```

Reads all `dagger_seed*` and `dagger_offline_seed*` evaluation logs and their expert benchmark summaries. One curve per initialization averages actor/expert finishing-step ratios across training seeds; bands show ±1 sample standard deviation across those seeds (not across evaluation laps). Expert mean is treated as a fixed reference; 1 means expert speed, below 1 faster, above 1 slower. X-axis excludes initial data and measures new online collection steps. Runs are aligned every 5000 steps using their latest available evaluation and limited to their common budget. Successful laps only; completion means are included in the accompanying CSV. This compares online sample use, not equal total initial-data budgets.

Saves `datasets/protocol/plots/relative_steps_comparison.svg`, and `.csv`. Optional `--output-dir` changes the destination. Existing per-experiment plotting commands still work.

## Gaussian or random exploration during DAgger collection

Learner collection (iterations 2…N) now uses the YAML `exploration` settings:

```yaml
exploration:
  mode: gaussian
  probability: 1.0
  acceleration_std: 0.05
  steer_std: 0.1
```

Gaussian noise is added independently to acceleration and steering, clipped to their valid ranges. Boolean predictions stay unchanged, with rescue always false. `probability` gates Gaussian perturbations per step; `online_epsilon` is used only in random mode. Initialization still uses its original expert/random epsilon mixtures; loaded initial data is unchanged. Evaluation has no exploration. Saved expert labels remain unperturbed; episode metadata and `noise_steps`/`random_steps` identify exploration.

```bash
# Gaussian noise; choose a fresh output folder.
uv run src/main/DAgger.py exploration.mode=gaussian exploration.acceleration_std=0.05 exploration.steer_std=0.1 output_dir=datasets/protocol/dagger_gaussian_seed1000
# Original epsilon whole-action replacement remains available.
uv run src/main/DAgger.py exploration.mode=random online_epsilon=0.1 output_dir=datasets/protocol/dagger_random_seed1000
```

`load_offline` applies the same whole-episode deduplication as RLPD before converting expert labels. Both reuse `utils/episode_deduplication.py`: identical executed actions, rewards, and distance-down-track sequences identify duplicates; first occurrence is retained. Metadata differences alone do not distinguish episodes. Source files are unchanged. This is not transition-level deduplication; generated initialization and new online episodes are unaffected. Loader prints retained episode/transition counts. Current Team4 abyss source has 30 episodes; the shared rule retains 28 episodes / 17,560 transitions.

The same `--compare-protocol` command also creates:

- `initialization_seed1000.svg`, `initialization_seed2000.svg`, `initialization_seed3000.svg` (SVG only): one figure per training seed, with two initialization curves comparing completion, all-episode steps, and successful finishing steps. Bands show evaluation-episode standard deviation.
- `initialization_best_actors.svg`: each method's selected best actor, evaluated on matching seeds; completion and per-episode steps appear side by side, with mean-step reference lines.
- `initialization_best_actors.csv`: selected training seed, iteration, source, and saved metrics. Selection uses completion first, finishing steps second, then lowest seed for exact ties.

These use the existing recorded evaluations, without rerunning actors. Best-actor plots are selected validation results; use all-run curves for comparing methods. Initial data sizes differ and are excluded from the online-step axis.

### Best-actor comparison on 20 held-out seeds

Select the winning training seed using validation metrics first; do not select using final-test outcomes. For the current runs, winners are generated seed 3000 and loaded-offline seed 1000:

```bash
uv run src/main/final_test.py datasets/protocol/dagger_seed3000 datasets/protocol/dagger_offline_seed1000 --checkpoints best --runs 20 --seed-start 20000
uv run src/main/plot_dagger.py --compare-protocol datasets/protocol --seeds 1000 2000 3000
```

`--checkpoints best` evaluates only best weights (default remains best and last). Final-test results save in each selected run's `final_test.csv` and `final_test_episodes.csv`. Existing results are not overwritten. The `initialization_best_actors` figure and CSV now use these 20 matched held-out seeds, never the five validation seeds; it skips that figure if final results are missing. Seed-by-seed training plots still use validation history. The final test must contain 20 distinct matching seeds for both actors.

## Final DAgger versus RLPD comparison

```bash
uv run src/main/plot_final_comparison.py --dagger datasets/protocol/dagger_offline_seed1000 --rlpd datasets/protocol/rlpd_seed3000
```

Reads only `checkpoint=best` rows from each run's `final_test_episodes.csv`; requires matching test seeds. These runs were chosen by validation, not test outcomes. Creates paired seed curves with mean/std, boxplots with episode points, and RLPD-minus-DAgger differences on jointly successful laps. Incomplete episodes are marked with crosses; all-episode step statistics include timeouts. Saves `dagger_rlpd_final_comparison.svg`, `dagger_rlpd_final_summary.csv`, and `dagger_rlpd_final_paired.csv` under `datasets/protocol/plots/`. Optional `--output-dir` changes destination. No simulator rerun is required. This compares selected checkpoints, not mean performance over all training seeds.

## Saved RLPD reward evolution

```bash
uv run src/main/plot_rlpd_rewards.py --run datasets/protocol/rlpd_seed3000
```

Plots training episode returns from `online_episodes.csv`, plus return versus episode duration. Saves `<run>/plots/episode_returns.svg` (SVG only); optional `--output-dir` overrides destination. These are stochastic training-policy returns, not best-checkpoint final-test rewards. Existing logs do not store within-lap per-step rewards; those require a new recorded rollout.


## Compare checkpoint generalization on two maps

```bash
uv run src/main/compare_generalization.py \
  --dagger datasets/protocol/dagger_offline_seed1000/actor_best.pt \
  --rlpd datasets/protocol/rlpd_seed3000/agent_best.pt
```

Inputs are actual checkpoint paths. DAgger checkpoint includes its encoding/normalization; RLPD also needs its training `config.json` beside the checkpoint (normalization loads from checkpoint). Uses deterministic policies, no noise, rescue false, one kart/one lap, and the YAML task step duration and timeout. Change independent test maps/seeds in `dagger_config.yaml`:

```yaml
comparison:
  tracks: [abyss, snowtuxpeak]
  seeds: [20000, 20001, 20002, 20003, 20004]
  output_dir: datasets/protocol/comparison
```

Runs five episodes per model per map (20 total). Saves `episodes.csv`, `summary.csv`, `config.json`, and grouped `generalization_boxplots.svg`. One x-axis group per map, two boxes per group. Boxes include timeout episode lengths; crosses mark failures and labels report completion counts. Failed episodes must not be interpreted as successful lap times. Normalization stays frozen to training data on every map. This is an exploratory generalization test; the default seeds overlap earlier final-test seeds on abyss.

Options: `--config` selects another YAML, `--output-dir` selects a fresh destination, `--device auto|cpu|mps` selects inference device. Existing result folders are protected by their config file. Replot without rerunning policies:

```bash
uv run src/main/compare_generalization.py --plot-only
```

To test a checkpoint with observation normalization disabled in the display runner, add `--no-normalization`:

```bash
uv run src/main/single_track_race_display.py --checkpoint datasets/protocol/dagger_offline_seed1000/actor_best.pt --track abyss --no-normalization
```

Works for DAgger and RLPD checkpoints supported by the runner. Encoding remains active; numeric inputs keep their raw units. Saved checkpoint statistics are unchanged. Normalization is enabled by default. This deliberately changes the input distribution if the model was trained with normalization.

### Gaussian DAgger winner versus RLPD seed 1000, 20 test seeds

```bash
uv run src/main/final_test.py datasets/protocol/best_dagger_gaussian --checkpoints best --runs 20 --seed-start 20000
uv run src/main/plot_final_comparison.py \
  --dagger datasets/protocol/best_dagger_gaussian \
  --rlpd datasets/protocol/rlpd_seed1000 \
  --dagger-label 'DAgger Gaussian (offline, seed 1000)' \
  --rlpd-label 'RLPD best (seed 1000)' \
  --output-dir datasets/protocol/plots/gaussian_vs_rlpd_seed1000
```

The test results are already saved; do not rerun the first command unless using a fresh test destination. Plotting is repeatable. Adds `dagger_rlpd_mean_std_and_episodes.svg`: constant mean lines with ±1 episode-std bands above and actual paired seed results below. Other comparison figures and CSVs are also written. This explicitly compares RLPD seed 1000 as requested; RLPD seed 3000 remains the validation winner across its three training seeds.

## Completion over evaluation rounds (Gaussian DAgger vs RLPD seed 1000)

```bash
uv run src/main/plot_evaluation_completion.py
```

Reads both evaluation histories and saves `datasets/protocol/plots/gaussian_dagger_rlpd_completion.svg` and CSV. X labels are nominal 5k budget rounds; CSV retains actual online steps. DAgger's zero point follows offline training; RLPD has no recorded zero-point evaluation. Both current runs have 100% completion at all recorded evaluations. Optional `--dagger`, `--rlpd` select run folders; `--output-dir` changes destination.

## Final-use figure style

New final-use figures use Seaborn darkgrid styling, separate images per plot, and SVG vector exports under `datasets/plots/`. The final comparison uses Matplotlib's bundled `seaborn-v0_8-darkgrid` style, without requiring a seaborn installation.

```bash
uv run src/main/plot_final_comparison.py \
  --dagger datasets/protocol/best_dagger_gaussian \
  --rlpd datasets/protocol/rlpd_seed1000 \
  --dagger-label 'DAgger Gaussian' --rlpd-label 'RLPD'
```

Writes `dagger_rlpd_paired_episodes.svg` (wide seed-by-seed lines, mean/std) and `dagger_rlpd_step_distribution.svg` (filled boxes and actual episode points). No step-difference plot or combined subplot image is produced. Generated PDF/PNG copies have been removed; static plots now export SVG only. Default output directory is now `datasets/plots/`; omit older `--output-dir datasets/protocol/...` overrides to use it. Box medians are dark, not the default orange. Standard 1.5×IQR whiskers can coincide with box edges when all more extreme observations are outliers; all observations remain visible as points.

### Final relative learning curves: Expert, Gaussian DAgger, RLPD

```bash
uv run src/main/plot_dagger.py --relative-best datasets/protocol
```

Selects the best validation run across seeds 1000/2000/3000 for Gaussian loaded-offline DAgger and RLPD (currently DAgger 1000 and RLPD 3000). Plots their full validation histories at actual online-step counts, using a common Team4 validation benchmark mean as denominator. Labels: Expert (team4), DAgger, RLPD. Bands are evaluation episode std divided by expert mean, not std across training runs. No invented RLPD point at zero. Writes SVG and CSV under `datasets/plots/` by default, with darkgrid style and no footer. This replaces the earlier relative initialization-comparison function; use per-seed initialization figures for those comparisons.

For the same learning curves in actual finishing steps, without expert normalization:

```bash
uv run src/main/plot_dagger.py --steps-best datasets/protocol
```

Saves `datasets/plots/finishing_steps_comparison.svg`, and `.csv`. Same selected runs, actual online-step x-axis, Team4 mean reference line, and evaluation-episode standard-deviation bands as the relative plot.

## Matched 2D trajectories - Expert, DAgger, RLPD

```bash
uv run src/main/compare_trajectories.py
```

Runs each policy separately, alone on abyss with seed 20000, one lap, then overlays all three on the same world X-Z overhead map. Defaults use `best_dagger_gaussian/actor_best.pt` and available `rlpd_seed1000/agent_best.pt`. Override `--dagger`, `--rlpd`, `--track`, `--seed`, `--max-steps`, `--step-size`, or `--device`. All actor decisions are deterministic; exploration and manual rescue are disabled. RLPD needs adjacent training `config.json`; normalization comes from checkpoint.

Data default: `datasets/protocol/trajectories_abyss_seed20000/`. Saves per-policy CSVs with positions, velocity, speed, track progress, on-road state, rewards and executed controls, plus track geometry, settings, and rollout summaries. Traveled distance includes any simulator teleports, so treat it as diagnostic until rescue/jump handling is refined. Off-road steps are not fall counts. Existing data is protected by exclusive configuration creation; pass a fresh `--data-dir` for new rollouts.

Final figure: `datasets/plots/trajectories_abyss_seed20000.svg`, with darkgrid styling, approximate road strips, matched paths, a start marker, and endpoint dots. Overhead projection hides elevation/overpasses. A gap is drawn for consecutive positions more than 20 m apart to avoid connecting teleports. Replot without running the simulator:

```bash
uv run src/main/compare_trajectories.py --plot-only
```

Trajectory recordings now also save `<policy>_items.csv`: initial world-space item IDs, types and XYZ positions. Plot uses the expert rollout's initial item layout, with different markers/colors for bonus boxes, bananas and nitro pickups. This is not a depiction of later collected/spawned items. Triangles on each colored policy path show recorded kart positions at 25%, 50%, 75%, and 100% of that rollout's steps; they are not synchronized snapshots across policies. They share the path color. Existing recordings without item CSVs still plot normally.

```bash
uv run src/main/compare_trajectories.py --data-dir datasets/protocol/trajectories_abyss_seed20000_items
uv run src/main/compare_trajectories.py --plot-only --data-dir datasets/protocol/trajectories_abyss_seed20000_items
```

The 2D trajectory map uses a race-inspired illustration instead of darkgrid: green decorative grass stripes, asphalt-colored road strips, pale edges, colored paths and small kart glyphs. Grass texture is decorative, not STK scenery; geometry and item positions come from recorded simulator data. Kart glyph orientation follows recorded velocity. Other scientific plots retain darkgrid styling.

### Screenshot with a future-position marker

```bash
uv run src/main/plot_trajectory_screenshot.py --model rlpd --step 200 --future-steps 10
```

Replays the selected model (`rlpd`, `dagger`, or `expert`) using the track, seed and
checkpoint paths in `--data-dir/config.json`. The default data directory is
`datasets/protocol/trajectories_abyss_seed20000_normalized`. Saves SVG
and JSON metadata in `datasets/plots/` (`--output-dir` overrides this).
The JSON records the current and future world positions from the same replay.
The installed PySTK build returns no player camera matrices. A tested C++ patch
restores them without moving the camera: see `patches/pystk2-camera-state.patch`
and `patches/README.md`. Use the separate local test build to get the projected
orange dots for every actual future position from +1
through +N steps:

```bash
PYTHONPATH=/tmp/pystk-camera-build/build312 \
  ../.venv/bin/python src/main/plot_trajectory_screenshot.py \
  --model rlpd --step 385 --future-steps 10
```

This leaves the installed training environment unchanged. The test build is in
`/tmp`, so it must be rebuilt if removed. Without the patched binding, the script
reports the unavailable camera projection and does not guess future dots.
With camera matrices, it also saves an NPZ containing the image, matrices and
current/future world positions. `--future-steps 20` displays +1 through +20;
only the first and last dots are labeled to keep the image readable. Dots are
screen overlays, without depth occlusion.


### Record policy videos

Build the isolated camera-enabled PySTK binding (requires the existing macOS
build dependencies, `uv`, `patch`, Python 3.12 and network access for a fresh build):

```bash
../.venv/bin/python src/main/build_camera_binding.py
```

Record both selected seed-1000 actors with evaluation seed 20000:

```bash
PYTHONPATH=/tmp/pystk-camera-build/build312 \
  ../.venv/bin/python src/main/record_policy_videos.py \
  --track abyss --seed 20000 --future-steps 10
```

Requires FFmpeg and Pillow. Defaults are Gaussian DAgger
`datasets/protocol/best_dagger_gaussian/actor_best.pt` and RLPD
`datasets/protocol/rlpd_seed1000/agent_best.pt`. Override with `--dagger` and
`--rlpd`. `--models rlpd` or `--models dagger` records only one.

- `--future-steps N`: dots for each actual future position from +1 through +N.
- `--seed`, `--track`, `--max-steps`, `--step-size`, `--device`: rollout settings;
  defaults 20000, abyss, 1000, 0.1 and auto.
- `--preview-start`, `--preview-seconds`: GIF preview window, defaults 30 and 16
  seconds. The MP4 always contains the full recorded lap.
- `--output-dir`: MP4 and GIF files, default `assets/rollouts`.
- `--record-dir`: positions, camera matrices, times, executed actions and run
  metadata, default `datasets/protocol/rollout_videos`. Existing run directories
  are refused; choose another directory to record again.

Each frame and all its future positions belong to the same graphical lap; no
second replay is used. MP4 playback uses the simulation step size, and the GIF
preview is 5 fps. Near the finish there are fewer future dots. The overlay does
not test depth occlusion against scenery. Graphical rollouts may differ from
headless evaluation logs; these videos are demonstrations, not replacement metrics.

### Multi-map training and diagnostics

See [MULTIMAP.md](MULTIMAP.md) for the shared 3-train / 3-unseen map setup,
common offline collection, 5k-step commands for both algorithms, saved diagnostics
and comparison plots. This is a separate experiment from the original Abyss runs.

### Full multi-map experiment - 50k

Use [MULTIMAP_50K.md](MULTIMAP_50K.md) for the complete command sequence:
prepare the shared expert timeouts, train DAgger and RLPD on seeds 1000/2000/3000,
evaluate five fixed seeds on all six maps, and generate SVG comparisons per seed.
Configuration: `multimap_50k.yaml`. Both methods reuse `datasets/multimap/offline/`
(27 unique episodes, 19,493 transitions); no offline regeneration is needed.
