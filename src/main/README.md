# Running experiments

Commands below start from the repository root (`pystk_project/`), with dependencies installed in the environment used by `uv run`. DAgger additionally needs PyTorch (`uv pip install torch`).

## DAgger training — `DAgger.py`

Collect expert/learner episodes, aggregate expert labels, train the actor, and evaluate the learner alone after each iteration. Runs headless, with Team4 as expert and one kart doing one lap.

```bash
uv run src/main/DAgger.py
uv run src/main/DAgger.py --config src/main/dagger_config.yaml
```

Only CLI option: `--config`, the YAML configuration path. Defaults to `dagger_config.yaml` beside the script. Change experiment settings in YAML:

| Parameter | Meaning |
|---|---|
| `N` | Maximum DAgger iterations: collect, train, evaluate |
| `T` | Learner-driven episodes per training map per later iteration |
| `initial_trajectories` | Expert-driven episodes per training map in iteration 1 |
| `max_steps` | Step limit per collection/evaluation episode |
| `seed` | Base collection seed; increments each episode; also seeds actor initialization |
| `train_tracks` | Explicit list of training maps; one shared actor learns from all |
| `unseen_tracks` | Diagnostic evaluation maps, excluded from training and checkpoint selection |
| `step_size` | Simulation step duration in seconds |
| `output_dir` | Experiment directory; relative paths resolve from repository root. Use a fresh directory per experiment. |
| `actor.num_paths`, `actor.num_items` | Number of path/item slots passed to NN; raw saved observations stay complete |
| `actor.hidden_sizes` | MLP hidden-layer widths, e.g. `[256, 256]` |
| `actor.learning_rate` | Adam learning rate |
| `actor.batch_size` | Training examples per gradient update |
| `actor.epochs` | Full passes through aggregated training data per iteration |
| `actor.continuous_weight`, `actor.boolean_weight` | Weights for continuous MSE and Boolean BCE losses |
| `actor.device` | `auto`: MPS when available, otherwise CPU; `cpu`: force CPU |
| `expert_benchmark.runs`, `expert_benchmark.seed_start` | Expert baseline episodes per map and first seed; benchmark runs automatically before training |
| `evaluation.seeds` | Fixed validation seeds, separate from all collection seeds |
| `evaluation.min_relative_improvement` | Minimum decrease in equal-map mean actor/expert finishing-step ratio at tied completion |

Saves episodes under `iteration_*/`, `episodes.json`, `config.json`, collection `metrics.csv`, `losses.csv`, `normalizer.npz`, and `actor_iteration_*.pt`. Evaluation writes `evaluation_episodes.csv`, `evaluation.csv`, `best_evaluation.json`, and `actor_best.pt`.

Best model: equally weighted mean completion rate across training maps first, then lower equally weighted mean actor/expert finishing-step ratio. Speed comparison requires identical completed (map, seed) cases and a defined ratio on every training map. Only one overall actor_best.pt is selected. All N iterations run, each training for actor.epochs; there is no early stopping. Evaluation CSVs include track identifiers. Evaluation reports steps for all episodes and successful episodes separately, with sample standard deviations. Evaluation episodes never enter training. Existing experiment configuration is not overwritten; resume is not implemented.

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

Run Team4 alone on all `train_tracks` and `unseen_tracks` maps, with a fresh expert per episode. Task settings (`max_steps`, `step_size`, one kart, one lap) match DAgger. Episodes are evaluated only and never added to training.

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

## Multi-track experiment outputs

Collection episodes use `iteration_<i>/<track>/episode_<n>.pkl.gz`. All contribute to one shared actor; minibatches sample uniformly over transitions, so longer episodes contribute more examples. Normalization fits all initial training-map episodes and then freezes.

The automatic baseline is saved under the experiment's `expert_benchmark/` directory (`episodes.csv`, `summary.csv`, `config.json`). It runs on both training and unseen maps. Training stops with an explanatory error if the expert never finishes a training map: a reference finishing-step ratio cannot be defined.

`evaluation.csv` reports per-map results including `relative_finish_steps` (actor mean finishing steps / expert mean finishing steps). `evaluation_aggregate.csv` reports training-map completion and relative speed used for checkpoint selection. Unseen maps never affect selection.

The baseline also saves `track_geometry/<track>.json` with official track length and segment statistics, plus `<track>_segments.csv` with each world-space segment's length in meters. Segment sums can differ from official length on branched tracks. With `num_paths: 20`, the count is fixed but physical lookahead varies with segment lengths.

## Plot DAgger results — `plot_dagger.py`

```bash
uv run src/main/plot_dagger.py --experiment datasets/dagger
```

- `--experiment`: required experiment directory containing CSV logs.
- `--output-dir`: optional destination; defaults to `<experiment>/plots/`.

Writes `losses.png` (MSE/BCE/total by logged epoch), `evaluation_by_track.png` (completion and all/successful episode steps, with expert reference lines), `relative_steps.png` (actor/expert finishing-step ratio), and `training_maps_aggregate.png` (aggregate training-map performance versus collected steps). The last two require the corresponding newer log columns/files. Standard-deviation bands show variation across episodes, not uncertainty across independent training runs. Unavailable statistics remain blank; successful-only speed must be read alongside completion rate. Run again after training to refresh plots.

## Track segment strips — `plot_track_segments.py`

```bash
uv run src/main/plot_track_segments.py --experiment datasets/dagger
```

One horizontal row per map, measured in meters, with vertical ticks separating segments. Reads `expert_benchmark/track_geometry/*_segments.csv`. Saves `plots/track_segments.png` and a zoomable `plots/track_segments.svg`. Optional `--output path/to/figure.png` overrides destination. Segments are concatenated in segment-ID order; branched tracks need not sum to official track length, which is also shown in the labels.

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
