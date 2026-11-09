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

## Compare DAgger initializations: relative finishing steps

```bash
uv run src/main/plot_dagger.py --compare-protocol datasets/protocol --seeds 1000 2000 3000
```

Reads all `dagger_seed*` and `dagger_offline_seed*` evaluation logs and their expert benchmark summaries. One curve per initialization averages actor/expert finishing-step ratios across training seeds; bands show ±1 sample standard deviation across those seeds (not across evaluation laps). Expert mean is treated as a fixed reference; 1 means expert speed, below 1 faster, above 1 slower. X-axis excludes initial data and measures new online collection steps. Runs are aligned every 5000 steps using their latest available evaluation and limited to their common budget. Successful laps only; completion means are included in the accompanying CSV. This compares online sample use, not equal total initial-data budgets.

Saves `datasets/protocol/plots/relative_steps_comparison.png`, `.svg`, and `.csv`. Optional `--output-dir` changes the destination. Existing per-experiment plotting commands still work.
