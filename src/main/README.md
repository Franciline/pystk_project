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
| `T` | Collection episodes per iteration |
| `max_steps` | Step limit per collection/evaluation episode |
| `beta_decay` | Expert execution probability: `beta_decay ** (iteration - 1)`; first iteration expert-only. Expert labels every collection step. |
| `seed` | Base collection seed; increments each episode; also seeds actor initialization |
| `track` | Track identifier, currently `abyss` |
| `step_size` | Simulation step duration in seconds |
| `output_dir` | Experiment directory; relative paths resolve from repository root. Use a fresh directory per experiment. |
| `actor.num_paths`, `actor.num_items` | Number of path/item slots passed to NN; raw saved observations stay complete |
| `actor.hidden_sizes` | MLP hidden-layer widths, e.g. `[256, 256]` |
| `actor.learning_rate` | Adam learning rate |
| `actor.batch_size` | Training examples per gradient update |
| `actor.epochs` | Full passes through aggregated training data per iteration |
| `actor.continuous_weight`, `actor.boolean_weight` | Weights for continuous MSE and Boolean BCE losses |
| `actor.device` | `auto`: MPS when available, otherwise CPU; `cpu`: force CPU |
| `evaluation.seeds` | Fixed validation seeds, separate from all collection seeds |
| `evaluation.min_steps_improvement` | Required decrease in mean successful finishing steps when completion rate is tied |
| `evaluation.patience` | Stop after this many consecutive evaluations without improvement |

Saves episodes under `iteration_*/`, `episodes.json`, `config.json`, collection `metrics.csv`, `losses.csv`, `normalizer.npz`, and `actor_iteration_*.pt`. Evaluation writes `evaluation_episodes.csv`, `evaluation.csv`, `best_evaluation.json`, and `actor_best.pt`.

Best model: higher completion rate first, then fewer mean finishing steps. Evaluation reports steps for all episodes and successful episodes separately, with sample standard deviations. Evaluation episodes never enter training. Existing experiment configuration is not overwritten; resume is not implemented.

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

Loads network weights and normalization from the checkpoint. Reads track, step duration, and step limit from `config.json` beside it when available; otherwise uses abyss, 0.1 seconds, and 1000 steps. Keep the experiment config beside the checkpoint. Drives one kart for one lap with graphics.

- `--checkpoint`: actor checkpoint; omit to drive with Team4.
- `--seed`: optional simulator seed.
- `--device`: `auto` (MPS if available, otherwise CPU), `cpu`, or `mps`.
- `--output`: optional episode recording. When recording actor driving, Team4 supplies separate expert labels; it does not control the kart.
