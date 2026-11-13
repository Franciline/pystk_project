"""Run separate races and pass a numbered dataset path to each process."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "datasets",
    )
    args = parser.parse_args()
    if args.runs < 1 or args.start < 1:
        parser.error("--runs and --start must be positive")

    outputs = [
        args.output_dir / f"episode_{number}.pkl.gz"
        for number in range(args.start, args.start + args.runs)
    ]
    for output in outputs:
        if output.exists():
            parser.error(f"Output already exists: {output}; choose another --start or --output-dir")

    runner = Path(__file__).with_name("single_track_race_display.py")
    for run, output in enumerate(outputs, start=1):
        print(f"Run {run}/{args.runs}: {output.name}", flush=True)
        subprocess.run(
            [sys.executable, str(runner), "--output", str(output)], check=True
        )


if __name__ == "__main__":
    main()
