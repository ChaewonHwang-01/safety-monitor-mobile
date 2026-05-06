from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show YOLO training progress.")
    parser.add_argument("--run", type=Path, default=Path("runs/train/hardhat-plus-yolov8n-80e"))
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROGRESS_RE = re.compile(
    r"(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+).*?"
    r"(?P<batch>\d+)\s*/\s*(?P<batches>\d+)"
)


def read_start_time(run_path: Path, start_path: Path) -> datetime:
    start_time = datetime.fromtimestamp(run_path.stat().st_ctime)
    if start_path.exists():
        try:
            start_time = datetime.fromisoformat(start_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return start_time


def now_like(value: datetime) -> datetime:
    return datetime.now(value.tzinfo) if value.tzinfo else datetime.now()


def show_live_log_progress(run_path: Path, start_path: Path, epochs: int) -> bool:
    log_path = run_path / "train.log"
    if not log_path.exists():
        return False

    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
    matches = []
    for chunk in re.split(r"[\r\n]+", ANSI_RE.sub("", text)):
        match = PROGRESS_RE.search(chunk)
        if match:
            matches.append(match)

    if not matches:
        return False

    last = matches[-1]
    current_epoch = int(last.group("epoch"))
    total_epochs = int(last.group("epochs")) or epochs
    batch = int(last.group("batch"))
    total_batches = int(last.group("batches"))

    start_time = read_start_time(run_path, start_path)
    now = now_like(start_time)
    elapsed = now - start_time
    epoch_fraction = batch / total_batches if total_batches else 0
    completed_fraction = max(0.0, min(1.0, ((current_epoch - 1) + epoch_fraction) / total_epochs))

    if completed_fraction > 0:
        total_seconds = elapsed.total_seconds() / completed_fraction
        eta_seconds = max(0.0, total_seconds - elapsed.total_seconds())
        eta = now.timestamp() + eta_seconds
        print(f"run: {run_path}")
        print(f"current epoch: {current_epoch}/{total_epochs}")
        print(f"current epoch progress: {batch}/{total_batches} ({epoch_fraction * 100:.1f}%)")
        print(f"overall progress: {completed_fraction * 100:.1f}%")
        print(f"elapsed: {elapsed}")
        print(f"estimated remaining: {eta_seconds / 3600:.2f} hours")
        print(f"estimated finish: {datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')}")
        print("metrics: waiting for YOLO to write results.csv")
    else:
        print(f"Training has started: {run_path}")
    return True


def main() -> None:
    args = parse_args()
    results_path = args.run / "results.csv"
    start_path = args.run / "start_time.txt"

    if not args.run.exists():
        print(f"Training run not found yet: {args.run}")
        return

    if not results_path.exists():
        if not show_live_log_progress(args.run, start_path, args.epochs):
            print(f"Training has started, but no completed epoch is recorded yet: {args.run}")
        return

    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    completed = len(rows)
    if completed == 0:
        print("No completed epochs yet.")
        return

    start_time = read_start_time(args.run, start_path)

    now = now_like(start_time)
    elapsed = now - start_time
    seconds_per_epoch = elapsed.total_seconds() / completed
    remaining_epochs = max(0, args.epochs - completed)
    eta_seconds = seconds_per_epoch * remaining_epochs
    eta = now.timestamp() + eta_seconds

    last = rows[-1]
    map50 = last.get("metrics/mAP50(B)", "").strip()
    map5095 = last.get("metrics/mAP50-95(B)", "").strip()

    print(f"run: {args.run}")
    print(f"completed epochs: {completed}/{args.epochs}")
    print(f"elapsed: {elapsed}")
    print(f"avg time/epoch: {seconds_per_epoch / 60:.2f} min")
    print(f"estimated remaining: {eta_seconds / 3600:.2f} hours")
    print(f"estimated finish: {datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"latest mAP50: {map50}")
    print(f"latest mAP50-95: {map5095}")


if __name__ == "__main__":
    main()
