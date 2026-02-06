from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import upload_folder
from safetensors.torch import save_file


def find_latest_b_sweep(sweep_root: Path) -> Path:
    for sweep_dir in sorted(sweep_root.glob("sweep_*"), reverse=True):
        if list(sweep_dir.glob("trial_*_B_*")):
            return sweep_dir
        summary = sweep_dir / "summary.csv"
        if not summary.exists():
            continue
        with open(summary, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("stage") == "B":
                    return sweep_dir
    raise FileNotFoundError("no B sweep found under outputs/train_runs")


def select_best_b_trial(sweep_dir: Path) -> tuple[Path | None, dict]:
    summary = sweep_dir / "summary.csv"
    if summary.exists():
        best_row = None
        best_score = None
        with open(summary, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("stage") != "B":
                    continue
                score = float(row["score"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_row = row
        if best_row is not None:
            trial_id = int(float(best_row["trial"]))
            trial_dirs = sorted(sweep_dir.glob(f"trial_{trial_id:02d}_B_*"))
            trial_dir = trial_dirs[0] if trial_dirs else None
            return trial_dir, {"source": "summary.csv", "row": best_row}

    best_trial_dir = None
    best_score = None
    best_trial_id = None
    for trial_dir in sorted(sweep_dir.glob("trial_*_B_*")):
        metrics = trial_dir / "metrics.jsonl"
        if not metrics.exists():
            continue
        with open(metrics) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                score = float(row.get("score", 0.0))
                if best_score is None or score > best_score:
                    best_score = score
                    best_trial_dir = trial_dir
                    m = re.search(r"trial_(\d+)_B_", trial_dir.name)
                    best_trial_id = int(m.group(1)) if m else None

    if best_trial_dir is None:
        raise FileNotFoundError(f"no usable B trial found in {sweep_dir}")

    return best_trial_dir, {
        "source": "trial_metrics.jsonl",
        "row": {"stage": "B", "trial": best_trial_id, "score": best_score},
    }


def resolve_model_path(sweep_dir: Path, best_trial_dir: Path | None) -> Path:
    root_model = sweep_dir / "best_model.pt"
    if root_model.exists():
        return root_model
    if best_trial_dir is not None and (best_trial_dir / "best_model.pt").exists():
        return best_trial_dir / "best_model.pt"
    raise FileNotFoundError(f"best_model.pt not found in {sweep_dir}")


def resolve_config_path(sweep_dir: Path, best_trial_dir: Path | None) -> Path:
    if (sweep_dir / "best_config.json").exists():
        return sweep_dir / "best_config.json"
    if best_trial_dir is not None and (best_trial_dir / "config.json").exists():
        return best_trial_dir / "config.json"
    if (sweep_dir / "config.json").exists():
        return sweep_dir / "config.json"
    raise FileNotFoundError(f"config not found in {sweep_dir}")


def resolve_label_map_path(sweep_dir: Path, best_trial_dir: Path | None) -> Path:
    if best_trial_dir is not None and (best_trial_dir / "label_map.json").exists():
        return best_trial_dir / "label_map.json"
    if (sweep_dir / "label_map.json").exists():
        return sweep_dir / "label_map.json"
    raise FileNotFoundError(f"label_map.json not found in {sweep_dir}")


def main() -> None:
    load_dotenv(".env")
    token = os.environ["HF_TOKEN"]
    repo_id = "teru00801/deberta-20260206"

    sweep_root = Path("outputs/train_runs")
    sweep_dir = find_latest_b_sweep(sweep_root)
    best_trial_dir, selected = select_best_b_trial(sweep_dir)
    model_path = resolve_model_path(sweep_dir, best_trial_dir)
    config_path = resolve_config_path(sweep_dir, best_trial_dir)
    label_map_path = resolve_label_map_path(sweep_dir, best_trial_dir)

    export_dir = Path("outputs/hf_export_deberta-20260206")
    shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(model_path, map_location="cpu")
    save_file(state, export_dir / "best_model.safetensors")

    shutil.copy2(config_path, export_dir / "config.json")
    shutil.copy2(label_map_path, export_dir / "label_map.json")
    if (sweep_dir / "summary.csv").exists():
        shutil.copy2(sweep_dir / "summary.csv", export_dir / "summary.csv")
    with open(export_dir / "selected_b_trial.json", "w") as f:
        json.dump(
            {
                "sweep_dir": str(sweep_dir),
                "best_trial_dir": str(best_trial_dir) if best_trial_dir is not None else None,
                "model_path": str(model_path),
                "config_path": str(config_path),
                "label_map_path": str(label_map_path),
                "selected": selected,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    base_model_dir = Path("models/deberta_v3_mlm")
    shutil.copytree(base_model_dir, export_dir / "base_model", dirs_exist_ok=True)

    upload_folder(
        repo_id=repo_id,
        folder_path=str(export_dir),
        repo_type="model",
        token=token,
    )
    print(f"uploaded: {repo_id}")
    print(f"source sweep: {sweep_dir}")
    print(f"model path: {model_path}")


if __name__ == "__main__":
    main()
