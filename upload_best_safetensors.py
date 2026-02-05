from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import upload_folder
from safetensors.torch import save_file


def main() -> None:
    load_dotenv(".env")
    token = os.environ["HF_TOKEN"]
    repo_id = "teru00801/deberta-20260206"

    sweep_root = Path("outputs/train_runs")
    sweep_dirs = sorted(sweep_root.glob("sweep_*"))
    best = None
    for sweep_dir in sweep_dirs:
        summary = sweep_dir / "summary.csv"
        if not summary.exists():
            continue
        with open(summary, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                score = float(row["score"])
                if best is None or score > best["score"]:
                    best = {"score": score, "sweep_dir": sweep_dir}
    if best is None:
        raise FileNotFoundError("no sweep summary.csv found under outputs/train_runs")
    sweep_dir = best["sweep_dir"]

    best_pt = sweep_dir / "best_model.pt"

    export_dir = Path("outputs/hf_export_deberta-20260206")
    export_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(best_pt, map_location="cpu")
    save_file(state, export_dir / "best_model.safetensors")

    for name in ["label_map.json", "best_config.json", "summary.csv"]:
        shutil.copy2(sweep_dir / name, export_dir / name)

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


if __name__ == "__main__":
    main()
