from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from main import DebertaMultiTask, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latest B-best sweep model on fixed test CSVs")
    parser.add_argument("--sweep-root", type=Path, default=Path("outputs/train_runs"))
    parser.add_argument(
        "--tests",
        nargs="+",
        default=[
            "Data_for_deberta/tests/Hawks4.0_refactored.csv",
            "Data_for_deberta/tests/Hawks_ver5.0_refactored.csv",
            "Data_for_deberta/tests/Hawks_ver5.1_refactored.csv",
        ],
    )
    parser.add_argument("--output-json", type=Path, default=Path("outputs/test_eval/b_best_fixed_tests.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def find_latest_b_trial(sweep_root: Path) -> tuple[Path, Path, Dict[str, str]]:
    for sweep_dir in sorted(sweep_root.glob("sweep_*"), reverse=True):
        summary_path = sweep_dir / "summary.csv"
        if not summary_path.exists():
            continue
        best_row = None
        best_score = None
        with open(summary_path, newline="") as f:
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
            if not trial_dirs:
                return sweep_dir, sweep_dir, best_row
            return sweep_dir, trial_dirs[0], best_row
    raise FileNotFoundError("No sweep with stage B rows found under outputs/train_runs")


def read_json(path: Path) -> Dict[str, object]:
    with open(path) as f:
        return json.load(f)


def resolve_config(sweep_dir: Path, b_trial_dir: Path) -> Dict[str, object]:
    if (b_trial_dir / "config.json").exists():
        return read_json(b_trial_dir / "config.json")
    if (sweep_dir / "best_config.json").exists():
        return read_json(sweep_dir / "best_config.json")
    if (sweep_dir / "config.json").exists():
        return read_json(sweep_dir / "config.json")
    raise FileNotFoundError("config not found in B trial dir or sweep dir")


def resolve_label_map(sweep_dir: Path, b_trial_dir: Path) -> Dict[str, int]:
    if (b_trial_dir / "label_map.json").exists():
        return read_json(b_trial_dir / "label_map.json")
    if (sweep_dir / "label_map.json").exists():
        return read_json(sweep_dir / "label_map.json")
    raise FileNotFoundError("label_map.json not found in B trial dir or sweep dir")


def resolve_model_path(sweep_dir: Path, b_trial_dir: Path) -> Path:
    if (b_trial_dir / "best_model.pt").exists():
        return b_trial_dir / "best_model.pt"
    if (sweep_dir / "best_model.pt").exists():
        return sweep_dir / "best_model.pt"
    raise FileNotFoundError("best_model.pt not found in B trial dir or sweep dir")


def pick_text_columns(df: pd.DataFrame) -> tuple[str, str]:
    title_col = "title" if "title" in df.columns else "title_original"
    body_col = "body" if "body" in df.columns else "body_original"
    return title_col, body_col


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, label2id: Dict[str, int], title_empty_token: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.title_empty_token = title_empty_token
        self.title_col, self.body_col = pick_text_columns(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        title = normalize_text(row.get(self.title_col, ""))
        body = normalize_text(row.get(self.body_col, ""))
        if not title:
            title = self.title_empty_token
        text = f"タイトル: {title}\n本文: {body}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        pick = str(row.get("pick", ""))
        label_t1 = -100
        if pick == "Pick":
            label_t1 = 1
        elif pick == "Decline":
            label_t1 = 0

        category = str(row.get("category", ""))
        label_t2 = self.label2id[category] if category in self.label2id else -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_t1": torch.tensor(label_t1, dtype=torch.long),
            "labels_t2": torch.tensor(label_t2, dtype=torch.long),
        }


@torch.no_grad()
def evaluate_one_file(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    t1_true: List[int] = []
    t1_pred: List[int] = []
    t2_true: List[int] = []
    t2_pred: List[int] = []

    for batch in tqdm(loader, desc="test", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t1 = batch["labels_t1"].to(device)
        labels_t2 = batch["labels_t2"].to(device)

        logits_t1, logits_t2 = model(input_ids, attention_mask)
        pred_t1 = logits_t1.argmax(dim=-1)
        pred_t2 = logits_t2.argmax(dim=-1)

        mask_t1 = labels_t1 != -100
        if mask_t1.any():
            t1_true.extend(labels_t1[mask_t1].cpu().tolist())
            t1_pred.extend(pred_t1[mask_t1].cpu().tolist())

        mask_t2 = labels_t2 != -100
        if mask_t2.any():
            t2_true.extend(labels_t2[mask_t2].cpu().tolist())
            t2_pred.extend(pred_t2[mask_t2].cpu().tolist())

    t1_precision, t1_recall, t1_f1, _ = precision_recall_fscore_support(
        t1_true, t1_pred, average="binary", zero_division=0
    )

    return {
        "task1_n": len(t1_true),
        "task1_acc": float(accuracy_score(t1_true, t1_pred)) if t1_true else 0.0,
        "task1_precision": float(t1_precision),
        "task1_recall": float(t1_recall),
        "task1_f1": float(t1_f1),
        "task2_n": len(t2_true),
        "task2_acc": float(accuracy_score(t2_true, t2_pred)) if t2_true else 0.0,
        "task2_f1_macro": float(f1_score(t2_true, t2_pred, average="macro")) if t2_true else 0.0,
        "task2_f1_weighted": float(f1_score(t2_true, t2_pred, average="weighted")) if t2_true else 0.0,
    }


def main() -> None:
    args = parse_args()

    sweep_dir, b_trial_dir, best_b_row = find_latest_b_trial(args.sweep_root)
    best_cfg = resolve_config(sweep_dir, b_trial_dir)
    label2id = resolve_label_map(sweep_dir, b_trial_dir)
    model_path = resolve_model_path(sweep_dir, b_trial_dir)

    model_dir = Path(best_cfg["model_dir"])
    max_length = int(best_cfg.get("max_length", 384))
    title_empty_token = str(best_cfg.get("title_empty_token", "[TITLE_EMPTY]"))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": [title_empty_token]})

    model = DebertaMultiTask(str(model_dir), num_task2=len(label2id))
    model.encoder.resize_token_embeddings(len(tokenizer))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    test_reports = []
    for test_path_str in args.tests:
        test_path = Path(test_path_str)
        df = pd.read_csv(test_path)
        dataset = EvalDataset(df, tokenizer, max_length, label2id, title_empty_token)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate_one_file(model, loader, device)
        report = {"test_file": str(test_path), "rows": int(len(df)), **metrics}
        test_reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_sweep_dir": str(sweep_dir),
        "selected_b_trial_dir": str(b_trial_dir),
        "selected_model_path": str(model_path),
        "selected_b_trial": best_b_row,
        "device": str(device),
        "reports": test_reports,
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
