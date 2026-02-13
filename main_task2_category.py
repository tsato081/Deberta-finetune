from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


@dataclass
class Config:
    model_id: str = "teru00801/deberta-v3-mlm"
    hf_dataset_repo: str = "teru00801/data-plus-high-prob"
    hf_filename: str = "all_categories_merged.csv"
    output_dir: Path = Path("outputs/task2_category_runs")
    tests: List[str] = None  # type: ignore[assignment]

    max_length: int = 384
    val_ratio: float = 0.1
    seed: int = 42

    batch_size: int = 16
    max_epochs: int = 10
    min_epochs: int = 3
    early_stop_patience: int = 2

    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    label_smoothing_task2: float = 0.05
    task2_balance_power: float = 0.5
    task2_balance_min: float = 0.2
    task2_balance_max: float = 5.0
    focal_gamma_task2: float = 0.0
    rdrop_alpha_task2: float = 0.0

    title_empty_token: str = "[TITLE_EMPTY]"
    use_amp: bool = True

    def __post_init__(self) -> None:
        if self.tests is None:
            self.tests = [
                "data_for_deberta/tests/Hawks4.0_refactored.csv",
                "data_for_deberta/tests/Hawks_ver5.0_refactored.csv",
                "data_for_deberta/tests/Hawks_ver5.1_refactored.csv",
                "data_for_deberta/tests/Hawks_ver6.0 csv出力用.csv",
            ]


def parse_args(cfg: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeBERTa Task2 category-only fine-tuning")
    parser.add_argument("--model-id", type=str, default=cfg.model_id)
    parser.add_argument("--hf-dataset-repo", type=str, default=cfg.hf_dataset_repo)
    parser.add_argument("--hf-filename", type=str, default=cfg.hf_filename)
    parser.add_argument("--output-dir", type=str, default=str(cfg.output_dir))
    parser.add_argument("--tests", nargs="+", default=cfg.tests)
    parser.add_argument("--max-length", type=int, default=cfg.max_length)
    parser.add_argument("--val-ratio", type=float, default=cfg.val_ratio)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--max-epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--min-epochs", type=int, default=cfg.min_epochs)
    parser.add_argument("--early-stop-patience", type=int, default=cfg.early_stop_patience)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--label-smoothing-task2", type=float, default=cfg.label_smoothing_task2)
    parser.add_argument("--task2-balance-power", type=float, default=cfg.task2_balance_power)
    parser.add_argument("--task2-balance-min", type=float, default=cfg.task2_balance_min)
    parser.add_argument("--task2-balance-max", type=float, default=cfg.task2_balance_max)
    parser.add_argument("--focal-gamma-task2", type=float, default=cfg.focal_gamma_task2)
    parser.add_argument("--rdrop-alpha-task2", type=float, default=cfg.rdrop_alpha_task2)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-use-amp", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-run-dir", type=str, default="")
    parser.add_argument("--eval-model-path", type=str, default="")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def resolve_text_columns(df: pd.DataFrame) -> Tuple[str, str]:
    title_col = "title" if "title" in df.columns else "title_original"
    body_col = "body" if "body" in df.columns else "body_original"
    if title_col not in df.columns or body_col not in df.columns:
        raise KeyError("CSV must include title/body or title_original/body_original columns")
    return title_col, body_col


def resolve_hf_token() -> str:
    load_dotenv(".env")
    return os.environ["HF_TOKEN"]


def download_train_csv(cfg: Config, token: str) -> Path:
    downloaded = hf_hub_download(
        repo_id=cfg.hf_dataset_repo,
        repo_type="dataset",
        filename=cfg.hf_filename,
        token=token,
    )
    return Path(downloaded)


def load_task2_train(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "category" not in raw.columns:
        raise KeyError("Training CSV must include category column")

    title_col, body_col = resolve_text_columns(raw)
    df = pd.DataFrame(
        {
            "title": raw[title_col].map(normalize_text),
            "body": raw[body_col].map(normalize_text),
            "category": raw["category"].map(normalize_text),
        }
    )
    df = df[df["category"] != ""].copy()
    df = df.drop_duplicates(subset=["title", "body", "category"], keep="first").reset_index(drop=True)
    return df


def split_train_val_by_category(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, group in df.groupby("category", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n = len(indices)
        if n <= 1:
            train_indices.extend(indices.tolist())
            continue
        n_val = max(1, int(n * val_ratio))
        n_val = min(n_val, n - 1)
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    train_df = df.loc[train_indices].copy().reset_index(drop=True)
    val_df = df.loc[val_indices].copy().reset_index(drop=True)

    if len(val_df) == 0:
        raise RuntimeError("Validation set is empty; training categories are too sparse.")
    return train_df, val_df


class Task2Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        label2id: Dict[str, int],
        title_empty_token: str,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.title_empty_token = title_empty_token

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        title = row["title"] if isinstance(row["title"], str) else ""
        body = row["body"] if isinstance(row["body"], str) else ""
        if title.strip() == "":
            title = self.title_empty_token
        text = f"タイトル: {title}\n本文: {body}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        label = self.label2id[row["category"]]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class Task2EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, title_empty_token: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.title_empty_token = title_empty_token

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        title = row["title"] if isinstance(row["title"], str) else ""
        body = row["body"] if isinstance(row["body"], str) else ""
        if title.strip() == "":
            title = self.title_empty_token
        text = f"タイトル: {title}\n本文: {body}"

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


class DebertaTask2Category(nn.Module):
    def __init__(self, model_id: str, num_labels: int, token: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, token=token)
        hidden = self.encoder.config.hidden_size
        dropout_prob = getattr(self.encoder.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return self.head(pooled)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    label_smoothing: float,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    if gamma <= 0:
        return F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing)
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none", label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def rdrop_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    logp_a = F.log_softmax(logits_a, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    kl_ab = F.kl_div(logp_a, p_b, reduction="batchmean")
    kl_ba = F.kl_div(logp_b, p_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def build_task2_class_weights(
    train_df: pd.DataFrame,
    label2id: Dict[str, int],
    power: float,
    min_w: float,
    max_w: float,
) -> torch.Tensor:
    counts = train_df["category"].value_counts().to_dict()
    median_count = float(np.median(list(counts.values()))) if counts else 1.0
    weights = torch.ones(len(label2id), dtype=torch.float32)
    for category, idx in label2id.items():
        count = max(counts.get(category, 1), 1)
        raw = (median_count / count) ** power
        weights[idx] = float(np.clip(raw, min_w, max_w))
    return weights


def compute_task2_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    if not labels:
        return {"acc": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Config,
    class_weights: torch.Tensor,
    scaler: GradScaler | None,
    epoch: int,
    max_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"train {epoch + 1}/{max_epochs}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type="cuda", enabled=cfg.use_amp):
            logits = model(input_ids, attention_mask)
            loss = focal_loss(
                logits,
                labels,
                gamma=cfg.focal_gamma_task2,
                label_smoothing=cfg.label_smoothing_task2,
                weight=class_weights,
            )
            if cfg.rdrop_alpha_task2 > 0:
                logits_b = model(input_ids, attention_mask)
                loss_b = focal_loss(
                    logits_b,
                    labels,
                    gamma=cfg.focal_gamma_task2,
                    label_smoothing=cfg.label_smoothing_task2,
                    weight=class_weights,
                )
                kl = rdrop_kl(logits, logits_b)
                loss = 0.5 * (loss + loss_b) + cfg.rdrop_alpha_task2 * kl

        optimizer.zero_grad()
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, List[int] | List[float]]:
    model.eval()
    labels: List[int] = []
    preds: List[int] = []
    pred_probs: List[float] = []

    for batch in tqdm(loader, desc="predict", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        prob = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

        labels.extend(batch_labels.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        pred_probs.extend(prob.cpu().tolist())

    return {"labels": labels, "preds": preds, "pred_probs": pred_probs}


def load_task2_eval_rows(path: Path, label2id: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    raw = pd.read_csv(path)
    if "category" not in raw.columns:
        raise KeyError(f"{path} must include category column")

    title_col, body_col = resolve_text_columns(raw)
    view = raw.copy()
    view["_row_id"] = np.arange(len(view))
    view["_title_norm"] = view[title_col].map(normalize_text)
    view["_body_norm"] = view[body_col].map(normalize_text)
    view["_category_norm"] = view["category"].map(normalize_text)
    has_category = view["_category_norm"] != ""
    known = has_category & view["_category_norm"].isin(label2id)

    eval_df = pd.DataFrame(
        {
            "row_id": view.loc[known, "_row_id"].to_numpy(),
            "title": view.loc[known, "_title_norm"].to_numpy(),
            "body": view.loc[known, "_body_norm"].to_numpy(),
            "category": view.loc[known, "_category_norm"].to_numpy(),
        }
    )
    eval_df["label_id"] = eval_df["category"].map(label2id).astype(int)

    stats = {
        "rows_total": int(len(raw)),
        "rows_with_category": int(has_category.sum()),
        "rows_evaluated": int(len(eval_df)),
    }
    return raw, eval_df, stats


def build_eval_output_rows(
    raw_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pred: Dict[str, List[int] | List[float]],
    id2label: Dict[int, str],
) -> pd.DataFrame:
    out = raw_df.copy()
    out["category_true"] = pd.NA
    out["category_pred"] = pd.NA
    out["category_prob"] = np.nan

    row_ids = eval_df["row_id"].to_numpy()
    labels = pred["labels"]
    preds = pred["preds"]
    probs = pred["pred_probs"]

    out.loc[row_ids, "category_true"] = [id2label[int(x)] for x in labels]
    out.loc[row_ids, "category_pred"] = [id2label[int(x)] for x in preds]
    out.loc[row_ids, "category_prob"] = probs
    return out


def safe_test_id(path: Path) -> str:
    test_id = re.sub(r"\W+", "_", path.stem, flags=re.UNICODE).strip("_")
    return test_id if test_id else "test"


def save_confusion_matrix_csv(
    labels: List[int],
    preds: List[int],
    id2label: Dict[int, str],
    out_path: Path,
) -> None:
    label_ids = list(range(len(id2label)))
    matrix = confusion_matrix(labels, preds, labels=label_ids)
    names = [id2label[idx] for idx in label_ids]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(out_path, index=True)


def run_test_evaluation(
    cfg: Config,
    run_dir: Path,
    tokenizer,
    model: nn.Module,
    label2id: Dict[str, int],
    suffix: str,
) -> List[Dict[str, object]]:
    id2label = {idx: label for label, idx in label2id.items()}
    device = torch.device("cuda")
    reports: List[Dict[str, object]] = []

    for test_path_str in cfg.tests:
        test_path = Path(test_path_str)
        raw_df, eval_df, stats = load_task2_eval_rows(test_path, label2id)
        eval_ds = Task2EvalDataset(eval_df, tokenizer, cfg.max_length, cfg.title_empty_token)
        eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        pred = predict(model, eval_loader, device)
        metrics = compute_task2_metrics(pred["labels"], pred["preds"])

        test_id = safe_test_id(test_path)
        pred_path = run_dir / f"test_predictions_{test_id}{suffix}.csv"
        conf_path = run_dir / f"confusion_matrix_{test_id}{suffix}.csv"

        out_df = build_eval_output_rows(raw_df, eval_df, pred, id2label)
        out_df.to_csv(pred_path, index=False)
        save_confusion_matrix_csv(pred["labels"], pred["preds"], id2label, conf_path)

        report = {
            "test_file": str(test_path),
            "rows_total": stats["rows_total"],
            "rows_with_category": stats["rows_with_category"],
            "rows_evaluated": stats["rows_evaluated"],
            "accuracy": metrics["acc"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "prediction_csv": str(pred_path),
            "confusion_matrix_csv": str(conf_path),
        }
        reports.append(report)

    json_path = run_dir / f"test_eval_summary{suffix}.json"
    csv_path = run_dir / f"test_eval_summary{suffix}.csv"
    with open(json_path, "w") as f:
        json.dump({"reports": reports}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(reports).to_csv(csv_path, index=False)
    return reports


def resolve_eval_run_dir(output_dir: Path, eval_run_dir: Path | None) -> Path:
    if eval_run_dir is not None:
        return eval_run_dir
    candidates = sorted([path for path in output_dir.iterdir() if path.is_dir()], key=lambda p: p.name, reverse=True)
    for path in candidates:
        if (path / "best_model.pt").exists() or (path / "best_model.safetensors").exists():
            return path
    raise FileNotFoundError(f"No run directory with best model found under: {output_dir}")


def resolve_eval_model_path(run_dir: Path, eval_model_path: Path | None) -> Path:
    if eval_model_path is not None:
        return eval_model_path
    pt = run_dir / "best_model.pt"
    if pt.exists():
        return pt
    safe = run_dir / "best_model.safetensors"
    if safe.exists():
        return safe
    raise FileNotFoundError(f"No best model file found in run dir: {run_dir}")


def load_state_dict(model_path: Path) -> Dict[str, torch.Tensor]:
    if model_path.suffix == ".safetensors":
        return load_safetensors(str(model_path))
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        return state
    raise RuntimeError(f"Unsupported checkpoint format: {model_path}")


def run_eval_only(cfg: Config, eval_run_dir: Path | None, eval_model_path: Path | None) -> None:
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (expected A100/H100).")

    token = resolve_hf_token()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_eval_run_dir(cfg.output_dir, eval_run_dir)
    model_path = resolve_eval_model_path(run_dir, eval_model_path)

    run_cfg_path = run_dir / "config.json"
    if run_cfg_path.exists():
        with open(run_cfg_path) as f:
            run_cfg = json.load(f)
        cfg.model_id = str(run_cfg.get("model_id", cfg.model_id))
        cfg.max_length = int(run_cfg.get("max_length", cfg.max_length))
        cfg.title_empty_token = str(run_cfg.get("title_empty_token", cfg.title_empty_token))
        cfg.tests = list(run_cfg.get("tests", cfg.tests))

    with open(run_dir / "label_map.json") as f:
        label2id = {str(k): int(v) for k, v in json.load(f).items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=token)
    tokenizer.add_special_tokens({"additional_special_tokens": [cfg.title_empty_token]})

    model = DebertaTask2Category(cfg.model_id, num_labels=len(label2id), token=token)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(load_state_dict(model_path), strict=True)
    model.to(torch.device("cuda"))

    reports = run_test_evaluation(
        cfg=cfg,
        run_dir=run_dir,
        tokenizer=tokenizer,
        model=model,
        label2id=label2id,
        suffix="_evalonly",
    )

    print("Saved:")
    print(f"  {run_dir / 'test_eval_summary_evalonly.json'}")
    print(f"  {run_dir / 'test_eval_summary_evalonly.csv'}")
    print(f"  reports={len(reports)}")


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (expected A100/H100).")
    device = torch.device("cuda")
    token = resolve_hf_token()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
        json.dump(payload, f, ensure_ascii=False, indent=2)

    train_csv_path = download_train_csv(cfg, token)
    train_all_df = load_task2_train(train_csv_path)
    train_df, val_df = split_train_val_by_category(train_all_df, cfg.val_ratio, cfg.seed)
    labels = sorted(train_all_df["category"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}

    with open(run_dir / "label_map.json", "w") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=token)
    tokenizer.add_special_tokens({"additional_special_tokens": [cfg.title_empty_token]})

    train_ds = Task2Dataset(train_df, tokenizer, cfg.max_length, label2id, cfg.title_empty_token)
    val_ds = Task2Dataset(val_df, tokenizer, cfg.max_length, label2id, cfg.title_empty_token)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = DebertaTask2Category(cfg.model_id, num_labels=len(label2id), token=token)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    class_weights = build_task2_class_weights(
        train_df=train_df,
        label2id=label2id,
        power=cfg.task2_balance_power,
        min_w=cfg.task2_balance_min,
        max_w=cfg.task2_balance_max,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_loader) * cfg.max_epochs, 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler("cuda") if cfg.use_amp else None

    metrics_path = run_dir / "metrics.jsonl"
    best_score = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    patience = 0
    best_pt_path = run_dir / "best_model.pt"

    for epoch in tqdm(range(cfg.max_epochs), desc="epochs"):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            class_weights=class_weights,
            scaler=scaler,
            epoch=epoch,
            max_epochs=cfg.max_epochs,
        )

        val_pred = predict(model, val_loader, device)
        val_metrics = compute_task2_metrics(val_pred["labels"], val_pred["preds"])
        score = val_metrics["acc"]

        with open(metrics_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_acc": val_metrics["acc"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_weighted_f1": val_metrics["weighted_f1"],
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = val_metrics
            torch.save(model.state_dict(), best_pt_path)
            patience = 0
        else:
            if epoch + 1 >= cfg.min_epochs:
                patience += 1

        if epoch + 1 >= cfg.min_epochs and patience >= cfg.early_stop_patience:
            break

    model.load_state_dict(torch.load(best_pt_path, map_location="cpu"), strict=True)
    model.to(device)
    safe_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_safetensors(safe_state, str(run_dir / "best_model.safetensors"))

    reports = run_test_evaluation(
        cfg=cfg,
        run_dir=run_dir,
        tokenizer=tokenizer,
        model=model,
        label2id=label2id,
        suffix="",
    )

    with open(run_dir / "summary.json", "w") as f:
        json.dump(
            {
                "train_source_repo": cfg.hf_dataset_repo,
                "train_source_file": cfg.hf_filename,
                "downloaded_train_csv": str(train_csv_path),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "num_categories": int(len(label2id)),
                "val_metrics": best_metrics,
                "test_reports_path_json": str(run_dir / "test_eval_summary.json"),
                "test_reports_path_csv": str(run_dir / "test_eval_summary.csv"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved:")
    print(f"  {run_dir / 'best_model.pt'}")
    print(f"  {run_dir / 'best_model.safetensors'}")
    print(f"  {run_dir / 'summary.json'}")
    print(f"  {run_dir / 'test_eval_summary.json'}")
    print(f"  {run_dir / 'test_eval_summary.csv'}")
    print(f"  reports={len(reports)}")


def main() -> None:
    cfg = Config()
    args = parse_args(cfg)

    cfg.model_id = args.model_id
    cfg.hf_dataset_repo = args.hf_dataset_repo
    cfg.hf_filename = args.hf_filename
    cfg.output_dir = Path(args.output_dir)
    cfg.tests = list(args.tests)
    cfg.max_length = args.max_length
    cfg.val_ratio = args.val_ratio
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.min_epochs = args.min_epochs
    cfg.early_stop_patience = args.early_stop_patience
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.warmup_ratio = args.warmup_ratio
    cfg.label_smoothing_task2 = args.label_smoothing_task2
    cfg.task2_balance_power = args.task2_balance_power
    cfg.task2_balance_min = args.task2_balance_min
    cfg.task2_balance_max = args.task2_balance_max
    cfg.focal_gamma_task2 = args.focal_gamma_task2
    cfg.rdrop_alpha_task2 = args.rdrop_alpha_task2
    if args.use_amp:
        cfg.use_amp = True
    if args.no_use_amp:
        cfg.use_amp = False

    if args.eval_only:
        eval_run_dir = Path(args.eval_run_dir) if args.eval_run_dir else None
        eval_model_path = Path(args.eval_model_path) if args.eval_model_path else None
        run_eval_only(cfg, eval_run_dir=eval_run_dir, eval_model_path=eval_model_path)
        return

    run(cfg)


if __name__ == "__main__":
    main()
