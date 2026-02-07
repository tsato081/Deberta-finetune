from __future__ import annotations

import argparse
import hashlib
import json
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
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


@dataclass
class Config:
    model_dir: Path = Path("models/deberta_v3_mlm")
    task1_csv: Path = Path("Data_for_deberta/processed/task1_ready/train.csv")
    task2_csv: Path = Path("Data_for_deberta/processed/task2_ready/train.csv")
    output_dir: Path = Path("outputs/train_runs")

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

    boost_mult: float = 1.0
    focal_gamma_task1: float = 0.0
    rdrop_alpha_task2: float = 0.2

    lambda_task1: float = 1.0
    lambda_task2: float = 1.0
    label_smoothing_task1: float = 0.02
    label_smoothing_task2: float = 0.05
    task2_balance_power: float = 0.5
    task2_balance_min: float = 0.2
    task2_balance_max: float = 5.0

    title_empty_token: str = "[TITLE_EMPTY]"
    use_amp: bool = True
    cartography: bool = False
    cartography_lowmean_q: float = 0.005
    cartography_task2_per_category_cap: int = 30


def parse_args(cfg: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeBERTa multi-task fine-tuning (Task1 + Task2)")
    parser.add_argument("--model-dir", type=str, default=str(cfg.model_dir))
    parser.add_argument("--task1-csv", type=str, default=str(cfg.task1_csv))
    parser.add_argument("--task2-csv", type=str, default=str(cfg.task2_csv))
    parser.add_argument("--output-dir", type=str, default=str(cfg.output_dir))
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
    parser.add_argument("--boost-mult", type=float, default=cfg.boost_mult)
    parser.add_argument("--focal-gamma-task1", type=float, default=cfg.focal_gamma_task1)
    parser.add_argument("--rdrop-alpha-task2", type=float, default=cfg.rdrop_alpha_task2)
    parser.add_argument("--task2-balance-power", type=float, default=cfg.task2_balance_power)
    parser.add_argument("--task2-balance-min", type=float, default=cfg.task2_balance_min)
    parser.add_argument("--task2-balance-max", type=float, default=cfg.task2_balance_max)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--cartography", action="store_true")
    parser.add_argument("--cartography-lowmean-q", type=float, default=cfg.cartography_lowmean_q)
    parser.add_argument(
        "--cartography-task2-per-category-cap",
        type=int,
        default=cfg.cartography_task2_per_category_cap,
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-stage", type=str, default="AB", choices=["A", "B", "AB"])
    parser.add_argument("--stage-b-lr", type=float, default=None)
    parser.add_argument("--stage-b-boost", type=float, default=None)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def example_id(title: str, body: str) -> str:
    return hashlib.md5(f"{title}\n{body}".encode("utf-8")).hexdigest()


def load_task1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["title", "body", "pick"]].copy()


def load_task2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["title", "body", "category"]].copy()


def build_combined(task1_df: pd.DataFrame, task2_df: pd.DataFrame) -> pd.DataFrame:
    t1 = task1_df.copy()
    t2 = task2_df.copy()

    t1["title"] = t1["title"].map(normalize_text)
    t1["body"] = t1["body"].map(normalize_text)
    t2["title"] = t2["title"].map(normalize_text)
    t2["body"] = t2["body"].map(normalize_text)

    t1["example_id"] = [example_id(a, b) for a, b in zip(t1["title"], t1["body"])]
    t2["example_id"] = [example_id(a, b) for a, b in zip(t2["title"], t2["body"])]

    t1 = t1.drop_duplicates(subset=["example_id", "pick"], keep="first")
    t2 = t2.drop_duplicates(subset=["example_id", "category"], keep="first")

    t1["category"] = ""
    t2["pick"] = "Pick"

    merged = pd.concat(
        [t1[["example_id", "title", "body", "pick", "category"]], t2[["example_id", "title", "body", "pick", "category"]]],
        ignore_index=True,
    )

    def agg_first_nonempty(series: pd.Series) -> str:
        for v in series:
            if isinstance(v, str) and v.strip():
                return v
        return ""

    def agg_pick(series: pd.Series) -> str:
        vals = {v for v in series if isinstance(v, str) and v.strip()}
        if "Pick" in vals:
            return "Pick"
        if "Decline" in vals:
            return "Decline"
        return ""

    def agg_category(series: pd.Series) -> str:
        for v in series:
            if isinstance(v, str) and v.strip():
                return v
        return ""

    grouped = merged.groupby("example_id", as_index=False).agg(
        title=("title", agg_first_nonempty),
        body=("body", agg_first_nonempty),
        pick=("pick", agg_pick),
        category=("category", agg_category),
    )

    grouped.loc[grouped["category"] != "", "pick"] = "Pick"
    return grouped


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = df["example_id"].values
    n_val = max(1, int(len(ids) * val_ratio))
    val_idx = rng.choice(len(ids), size=n_val, replace=False)
    val_mask = np.zeros(len(ids), dtype=bool)
    val_mask[val_idx] = True
    df_val = df[val_mask].copy()
    df_train = df[~val_mask].copy()

    missing_cats = set(df["category"].unique()) - set(df_val["category"].unique())
    missing_cats.discard("")
    for cat in missing_cats:
        candidates = df_train[df_train["category"] == cat]
        if len(candidates) == 0:
            continue
        pick_one = candidates.sample(n=1, random_state=seed)
        df_val = pd.concat([df_val, pick_one], ignore_index=True)
        df_train = df_train.drop(pick_one.index)

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, label2id: Dict[str, int], title_empty_token: str):
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

        cat = row.get("category", "")
        label_t2 = -100
        if isinstance(cat, str) and cat in self.label2id:
            label_t2 = self.label2id[cat]
        is_task1_only = isinstance(cat, str) and cat == ""

        pick = row.get("pick", "")
        label_t1 = -100
        if is_task1_only:
            if pick == "Pick":
                label_t1 = 1
            elif pick == "Decline":
                label_t1 = 0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_t1": torch.tensor(label_t1, dtype=torch.long),
            "labels_t2": torch.tensor(label_t2, dtype=torch.long),
            "is_task1_only": torch.tensor(1 if is_task1_only else 0, dtype=torch.long),
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }


class DebertaMultiTask(nn.Module):
    def __init__(self, base_model: str, num_task2: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        dropout_prob = getattr(self.encoder.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.head_t1 = nn.Linear(hidden, 2)
        self.head_t2 = nn.Linear(hidden, num_task2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, 0]
        pooled = self.dropout(pooled)
        return self.head_t1(pooled), self.head_t2(pooled)


def build_sampler(
    df: pd.DataFrame,
    boost_mult: float,
    task2_balance_power: float,
    task2_balance_min: float,
    task2_balance_max: float,
) -> WeightedRandomSampler:
    pick = df["pick"].fillna("").astype(str)
    cat = df["category"].fillna("").astype(str)

    task1_mask = cat == ""
    task2_mask = cat != ""

    pick_cnt = pick[task1_mask].value_counts().to_dict()
    cat_cnt = cat[task2_mask].value_counts().to_dict()
    median_cat = float(np.median(list(cat_cnt.values()))) if cat_cnt else 1.0

    n_task1 = int(task1_mask.sum())
    n_task2 = int(task2_mask.sum())
    task2_boost = (n_task1 / n_task2) * boost_mult if n_task2 > 0 else 1.0
    per_class_total = (n_task1 / len(pick_cnt)) if pick_cnt else 1.0

    def w_pick(p: str) -> float:
        if (not p) or (not pick_cnt):
            return 1.0
        return per_class_total / max(pick_cnt.get(p, 1), 1)

    def w_cat(c: str) -> float:
        if not c:
            return 1.0
        raw = (median_cat / max(cat_cnt.get(c, 1), 1)) ** task2_balance_power
        return float(np.clip(raw, task2_balance_min, task2_balance_max))

    weights = []
    for p, c in zip(pick, cat):
        if c:
            w = w_cat(c) * task2_boost
        else:
            w = w_pick(p)
        weights.append(w)

    weights = np.array(weights, dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float, label_smoothing: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    logp_y = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    p_y = logp_y.exp()
    if gamma <= 0:
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    ce = -logp_y
    loss = ((1.0 - p_y) ** gamma) * ce
    return loss.mean()


def rdrop_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    logp_a = F.log_softmax(logits_a, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    kl_ab = F.kl_div(logp_a, p_b, reduction="batchmean")
    kl_ba = F.kl_div(logp_b, p_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {"acc": 0.0, "f1": 0.0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
    }


def build_task2_class_weights(
    train_df: pd.DataFrame,
    label2id: Dict[str, int],
    task2_balance_power: float,
    task2_balance_min: float,
    task2_balance_max: float,
) -> torch.Tensor:
    cat_series = train_df["category"].fillna("").astype(str)
    cat_series = cat_series[cat_series != ""]
    cat_cnt = cat_series.value_counts().to_dict()
    median_cat = float(np.median(list(cat_cnt.values()))) if cat_cnt else 1.0

    weights = torch.ones(len(label2id), dtype=torch.float32)
    for cat, idx in label2id.items():
        count = max(cat_cnt.get(cat, 1), 1)
        raw = (median_cat / count) ** task2_balance_power
        weight = float(np.clip(raw, task2_balance_min, task2_balance_max))
        weights[idx] = weight
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Config,
    task2_class_weights: torch.Tensor,
    scaler: GradScaler | None,
    epoch: int,
    max_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"train {epoch + 1}/{max_epochs}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t1 = batch["labels_t1"].to(device)
        labels_t2 = batch["labels_t2"].to(device)

        with autocast(device_type="cuda", enabled=cfg.use_amp):
            logits_t1, logits_t2 = model(input_ids, attention_mask)

            loss_t1 = logits_t1.new_tensor(0.0)
            loss_t2 = logits_t2.new_tensor(0.0)

            mask_t1 = labels_t1 != -100
            if mask_t1.any():
                ls_t1 = 0.0 if cfg.focal_gamma_task1 > 0 else cfg.label_smoothing_task1
                loss_t1 = focal_loss(
                    logits_t1[mask_t1],
                    labels_t1[mask_t1],
                    gamma=cfg.focal_gamma_task1,
                    label_smoothing=ls_t1,
                )

            mask_t2 = labels_t2 != -100
            if mask_t2.any():
                loss_t2 = F.cross_entropy(
                    logits_t2[mask_t2],
                    labels_t2[mask_t2],
                    weight=task2_class_weights,
                    label_smoothing=cfg.label_smoothing_task2,
                )

                if cfg.rdrop_alpha_task2 > 0:
                    logits_t1_b, logits_t2_b = model(input_ids, attention_mask)
                    loss_t2_b = F.cross_entropy(
                        logits_t2_b[mask_t2],
                        labels_t2[mask_t2],
                        weight=task2_class_weights,
                        label_smoothing=cfg.label_smoothing_task2,
                    )
                    kl = rdrop_kl(logits_t2[mask_t2], logits_t2_b[mask_t2])
                    loss_t2 = 0.5 * (loss_t2 + loss_t2_b) + cfg.rdrop_alpha_task2 * kl

            loss = cfg.lambda_task1 * loss_t1 + cfg.lambda_task2 * loss_t2

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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    t1_labels: List[int] = []
    t1_preds: List[int] = []
    t1_is_task1_only: List[int] = []
    t2_labels: List[int] = []
    t2_preds: List[int] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t1 = batch["labels_t1"].to(device)
        labels_t2 = batch["labels_t2"].to(device)
        is_task1_only = batch["is_task1_only"].to(device)

        logits_t1, logits_t2 = model(input_ids, attention_mask)
        pred_t1 = logits_t1.argmax(dim=-1)
        pred_t2 = logits_t2.argmax(dim=-1)

        mask_t1 = labels_t1 != -100
        if mask_t1.any():
            t1_labels.extend(labels_t1[mask_t1].cpu().tolist())
            t1_preds.extend(pred_t1[mask_t1].cpu().tolist())
            t1_is_task1_only.extend(is_task1_only[mask_t1].cpu().tolist())

        mask_t2 = labels_t2 != -100
        if mask_t2.any():
            t2_labels.extend(labels_t2[mask_t2].cpu().tolist())
            t2_preds.extend(pred_t2[mask_t2].cpu().tolist())

    t1_all = compute_metrics(t1_labels, t1_preds)
    t1_task1only_labels = [y for y, m in zip(t1_labels, t1_is_task1_only) if m == 1]
    t1_task1only_preds = [y for y, m in zip(t1_preds, t1_is_task1_only) if m == 1]
    t1_task1only = compute_metrics(t1_task1only_labels, t1_task1only_preds)
    t2 = {
        "acc": float(accuracy_score(t2_labels, t2_preds)) if t2_labels else 0.0,
        "f1": float(f1_score(t2_labels, t2_preds, average="macro")) if t2_labels else 0.0,
    }
    return {
        "task1_acc": t1_all["acc"],
        "task1_f1": t1_all["f1"],
        "task1_acc_task1only": t1_task1only["acc"],
        "task1_f1_task1only": t1_task1only["f1"],
        "task2_acc": t2["acc"],
        "task2_f1": t2["f1"],
    }


@torch.no_grad()
def collect_cartography_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    n = len(loader.dataset)
    p_t1 = np.full(n, np.nan, dtype=np.float32)
    c_t1 = np.full(n, np.nan, dtype=np.float32)
    p_t2 = np.full(n, np.nan, dtype=np.float32)
    c_t2 = np.full(n, np.nan, dtype=np.float32)

    for batch in tqdm(loader, desc="cartography", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t1 = batch["labels_t1"].to(device)
        labels_t2 = batch["labels_t2"].to(device)
        sample_idx = batch["sample_idx"].cpu().numpy()

        logits_t1, logits_t2 = model(input_ids, attention_mask)
        probs_t1 = torch.softmax(logits_t1, dim=-1)
        probs_t2 = torch.softmax(logits_t2, dim=-1)

        mask_t1 = labels_t1 != -100
        if mask_t1.any():
            sid = sample_idx[mask_t1.cpu().numpy()]
            labels = labels_t1[mask_t1]
            p = probs_t1[mask_t1].gather(1, labels.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
            pred = logits_t1[mask_t1].argmax(dim=-1)
            corr = (pred == labels).float().detach().cpu().numpy()
            p_t1[sid] = p
            c_t1[sid] = corr

        mask_t2 = labels_t2 != -100
        if mask_t2.any():
            sid = sample_idx[mask_t2.cpu().numpy()]
            labels = labels_t2[mask_t2]
            p = probs_t2[mask_t2].gather(1, labels.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
            pred = logits_t2[mask_t2].argmax(dim=-1)
            corr = (pred == labels).float().detach().cpu().numpy()
            p_t2[sid] = p
            c_t2[sid] = corr

    return {"p_t1": p_t1, "c_t1": c_t1, "p_t2": p_t2, "c_t2": c_t2}


def build_cartography_table(
    train_df: pd.DataFrame,
    probs: np.ndarray,
    correct: np.ndarray,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    seen = np.sum(~np.isnan(probs), axis=0)
    sum_probs = np.nansum(probs, axis=0)
    confidence = np.divide(sum_probs, seen, out=np.zeros_like(sum_probs, dtype=np.float32), where=seen > 0)

    centered = probs - confidence[None, :]
    centered[np.isnan(probs)] = 0.0
    sq = centered**2
    variability = np.sqrt(np.divide(np.sum(sq, axis=0), seen, out=np.zeros_like(sum_probs, dtype=np.float32), where=seen > 0))

    sum_correct = np.nansum(correct, axis=0)
    correctness = np.divide(sum_correct, seen, out=np.zeros_like(sum_correct, dtype=np.float32), where=seen > 0)

    idx = np.where(valid_mask)[0]
    out = train_df.loc[idx, ["example_id", "title", "body", "pick", "category"]].copy()
    out["confidence"] = confidence[idx]
    out["variability"] = variability[idx]
    out["correctness"] = correctness[idx]
    out["epochs_seen"] = seen[idx]
    return out


def save_cartography_scatter(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        df["variability"].values,
        df["confidence"].values,
        c=df["correctness"].values,
        cmap="viridis",
        s=8,
        alpha=0.45,
    )
    ax.set_xlabel("Variability")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="Correctness")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def save_cartography_outputs(
    run_dir: Path,
    train_df: pd.DataFrame,
    probs_t1_epochs: List[np.ndarray],
    corr_t1_epochs: List[np.ndarray],
    probs_t2_epochs: List[np.ndarray],
    corr_t2_epochs: List[np.ndarray],
    lowmean_q: float,
    task2_cap: int,
) -> None:
    cartography_dir = run_dir / "cartography"
    suspects_dir = run_dir / "suspects"
    cartography_dir.mkdir(parents=True, exist_ok=True)
    suspects_dir.mkdir(parents=True, exist_ok=True)

    probs_t1 = np.stack(probs_t1_epochs, axis=0)
    corr_t1 = np.stack(corr_t1_epochs, axis=0)
    probs_t2 = np.stack(probs_t2_epochs, axis=0)
    corr_t2 = np.stack(corr_t2_epochs, axis=0)

    task1_valid = train_df["category"].fillna("").astype(str).values == ""
    task2_valid = train_df["category"].fillna("").astype(str).values != ""

    t1_df = build_cartography_table(train_df, probs_t1, corr_t1, task1_valid)
    t2_df = build_cartography_table(train_df, probs_t2, corr_t2, task2_valid)

    t1_df.to_csv(cartography_dir / "task1_cartography.csv", index=False)
    t2_df.to_csv(cartography_dir / "task2_cartography.csv", index=False)

    q = float(np.clip(lowmean_q, 0.0, 1.0))
    t1_thr = float(t1_df["confidence"].quantile(q))
    t2_thr = float(t2_df["confidence"].quantile(q))

    t1_suspects = t1_df[t1_df["confidence"] <= t1_thr].sort_values(
        ["confidence", "variability", "correctness"],
        ascending=[True, False, True],
    )
    t1_suspects.to_csv(suspects_dir / "cartography_task1_lowmean.csv", index=False)

    t2_suspects = t2_df[t2_df["confidence"] <= t2_thr].sort_values(
        ["category", "confidence", "variability", "correctness"],
        ascending=[True, True, False, True],
    )
    t2_suspects = t2_suspects.groupby("category", as_index=False).head(task2_cap)
    t2_suspects.to_csv(suspects_dir / "cartography_task2_lowmean.csv", index=False)

    plot_t1 = save_cartography_scatter(t1_df, cartography_dir / "task1_cartography.png", "Task1 Cartography")
    plot_t2 = save_cartography_scatter(t2_df, cartography_dir / "task2_cartography.png", "Task2 Cartography")

    with open(cartography_dir / "summary.json", "w") as f:
        json.dump(
            {
                "epochs": int(probs_t1.shape[0]),
                "task1_rows": int(len(t1_df)),
                "task2_rows": int(len(t2_df)),
                "task1_lowmean_threshold": t1_thr,
                "task2_lowmean_threshold": t2_thr,
                "task1_suspects": int(len(t1_suspects)),
                "task2_suspects": int(len(t2_suspects)),
                "task2_per_category_cap": int(task2_cap),
                "plots_saved": {"task1": plot_t1, "task2": plot_t2},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def train_and_eval(cfg: Config, run_dir: Path, save_best_path: Path | None = None) -> Dict[str, float]:
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (expected A100/H100).")
    device = torch.device("cuda")

    task1_df = load_task1(cfg.task1_csv)
    task2_df = load_task2(cfg.task2_csv)
    combined = build_combined(task1_df, task2_df)

    label2id = {c: i for i, c in enumerate(sorted([c for c in combined["category"].unique() if c]))}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": [cfg.title_empty_token]})

    train_df, val_df = split_train_val(combined, cfg.val_ratio, cfg.seed)
    train_ds = MultiTaskDataset(train_df, tokenizer, cfg.max_length, label2id, cfg.title_empty_token)
    val_ds = MultiTaskDataset(val_df, tokenizer, cfg.max_length, label2id, cfg.title_empty_token)

    sampler = build_sampler(
        train_df,
        cfg.boost_mult,
        cfg.task2_balance_power,
        cfg.task2_balance_min,
        cfg.task2_balance_max,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    cartography_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = DebertaMultiTask(str(cfg.model_dir), num_task2=len(label2id))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)
    task2_class_weights = build_task2_class_weights(
        train_df,
        label2id,
        cfg.task2_balance_power,
        cfg.task2_balance_min,
        cfg.task2_balance_max,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_loader) * cfg.max_epochs, 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler("cuda") if cfg.use_amp else None

    best_score = -1.0
    best_metrics = {}
    patience = 0
    probs_t1_epochs: List[np.ndarray] = []
    corr_t1_epochs: List[np.ndarray] = []
    probs_t2_epochs: List[np.ndarray] = []
    corr_t2_epochs: List[np.ndarray] = []

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in tqdm(range(cfg.max_epochs), desc="epochs"):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, task2_class_weights, scaler, epoch, cfg.max_epochs
        )
        metrics = evaluate(model, val_loader, device)
        if cfg.cartography:
            cartography_epoch = collect_cartography_epoch(model, cartography_loader, device)
            probs_t1_epochs.append(cartography_epoch["p_t1"])
            corr_t1_epochs.append(cartography_epoch["c_t1"])
            probs_t2_epochs.append(cartography_epoch["p_t2"])
            corr_t2_epochs.append(cartography_epoch["c_t2"])
        task1_acc_for_score = metrics.get("task1_acc_task1only", metrics.get("task1_acc", 0.0))
        score = task1_acc_for_score + metrics.get("task2_acc", 0.0)

        with open(metrics_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        **metrics,
                        "task1_acc_for_score": task1_acc_for_score,
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if score > best_score:
            best_score = score
            best_metrics = {**metrics, "task1_acc_for_score": task1_acc_for_score, "score": score, "epoch": epoch + 1}
            if save_best_path is not None:
                torch.save(model.state_dict(), save_best_path)
            else:
                torch.save(model.state_dict(), run_dir / "best_model.pt")
            patience = 0
        else:
            if epoch + 1 >= cfg.min_epochs:
                patience += 1
        if epoch + 1 >= cfg.min_epochs and patience >= cfg.early_stop_patience:
            break

    with open(run_dir / "label_map.json", "w") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    if cfg.cartography and probs_t1_epochs and probs_t2_epochs:
        save_cartography_outputs(
            run_dir=run_dir,
            train_df=train_df,
            probs_t1_epochs=probs_t1_epochs,
            corr_t1_epochs=corr_t1_epochs,
            probs_t2_epochs=probs_t2_epochs,
            corr_t2_epochs=corr_t2_epochs,
            lowmean_q=cfg.cartography_lowmean_q,
            task2_cap=cfg.cartography_task2_per_category_cap,
        )

    return best_metrics


def run_single(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, f, ensure_ascii=False, indent=2)
    train_and_eval(cfg, run_dir)


def run_sweep(
    cfg: Config,
    sweep_stage: str,
    stage_b_lr: float | None = None,
    stage_b_boost: float | None = None,
) -> None:
    sweep_dir = cfg.output_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_path = sweep_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "stage,trial,learning_rate,boost_mult,focal_gamma_task1,rdrop_alpha_task2,task1_acc_task1only,task1_acc,task2_acc,score,epoch\n"
        )

    global_best = -1.0
    global_best_cfg = None
    global_best_path = sweep_dir / "best_model.pt"

    best_a = None
    trial_id = 0
    if sweep_stage in {"A", "AB"}:
        stage_a_lrs = [1.5e-5, 2.0e-5, 3.0e-5]
        stage_a_boost = [0.8, 1.0, 1.2]
        stage_a_trials = []
        for lr in stage_a_lrs:
            for boost in stage_a_boost:
                stage_a_trials.append((lr, boost))

        for lr, boost in stage_a_trials:
            trial_id += 1
            trial_dir = sweep_dir / f"trial_{trial_id:02d}_A_lr{lr}_boost{boost}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_cfg = Config(**asdict(cfg))
            trial_cfg.learning_rate = lr
            trial_cfg.boost_mult = boost
            trial_cfg.focal_gamma_task1 = 0.0
            trial_cfg.rdrop_alpha_task2 = 0.2
            with open(trial_dir / "config.json", "w") as f:
                json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(trial_cfg).items()}, f, ensure_ascii=False, indent=2)

            metrics = train_and_eval(trial_cfg, trial_dir, save_best_path=global_best_path)
            score = metrics.get("score", 0.0)
            if score > global_best:
                global_best = score
                global_best_cfg = asdict(trial_cfg)
            if best_a is None or score > best_a["score"]:
                best_a = {"lr": lr, "boost": boost, "score": score}

            with open(summary_path, "a") as f:
                f.write(
                    f"A,{trial_id},{lr},{boost},0.0,0.2,{metrics.get('task1_acc_task1only',0.0)},{metrics.get('task1_acc',0.0)},{metrics.get('task2_acc',0.0)},{score},{metrics.get('epoch',0)}\n"
                )

    stage_b_gamma = [0.0, 1.0, 2.0]
    stage_b_rdrop = [0.3, 0.7]

    if sweep_stage in {"B", "AB"}:
        if best_a is None:
            if (stage_b_lr is None) ^ (stage_b_boost is None):
                raise ValueError("--stage-b-lr and --stage-b-boost must be specified together.")
            if (stage_b_lr is not None) and (stage_b_boost is not None):
                best_a = {"lr": stage_b_lr, "boost": stage_b_boost, "score": None}

            latest = None
            if best_a is None:
                for prev in sorted(cfg.output_dir.glob("sweep_*"), reverse=True):
                    if prev == sweep_dir:
                        continue
                    summary = prev / "summary.csv"
                    if not summary.exists():
                        continue
                    with open(summary) as f:
                        header = f.readline().strip().split(",")
                        if "stage" not in header or "learning_rate" not in header or "boost_mult" not in header or "score" not in header:
                            continue
                        idx = {k: i for i, k in enumerate(header)}
                        for line in f:
                            parts = line.strip().split(",")
                            if not parts or parts[idx["stage"]] != "A":
                                continue
                            score = float(parts[idx["score"]])
                            lr = float(parts[idx["learning_rate"]])
                            boost = float(parts[idx["boost_mult"]])
                            if (latest is None) or (score > latest["score"]):
                                latest = {"lr": lr, "boost": boost, "score": score}
                    if latest is not None:
                        break
                best_a = latest or {"lr": cfg.learning_rate, "boost": cfg.boost_mult, "score": None}

        for gamma in stage_b_gamma:
            for rdrop in stage_b_rdrop:
                trial_id += 1
                lr = best_a["lr"]
                boost = best_a["boost"]
                trial_dir = sweep_dir / f"trial_{trial_id:02d}_B_gamma{gamma}_rdrop{rdrop}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                trial_cfg = Config(**asdict(cfg))
                trial_cfg.learning_rate = lr
                trial_cfg.boost_mult = boost
                trial_cfg.focal_gamma_task1 = gamma
                trial_cfg.rdrop_alpha_task2 = rdrop
                with open(trial_dir / "config.json", "w") as f:
                    json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(trial_cfg).items()}, f, ensure_ascii=False, indent=2)

                metrics = train_and_eval(trial_cfg, trial_dir, save_best_path=global_best_path)
                score = metrics.get("score", 0.0)
                if score > global_best:
                    global_best = score
                    global_best_cfg = asdict(trial_cfg)

                with open(summary_path, "a") as f:
                    f.write(
                        f"B,{trial_id},{lr},{boost},{gamma},{rdrop},{metrics.get('task1_acc_task1only',0.0)},{metrics.get('task1_acc',0.0)},{metrics.get('task2_acc',0.0)},{score},{metrics.get('epoch',0)}\n"
                    )

    if global_best_cfg is not None:
        with open(sweep_dir / "best_config.json", "w") as f:
            json.dump({k: str(v) if isinstance(v, Path) else v for k, v in global_best_cfg.items()}, f, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = Config()
    args = parse_args(cfg)

    cfg.model_dir = Path(args.model_dir)
    cfg.task1_csv = Path(args.task1_csv)
    cfg.task2_csv = Path(args.task2_csv)
    cfg.output_dir = Path(args.output_dir)
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
    cfg.boost_mult = args.boost_mult
    cfg.focal_gamma_task1 = args.focal_gamma_task1
    cfg.rdrop_alpha_task2 = args.rdrop_alpha_task2
    cfg.task2_balance_power = args.task2_balance_power
    cfg.task2_balance_min = args.task2_balance_min
    cfg.task2_balance_max = args.task2_balance_max
    cfg.use_amp = args.use_amp or cfg.use_amp
    cfg.cartography = args.cartography
    cfg.cartography_lowmean_q = args.cartography_lowmean_q
    cfg.cartography_task2_per_category_cap = args.cartography_task2_per_category_cap

    if args.sweep:
        run_sweep(cfg, args.sweep_stage, stage_b_lr=args.stage_b_lr, stage_b_boost=args.stage_b_boost)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
