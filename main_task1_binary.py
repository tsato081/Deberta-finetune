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
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed


@dataclass
class Config:
    model_dir: Path = Path("models/deberta_v3_mlm")
    train_csv: Path = Path("data_for_deberta/Risk-only-pick/task1_risk_training_0209.csv")
    final_eval_csv: Path = Path("data_for_deberta/Risk-only-pick/Hawks_ver6.0 csv出力用_pickflip.csv")
    output_dir: Path = Path("outputs/task1_binary_runs")

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

    focal_gamma_task1: float = 0.0
    label_smoothing_task1: float = 0.02

    title_empty_token: str = "[TITLE_EMPTY]"
    use_amp: bool = True

    cartography: bool = True
    cartography_lowmean_q: float = 0.005


def parse_args(cfg: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeBERTa Task1 binary fine-tuning (Pick/Decline only)")
    parser.add_argument("--model-dir", type=str, default=str(cfg.model_dir))
    parser.add_argument("--train-csv", type=str, default=str(cfg.train_csv))
    parser.add_argument("--final-eval-csv", type=str, default=str(cfg.final_eval_csv))
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
    parser.add_argument("--focal-gamma-task1", type=float, default=cfg.focal_gamma_task1)
    parser.add_argument("--label-smoothing-task1", type=float, default=cfg.label_smoothing_task1)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-use-amp", action="store_true")
    parser.add_argument("--cartography", dest="cartography", action="store_true", default=cfg.cartography)
    parser.add_argument("--no-cartography", dest="cartography", action="store_false")
    parser.add_argument("--cartography-lowmean-q", type=float, default=cfg.cartography_lowmean_q)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-run-dir", type=str, default="")
    parser.add_argument("--eval-model-path", type=str, default="")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def normalize_pick_label(value: object) -> str:
    label = normalize_text(value).lower()
    if label in {"pick", "others", "1", "true"}:
        return "Pick"
    if label in {"decline", "0", "false"}:
        return "Decline"
    return ""


def example_id(title: str, body: str) -> str:
    return hashlib.md5(f"{title}\n{body}".encode("utf-8")).hexdigest()


def load_task1_binary(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = raw[["title", "body", "pick"]].copy()
    df["title"] = df["title"].map(normalize_text)
    df["body"] = df["body"].map(normalize_text)
    df["pick"] = df["pick"].map(normalize_pick_label)
    df = df[df["pick"].isin(["Decline", "Pick"])].copy()
    df["example_id"] = [example_id(title, body) for title, body in zip(df["title"], df["body"])]

    def agg_first_nonempty(series: pd.Series) -> str:
        for value in series:
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def agg_pick(series: pd.Series) -> str:
        values = {value for value in series if isinstance(value, str) and value.strip()}
        if "Pick" in values:
            return "Pick"
        if "Decline" in values:
            return "Decline"
        return ""

    grouped = df.groupby("example_id", as_index=False).agg(
        title=("title", agg_first_nonempty),
        body=("body", agg_first_nonempty),
        pick=("pick", agg_pick),
    )
    grouped = grouped[grouped["pick"].isin(["Decline", "Pick"])].reset_index(drop=True)
    return grouped


def resolve_text_columns(df: pd.DataFrame) -> Tuple[str, str]:
    title_col = "title" if "title" in df.columns else "title_original"
    body_col = "body" if "body" in df.columns else "body_original"
    if title_col not in df.columns or body_col not in df.columns:
        raise KeyError("CSV must include title/body or title_original/body_original columns")
    return title_col, body_col


def load_task1_eval_rows(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    title_col, body_col = resolve_text_columns(raw)
    view = raw.copy()
    view["_row_id"] = np.arange(len(view))
    view["_title_norm"] = view[title_col].map(normalize_text)
    view["_body_norm"] = view[body_col].map(normalize_text)
    view["_pick_norm"] = view["pick"].map(normalize_pick_label)
    eval_df = pd.DataFrame(
        {
            "row_id": view["_row_id"],
            "title": view["_title_norm"],
            "body": view["_body_norm"],
            "pick": view["_pick_norm"],
        }
    )
    eval_df = eval_df[eval_df["pick"].isin(["Decline", "Pick"])].reset_index(drop=True)
    return raw, eval_df


def build_final_eval_output_rows(
    raw_eval_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_pred: Dict[str, List[float] | List[int]],
) -> pd.DataFrame:
    out = raw_eval_df.copy()
    out["pick_true"] = pd.NA
    out["pick_pred"] = pd.NA
    out["pick_prob"] = np.nan
    row_ids = eval_df["row_id"].to_numpy()
    out.loc[row_ids, "pick_true"] = ["Pick" if label == 1 else "Decline" for label in eval_pred["labels"]]
    out.loc[row_ids, "pick_pred"] = ["Pick" if pred == 1 else "Decline" for pred in eval_pred["preds"]]
    out.loc[row_ids, "pick_prob"] = eval_pred["prob_pick"]
    return out


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(df) * val_ratio))
    val_idx = rng.choice(len(df), size=n_val, replace=False)
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[val_idx] = True
    df_val = df[val_mask].copy()
    df_train = df[~val_mask].copy()

    for label in ["Decline", "Pick"]:
        if label in set(df["pick"]) and label not in set(df_val["pick"]):
            candidates = df_train[df_train["pick"] == label]
            if len(candidates) > 0:
                pick_one = candidates.sample(n=1, random_state=seed)
                df_val = pd.concat([df_val, pick_one], ignore_index=True)
                df_train = df_train.drop(pick_one.index)

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


class Task1Dataset(Dataset):
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
        label = 1 if row["pick"] == "Pick" else 0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "sample_idx": torch.tensor(idx, dtype=torch.long),
        }


class DebertaTask1Binary(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        dropout_prob = getattr(self.encoder.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.head = nn.Linear(hidden, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return self.head(pooled)


def build_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df["pick"].value_counts().to_dict()
    per_class_total = len(df) / max(len(counts), 1)
    weights = []
    for label in df["pick"]:
        weight = per_class_total / max(counts.get(label, 1), 1)
        weights.append(weight)
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


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if not y_true:
        return {"acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Config,
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
            smoothing = 0.0 if cfg.focal_gamma_task1 > 0 else cfg.label_smoothing_task1
            loss = focal_loss(
                logits,
                labels,
                gamma=cfg.focal_gamma_task1,
                label_smoothing=smoothing,
            )

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
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, List[float] | List[int]]:
    model.eval()
    labels_all: List[int] = []
    preds_all: List[int] = []
    prob_pick_all: List[float] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        prob_pick_all.extend(probs[:, 1].detach().cpu().tolist())

    return {"labels": labels_all, "preds": preds_all, "prob_pick": prob_pick_all}


@torch.no_grad()
def collect_cartography_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    n = len(loader.dataset)
    probs_gold = np.full(n, np.nan, dtype=np.float32)
    correctness = np.full(n, np.nan, dtype=np.float32)

    for batch in tqdm(loader, desc="cartography", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sample_idx = batch["sample_idx"].cpu().numpy()

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        p = probs.gather(1, labels.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
        c = (pred == labels).float().detach().cpu().numpy()

        probs_gold[sample_idx] = p
        correctness[sample_idx] = c

    return {"p": probs_gold, "c": correctness}


def build_cartography_table(train_df: pd.DataFrame, probs: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    seen = np.sum(~np.isnan(probs), axis=0)
    sum_probs = np.nansum(probs, axis=0)
    confidence = np.divide(sum_probs, seen, out=np.zeros_like(sum_probs, dtype=np.float32), where=seen > 0)

    centered = probs - confidence[None, :]
    centered[np.isnan(probs)] = 0.0
    sq = centered**2
    variability = np.sqrt(np.divide(np.sum(sq, axis=0), seen, out=np.zeros_like(sum_probs, dtype=np.float32), where=seen > 0))

    sum_correct = np.nansum(correct, axis=0)
    correctness = np.divide(sum_correct, seen, out=np.zeros_like(sum_correct, dtype=np.float32), where=seen > 0)

    out = train_df[["example_id", "title", "body", "pick"]].copy()
    out["confidence"] = confidence
    out["variability"] = variability
    out["correctness"] = correctness
    out["epochs_seen"] = seen
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
    probs_epochs: List[np.ndarray],
    corr_epochs: List[np.ndarray],
    lowmean_q: float,
) -> None:
    cartography_dir = run_dir / "cartography"
    suspects_dir = run_dir / "suspects"
    cartography_dir.mkdir(parents=True, exist_ok=True)
    suspects_dir.mkdir(parents=True, exist_ok=True)

    probs = np.stack(probs_epochs, axis=0)
    corr = np.stack(corr_epochs, axis=0)
    cart_df = build_cartography_table(train_df, probs, corr)
    cart_df.to_csv(cartography_dir / "task1_cartography.csv", index=False)

    q = float(np.clip(lowmean_q, 0.0, 1.0))
    threshold = float(cart_df["confidence"].quantile(q))
    suspects = cart_df[cart_df["confidence"] <= threshold].sort_values(
        ["confidence", "variability", "correctness"],
        ascending=[True, False, True],
    )
    suspects.to_csv(suspects_dir / "cartography_task1_lowmean.csv", index=False)

    plot_saved = save_cartography_scatter(cart_df, cartography_dir / "task1_cartography.png", "Task1 Cartography")

    with open(cartography_dir / "summary.json", "w") as f:
        json.dump(
            {
                "epochs": int(probs.shape[0]),
                "task1_rows": int(len(cart_df)),
                "task1_lowmean_threshold": threshold,
                "task1_suspects": int(len(suspects)),
                "plot_saved": plot_saved,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def resolve_eval_run_dir(output_dir: Path, eval_run_dir: Path | None) -> Path:
    if eval_run_dir is not None:
        return eval_run_dir
    candidates = [path for path in output_dir.iterdir() if path.is_dir()]
    candidates = sorted(candidates, key=lambda path: path.name, reverse=True)
    for path in candidates:
        if (path / "best_model.pt").exists() or (path / "best_model.safetensors").exists():
            return path
    raise FileNotFoundError(f"No run directory with best_model found under: {output_dir}")


def resolve_eval_model_path(run_dir: Path, eval_model_path: Path | None) -> Path:
    if eval_model_path is not None:
        return eval_model_path
    pt_path = run_dir / "best_model.pt"
    if pt_path.exists():
        return pt_path
    safe_path = run_dir / "best_model.safetensors"
    if safe_path.exists():
        return safe_path
    raise FileNotFoundError(f"No best model found in run dir: {run_dir}")


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
    device = torch.device("cuda")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_eval_run_dir(cfg.output_dir, eval_run_dir)
    model_path = resolve_eval_model_path(run_dir, eval_model_path)

    run_cfg_path = run_dir / "config.json"
    if run_cfg_path.exists():
        with open(run_cfg_path) as f:
            run_cfg = json.load(f)
        if "model_dir" in run_cfg:
            cfg.model_dir = Path(run_cfg["model_dir"])
        if "max_length" in run_cfg:
            cfg.max_length = int(run_cfg["max_length"])
        if "title_empty_token" in run_cfg:
            cfg.title_empty_token = str(run_cfg["title_empty_token"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": [cfg.title_empty_token]})
    final_raw_df, final_eval_df = load_task1_eval_rows(cfg.final_eval_csv)
    final_ds = Task1Dataset(final_eval_df[["title", "body", "pick"]], tokenizer, cfg.max_length, cfg.title_empty_token)
    final_loader = DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = DebertaTask1Binary(str(cfg.model_dir))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(load_state_dict(model_path), strict=True)
    model.to(device)

    final_pred = predict(model, final_loader, device)
    final_metrics = compute_metrics(final_pred["labels"], final_pred["preds"])
    baseline_always_pick = float(np.mean(np.array(final_pred["labels"]) == 1)) if final_pred["labels"] else 0.0
    baseline_always_decline = float(np.mean(np.array(final_pred["labels"]) == 0)) if final_pred["labels"] else 0.0

    final_out = build_final_eval_output_rows(final_raw_df, final_eval_df, final_pred)
    pred_out = run_dir / "final_eval_predictions_evalonly.csv"
    sum_out = run_dir / "final_eval_summary_evalonly.csv"
    final_out.to_csv(pred_out, index=False)

    pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "model_path": str(model_path),
                "dataset": str(cfg.final_eval_csv),
                "rows_raw": int(len(final_raw_df)),
                "rows_evaluated": int(len(final_pred["labels"])),
                "acc": final_metrics["acc"],
                "f1": final_metrics["f1"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "always_pick_acc": baseline_always_pick,
                "always_decline_acc": baseline_always_decline,
            }
        ]
    ).to_csv(sum_out, index=False)

    print("Saved:")
    print(f"  {pred_out}")
    print(f"  {sum_out}")


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (expected A100/H100).")
    device = torch.device("cuda")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, f, ensure_ascii=False, indent=2)

    train_all_df = load_task1_binary(cfg.train_csv)
    train_df, val_df = split_train_val(train_all_df, cfg.val_ratio, cfg.seed)
    final_raw_df, final_eval_df = load_task1_eval_rows(cfg.final_eval_csv)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": [cfg.title_empty_token]})

    train_ds = Task1Dataset(train_df, tokenizer, cfg.max_length, cfg.title_empty_token)
    val_ds = Task1Dataset(val_df, tokenizer, cfg.max_length, cfg.title_empty_token)
    final_ds = Task1Dataset(final_eval_df[["title", "body", "pick"]], tokenizer, cfg.max_length, cfg.title_empty_token)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=build_sampler(train_df), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    final_loader = DataLoader(final_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    cartography_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = DebertaTask1Binary(str(cfg.model_dir))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_loader) * cfg.max_epochs, 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler("cuda") if cfg.use_amp else None

    best_score = -1.0
    best_epoch = 0
    patience = 0
    best_path = run_dir / "best_model.pt"
    probs_epochs: List[np.ndarray] = []
    corr_epochs: List[np.ndarray] = []

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in tqdm(range(cfg.max_epochs), desc="epochs"):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            scaler=scaler,
            epoch=epoch,
            max_epochs=cfg.max_epochs,
        )

        val_pred = predict(model, val_loader, device)
        val_metrics = compute_metrics(val_pred["labels"], val_pred["preds"])
        score = val_metrics["acc"]

        if cfg.cartography:
            cart = collect_cartography_epoch(model, cartography_loader, device)
            probs_epochs.append(cart["p"])
            corr_epochs.append(cart["c"])

        with open(metrics_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        **val_metrics,
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            if epoch + 1 >= cfg.min_epochs:
                patience += 1
        if epoch + 1 >= cfg.min_epochs and patience >= cfg.early_stop_patience:
            break

    best_state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.to(device)

    model_for_save = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_safetensors(model_for_save, str(run_dir / "best_model.safetensors"))

    val_pred = predict(model, val_loader, device)
    val_metrics = compute_metrics(val_pred["labels"], val_pred["preds"])

    final_pred = predict(model, final_loader, device)
    final_metrics = compute_metrics(final_pred["labels"], final_pred["preds"])

    final_out = build_final_eval_output_rows(final_raw_df, final_eval_df, final_pred)
    final_out.to_csv(run_dir / "final_eval_predictions.csv", index=False)

    if cfg.cartography and probs_epochs:
        save_cartography_outputs(
            run_dir=run_dir,
            train_df=train_df,
            probs_epochs=probs_epochs,
            corr_epochs=corr_epochs,
            lowmean_q=cfg.cartography_lowmean_q,
        )

    with open(run_dir / "label_map.json", "w") as f:
        json.dump({"Decline": 0, "Pick": 1}, f, ensure_ascii=False, indent=2)

    baseline_always_pick = float(np.mean(np.array(final_pred["labels"]) == 1)) if final_pred["labels"] else 0.0
    baseline_always_decline = float(np.mean(np.array(final_pred["labels"]) == 0)) if final_pred["labels"] else 0.0

    with open(run_dir / "summary.json", "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "final_eval_rows": int(len(final_eval_df)),
                "val_metrics": val_metrics,
                "final_eval_metrics": final_metrics,
                "final_eval_baseline": {
                    "always_pick_acc": baseline_always_pick,
                    "always_decline_acc": baseline_always_decline,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        [
            {
                "dataset": str(cfg.final_eval_csv),
                "rows_evaluated": int(len(final_pred["labels"])),
                "acc": final_metrics["acc"],
                "f1": final_metrics["f1"],
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "always_pick_acc": baseline_always_pick,
                "always_decline_acc": baseline_always_decline,
            }
        ]
    ).to_csv(run_dir / "final_eval_summary.csv", index=False)

    print("Saved:")
    print(f"  {run_dir / 'best_model.pt'}")
    print(f"  {run_dir / 'best_model.safetensors'}")
    print(f"  {run_dir / 'summary.json'}")
    print(f"  {run_dir / 'final_eval_predictions.csv'}")
    print(f"  {run_dir / 'final_eval_summary.csv'}")


def main() -> None:
    cfg = Config()
    args = parse_args(cfg)

    cfg.model_dir = Path(args.model_dir)
    cfg.train_csv = Path(args.train_csv)
    cfg.final_eval_csv = Path(args.final_eval_csv)
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
    cfg.focal_gamma_task1 = args.focal_gamma_task1
    cfg.label_smoothing_task1 = args.label_smoothing_task1
    if args.use_amp:
        cfg.use_amp = True
    if args.no_use_amp:
        cfg.use_amp = False
    cfg.cartography = args.cartography
    cfg.cartography_lowmean_q = args.cartography_lowmean_q

    eval_run_dir = Path(args.eval_run_dir) if args.eval_run_dir else None
    eval_model_path = Path(args.eval_model_path) if args.eval_model_path else None
    if args.eval_only:
        run_eval_only(cfg, eval_run_dir=eval_run_dir, eval_model_path=eval_model_path)
        return

    run(cfg)


if __name__ == "__main__":
    main()
