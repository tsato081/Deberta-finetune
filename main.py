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
from torch.cuda.amp import GradScaler, autocast
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

    title_empty_token: str = "[TITLE_EMPTY]"
    use_amp: bool = True


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
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--sweep", action="store_true")
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

        pick = row.get("pick", "")
        label_t1 = -100
        if pick == "Pick":
            label_t1 = 1
        elif pick == "Decline":
            label_t1 = 0

        cat = row.get("category", "")
        label_t2 = -100
        if isinstance(cat, str) and cat in self.label2id:
            label_t2 = self.label2id[cat]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels_t1": torch.tensor(label_t1, dtype=torch.long),
            "labels_t2": torch.tensor(label_t2, dtype=torch.long),
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


def build_sampler(df: pd.DataFrame, boost_mult: float) -> WeightedRandomSampler:
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

    def w_pick(p: str) -> float:
        if not p:
            return 1.0
        return 1.0 / max(pick_cnt.get(p, 1), 1)

    def w_cat(c: str) -> float:
        if not c:
            return 1.0
        raw = (median_cat / max(cat_cnt.get(c, 1), 1)) ** 0.5
        return float(np.clip(raw, 1.0, 5.0))

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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Config,
    scaler: GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t1 = batch["labels_t1"].to(device)
        labels_t2 = batch["labels_t2"].to(device)

        with autocast(enabled=cfg.use_amp and device.type == "cuda"):
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
                    label_smoothing=cfg.label_smoothing_task2,
                )

                if cfg.rdrop_alpha_task2 > 0:
                    logits_t1_b, logits_t2_b = model(input_ids, attention_mask)
                    loss_t2_b = F.cross_entropy(
                        logits_t2_b[mask_t2],
                        labels_t2[mask_t2],
                        label_smoothing=cfg.label_smoothing_task2,
                    )
                    kl = rdrop_kl(logits_t2[mask_t2], logits_t2_b[mask_t2])
                    loss_t2 = 0.5 * (loss_t2 + loss_t2_b) + cfg.rdrop_alpha_task2 * kl

            loss = cfg.lambda_task1 * loss_t1 + cfg.lambda_task2 * loss_t2

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
    t2_labels: List[int] = []
    t2_preds: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t1 = batch["labels_t1"].to(device)
        labels_t2 = batch["labels_t2"].to(device)

        logits_t1, logits_t2 = model(input_ids, attention_mask)
        pred_t1 = logits_t1.argmax(dim=-1)
        pred_t2 = logits_t2.argmax(dim=-1)

        mask_t1 = labels_t1 != -100
        if mask_t1.any():
            t1_labels.extend(labels_t1[mask_t1].cpu().tolist())
            t1_preds.extend(pred_t1[mask_t1].cpu().tolist())

        mask_t2 = labels_t2 != -100
        if mask_t2.any():
            t2_labels.extend(labels_t2[mask_t2].cpu().tolist())
            t2_preds.extend(pred_t2[mask_t2].cpu().tolist())

    t1 = compute_metrics(t1_labels, t1_preds)
    t2 = {
        "acc": float(accuracy_score(t2_labels, t2_preds)) if t2_labels else 0.0,
        "f1": float(f1_score(t2_labels, t2_preds, average="macro")) if t2_labels else 0.0,
    }
    return {"task1_acc": t1["acc"], "task1_f1": t1["f1"], "task2_acc": t2["acc"], "task2_f1": t2["f1"]}


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

    sampler = build_sampler(train_df, cfg.boost_mult)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = DebertaMultiTask(str(cfg.model_dir), num_task2=len(label2id))
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(len(train_loader) * cfg.max_epochs, 1)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler() if cfg.use_amp else None

    best_score = -1.0
    best_metrics = {}
    patience = 0

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg, scaler)
        metrics = evaluate(model, val_loader, device)
        score = metrics["task1_acc"] + metrics["task2_acc"]

        with open(metrics_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        **metrics,
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if score > best_score:
            best_score = score
            best_metrics = {**metrics, "score": score, "epoch": epoch + 1}
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

    return best_metrics


def run_single(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, f, ensure_ascii=False, indent=2)
    train_and_eval(cfg, run_dir)


def run_sweep(cfg: Config) -> None:
    sweep_dir = cfg.output_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_path = sweep_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write(
            "stage,trial,learning_rate,boost_mult,focal_gamma_task1,rdrop_alpha_task2,task1_acc,task2_acc,score,epoch\n"
        )

    global_best = -1.0
    global_best_cfg = None
    global_best_path = sweep_dir / "best_model.pt"

    stage_a_lrs = [1.5e-5, 2.0e-5, 3.0e-5]
    stage_a_boost = [0.8, 1.0, 1.2]
    stage_a_trials = []
    for lr in stage_a_lrs:
        for boost in stage_a_boost:
            stage_a_trials.append((lr, boost))

    best_a = None
    trial_id = 0
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
                f"A,{trial_id},{lr},{boost},0.0,0.2,{metrics.get('task1_acc',0.0)},{metrics.get('task2_acc',0.0)},{score},{metrics.get('epoch',0)}\n"
            )

    stage_b_gamma = [0.0, 1.0, 2.0]
    stage_b_rdrop = [0.1, 0.2, 0.3]

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
                    f"B,{trial_id},{lr},{boost},{gamma},{rdrop},{metrics.get('task1_acc',0.0)},{metrics.get('task2_acc',0.0)},{score},{metrics.get('epoch',0)}\n"
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
    cfg.use_amp = args.use_amp or cfg.use_amp

    if args.sweep:
        run_sweep(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
