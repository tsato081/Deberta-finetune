以下の2タスクの分類器を作る。
入力はニュース記事で、タイトル・本文がそれぞれある。"title", "body".
ベースはDeberta V3 (ku-nlp).
一応ニュース記事で継続事前学習を少ししたものがある（teru00801/deberta-v3-mlm on huggingface, private）

## Task 1
    Task2で分類するカテゴリに該当するかどうかを判定する2値分類。（どのカテゴリに対応するかは問題外で、Task2分類対象かそうでないかの二値分類。）
    Task2対象：Pick, そうでない：Declineとして呼ぶ

## Task 2
    96？もうちょい少ない？カテゴリの分類。
    task2_cleaned.csvは、カテゴリを統合する前のもの（昔のやつ）。96カテゴリある。
    分類器を作りたいのはrefactoredの方。

## Objective
BenchMark to beat:
90%+ on both (preferably 92%+)

Multitask BERT or We can train Deberta for each task.

## Problems
Task1 なんかデータ少ない
いっぱいテキストデータあるけどどこまでが正しいやつでどこまでが正しくないとかわかんない

どうしよ
Task1 90% acc, Task2 85%のモデルは持ってこようと思えばできる
/Users/terusato/BaseConnect/hawks/var/files/reducer/models
ここにあったわ

---

## Cloud (Ubuntu + A100/H100) 手順

前提:
- GPU は CUDA が使える状態（A100/H100, 80GB想定）
- `HF_TOKEN` が必要（private repo を落とす場合）

### 1) セットアップ（tmux / uv / repo）
```bash
sudo apt-get update
sudo apt-get install -y git tmux curl ca-certificates

# uv install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# repo
git clone <YOUR_REPO_URL>
cd Deberta-finetune

# python + deps
uv python install 3.11
uv sync

# GPU (A100/H100) 用 PyTorch をインストール
# ※PyPIのtorchはCPU版になるため、必ず公式CUDA wheelを使う

# 1) CUDAバージョン確認
nvidia-smi | head -n 5

# 2) 出たCUDAに合わせてインストール
# nvidia-smi の CUDA 13.0 は「ドライバの最大対応」なので、
# torch は cu128（CUDA 12.8 同梱）で問題なし
uv pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade torch torchvision torchaudio

# CUDA 12.x → cu121
uv pip install --index-url https://download.pytorch.org/whl/cu121 --upgrade torch torchvision torchaudio

# CUDA 11.8 → cu118
# uv pip install --index-url https://download.pytorch.org/whl/cu118 --upgrade torch torchvision torchaudio

# 3) 動作確認（GPUを認識しているか）
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("is_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

### 2) HF token 設定
```bash
export HF_TOKEN="<YOUR_HF_TOKEN>"
```

### 3) 継続事前学習済みモデルをローカルへ配置
`main.py` はローカルパス `models/deberta_v3_mlm` を読む想定。

#### A) すでに手元でダウンロード済みならアップロード
```bash
mkdir -p models/deberta_v3_mlm
# scp/rsync などで models/deberta_v3_mlm/ に config.json / tokenizer.* / *.safetensors を配置
```

#### B) HuggingFace から落とす（スクリプト利用）
```bash
uv run python download_mlm.py
```

確認:
```bash
ls -la models/deberta_v3_mlm | head
```

### 4) 学習データ（ready）を用意
学習は URL 置換済みの “ready” を使う。

#### A) HuggingFace Dataset からダウンロード
```bash
uv run python download_data.py
```

#### B) ローカル生成は省略（クラウドは `download_data.py` のみでOK）

### 5) 学習（単発 / sweep）
tmux 推奨:
```bash
tmux new -s deberta
```

単発:
```bash
uv run python main.py \
  --model-dir models/deberta_v3_mlm \
  --task1-csv Data_for_deberta/processed/task1_ready/train.csv \
  --task2-csv Data_for_deberta/processed/task2_ready/train.csv \
  --output-dir outputs/train_runs
```

sweep（Stage A→B）:
```bash
uv run python main.py \
  --model-dir models/deberta_v3_mlm \
  --task1-csv Data_for_deberta/processed/task1_ready/train.csv \
  --task2-csv Data_for_deberta/processed/task2_ready/train.csv \
  --output-dir outputs/train_runs \
  --sweep
```
