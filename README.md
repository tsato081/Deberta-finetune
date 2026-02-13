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

上記で `Risk-only-pick` も取得される（`main_task1_binary.py` 用）:
```bash
data_for_deberta/Risk-only-pick/task1_risk_training_0209.csv
data_for_deberta/Risk-only-pick/Hawks_ver6.0 csv出力用_pickflip.csv
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

Stage Aのみ:
```bash
uv run python main.py --sweep --sweep-stage A
```

Stage Bのみ（最新sweepのAベストを自動利用）:
```bash
uv run python main.py --sweep --sweep-stage B
```
※ Stage Aを途中で止めても、`summary.csv`内のAベストを自動で拾ってBに入る

Stage Bのみ（クラウド再作成後: Aベスト手動指定 + Cartography）:
```bash
uv run python main.py \
  --model-dir models/deberta_v3_mlm \
  --task1-csv Data_for_deberta/processed/task1_ready/train.csv \
  --task2-csv Data_for_deberta/processed/task2_ready/train.csv \
  --output-dir outputs/train_runs \
  --sweep \
  --sweep-stage B \
  --stage-b-lr 1.5e-5 \
  --stage-b-boost 1.0 \
  --cartography \
  --cartography-lowmean-q 0.005 \
  --cartography-task2-per-category-cap 30
```

Task1のみ（二値分類, Risk-only-pick）:
```bash
uv run python main_task1_binary.py
```

Task1の評価だけ（最新runの `best_model.pt` を自動使用）:
```bash
uv run python main_task1_binary.py \
  --eval-only \
  --final-eval-csv "data_for_deberta/Risk-only-pick/Hawks_ver6.0 csv出力用_pickflip.csv"
```

Task1の評価だけ（runを明示）:
```bash
uv run python main_task1_binary.py \
  --eval-only \
  --eval-run-dir outputs/task1_binary_runs/20260209_103015 \
  --final-eval-csv "data_for_deberta/Risk-only-pick/Hawks_ver6.0 csv出力用_pickflip.csv"
```

Task2のみ（カテゴリ分類専用 / HF dataset: `teru00801/data-plus-high-prob`）:
```bash
uv run python main_task2_category.py \
  --model-id teru00801/deberta-v3-mlm \
  --hf-dataset-repo teru00801/data-plus-high-prob \
  --hf-filename all_categories_merged.csv \
  --output-dir outputs/task2_category_runs
```

Task2の評価だけ（最新runの best_model を自動使用）:
```bash
uv run python main_task2_category.py --eval-only
```

Task2の評価だけ（runを明示）:
```bash
uv run python main_task2_category.py \
  --eval-only \
  --eval-run-dir outputs/task2_category_runs/<timestamp>
```

Task2スクリプトの構文チェック:
```bash
uv run python -m py_compile main_task2_category.py
```

出力:
```bash
outputs/task1_binary_runs/<timestamp>/final_eval_predictions.csv
outputs/task1_binary_runs/<timestamp>/final_eval_summary.csv
outputs/task1_binary_runs/<run_timestamp>/final_eval_predictions_evalonly.csv
outputs/task1_binary_runs/<run_timestamp>/final_eval_summary_evalonly.csv
outputs/task2_category_runs/<timestamp>/summary.json
outputs/task2_category_runs/<timestamp>/test_eval_summary.json
outputs/task2_category_runs/<timestamp>/test_eval_summary.csv
outputs/task2_category_runs/<timestamp>/test_predictions_<test_id>.csv
outputs/task2_category_runs/<timestamp>/confusion_matrix_<test_id>.csv
outputs/task2_category_runs/<run_timestamp>/test_eval_summary_evalonly.json
outputs/task2_category_runs/<run_timestamp>/test_eval_summary_evalonly.csv
```

### 6) エクスポート後ファイルの意味（`upload_best_safetensors.py`）

エクスポート先:
```bash
outputs/hf_export_deberta-20260206
```

主なファイル:
```bash
outputs/hf_export_deberta-20260206/best_model.safetensors
```
上記はファインチューニング済みの重み（Task1/Task2ヘッドを含む）。

```bash
outputs/hf_export_deberta-20260206/base_model/
```
上記はベースモデル一式（トークナイザ・HFモデル設定など）。

```bash
outputs/hf_export_deberta-20260206/base_model/config.json
```
上記はHFモデル定義用のconfig（モデル構造の設定）。

```bash
outputs/hf_export_deberta-20260206/config.json
```
上記は学習時の実験設定（`max_length`, `title_empty_token`, `learning_rate` など）。

```bash
outputs/hf_export_deberta-20260206/label_map.json
```
上記はTask2カテゴリのIDマップ（推論時に必須）。

```bash
outputs/hf_export_deberta-20260206/selected_b_trial.json
```
上記はどのB trialを選んだかの監査情報。

```bash
outputs/hf_export_deberta-20260206/summary.csv
```
上記はsweep結果一覧（存在する場合のみ同梱）。

推論で最低限使うセット:
```bash
outputs/hf_export_deberta-20260206/best_model.safetensors
outputs/hf_export_deberta-20260206/label_map.json
outputs/hf_export_deberta-20260206/config.json
outputs/hf_export_deberta-20260206/base_model/
```
