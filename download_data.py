from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import shutil

load_dotenv(".env")
token = os.environ["HF_TOKEN"]
repo_id = "teru00801/New-Hawks-Data"

files = {
    "task1/train.csv": "Data_for_deberta/processed/task1_ready/train.csv",
    "task2/train.csv": "Data_for_deberta/processed/task2_ready/train.csv",
    "tests/Hawks4.0_refactored.csv": "Data_for_deberta/tests/Hawks4.0_refactored.csv",
    "tests/Hawks_ver5.0_refactored.csv": "Data_for_deberta/tests/Hawks_ver5.0_refactored.csv",
    "tests/Hawks_ver5.1_refactored.csv": "Data_for_deberta/tests/Hawks_ver5.1_refactored.csv",
    "tests/Hawks_ver6.0 csv出力用.csv": "Data_for_deberta/tests/Hawks_ver6.0 csv出力用.csv",
}

for src, dst in files.items():
    downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=src, token=token)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, dst_path)

print("done: dataset files")
