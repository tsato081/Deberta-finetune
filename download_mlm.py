from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import os

load_dotenv(".env")
token = os.environ["HF_TOKEN"]
snapshot_download(
    repo_id="teru00801/deberta-v3-mlm",
    token=token,
    local_dir="models/deberta_v3_mlm",
    local_dir_use_symlinks=False,
)
print("done: models/deberta_v3_mlm")

