from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def _safe_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _has_any_files(path: Path, patterns: list[str]) -> bool:
    for pattern in patterns:
        if any(path.glob(pattern)):
            return True
    return False


def _verify_snapshot(local_dir: Path) -> tuple[bool, str]:
    required_any = ["config.json", "tokenizer.json", "tokenizer.model"]
    weight_any = ["*.safetensors", "*.bin"]

    has_required = _has_any_files(local_dir, required_any)
    has_weights = _has_any_files(local_dir, weight_any)

    if has_required and has_weights:
        return True, "OK"

    files = sorted(p.relative_to(local_dir) for p in local_dir.rglob("*") if p.is_file())
    preview = "\n".join(f"- {p}" for p in files[:50])
    return (
        False,
        "Snapshot downloaded but expected model files are missing.\n"
        "Expected at least one of: config.json / tokenizer.json / tokenizer.model and at least one weight file: *.safetensors / *.bin.\n"
        f"Found {len(files)} files under local_dir:\n{preview}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a HuggingFace Hub model snapshot to a local directory.")
    parser.add_argument(
        "--repo-id",
        default="teru00801/deberta-v3-mlm",
        help="Hugging Face model repo id (e.g. 'org/name').",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Where to materialize the snapshot (default: models/hf/<repo-id with '/' replaced>).",
    )
    parser.add_argument(
        "--cache-dir",
        default="models/hf_cache",
        help="Hugging Face cache dir (default: models/hf_cache).",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Env var name that stores the Hugging Face token (default: HF_TOKEN).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download.",
    )
    args = parser.parse_args()

    load_dotenv(".env")
    token = os.getenv(args.token_env)
    if not token:
        raise SystemExit(f"Missing token. Set {args.token_env} in .env or environment variables.")

    repo_id = args.repo_id
    local_dir = Path(args.local_dir) if args.local_dir else Path("models") / "hf" / _safe_dirname(repo_id)
    cache_dir = Path(args.cache_dir)

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=args.revision,
        token=token,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        resume_download=not args.force,
        force_download=args.force,
    )

    print("Downloaded snapshot:")
    print(f"  repo_id={repo_id}")
    if args.revision:
        print(f"  revision={args.revision}")
    print(f"  local_dir={local_dir}")
    print(f"  snapshot_path={snapshot_path}")

    ok, message = _verify_snapshot(local_dir)
    if ok:
        print("Verify: OK (model files detected)")
    else:
        print("Verify: WARNING")
        print(message)


if __name__ == "__main__":
    main()
