"""下载 TinyStories 数据集（V2, GPT-4 生成版）到 data/tinystories/。

用法：
    python scripts/download_tinystories.py            # 只下 valid (~20MB)，开发迭代用
    python scripts/download_tinystories.py --train    # 加下 train (~2GB)，正式训练用
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEST = Path(__file__).resolve().parent.parent / "data" / "tinystories"
REPO = "roneneldan/TinyStories"


def fetch(filename: str):
    print(f"downloading {filename} ...")
    hf_hub_download(repo_id=REPO, filename=filename, repo_type="dataset",
                    local_dir=DEST)
    print(f"  -> {DEST / filename}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="同时下载 2GB 的 train split")
    args = ap.parse_args()

    fetch("TinyStoriesV2-GPT4-valid.txt")
    if args.train:
        fetch("TinyStoriesV2-GPT4-train.txt")
