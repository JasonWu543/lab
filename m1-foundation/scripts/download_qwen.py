"""
下载 Qwen2.5-0.5B 权重到 data/qwen2.5-0.5b/

用法：
    python3 scripts/download_qwen.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_qwen():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    save_dir = Path(__file__).parent.parent / 'data' / 'qwen2.5-0.5b'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"下载 Qwen2.5-0.5B 到 {save_dir} ...")
    snapshot_download(
        repo_id='Qwen/Qwen2.5-0.5B',
        local_dir=str(save_dir),
        ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
    )
    print("下载完成！")
    print(f"权重目录：{save_dir}")


if __name__ == '__main__':
    download_qwen()
