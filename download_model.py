# -*- coding: utf-8 -*-
"""从 hf-mirror.com 下载模型仓库到本地目录，绕开 huggingface_hub 的响应头校验。

用法: python download_model.py [repo_id] [目标目录]
默认: laion/clap-htsat-fused -> ./models/clap-htsat-fused
"""

import json
import os
import sys
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "laion/clap-htsat-fused"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "models" / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    r = requests.get(f"{MIRROR}/api/models/{repo}/tree/main", timeout=30)
    r.raise_for_status()
    files = [f["path"] for f in r.json()
             if f["type"] == "file"
             and not f["path"].endswith((".md", ".gitattributes"))
             and f["path"] != ".gitignore"]
    print(f"[repo] {repo}: {len(files)} files -> {dest}")

    session = requests.Session()
    for path in files:
        out = dest / path
        if out.exists():
            print(f"[skip] {path} (already exists)")
            continue
        url = f"{MIRROR}/{repo}/resolve/main/{path}"
        with session.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as fp:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fp.write(chunk)
                    done += len(chunk)
                    if total > 10_000_000:
                        print(f"\r[dl  ] {path} {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
            tmp.rename(out)
            print(f"\r[dl  ] {path} done ({total / 1e6:.1f} MB)      ")
    print("[done]")


if __name__ == "__main__":
    sys.exit(main())
