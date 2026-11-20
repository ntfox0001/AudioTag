# -*- coding: utf-8 -*-
"""音频向量索引构建器（增量式）

扫描文件夹，为每条音频计算 CLAP 嵌入（512 维），存入 index/ 目录。
已索引且未修改的文件自动跳过，只计算新增/变更文件。

用法:
    python build_index.py <音频文件夹> [--index-dir d:/audio_tag/index]
可多次调用不同文件夹，索引会合并累积。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import librosa
import numpy as np
import torch

torch.set_num_threads(4)

from transformers import ClapModel, ClapProcessor

SR = 48000
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff"}
EMB_DIM = 512


def fkey(path: Path) -> str:
    """文件唯一键：绝对路径的小写 + 大小 + mtime，任一变化即视为需重算。"""
    st = path.stat()
    return f"{str(path).lower()}|{st.st_size}|{int(st.st_mtime)}"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="构建 CLAP 音频向量索引（增量）")
    ap.add_argument("folder", help="音频文件夹")
    ap.add_argument("--index-dir", default=str(here / "index"))
    ap.add_argument("--model", default=str(here / "models" / "clap-htsat-fused"))
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / "meta.json"
    emb_path = index_dir / "embeddings.npy"

    # 读取旧索引
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        embs = list(np.load(emb_path))
    else:
        meta, embs = {"files": {}}, []
    old_by_key = {v["key"]: i for i, v in meta["files"].items()}

    folder = Path(args.folder)
    files = [f for f in sorted(folder.rglob("*"))
             if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise SystemExit(f"{folder} 下没有音频文件")

    todo = [f for f in files if fkey(f) not in old_by_key]
    print(f"[scan] {len(files)} files, {len(todo)} need embedding "
          f"(index has {len(meta['files'])})")

    if todo:
        model = ClapModel.from_pretrained(args.model).eval()
        proc = ClapProcessor.from_pretrained(args.model)
        t0 = time.time()
        for i, f in enumerate(todo, 1):
            try:
                wav, _ = librosa.load(str(f), sr=SR, mono=True)
                if len(wav) < 480:  # 过短补零，避免处理器报错
                    wav = np.pad(wav, (0, 480 - len(wav)))
                with torch.no_grad():
                    a = proc(audios=wav, return_tensors="pt", sampling_rate=SR)
                    emb = model.get_audio_features(**a)[0].cpu().numpy()
                    emb = emb / np.linalg.norm(emb)
                embs.append(emb.astype(np.float32))
                meta["files"][str(f).lower()] = {
                    "key": fkey(f), "name": f.name,
                    "duration": round(len(wav) / SR, 2)}
                print(f"\r[{i}/{len(todo)}] {f.name}", end="", flush=True)
                # 每 50 条落盘一次，中断后已算部分不丢失
                if i % 50 == 0:
                    np.save(emb_path, np.stack(embs))
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
            except Exception as e:
                print(f"\n[skip] {f}: {e!r}")
        print(f"\n[embed] {len(todo)} files in {time.time()-t0:.1f}s")

    np.save(emb_path, np.stack(embs))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[save] {len(meta['files'])} embeddings -> {index_dir}")


if __name__ == "__main__":
    sys.exit(main())
