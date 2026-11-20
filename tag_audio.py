# -*- coding: utf-8 -*-
"""
CLAP 音频打标工具（路线二：音频-文本对比模型）

给一批音频文件，按 labels.json 中自定义的标签逐个算相似度分数，
输出控制台排序表 + results.csv。

用法:
    python tag_audio.py <音频文件夹> [--labels labels.json] [--out results.csv]
                        [--model laion/clap-htsat-fused] [--top 10]

标签配置 (labels.json):
    positive : 期望听感的描述词，多条取平均，构成正向分
    negative : 排除项（混响尾、歧义声等），取最差一条扣分
    extra    : 仅展示分数、不参与排序的参考标签（如 grenade explosion）

聚合分 = mean(positive) - max(negative)，按聚合分降序输出。
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# 默认走 hf-mirror.com；本机可直连 huggingface.co 时,运行前设置 HF_ENDPOINT 覆盖即可
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import librosa
import torch

# 24 线程在部分 Windows 机器上会导致 CPU 推理异常缓慢，固定线程数
torch.set_num_threads(4)

# 排障用：若进程卡住，每 120s 把调用栈写入 stack.log
import faulthandler
faulthandler.dump_traceback_later(120, repeat=True,
                                  file=open(Path(__file__).resolve().parent / "stack.log", "w"))

from transformers import ClapModel, ClapProcessor

SR = 48000
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff"}


def load_labels(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("positive", "negative", "extra"):
        if not isinstance(data.get(key, []), list):
            raise ValueError(f"labels.json 的 {key} 必须是字符串数组")
    return data


def collect_audio(folder: Path):
    files = [f for f in sorted(folder.iterdir())
             if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise SystemExit(f"{folder} 下没有找到音频文件（支持: {', '.join(sorted(AUDIO_EXTS))}）")
    return files


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="CLAP 音频打标")
    ap.add_argument("folder", help="音频文件夹")
    ap.add_argument("--labels", default=str(here / "labels.json"), help="标签配置 json")
    ap.add_argument("--out", default=str(here / "results.csv"), help="输出 csv")
    ap.add_argument("--model", default="laion/clap-htsat-fused")
    ap.add_argument("--top", type=int, default=10, help="控制台只显示前 N 条")
    args = ap.parse_args()

    labels = load_labels(Path(args.labels))
    all_labels = labels.get("positive", []) + labels.get("negative", []) + labels.get("extra", [])
    if not all_labels:
        raise SystemExit("labels.json 里没有任何标签")

    folder = Path(args.folder)
    files = collect_audio(folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] model={args.model} device={device}")
    t0 = time.time()
    model = ClapModel.from_pretrained(args.model).to(device).eval()
    processor = ClapProcessor.from_pretrained(args.model)
    print(f"[load] done in {time.time() - t0:.1f}s")

    # 文本向量只算一次
    with torch.no_grad():
        t_in = processor(text=all_labels, padding=True, return_tensors="pt").to(device)
        text_emb = model.get_text_features(**t_in)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    n_pos = len(labels.get("positive", []))
    n_neg = len(labels.get("negative", []))

    rows = []
    for i, f in enumerate(files, 1):
        try:
            wav, _ = librosa.load(str(f), sr=SR, mono=True)
        except Exception as e:
            print(f"[skip] {f.name}: 解码失败 {e!r}")
            continue
        dur = len(wav) / SR
        with torch.no_grad():
            a_in = processor(audios=wav, return_tensors="pt", sampling_rate=SR).to(device)
            aud_emb = model.get_audio_features(**a_in)
            aud_emb = aud_emb / aud_emb.norm(dim=-1, keepdim=True)
        sims = (aud_emb @ text_emb.T).squeeze(0).cpu().tolist()

        pos_scores = sims[:n_pos]
        neg_scores = sims[n_pos:n_pos + n_neg]
        agg = (sum(pos_scores) / len(pos_scores) if pos_scores else 0.0) \
            - (max(neg_scores) if neg_scores else 0.0)
        best = all_labels[sims.index(max(sims))]
        rows.append({"file": f.name, "duration": round(dur, 2),
                     "aggregate": round(agg, 4), "best_label": best,
                     "scores": dict(zip(all_labels, [round(s, 4) for s in sims]))})
        print(f"[{i}/{len(files)}] {f.name}  {dur:.1f}s  agg={agg:.3f}  best: {best}")

    rows.sort(key=lambda r: -r["aggregate"])

    # 控制台表格
    col_w = max(len(l) for l in all_labels) + 2
    print("\n===== 排序结果（按聚合分降序）=====")
    print(f"{'file':<40}{'dur':>6}{'agg':>8}  best_label")
    for r in rows[:args.top]:
        print(f"{r['file'][:38]:<40}{r['duration']:>5.1f}s{r['aggregate']:>8.3f}  {r['best_label']}")

    # CSV
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "duration_s", "aggregate"] + [f"score:{l}" for l in all_labels])
        for r in rows:
            w.writerow([r["file"], r["duration"], r["aggregate"]] +
                       [r["scores"][l] for l in all_labels])
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    sys.exit(main())
