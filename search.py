# -*- coding: utf-8 -*-
"""音频检索：输入标签/描述，按相似度列出索引中的音频，生成一键播放页面。

索引里的音频嵌入是预计算好的，检索只算一次文本向量 + 一次矩阵乘法，
几千个文件也是毫秒级。查询文本向量按查询词缓存，重复查询零开销。

用法:
    python search.py "散弹枪 短促" [--top 30] [--open]
    python search.py "shotgun blast, dry" --top 50

中文词会经 vocab.json 映射为英文提示词；直接写英文也可以，英文效果最佳。
--open 自动用默认浏览器打开结果页。
"""

import argparse
import hashlib
import json
import os
import sys
import webbrowser
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

torch.set_num_threads(4)

from transformers import ClapModel, ClapProcessor


_MT_CACHE = None


def mt_translate(text: str, mt_dir: Path) -> str:
    """本地中→英翻译（Helsinki-NLP/opus-mt-zh-en），进程内只加载一次。"""
    global _MT_CACHE
    if _MT_CACHE is None:
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(str(mt_dir))
        model = MarianMTModel.from_pretrained(str(mt_dir)).eval()
        _MT_CACHE = (tok, model)
    tok, model = _MT_CACHE
    with torch.no_grad():
        t = tok([text], return_tensors="pt", padding=True)
        out = model.generate(**t, max_new_tokens=64)
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def translate_query(q: str, vocab_path: Path, mt_dir: Path) -> list:
    """查询词处理：vocab.json 命中的词直接用映射（值可为字符串或同义变体数组）；
    命不中的中文词拼接后走本地 MT 翻译成英文描述；英文词原样使用。
    多词之间笛卡尔积组合成提示词列表，每条音频取所有提示词的最高分。
    """
    vocab = json.loads(vocab_path.read_text(encoding="utf-8-sig"))
    word_variants, cjk_left = [], []
    for w in q.replace("，", " ").replace(",", " ").split():
        if w in vocab:
            v = vocab[w]
            word_variants.append(v if isinstance(v, list) else [v])
        elif any("\u4e00" <= ch <= "\u9fff" for ch in w):
            cjk_left.append(w)
        else:
            word_variants.append([w])
    if cjk_left:
        en = mt_translate(" ".join(cjk_left), mt_dir)
        print(f"[mt  ] {' '.join(cjk_left)} -> \"{en}\"")
        # CLAP 对标签式短语（如 "metal, impact"）比整句更敏感，
        # 保留原句与标签式两个变体，取每条音频的最高分
        word_variants.append([en, ", ".join(w for w in en.split() if len(w) > 2) or en])
    if not word_variants:
        raise SystemExit("查询词为空或无法解析")
    # 笛卡尔积组合（限制上限防爆炸）
    prompts = [""]
    for variants in word_variants:
        prompts = [f"{a}, {b}".strip(", ") for a in prompts for b in variants]
    if len(prompts) > 24:
        prompts = prompts[:24]
        print("[warn] 组合超过 24 条提示词，已截断")
    return prompts


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="CLAP 音频检索")
    ap.add_argument("query", help="查询词（中文或英文）")
    ap.add_argument("--index-dir", default=str(here / "index"))
    ap.add_argument("--model", default=str(here / "models" / "clap-htsat-fused"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=None, help="输出 html 路径")
    ap.add_argument("--open", action="store_true", help="生成后自动打开浏览器")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8-sig"))
    embs = np.load(index_dir / "embeddings.npy")
    paths = list(meta["files"].keys())  # 与 embeddings 行序一致
    if len(paths) != len(embs):
        raise SystemExit("索引不一致，请重新运行 build_index.py")
    if not paths:
        raise SystemExit("索引为空，请先运行 build_index.py")

    prompts = translate_query(args.query, here / "vocab.json", here / "models" / "opus-mt-zh-en")
    print(f"[query] {args.query}")
    for pr in prompts:
        print(f"        prompt: \"{pr}\"")

    # 查询向量缓存：每个提示词独立缓存，同一提示词第二次起零开销
    cache_dir = index_dir / "query_cache"
    cache_dir.mkdir(exist_ok=True)
    model, proc = None, None
    qembs = []
    for pr in prompts:
        cache_file = cache_dir / f"{hashlib.md5(pr.encode()).hexdigest()}.npy"
        if cache_file.exists():
            qembs.append(np.load(cache_file))
            continue
        if model is None:
            model = ClapModel.from_pretrained(args.model).eval()
            proc = ClapProcessor.from_pretrained(args.model)
        with torch.no_grad():
            t = proc(text=[pr], padding=True, return_tensors="pt")
            q = model.get_text_features(**t)[0].cpu().numpy()
            q = q / np.linalg.norm(q)
        np.save(cache_file, q.astype(np.float32))
        qembs.append(q.astype(np.float32))

    # 每条音频取所有提示词的最高分
    scores = np.max(np.stack([embs @ q for q in qembs]), axis=0)
    order = np.argsort(-scores)[: args.top]

    print(f"\n===== Top {len(order)} / {len(paths)} =====")
    print(f"{'score':>7}  file")
    rows = []
    for r, i in enumerate(order, 1):
        info = meta["files"][paths[i]]
        rows.append({"rank": r, "score": round(float(scores[i]), 4),
                     "path": paths[i], "name": info["name"],
                     "duration": info["duration"]})
        print(f"{scores[i]:7.4f}  {info['name']}")

    out = Path(args.out) if args.out else here / "search_result.html"
    audio_rows = "\n".join(
        f'<tr><td>{r["rank"]}</td><td>{r["score"]:.4f}</td>'
        f'<td>{r["name"]}</td><td>{r["duration"]}s</td>'
        f'<td><audio controls preload="none" src="file:///{r["path"].replace(chr(92), "/")}"></audio></td></tr>'
        for r in rows)
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>检索: {args.query}</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#1e1e28;color:#ddd;margin:24px}}
h1{{font-size:20px;color:#7ec8ff}} table{{border-collapse:collapse;width:100%}}
th,td{{padding:6px 12px;border-bottom:1px solid #333;text-align:left;vertical-align:middle}}
tr:hover{{background:#2a2a38}} audio{{width:320px;height:32px}}
.score{{color:#8bd450;font-weight:600}}
</style></head><body>
<h1>检索: {args.query}</h1>
<p>提示词: "{" &nbsp;|&nbsp; ".join(prompts)}" &nbsp;|&nbsp; 共 {len(paths)} 条索引，显示 Top {len(rows)}</p>
<table><tr><th>#</th><th>相似度</th><th>文件名</th><th>时长</th><th>播放</th></tr>
{audio_rows}
</table></body></html>"""
    out.write_text(html, encoding="utf-8")
    print(f"\n[out] {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    sys.exit(main())
