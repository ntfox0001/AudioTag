# -*- coding: utf-8 -*-
"""音效检索 Web 服务

后台 HTTP 服务 + 前台网页（web/index.html）：
  - 搜索页：输入描述文本，返回相关音效列表（可在线播放）
  - 索引页：对指定目录构建 CLAP 向量索引，支持查询进度 / 暂停 / 停止

用法:  python server.py [--port 8000]
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

here = Path(__file__).resolve().parent
# 关键：numba 缓存必须重定向，否则在受限环境下会陷入"创建临时文件被拒"风暴
os.environ.setdefault("NUMBA_CACHE_DIR", str(here / ".numbacache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor

torch.set_num_threads(4)

from build_index import fkey, AUDIO_EXTS, SR
from search import translate_query

INDEX_DIR = here / "index"
MODEL_DIR = here / "models" / "clap-htsat-fused"
MT_DIR = here / "models" / "opus-mt-zh-en"
WEB_DIR = here / "web"

# ---------------------------------------------------------------- 全局模型
_model = None
_proc = None
_model_lock = threading.Lock()


def get_model():
    """惰性加载 CLAP 模型，搜索与构建共用同一实例。"""
    global _model, _proc
    with _model_lock:
        if _model is None:
            _model = ClapModel.from_pretrained(MODEL_DIR).eval()
            _proc = ClapProcessor.from_pretrained(MODEL_DIR)
    return _model, _proc


# ---------------------------------------------------------------- 索引 IO 锁
io_lock = threading.RLock()  # 构建(写索引)与搜索(读索引)互斥


def load_index():
    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8-sig"))
    embs = np.load(INDEX_DIR / "embeddings.npy")
    return meta, embs


# ---------------------------------------------------------------- 构建任务
class BuildJob:
    """后台索引构建任务：可查询进度，可暂停/继续/停止。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "status": "idle",   # idle/running/paused/stopped/done/error
            "phase": "",        # 扫描/加载模型/计算嵌入
            "folder": "",
            "processed": 0,
            "total": 0,
            "skipped": 0,
            "current": "",
            "message": "",
        }
        self.pause_evt = threading.Event()   # 置位=暂停
        self.stop_evt = threading.Event()
        self.thread = None

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
        # 暂停时循环阻塞在文件边界，状态保持 running，这里如实呈现
        if s["status"] == "running" and self.pause_evt.is_set():
            s["status"] = "paused"
        return s

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def start(self, folder: str) -> str:
        with self.lock:
            if self.state["status"] in ("running", "paused"):
                return "已有构建任务在进行中"
            self.pause_evt.clear()
            self.stop_evt.clear()
            self.state.update({
                "status": "running", "phase": "扫描", "folder": folder,
                "processed": 0, "total": 0, "skipped": 0,
                "current": "", "message": ""})
            self.thread = threading.Thread(target=self._run, args=[folder], daemon=True)
            self.thread.start()
        return ""

    def _run(self, folder: str):
        try:
            self._build(folder)
        except Exception as e:
            self._set(status="error", message=repr(e))

    def _wait_if_paused(self):
        while self.pause_evt.is_set() and not self.stop_evt.is_set():
            time.sleep(0.3)
        return self.stop_evt.is_set()

    def _save(self, embs, meta):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if embs:
            np.save(INDEX_DIR / "embeddings.npy", np.stack(embs))
        (INDEX_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def _build(self, folder: str):
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise RuntimeError(f"目录不存在: {folder}")

        self._set(phase="扫描")
        files = [f for f in sorted(folder_path.rglob("*"))
                 if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        if not files:
            raise RuntimeError(f"{folder} 下没有音频文件")

        with io_lock:
            meta, _embs = load_index()
            embs = list(_embs)  # 转为行列表，便于逐条 append
            old_by_key = {v["key"]: i for i, v in meta["files"].items()}
        todo = [f for f in files if fkey(f) not in old_by_key]
        self._set(total=len(todo), phase="加载模型" if todo else "",
                  message=f"共 {len(files)} 个音频，需新建 {len(todo)} 条索引")
        self._register_root(folder_path)

        if todo:
            get_model()
            self._set(phase="计算嵌入")
            t0 = time.time()
            for i, f in enumerate(todo, 1):
                if self._wait_if_paused():
                    self._set(status="stopped", phase="",
                              message=f"已停止：完成 {i-1}/{len(todo)}，进度已保存")
                    return
                self._set(current=f.name, processed=i - 1)
                try:
                    wav, _ = librosa.load(str(f), sr=SR, mono=True)
                    if len(wav) < 480:
                        wav = np.pad(wav, (0, 480 - len(wav)))
                    model, proc = get_model()
                    with torch.no_grad():
                        a = proc(audios=wav, return_tensors="pt", sampling_rate=SR)
                        emb = model.get_audio_features(**a)[0].cpu().numpy()
                        emb = emb / np.linalg.norm(emb)
                    with io_lock:
                        embs.append(emb.astype(np.float32))
                        meta["files"][str(f).lower()] = {
                            "key": fkey(f), "name": f.name,
                            "duration": round(len(wav) / SR, 2)}
                    self._set(processed=i)
                    # 每 25 条落盘一次；暂停/停止时也会保存
                    if i % 25 == 0:
                        with io_lock:
                            self._save(embs, meta)
                except Exception as e:
                    with self.lock:
                        self.state["skipped"] += 1
                    _log_error(f, e)

            with io_lock:
                self._save(embs, meta)
            self._set(status="done", phase="", current="",
                      message=f"完成：新增 {len(todo)} 条，跳过 {self.snapshot()['skipped']} 条，"
                              f"索引共 {len(meta['files'])} 条，耗时 {time.time()-t0:.0f}s")

    @staticmethod
    def _register_root(folder: Path):
        """记录已索引根目录，供 /api/audio 做访问范围校验。"""
        roots_file = INDEX_DIR / "roots.json"
        roots = set(json.loads(roots_file.read_text(encoding="utf-8"))) \
            if roots_file.exists() else set()
        roots.add(str(folder))
        roots_file.write_text(json.dumps(sorted(roots)), encoding="utf-8")


_log_file = here / "server.log"


def _log_error(f, e):
    import traceback
    with open(_log_file, "a", encoding="utf-8") as fp:
        fp.write(f"\n==== {time.strftime('%H:%M:%S')} {f}\n")
        traceback.print_exc(file=fp)


job = BuildJob()


def seed_roots():
    """首次启动时从现有索引推导已索引根目录。"""
    roots_file = INDEX_DIR / "roots.json"
    if roots_file.exists():
        return
    try:
        meta, _ = load_index()
        roots = set()
        for p in meta["files"]:
            parts = Path(p).parts
            if len(parts) >= 3:
                roots.add(str(Path(*parts[:3])))
        roots_file.write_text(json.dumps(sorted(roots)), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------- 搜索
def do_search(q: str, top: int):
    with io_lock:
        meta, embs = load_index()
    paths = list(meta["files"].keys())
    if not paths or len(paths) != len(embs):
        raise RuntimeError("索引为空或不一致，请先在“索引”页构建")

    prompts = translate_query(q, here / "vocab.json", MT_DIR)

    cache_dir = INDEX_DIR / "query_cache"
    cache_dir.mkdir(exist_ok=True)
    qembs = []
    for pr in prompts:
        cf = cache_dir / f"{hashlib.md5(pr.encode()).hexdigest()}.npy"
        if cf.exists():
            qembs.append(np.load(cf))
            continue
        model, proc = get_model()
        with torch.no_grad():
            t = proc(text=[pr], padding=True, return_tensors="pt")
            vec = model.get_text_features(**t)[0].cpu().numpy()
            vec = vec / np.linalg.norm(vec)
        vec = vec.astype(np.float32)
        np.save(cf, vec)
        qembs.append(vec)

    scores = np.max(np.stack([embs @ vec for vec in qembs]), axis=0)
    order = np.argsort(-scores)[:top]
    return {
        "query": q,
        "prompts": prompts,
        "total": len(paths),
        "results": [{
            "rank": r, "score": round(float(scores[i]), 4),
            "path": paths[i], "name": meta["files"][paths[i]]["name"],
            "duration": meta["files"][paths[i]]["duration"],
        } for r, i in enumerate(order, 1)],
    }


# ---------------------------------------------------------------- HTTP
CONTENT_TYPES = {".ogg": "audio/ogg", ".wav": "audio/wav", ".mp3": "audio/mpeg",
                 ".flac": "audio/flac", ".m4a": "audio/mp4", ".aif": "audio/aiff",
                 ".aiff": "audio/aiff"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        try:
            body = path.read_bytes()
        except OSError:
            self._json({"error": "读取失败"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif url.path == "/api/search":
            q = unquote(qs.get("q", [""])[0]).strip()
            top = min(int(qs.get("top", ["30"])[0]), 200)
            if not q:
                self._json({"error": "查询词为空"}, 400)
                return
            try:
                self._json(do_search(q, top))
            except Exception as e:
                self._json({"error": repr(e)}, 500)
        elif url.path == "/api/build/status":
            self._json(job.snapshot())
        elif url.path == "/api/audio":
            p = Path(unquote(qs.get("p", [""])[0]))
            roots_file = INDEX_DIR / "roots.json"
            roots = json.loads(roots_file.read_text(encoding="utf-8")) \
                if roots_file.exists() else []
            allowed = any(_under(p, Path(r)) for r in roots)
            if (not allowed or not p.is_file()
                    or p.suffix.lower() not in CONTENT_TYPES):
                self._json({"error": "文件不允许访问"}, 403)
                return
            self._file(p, CONTENT_TYPES[p.suffix.lower()])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (ValueError, UnicodeDecodeError):
            self._json({"error": "请求体不是合法 JSON"}, 400)
            return
        if self.path == "/api/build/start":
            msg = job.start(payload.get("dir", ""))
            self._json({"error": msg}, 409 if msg else 200)
        elif self.path == "/api/build/pause":
            job.pause_evt.set()
            self._json({"ok": True})
        elif self.path == "/api/build/resume":
            job.pause_evt.clear()
            self._json({"ok": True})
        elif self.path == "/api/build/stop":
            job.stop_evt.set()
            job.pause_evt.clear()  # 若正处暂停态，唤醒以便检测到停止
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


def _under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description="音效检索 Web 服务")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    seed_roots()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    addr = f"http://127.0.0.1:{args.port}"
    print(f"[serve] {addr}  (Ctrl+C 退出)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(addr)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
