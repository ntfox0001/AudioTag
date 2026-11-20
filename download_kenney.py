# -*- coding: utf-8 -*-
"""下载 kenney.nl Audio 分类下的全部音效包（CC0 授权），并解压。

下载流程：资源页里 data-download-id 是 'assets/<slug>' 的 hex 编码，
通过 /download-link/<hex> 换取带签名的 /download/<hex>/<token>?expires=... 地址。
"""
import http.cookiejar
import json
import re
import zipfile
from pathlib import Path
from urllib.request import build_opener, HTTPCookieProcessor, Request

BASE = "https://kenney.nl"
OUT = Path(r"d:\audio_tag\kenney")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

_jar = http.cookiejar.CookieJar()
opener = build_opener(HTTPCookieProcessor(_jar))


def get(url: str, referer: str | None = None) -> bytes:
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    return opener.open(Request(url, headers=headers), timeout=120).read()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    html = get(f"{BASE}/assets/category:Audio").decode("utf-8", "ignore")

    # 分类页里的资源链接：绝对 URL 或 预览图 URL（media/pages/assets/<slug>/...）
    html_links = re.findall(r'kenney\.nl/assets/([a-z0-9-]+)', html)
    img_links = re.findall(r'media/pages/assets/([a-z0-9-]+)/', html)
    skip = {"assets", "donate", "feed", "games", "knowledge-base",
            "starter-kits", "support", "tools", "category", "series", "tag"}
    slugs = sorted(set(html_links + img_links) - skip)
    print(f"[scan] {len(slugs)} packs: {slugs}\n")

    ok, fail = 0, []
    for i, slug in enumerate(slugs, 1):
        try:
            page_url = f"{BASE}/assets/{slug}"
            dest, exdir = OUT / f"{slug}.zip", OUT / slug
            if not dest.exists():
                # 1) 换签名下载地址
                did = ("assets/" + slug).encode().hex()
                info = json.loads(get(f"{BASE}/download-link/{did}",
                                      referer=page_url).decode())
                url = info.get("url") if isinstance(info, dict) else None
                if not url:
                    print(f"[{i}/{len(slugs)}] {slug}: no signed url ({info})")
                    fail.append(slug)
                    continue
                # 2) 下载 zip
                data = get(url, referer=page_url)
                if data[:2] != b"PK":  # 防伪：必须是zip魔数
                    print(f"[{i}/{len(slugs)}] {slug}: not a zip")
                    fail.append(slug)
                    continue
                dest.write_bytes(data)
            if not exdir.exists():
                with zipfile.ZipFile(dest) as z:
                    z.extractall(exdir)
            n_audio = sum(1 for f in exdir.rglob("*")
                          if f.suffix.lower() in (".ogg", ".wav", ".mp3"))
            ok += 1
            print(f"[{i}/{len(slugs)}] {slug}: {dest.stat().st_size//1024}KB, {n_audio} audio files")
        except Exception as e:
            print(f"[{i}/{len(slugs)}] {slug}: FAIL {e!r}")
            fail.append(slug)

    print(f"\n[done] {ok}/{len(slugs)} packs -> {OUT}")
    if fail:
        print(f"[fail] {fail}")


if __name__ == "__main__":
    main()
