# AudioTag

基于 **CLAP 音频-文本对比模型** 的本地音效语义检索工具。用一句中文或英文描述（如"金属撞击"、"laser shot"），即可在本地音效库中找出最相关的音频，并在网页上直接试听。全程离线运行，不上传任何文件。

## 功能

- **网页搜索**：输入描述文本，返回 Top 50 相关音效（相似度、时长、在线播放）
- **中文支持**：内置中文→英文翻译模型 + 标签式提示词优化；常用词可在 `vocab.json` 固定为精准英文提示词
- **索引管理**：网页上对任意目录一键构建向量索引，支持进度查询、暂停、继续、停止；每 25 条自动落盘，中断后自动续建
- **命令行**：`search.py` 检索、`build_index.py` 建索引、`tag_audio.py` 按 `labels.json` 自定义标签批量打分

## 快速开始

环境要求：Windows、Python 3.10+（CPU 即可，无需 GPU）

```bat
:: 1. 克隆项目
git clone https://github.com/ntfox0001/AudioTag.git
cd AudioTag

:: 2. 一键部署：安装依赖 + 下载模型（约 1.8 GB，走 hf-mirror）
setup.bat

:: 3. 启动服务（自动打开浏览器）
start_audio_tag.bat
```

启动后在网页 **索引** 页填入音频目录（如示例库 `python download_kenney.py` 下载的 kenney 音效包），点击"开始构建"即可。

## 项目结构

```
server.py            Web 服务（标准库实现，无框架依赖）
web/index.html       前端单页（搜索 / 索引两个标签）
build_index.py       索引构建（CLI，支持增量、断点续建）
search.py            语义检索（CLI）
download_model.py    模型下载器（hf-mirror）
download_kenney.py   Kenney 音效包下载器
tag_audio.py         按自定义标签批量打分（CLI）
vocab.json           中文词 -> 英文提示词映射表
models/              模型目录（自动下载，不入库）
index/               向量索引（构建后生成，不入库）
```

## 检索原理

1. 索引阶段：每条音频经 CLAP 音频编码器算出 512 维向量（L2 归一化）
2. 检索阶段：查询文本翻译为英文 → CLAP 文本编码器编码 → 与全库向量做余弦相似度（矩阵乘法，毫秒级）
3. 中文查询会同时生成"整句"和"标签式"两个提示词变体，取每条音频的最高分（CLAP 对 `metal, impact` 这类标签式短语更敏感）

## 注意事项

- 模型从 [hf-mirror.com](https://hf-mirror.com) 下载，国内网络无需代理
- `librosa/numba` 在受限目录写编译缓存会触发性能雪崩，项目已通过 `NUMBA_CACHE_DIR` 重定向到本地 `.numbacache`（已在 `.gitignore` 排除）
- 索引数据（`index/meta.json` + `index/embeddings.npy`）行序对齐，手工编辑 meta 时须保持一致

## 致谢

- 模型：[laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused)、[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
- 示例音效库：[Kenney.nl](https://kenney.nl/assets/category:Audio)（CC0 协议）
