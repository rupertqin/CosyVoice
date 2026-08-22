# CosyVoice 使用文档（本仓库新增功能）

本文档汇总本仓库针对"语音合成 + 字幕生成"新增的脚本与命令。

## 脚本总览

| 脚本 | 职责 |
|------|------|
| `gen.py` | 合成音频（可选逐句 srt） |
| `align_srt.py` | 对已有音频 + 已知文本做逐词/逐字时间戳字幕（ASR 只做时间戳） |
| `srt.py` | 底层封装（`CosyVoiceSRT` + 对齐函数） |

**核心原则**：`gen.py` 只合成音频；`align_srt.py` 只做时间戳对齐，且**字幕文字 100% 使用你提供的原文**，ASR 不修改文字。

---

## 1. 环境准备

所有命令都在 conda 的 `cosyvoice` 环境中运行。

```bash
# 基础依赖（CosyVoice 本体，模型已下载到 pretrained_models/ 时仍需装 modelscope，
# 因为 cosyvoice.py 顶层无条件 import 它）
conda run -n cosyvoice pip install funasr modelscope   # FunASR 引擎（ASR 时间戳）
conda run -n cosyvoice pip install mlx-whisper         # mlx-whisper 引擎（Apple Silicon）
conda run -n cosyvoice pip install jieba               # 中文分词（逐词粒度需要）
```

> 说明：`modelscope` 是 CosyVoice 的硬依赖（`cosyvoice/cli/cosyvoice.py` 顶层无条件 `from modelscope import snapshot_download`），无论模型是否已在本地都必须安装。

---

## 2. 合成音频（`gen.py`）

### 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--voice` | 是 | 参考音色（见下方"可用音色"） |
| `--text` / `--text-file` | 是 | 合成文本（二选一，优先 `--text-file`） |
| `--model` | 否 | `cosyvoice2`（默认）/ `cosyvoice3` |
| `--out` | 否 | 输出 wav 路径，默认 `output.wav` |
| `--srt` | 否 | 同步导出逐句 srt 路径（不传不导出） |
| `--srt-format` | 否 | `srt`（默认）/ `vtt` |
| `--srt-min-length` | 否 | 逐句字幕最短字数，相邻过短句向后合并（默认 12；传 0 不合并） |
| `--seed` | 否 | 随机种子，固定可复现 |

### 可用音色

`gen.py` 自动扫描 `asset/voices/` 下所有子目录（`man`、`nice`、`f-a`、`f-b`、`f-c`、`vivian`），格式为 `目录名_音色名`。例如：

```
man_surprise, man_normal, man_angry, man_happy, man_sad,
nice_caixukun, nice_guimi, nice_happy,
f-a_angry, f-b_wa, f-c_heihei, vivian_surprise, ...
```

运行 `python gen.py --help` 可查看完整列表。

### 示例

```bash
# 只合成音频
conda run -n cosyvoice python gen.py \
  --model cosyvoice2 --voice man_surprise \
  --text "什么？你说我图修歪了？不可能！" --out out.wav

# 从文件读文本
conda run -n cosyvoice python gen.py \
  --voice man_surprise --text-file tts.txt --out out.wav

# 合成音频 + 同步导出逐句 srt
conda run -n cosyvoice python gen.py \
  --voice man_surprise --text-file tts.txt \
  --out out.wav --srt out.sentence.srt
```

---

## 3. 逐词/逐字时间戳字幕（`align_srt.py`）

对**已合成**的音频 + 已知文本，用 ASR 提取时间戳并映射回原文，生成逐词/逐字 srt。

### 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--audio` | 是 | 已合成的 wav 路径 |
| `--text` / `--text-file` | 是 | 已知文本（二选一，优先 `--text-file`） |
| `--engine` | 是 | `funasr` 或 `mlx-whisper` |
| `--granularity` | 否 | `word`（默认，jieba 分词）/ `char`（逐字） |
| `--model` | 否 | mlx-whisper 模型 id（默认 `mlx-community/whisper-large-v3-mlx`）；funasr 忽略 |
| `--out` | 否 | 输出字幕路径（不传则打印到 stdout） |
| `--format` | 否 | `srt`（默认）/ `vtt` |

### 示例

```bash
# FunASR，词级（默认）
conda run -n cosyvoice python align_srt.py \
  --audio output/out.wav --text-file output/speech.txt \
  --engine funasr --out output/out.funasr.word.srt

# mlx-whisper，词级
conda run -n cosyvoice python align_srt.py \
  --audio output/out.wav --text-file output/speech.txt \
  --engine mlx-whisper --out output/out.mlx.word.srt

# 逐字粒度
conda run -n cosyvoice python align_srt.py \
  --audio output/out.wav --text-file output/speech.txt \
  --engine funasr --granularity char --out output/out.char.srt
```

---

## 4. 完整工作流（推荐）

```bash
# 1. 文本存成文件（避免命令行转义，长文本更稳）
cat > tts.txt <<'EOF'
今天天气真好。我们出去玩吧。晚上吃什么？
EOF

# 2. 合成音频
conda run -n cosyvoice python gen.py \
  --voice man_surprise --text-file tts.txt --out out.wav

# 3. 用两个 ASR 引擎分别对齐，对比效果与速度
conda run -n cosyvoice python align_srt.py --audio out.wav --text-file tts.txt \
  --engine funasr --out out.funasr.word.srt
conda run -n cosyvoice python align_srt.py --audio out.wav --text-file tts.txt \
  --engine mlx-whisper --out out.mlx.word.srt
```

---

## 5. 引擎效率参考（实测）

同一段约 100 秒中文音频、同机 Apple Silicon 实测：

| 引擎 | 端到端耗时 | 说明 |
|------|-----------|------|
| FunASR (`paraformer-zh`) | ~22s | CPU onnx 推理，快 |
| mlx-whisper (`large-v3`) | ~194s | 模型大（1550M），较慢 |

- **追求速度** → 用 `funasr`
- **追求效果**（复杂音频/口音/噪声）→ 用 `mlx-whisper`，或用 `--model` 指定小模型（如 `whisper-medium` / `whisper-small`）提速

---

## 6. 字幕断句规则（逐句模式）

`gen.py --srt` 的逐句字幕由 `srt.py` 处理，规则：
- 按逗号/句号/问号/感叹号等标点切分
- 相邻短句不足 `--srt-min-length`（默认 12 字）会向后合并成一句
- 每条字幕句尾只保留问号 `？?` 和感叹号 `！!`，去掉其它标点

---

## 7. 常见问题

**Q: 为什么必须装 modelscope？**
A: `cosyvoice/cli/cosyvoice.py` 顶层无条件 `from modelscope import snapshot_download`，不装就无法 import CosyVoice。

**Q: FunASR 报"ASR 未能提取任何字级时间戳"？**
A: 需使用 `paraformer-zh` 模型（`srt.py` 已默认），且**不能加标点模型**（否则 text 变连续文本无法逐字对齐）。当前 `srt.py` 已正确配置。

**Q: 字幕文字会被 ASR 改掉吗？**
A: 不会。`align_srt.py` 用你提供的 `--text`/`--text-file` 作为最终字幕文字，ASR 只提供声学时间戳，通过 `difflib` 对齐映射回原文。

**Q: 如何看完整参数列表？**
A: `python gen.py --help` / `python align_srt.py --help`（无需 conda 环境即可查看，重依赖已延迟加载）。
