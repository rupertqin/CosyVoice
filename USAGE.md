# CosyVoice 使用文档（本仓库新增功能）

本文档汇总本仓库针对"语音合成 + 字幕生成"新增的脚本与命令。

## 脚本总览

| 脚本           | 职责                                                         |
| -------------- | ------------------------------------------------------------ |
| `gen.py`       | 合成音频（可选逐句 srt）                                     |
| `align_srt.py` | 对已有音频 + 已知文本做逐词/逐字时间戳字幕（ASR 只做时间戳） |
| `srt.py`       | 底层封装（`CosyVoiceSRT` + 对齐函数）                        |

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

| 参数                     | 必填 | 说明                                                         |
| ------------------------ | ---- | ------------------------------------------------------------ |
| `--voice`                | 是   | 参考音色（见下方"可用音色"）                                 |
| `--text` / `--text-file` | 是   | 合成文本（二选一，优先 `--text-file`）                       |
| `--model`                | 否   | `cosyvoice2`（默认）/ `cosyvoice3`                           |
| `--out`                  | 否   | 输出路径，默认 `output.wav`。**按后缀自动选格式**：wav/flac/m4a/mp3 |
| `--bits`                 | 否   | wav 位深（默认 16 = PCM，体积为 32bit float 的一半；32 = IEEE Float） |
| `--bitrate`              | 否   | 有损格式码率（`.m4a`/`.mp3`，默认 128k；语音建议 96k~128k） |
| `--srt`                  | 否   | 同步导出逐句 srt 路径（不传不导出）                          |
| `--srt-format`           | 否   | `srt`（默认）/ `vtt`                                         |
| `--srt-min-length`       | 否   | 逐句字幕最短字数，相邻过短句向后合并（默认 12；传 0 不合并） |
| `--seed`                 | 否   | 随机种子，固定可复现                                         |
| `--flow-steps`           | 否   | Flow 扩散步数（默认 10，原版行为）。降到 5 约快 20~35%，音质略降 |
| `--speed`                | 否   | 语速倍率（默认 1.0）。>1 更快更短，字幕时间戳随之变化       |
| `--threads`              | 否   | CPU 线程数（默认 0 = PyTorch 默认）；多进程并行时建议调小   |
| `--jobs`                 | 否   | 并行进程数（默认 1 = 不并行）。按句切分多进程合成再拼接     |
| `--config`               | 否   | 配置文件路径（默认脚本同目录的 `gen_config.yaml`）          |
| `--no-config`            | 否   | 忽略配置文件，只用命令行 + 内置默认值                       |

### 配置文件 `gen_config.yaml`（可选）

仓库根目录的 `gen_config.yaml` 用来存放常用设置，省得每次敲一长串参数。

**优先级：命令行参数 > `gen_config.yaml` > 代码内置默认值**

```yaml
model: cosyvoice3            # 模型
voice: nice_mocangli         # 音色
text: ""                     # 直接写文本
text_file: output/speech.txt # 从文件读文本（优先）
out: output/audio.m4a        # 输出（后缀决定格式 wav/flac/m4a/mp3）
bits: 16                     # wav 位深
bitrate: 96k                 # m4a/mp3 码率
srt: output/audio.srt        # 字幕路径（空 = 不导出）
srt_format: srt
srt_min_length: 12
flow_steps: 5                # Flow 步数
speed: 1.0                   # 语速
threads: 0
jobs: 1
seed: 0
```

上面这份取值等价于以下命令（所以日常只需 `python gen.py`）：

```bash
python gen.py --model cosyvoice3 --voice nice_mocangli --text-file output/speech.txt \
  --out output/audio.m4a --srt output/audio.srt --flow-steps 5 --speed 1 --bitrate 96k
```

规则与用法：

- 文件里**没写的键** → 用代码默认值；写成**空字符串 `""`** → 视为"未设置"（该功能不启用）
- 命令行**显式传了某个参数** → 覆盖文件里的同名键（只有这一个参数被覆盖）
- 用 `--config other.yaml` 指定别的配置；用 `--no-config` 完全忽略配置
- 未识别的键会打印提示并忽略；数字写成字符串（如 `flow_steps: "5"`）会容错转换

```bash
# 日常：直接用配置（model/voice/text_file/out/srt/flow_steps/bitrate 全来自配置）
conda run -n cosyvoice python gen.py

# 临时覆盖其中几项，其余仍用配置
conda run -n cosyvoice python gen.py --voice man_surprise --text "换个音色试听"
conda run -n cosyvoice python gen.py --out output/audio.wav --bits 16   # 改回 16bit wav
conda run -n cosyvoice python gen.py --srt ""                          # 本次不导出字幕
conda run -n cosyvoice python gen.py --flow-steps 10                   # 本次用原版质量

# 完全不用配置（回到代码默认：cosyvoice2 / flow_steps 10 / 不导出字幕）
conda run -n cosyvoice python gen.py --no-config --voice man_surprise --text "测试"
```

> 注意：配置里的 `out` / `srt` 也是默认值，所以**不加 `--out` 时会写到配置指定的路径**
> （可能覆盖已有文件）。临时试验建议显式传 `--out`；要本次关闭字幕用 `--srt ""`。

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

| 参数                     | 必填 | 说明                                                                          |
| ------------------------ | ---- | ----------------------------------------------------------------------------- |
| `--audio`                | 是   | 已合成的 wav 路径                                                             |
| `--text` / `--text-file` | 是   | 已知文本（二选一，优先 `--text-file`）                                        |
| `--engine`               | 是   | `funasr` 或 `mlx-whisper`                                                     |
| `--granularity`          | 否   | `word`（默认，jieba 分词）/ `char`（逐字）                                    |
| `--model`                | 否   | mlx-whisper 模型 id（默认 `mlx-community/whisper-large-v3-mlx`）；funasr 忽略 |
| `--out`                  | 否   | 输出字幕路径（不传则打印到 stdout）                                           |
| `--format`               | 否   | `srt`（默认）/ `vtt`                                                          |

### 示例

```bash
# FunASR，词级（默认）
conda run -n cosyvoice python align_srt.py \
  --audio output/audio.wav --text-file output/speech.txt \
  --engine funasr --out output/audio.funasr.word.srt

# mlx-whisper，词级
conda run -n cosyvoice python align_srt.py \
  --audio output/audio.wav --text-file output/speech.txt \
  --engine mlx-whisper --out output/audio.mlx.word.srt

# 逐字粒度
conda run -n cosyvoice python align_srt.py \
  --audio output/out.wav --text-file output/speech.txt \
  --engine funasr --granularity char --out output/out.char.srt
```

### 后处理规则（自动生效，两种粒度都适用）

1. **纯标点不单独成条**：标点的时间并入前一条目；仅句尾的问号 `？?`、感叹号 `！!`
   保留文本追加到前一条（如 `好，真棒？` → `好` / `真棒？`）。开头的纯标点直接跳过。
2. **首尾未对齐段压缩**：ASR 未对齐到的首尾字符（多位于静音或文本与音频略有出入处）
   其时间会被压缩到锚点旁最多 0.6 秒的短窗口，**既不覆盖长静音、也不丢失文本**
   （如开头 0~0.9s 静音不会出现在字幕时间段里）。
3. **文字始终来自你的原文**，ASR 只提供时间戳。

> 注：句子级字幕（`gen.py --srt`）另有规则：句尾只保留问号/感叹号，见第 6 章。

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

| 引擎                     | 端到端耗时 | 说明                  |
| ------------------------ | ---------- | --------------------- |
| FunASR (`paraformer-zh`) | ~22s       | CPU onnx 推理，快     |
| mlx-whisper (`large-v3`) | ~194s      | 模型大（1550M），较慢 |

- **追求速度** → 用 `funasr`
- **追求效果**（复杂音频/口音/噪声）→ 用 `mlx-whisper`，或用 `--model` 指定小模型（如 `whisper-medium` / `whisper-small`）提速

---

## 6. 字幕断句规则（逐句模式）

`gen.py --srt` 的逐句字幕由 `srt.py` 处理，规则：

- 按逗号/句号/问号/感叹号等标点切分
- 相邻短句不足 `--srt-min-length`（默认 12 字）会向后合并成一句
- 每条字幕句尾只保留问号 `？?` 和感叹号 `！!`，去掉其它标点；会跨过结尾的引号/括号清理，
  如 `“很伤害脑子。”` → `“很伤害脑子”`
- **引号归位**：上游切分可能把闭合引号 `”` 切给下一句，导致出现只含一个 `”` 的孤立条目。
  这里会把字幕开头的收尾标点（`”`、`）`、`。` 等，不含开引号 `“`）挪回上一条结尾
- **去除换行**：映射回原文时若跨行，会把文本内换行去掉（含中文的片段直接去空白），
  避免一条字幕显示成多行

---

## 7. 发音注解（注释式音标）

读错的字可以用**注释式拼音注解**修正：在字后面用花括号标注带调拼音。

### 语法

```
信{xìn}    # 字 + {带调拼音}
给{jǐ}予   # 如 example.py 的 hotfix 示例"给予"读 jǐ
阿{ā}      # 零声母音节整体作为韵母
```

### 行为（自动处理，无需手动拆分）

| 用途 | 展开结果 |
|------|---------|
| TTS 合成 | `信{xìn}` → `[x][ìn]`（CosyVoice3 hotfix 拼音 token） |
| 字幕/对齐 | `信{xìn}` → `信`（剥除注解，保留干净原字） |

- `gen.py`：合成用展开拼音的文本；`--srt` 逐句字幕自动用剥除注解的干净文本
- `align_srt.py`：自动剥除注解，对齐与字幕均使用干净原文
- **仅 `--model cosyvoice3` 支持**拼音 hotfix（CosyVoice2 无拼音 token），用 cosyvoice2 时会打印警告
- 拼音必须带声调符号（如 `xìn` 不是 `xin`），声母/韵母需在 `CosyVoice3Tokenizer` 支持列表内（`cosyvoice/tokenizer/tokenizer.py:295`）

### 示例

```bash
# speech.txt 第20行：信{xìn}负责提高关系的可预测性
# 合成（音频里"信"读 xìn）
conda run -n cosyvoice python gen.py --model cosyvoice3 \
  --voice man_surprise --text-file speech.txt --out out.wav --srt out.sentence.srt

# 对齐（字幕里显示干净的"信"）
conda run -n cosyvoice python align_srt.py --audio out.wav \
  --text-file speech.txt --engine funasr --out out.word.srt
```

> 也可以直接用 CosyVoice3 原生替换式语法 `[x][ìn]`（字被拼音替换），但字幕与对齐需另备干净文本。推荐注释式 `信{xìn}`，一份文本同时服务合成与字幕。

---

## 8. 常见问题

**Q: 为什么必须装 modelscope？**
A: `cosyvoice/cli/cosyvoice.py` 顶层无条件 `from modelscope import snapshot_download`，不装就无法 import CosyVoice。

**Q: FunASR 报"ASR 未能提取任何字级时间戳"？**
A: 需使用 `paraformer-zh` 模型（`srt.py` 已默认），且**不能加标点模型**（否则 text 变连续文本无法逐字对齐）。当前 `srt.py` 已正确配置。

**Q: 字幕文字会被 ASR 改掉吗？**
A: 不会。`align_srt.py` 用你提供的 `--text`/`--text-file` 作为最终字幕文字，ASR 只提供声学时间戳，通过 `difflib` 对齐映射回原文。

**Q: 如何看完整参数列表？**
A: `python gen.py --help` / `python align_srt.py --help`（无需 conda 环境即可查看，重依赖已延迟加载）。

---

## 9. 合成性能调优（gen.py）

纯 CPU（如 Apple Silicon 未启用 GPU）下合成较慢。在 M3 / 8 核 / 16GB 上实测（RTF = 计算耗时 ÷ 音频时长）：

### 耗时构成

| 阶段 | 占比 | 说明 |
|------|------|------|
| **Flow（扩散）** | ~50% | 受 `--flow-steps` 控制 |
| **LLM 解码** | ~45% | 逐 token 自回归，难以调节 |
| HiFiGAN 声码器 | ~5% | — |

总体 RTF ≈ **6.4**（即 1 分钟音频需约 6.4 分钟计算）。

### 可用旋钮

| 参数 | 效果 | 质量影响 |
|------|------|---------|
| `--flow-steps 5` | 实测 **快 20~35%**（RTF 6.4→4.4 左右） | 细节略降；<5 步会变差（3 步实测反而更慢且失真） |
| `--speed 1.2` | 音频按倍率变短，Flow/声码器随之少算 | 只是语速变快，音质不变 |
| `--threads N` | 默认用全部核心；多进程并行时建议设 4 左右 | 无 |

```bash
# 推荐组合：质量与速度折中
conda run -n cosyvoice python gen.py --model cosyvoice3 --voice nice_mocangli_short \
  --text-file output/speech.txt --out output/audio.wav --srt output/audio.srt \
  --flow-steps 5
```

### 已排除的方案

- **Apple GPU（MPS）**：实测仅快约 5%，因为瓶颈是逐 token 算子调度而非算力，且上层代码有 `float64`（MPS 不支持）需回退。
- **fp16 / TensorRT / vLLM**：CosyVoice 仅在 CUDA 下支持，Mac 不可用。

### 并行模式 `--jobs`（慎用，内存敏感）

按句末标点把文本切成 N 段，启动 N 个子进程并行合成，父进程负责拼接音频、
并把各段字幕时间轴整体平移后合并（已实现 `--jobs N`，默认 1 不启用）。

**16GB 内存机器实测：`--jobs 2` 反而更慢**（344s vs 单进程约 165s），原因：

- 每个进程常驻内存约 **5GB**，2 个进程 + 系统已用内存导致 **swap 大量写入**（实测 swap 已用 10GB）
- CPU 利用率反而降到 164%（单进程是 340%），时间耗在换页上

结论：

| 机器内存 | 建议 |
|---------|------|
| 16GB | **不要用 `--jobs`**，用 `--flow-steps 5` 就够了 |
| ≥32GB | 可试 `--jobs 2`（配合 `--threads 4`） |

另外注意：各段是独立合成的，段落拼接处可能有**轻微韵律断层**（以及各段自带的首尾静音会叠加成略长的停顿），建议先小文本试听。

### 输出体积 / 音频格式

WAV 是**未压缩**格式，体积只取决于 采样率 × 位深 × 时长。上游原代码调用
`torchaudio.save(path, speech, sr)` 时**不指定位深**，而 torchaudio 对 float 张量默认写
IEEE Float(32bit)，因此体积翻倍：

| 位深 | 每小时 | 16.7 分钟 |
|------|--------|-----------|
| 32-bit float（上游默认） | ~330 MB | 91.5 MB |
| **16-bit PCM（本项目默认 `--bits 16`）** | ~165 MB | 45.7 MB |

**按扩展名自动选择格式**：`--out` 用什么后缀就输出什么格式（wav/flac/m4a/mp3），
有损格式可用 `--bitrate` 指定码率（默认 128k）。

```bash
# 无损
... --out audio.flac
# 有损压缩
... --out audio.m4a --bitrate 128k
... --out audio.mp3 --bitrate 96k
```

同一段 16.7 分钟语音（24kHz 单声道）实测：

| 格式 | 体积 | SNR（解码后 vs 原音频） | 说明 |
|------|------|------------------------|------|
| wav 32-bit float | 91 MB | — | 上游默认 |
| **wav 16-bit PCM** | 45.7 MB | ~81 dB | 本项目默认，听感无损（CD 标准） |
| **flac** | 48 MB | ~79 dB | **无损**（保留精度），体积≈16bit PCM |
| m4a AAC 128k | 13 MB | 42.7 dB | 有损，语音通常听不出 |
| m4a AAC 96k | 12.6 MB | 39.9 dB | 有损，语音一般可接受 |
| mp3 96k | 11.4 MB | 25.9 dB | 有损（同码率下不如 AAC） |
| m4a AAC 64k | 8.0 MB | 28.7 dB | 有损，可能出现可闻 artifact |

参考尺度：磁带 ~50 dB、黑胶 ~60 dB、CD ~96 dB。要点：

- **m4a/mp3 是"有损"，一定会降质量**（SNR 从 81 dB 掉到 26~43 dB），只是语音在
  ≥96k 时通常听不出。
- 想要"又小又无损"→ 用 **`.flac`**（48 MB，是 91 MB 的一半，且无质量损失）。
- 想进一步瘦身 → `.m4a --bitrate 96k`（12.6 MB）；低于 64k 开始有可闻瑕疵。
- 生成 `.m4a/.mp3` 需要 **ffmpeg**（macOS: `brew install ffmpeg`）。
