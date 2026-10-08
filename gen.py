#!/usr/bin/env python3
"""gen.py — CosyVoice 语音合成脚本（本项目二次开发）。

功能
----
1. 零样本复刻合成：--model cosyvoice2/cosyvoice3 + --voice 参考音色
2. 注释式发音纠音：文本写 信{xìn} → 合成读 xìn，字幕显示"信"（仅 cosyvoice3）
3. 可选逐句字幕：--srt（srt/vtt），自动处理句子合并、句尾标点、引号归位
4. 可选加速：--flow-steps（默认 10）、--speed、--threads、--jobs（并行）

配置
----
默认读取脚本同目录的 gen_config.yaml（用 --config 换路径，--no-config 忽略）。
优先级：命令行参数 > gen_config.yaml > 本文件内置默认值（CONFIG_DEFAULTS）。
配置文件里空字符串 "" 视为"未设置"；未识别的键会提示后忽略。

输出格式（按 --out 后缀自动选择）
--------------------------------
.wav   默认，16-bit PCM。--bits 16/32 可选；32bit = 上游 torchaudio 默认行为
       （注意：torchaudio.save 对 float 张量默认写 IEEE Float 32bit，体积翻倍）
.flac  无损压缩，体积 ≈ 16bit PCM，无质量损失
.m4a   AAC 有损，码率用 --bitrate（默认 128k）
.mp3   MP3 有损，码率用 --bitrate（默认 128k）

flac/m4a/mp3 需要 ffmpeg（macOS: brew install ffmpeg）：先落一个中间 wav，
再交给 ffmpeg 编码，完成后自动删除中间文件。

实测体积 / 质量（16.7 分钟、24kHz 单声道；SNR = 解码回放 vs 原音频）
--------------------------------------------------------------
wav 32bit   91    MB   （上游默认，体积最大）
wav 16bit   45.7  MB   ~81 dB  ← 本项目默认，听感无损
flac        48    MB   ~79 dB  ← 无损且体积减半
m4a 128k    13    MB   42.7 dB
m4a 96k     12.6  MB   39.9 dB  ← 语音推荐
m4a 64k     8.0   MB   28.7 dB  （可能出现可闻瑕疵）

与上游的差异：上游 example.py / runtime client 直接 torchaudio.save 且不指定位深
（因此是 32bit float）。本脚本默认 16bit 以减半体积；需要上游行为时传 --bits 32。
详细说明见 USAGE.md。

# 示例（用户常用命令）：
# python gen.py --model cosyvoice3 --voice nice_mocangli --text-file output/speech.txt --out output/audio.m4a --srt output/audio.srt --flow-steps 5 --speed 1 --bitrate 96k
"""
import sys
import os
import argparse
from typing import Any
sys.path.append('third_party/Matcha-TTS')
# 重依赖（torch / hyperpyyaml / onnxruntime 等）延迟到解析参数后再导入，
# 方便在未装依赖的环境里先跑 --help / 参数校验。
_IMPORTS = None


def _load_deps():
    global _IMPORTS
    if _IMPORTS is not None:
        return _IMPORTS
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed
    import torch
    import torchaudio
    _IMPORTS = (AutoModel, set_all_random_seed, torch, torchaudio)
    return _IMPORTS

# 支持的模型 -> (本地模型目录, 是否 chat 式 prompt)
MODELS = {
    'cosyvoice2': ('pretrained_models/CosyVoice2-0.5B', False),
    'cosyvoice3': ('pretrained_models/Fun-CosyVoice3-0.5B', True),
}

# CosyVoice3 是 chat 式 LLM，prompt 文本需要这个前缀
CV3_PROMPT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'

# 参考音色目录：本仓库 asset/voices（已从 F5-TTS/voices 拷入）。
# 如需调整，可改为绝对路径或其它目录。
VOICES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asset', 'voices')


def _discover_voice_dirs():
    """自动扫描 VOICES_ROOT 下所有子目录作为音色分组。"""
    if not os.path.isdir(VOICES_ROOT):
        return []
    return sorted(
        d for d in os.listdir(VOICES_ROOT)
        if os.path.isdir(os.path.join(VOICES_ROOT, d))
    )


def discover_voices():
    """扫描 VOICES_ROOT 所有子目录下的 {name}.wav + 同名 {name}.txt"""
    voices = {}
    for d in _discover_voice_dirs():
        base = os.path.join(VOICES_ROOT, d)
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if not f.endswith('.wav'):
                continue
            name = os.path.splitext(f)[0]
            txt = os.path.join(base, name + '.txt')
            if not os.path.exists(txt):
                print('警告：{} 缺少同名文字稿，跳过'.format(os.path.join(base, f)))
                continue
            with open(txt, 'r', encoding='utf-8') as fh:
                transcript = fh.read().strip()
            voices['{}_{}'.format(d, name)] = (os.path.join(base, f), transcript)
    return voices


def _fmt_desc(path, bits, bitrate):
    """输出格式的文字描述（仅用于打印）。"""
    ext = os.path.splitext(path)[1].lower().lstrip('.') or 'wav'
    if ext == 'wav':
        return '{}-bit PCM'.format(bits)
    if ext == 'flac':
        return 'FLAC 无损'
    return '{} {}'.format(ext.upper(), bitrate)


def _fmt_size(path):
    """文件体积文本（<1MB 用 KB）。"""
    size = os.path.getsize(path)
    return '{:.0f} KB'.format(size / 1024) if size < 1048576 else '{:.1f} MB'.format(size / 1048576)


def _save_audio(path, speech, sample_rate, bits, bitrate='128k'):
    """按输出扩展名保存音频。

    - .wav  ：直接写（bits=16 为 PCM，体积是 32bit float 的一半）
    - .flac ：无损压缩，用 ffmpeg 从 float 源编码（真正无损）
    - .m4a/.mp3：有损压缩，需要 ffmpeg（macOS: brew install ffmpeg）

    有损格式先落一个中间 wav 再交给 ffmpeg，避免依赖 torchaudio 的编码后端。
    """
    import shutil
    import subprocess
    import torchaudio

    ext = os.path.splitext(path)[1].lower().lstrip('.')

    if ext in ('', 'wav'):
        if bits == 16:
            torchaudio.save(path, speech, sample_rate,
                            encoding='PCM_S', bits_per_sample=16)
        else:
            torchaudio.save(path, speech, sample_rate)
        return

    if ext not in ('flac', 'm4a', 'mp3'):
        raise ValueError('不支持的输出格式 .{}（支持 wav/flac/m4a/mp3）'.format(ext))

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('输出 .{} 需要 ffmpeg，请先安装：brew install ffmpeg'.format(ext))

    codec = {'flac': 'flac', 'm4a': 'aac', 'mp3': 'libmp3lame'}[ext]
    # 无损格式用 32bit 中间件（保真），有损格式用 16bit 即可
    tmp_bits = 32 if ext == 'flac' else 16
    tmp = path + '.tmp.wav'
    if tmp_bits == 16:
        torchaudio.save(tmp, speech, sample_rate, encoding='PCM_S', bits_per_sample=16)
    else:
        torchaudio.save(tmp, speech, sample_rate)
    try:
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-i', tmp, '-c:a', codec]
        if ext != 'flac':
            cmd += ['-b:a', bitrate]
        cmd.append(path)
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _patch_flow_steps(cosyvoice, steps):
    """覆盖 Flow 扩散步数（上游硬编码 n_timesteps=10，见 cosyvoice/flow/flow.py）。

    步数越少，Flow（占总耗时约 50%）越快，但细节质量略降。
    这里用运行时包装实现，不改动上游源码。
    """
    decoder = cosyvoice.model.flow.decoder
    orig_forward = decoder.forward

    def _patched_forward(mu, mask, **kwargs):
        kwargs['n_timesteps'] = steps
        return orig_forward(mu, mask, **kwargs)

    decoder.forward = _patched_forward


_SENT_END_CHARS = '。？！…!?；;'


def _split_text_for_jobs(text, n):
    """把文本按句末标点尽量均分成 n 段（不切在发音注解 {} 内）。

    返回分段列表；可用的句末切点不足时会返回较少段数。
    """
    depth = 0
    cuts = []
    for i, ch in enumerate(text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth = max(0, depth - 1)
        elif depth == 0 and ch in _SENT_END_CHARS:
            if i + 1 < len(text):      # 末位切点会生成空段，跳过
                cuts.append(i + 1)
    n = min(n, len(cuts) + 1)
    if n <= 1:
        return [text]

    target = len(text) / float(n)
    parts = []
    start = 0
    for k in range(1, n):
        want = target * k
        remain = [c for c in cuts if c > start]
        if not remain:
            break
        best = min(remain, key=lambda c: abs(c - want))
        parts.append(text[start:best])
        start = best
    parts.append(text[start:])
    return [p for p in parts if p.strip()]


def _run_parallel(args, tts_text):
    """按句切分文本，多进程并行合成，再合并音频与字幕。

    返回 True 表示已接管输出（成功或失败）；False 表示应回退单进程。
    """
    import subprocess
    import shutil
    import tempfile
    import torch
    import torchaudio
    from srt import parse_srt, build_srt, build_vtt, shift_leading_punct

    parts = _split_text_for_jobs(tts_text, args.jobs)
    if len(parts) < 2:
        return False

    jobs = len(parts)
    cpu = os.cpu_count() or 4
    threads = args.threads if args.threads > 0 else max(1, cpu // jobs)
    if jobs > 2:
        print('提示：并行数 {} 较大，每个进程约占 5GB 内存，16GB 机器建议 ≤2'.format(jobs))

    out_dir = os.path.dirname(os.path.abspath(args.out)) or '.'
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix='.cvjobs_', dir=out_dir)
    print('并行合成：{} 段，每进程 {} 线程，工作目录 {}'.format(jobs, threads, work))
    for idx, p in enumerate(parts):
        print('  段{}: {} 字'.format(idx + 1, len(p)))

    script = os.path.abspath(__file__)
    procs = []
    for idx, part in enumerate(parts):
        tf = os.path.join(work, 'part{}.txt'.format(idx))
        with open(tf, 'w', encoding='utf-8') as fh:
            fh.write(part)
        # 父进程已完成配置合并，子进程用 --no-config 避免重复套用配置文件
        cmd = [sys.executable, script, '--no-config',
               '--model', args.model, '--voice', args.voice,
               '--text-file', tf,
               '--out', os.path.join(work, 'part{}.wav'.format(idx)),
               '--srt', os.path.join(work, 'part{}.srt'.format(idx)),
               '--srt-format', 'srt',
               '--srt-min-length', str(args.srt_min_length),
               '--bits', str(args.bits),
               '--bitrate', str(args.bitrate),
               '--flow-steps', str(args.flow_steps),
               '--speed', str(args.speed),
               '--threads', str(threads),
               '--seed', str(args.seed + idx),
               '--jobs', '1']
        procs.append(subprocess.Popen(cmd))

    failed = [i for i, p in enumerate(procs) if p.wait() != 0]
    if failed:
        print('并行合成失败（段 {}），未合并输出。工作目录保留：{}'.format(failed, work))
        return True

    # 合并音频 + 平移字幕时间轴
    audios = []
    segments = []
    offset = 0.0
    sr = 0
    for idx in range(jobs):
        w, sr = torchaudio.load(os.path.join(work, 'part{}.wav'.format(idx)))
        audios.append(w)
        duration = w.shape[1] / sr
        srt_file = os.path.join(work, 'part{}.srt'.format(idx))
        if os.path.exists(srt_file):
            with open(srt_file, encoding='utf-8') as fh:
                for e in parse_srt(fh.read()):
                    segments.append({'text': e['text'],
                                     'start': e['start'] + offset,
                                     'end': e['end'] + offset})
        offset += duration

    # 分段边界处可能残留开头的收尾标点（如闭合引号），归位到上一条
    segments = shift_leading_punct(segments)

    speech = torch.cat(audios, dim=1) if len(audios) > 1 else audios[0]
    _save_audio(args.out, speech, sr, args.bits, args.bitrate)
    print('已保存到 {}（时长 {:.2f}s，{} 段拼接，{}，{}）'.format(
        args.out, speech.shape[1] / sr, jobs,
        _fmt_desc(args.out, args.bits, args.bitrate), _fmt_size(args.out)))
    if args.srt:
        content = build_vtt(segments) if args.srt_format == 'vtt' else build_srt(segments)
        with open(args.srt, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print('已导出逐句字幕到 {}（{} 格式，{} 条）'.format(
            args.srt, args.srt_format, len(segments)))
    shutil.rmtree(work, ignore_errors=True)
    return True


CONFIG_DEFAULTS = {
    'model': 'cosyvoice2',
    'voice': None,
    'text': None,
    'text_file': None,
    'out': 'output.wav',
    'bits': 16,
    'bitrate': '128k',
    'srt': None,
    'srt_format': 'srt',
    'srt_min_length': 12,
    'flow_steps': 10,
    'speed': 1.0,
    'threads': 0,
    'jobs': 1,
    'seed': 0,
}
DEFAULT_CONFIG_NAME = 'gen_config.yaml'


def _read_yaml(path):
    """读取 YAML（优先 PyYAML，回退 ruamel.yaml）。"""
    try:
        import yaml
        with open(path, encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except ImportError:
        from ruamel.yaml import YAML
        with open(path, encoding='utf-8') as fh:
            return YAML(typ='safe').load(fh)


def _load_config(path):
    """读取配置文件；不存在/解析失败时返回 {} 并给出提示。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        cfg = _read_yaml(path)
    except Exception as e:
        print('警告：配置文件读取失败 {}（{}），改用内置默认值'.format(path, e))
        return {}
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        print('警告：配置文件顶层应为键值结构，已忽略 {}'.format(path))
        return {}
    return cfg


def _apply_config(args, cfg):
    """填参数：优先级 命令行 > 配置文件 > 代码默认值。

    命令行未显式给出的参数（值为 None）才会被配置/默认值填充；
    配置里写成空字符串 "" 视为"未设置"，回退到代码默认值。
    """
    unknown = sorted(k for k in cfg if k not in CONFIG_DEFAULTS)
    if unknown:
        print('提示：配置文件中未识别的键已忽略: {}'.format(', '.join(unknown)))
    for key, dflt in CONFIG_DEFAULTS.items():
        if getattr(args, key, None) is not None:
            continue                       # 命令行已显式指定，优先级最高
        val = cfg.get(key)
        if val is None or (isinstance(val, str) and val == ''):
            val = dflt
        # 配置文件写成字符串数字时做一次容错转换（如 flow_steps: "5"）
        try:
            if key in ('speed',):
                val = float(val) if val is not None else val
            elif key in ('bits', 'srt_min_length', 'flow_steps', 'threads', 'jobs', 'seed'):
                val = int(val) if val is not None else val
        except (TypeError, ValueError):
            print('警告：配置项 {} 的值 {!r} 类型不正确，已忽略'.format(key, val))
            val = dflt
        setattr(args, key, val)


def _validate_args(args, voices):
    """校验合并后的参数（配置值不经过 argparse 校验，需要自己查）。返回错误信息或 None。"""
    if args.model not in MODELS:
        return 'model 无效：{}（可选 {}）'.format(args.model, ' / '.join(MODELS))
    if args.voice not in voices:
        return '音色无效或未指定：{}（用 --voice 指定，或在 gen_config.yaml 中设置）'.format(args.voice)
    if args.bits not in (16, 32):
        return 'bits 只能是 16 或 32'
    if args.srt_format not in ('srt', 'vtt'):
        return 'srt_format 只能是 srt 或 vtt'
    if args.jobs < 1:
        return 'jobs 必须 >= 1'
    if args.flow_steps < 1:
        return 'flow_steps 必须 >= 1'
    if args.speed <= 0:
        return 'speed 必须 > 0'
    if args.text_file and not os.path.exists(args.text_file):
        return '文本文件不存在：{}'.format(args.text_file)
    if not args.text_file and not args.text:
        return '请通过 --text / --text-file 提供文本，或在 gen_config.yaml 中设置'
    return None


def _build_parser(voices):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='配置文件路径（默认取脚本同目录的 {}；不存在则忽略）'.format(DEFAULT_CONFIG_NAME))
    parser.add_argument('--no-config', action='store_true', help='忽略配置文件，只用命令行与内置默认值')
    parser.add_argument('--model', choices=list(MODELS.keys()), default=None,
                        help='切换模型：cosyvoice2 / cosyvoice3（默认 cosyvoice2）')
    parser.add_argument('--voice', default=None, choices=sorted(voices.keys()),
                        help='参考音色，例如 man_normal / man_surprise / nice_caixukun / nice_guimi 等')
    parser.add_argument('--text', default=None, help='外置目标文本（与 --text-file 二选一，优先 --text-file）')
    parser.add_argument('--text-file', default=None, help='从文件读取目标文本')
    parser.add_argument('--out', default=None,
                        help='输出路径，默认 output.wav。按后缀自动选格式：'
                             'wav（默认16bit）/ flac（无损）/ m4a、mp3（有损，需 ffmpeg）')
    parser.add_argument('--bits', type=int, choices=[16, 32], default=None,
                        help='输出位深（默认 16 = PCM，体积为 32bit float 的一半；32 = IEEE Float）')
    parser.add_argument('--bitrate', default=None,
                        help='有损格式码率（.m4a/.mp3 用，默认 128k）。语音建议 96k~128k；'
                             '输出 .flac 为无损、无需码率')
    parser.add_argument('--srt', default=None,
                        help='可选：同步导出的 srt/vtt 字幕路径，例如 output.srt（不传则不导出）')
    parser.add_argument('--srt-format', choices=['srt', 'vtt'], default=None,
                        help='字幕格式，默认 srt')
    parser.add_argument('--srt-min-length', type=int, default=None,
                        help='每条字幕的最短字符数，相邻过短句会向后合并（默认 12；传 0 表示仅按标点切不合并）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子，固定可复现')
    parser.add_argument('--flow-steps', type=int, default=None,
                        help='Flow 扩散步数（默认 10，越快越糊）。实测 10→5 约快 33%%，质量略降；建议 5~8')
    parser.add_argument('--speed', type=float, default=None,
                        help='语速倍率（默认 1.0）。>1 语速更快、音频更短，字幕时间戳随之变化')
    parser.add_argument('--threads', type=int, default=None,
                        help='CPU 线程数（默认 0 = 用 PyTorch 默认）。多进程并行时建议调小')
    parser.add_argument('--jobs', type=int, default=None,
                        help='并行进程数（默认 1 = 不并行）。按句切分文本、多进程合成再拼接；'
                             '每个进程约占 5GB 内存，16GB 机器建议 ≤2')
    return parser


def main():
    voices = discover_voices()
    if not voices:
        print('错误：未在 {} 的子目录下发现任何 (wav,txt) 对'.format(VOICES_ROOT))
        return

    parser = _build_parser(voices)
    args = parser.parse_args()

    # 配置合并：命令行 > 配置文件 > 代码默认值
    cfg_path = None
    if not args.no_config:
        cfg_path = args.config or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), DEFAULT_CONFIG_NAME)
    cfg = _load_config(cfg_path)
    _apply_config(args, cfg)
    if cfg:
        print('配置来源: {}（命令行参数优先）'.format(cfg_path))

    err = _validate_args(args, voices)
    if err:
        print('错误：{}'.format(err))
        return

    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as fh:
            tts_text = fh.read().strip()
    else:
        tts_text = args.text

    model_dir, is_chat_prompt = MODELS[args.model]
    prompt_wav, transcript = voices[args.voice]
    prompt_text = (CV3_PROMPT_PREFIX + transcript) if is_chat_prompt else transcript

    print('模型: {} ({})'.format(args.model, model_dir))
    print('参考音色: {} -> {}'.format(args.voice, prompt_wav))
    print('prompt 文本: {}'.format(prompt_text))
    print('目标文本: {}'.format(tts_text))

    # 并行模式（可选）：按句切分、多进程合成再拼接，父进程只负责调度与合并
    if args.jobs > 1:
        if _run_parallel(args, tts_text):
            return
        print('并行模式：句末切点不足，回退单进程')

    AutoModel, set_all_random_seed, torch, torchaudio = _load_deps()
    from srt import CosyVoiceSRT, expand_pinyin_annotations

    # 发音注解支持：信{xìn} -> TTS 用 [x][ìn]（仅 cosyvoice3），字幕保留"信"
    tts_text_in, subtitle_text = expand_pinyin_annotations(tts_text)
    if tts_text_in != tts_text:
        print('检测到发音注解，TTS 文本: {}'.format(tts_text_in))
        print('字幕文本（剥除注解）: {}'.format(subtitle_text))
        if args.model != 'cosyvoice3':
            print('警告：发音注解的拼音 hotfix 仅 cosyvoice3 支持，当前模型 {} 可能不生效'.format(args.model))

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    cosyvoice = AutoModel(model_dir=model_dir)
    if args.flow_steps != 10:
        _patch_flow_steps(cosyvoice, args.flow_steps)
        print('Flow 扩散步数: {}（默认 10；步数越少越快、质量略降）'.format(args.flow_steps))
    # CosyVoiceSRT 是动态委托封装（__getattr__ 转发），返回类型随参数变化，
    # 标注为 Any 以避免静态检查对联合类型的误报。
    cv: Any = CosyVoiceSRT(cosyvoice)
    set_all_random_seed(args.seed)

    # gen.py 只负责合成音频（可选逐句 srt）；逐词/逐字时间戳请用独立的 align_srt.py。
    # subtitle_text：字幕断句使用剥除注解后的干净原文，合成用展开拼音后的文本。
    subtitle = None
    if args.srt:
        # 走字幕路径：返回 (full_audio, subtitle_string) 并自动写文件
        speech, subtitle = cv.inference_zero_shot(
            tts_text_in, prompt_text, prompt_wav, stream=False, speed=args.speed,
            srt_path=args.srt, return_subtitles=args.srt_format,
            subtitle_min_length=args.srt_min_length,
            subtitle_text=subtitle_text,
        )
    else:
        # 不导出字幕：透明委托原生生成器，逐段收集后拼接音频
        chunks = [j['tts_speech'] for j in cv.inference_zero_shot(
            tts_text_in, prompt_text, prompt_wav, stream=False, speed=args.speed)]
        speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _save_audio(args.out, speech, cosyvoice.sample_rate, args.bits, args.bitrate)
    print('已保存到 {}（时长 {:.2f}s，{}，{}）'.format(
        args.out, speech.shape[1] / cosyvoice.sample_rate,
        _fmt_desc(args.out, args.bits, args.bitrate), _fmt_size(args.out)))
    if args.srt and subtitle is not None:
        with open(args.srt, 'w', encoding='utf-8') as fh:
            fh.write(subtitle)
        print('已导出逐句字幕到 {}（{} 格式，{} 条）'.format(
            args.srt, args.srt_format,
            subtitle.count('\n') // 3 + (1 if args.srt_format == 'vtt' else 0)))


if __name__ == '__main__':
    main()
