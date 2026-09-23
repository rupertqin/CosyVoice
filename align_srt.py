#!/usr/bin/env python3
"""align.py — 对"已合成音频 + 已知文本"做逐字时间戳字幕（SRT/VTT）。

与 gen.py 解耦：gen.py 只合成音频，本脚本只做 ASR 对齐。
支持两个 ASR 引擎：funasr / mlx-whisper，便于对比效果与速度。

核心保证：srt 里的文字 100% 来自 --text 提供的原文，
ASR 只用于提取声学时间戳并映射回原文，绝不改动文字。

用法示例：
    # 用 FunASR 对齐
    python align.py --audio output/audio.wav --text "今天天气真好。我们出去玩吧。" \\
                    --engine funasr --out output/audio.word.srt

    # 用 mlx-whisper 对齐（可指定模型）
    python align.py --audio output/audio.wav --text "..." --engine mlx-whisper \\
                    --model mlx-community/whisper-large-v3-mlx --out output/audio.mlx.srt
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='对已合成音频 + 已知文本做逐词/逐字时间戳字幕（ASR 只做时间戳，不改文字）')
    parser.add_argument('--audio', required=True, help='已合成的 wav 音频路径')
    parser.add_argument('--text', help='已知文本（与 --text-file 二选一，优先 --text-file）')
    parser.add_argument('--text-file', help='从文件读取已知文本')
    parser.add_argument('--engine', required=True, choices=['funasr', 'mlx-whisper'],
                        help='ASR 引擎：funasr 或 mlx-whisper')
    parser.add_argument('--model', default=None,
                        help='mlx-whisper 的模型 id（默认 mlx-community/whisper-large-v3-mlx）；funasr 忽略')
    parser.add_argument('--out', help='输出字幕路径（不传则打印到 stdout）')
    parser.add_argument('--format', choices=['srt', 'vtt'], default='srt',
                        help='字幕格式，默认 srt')
    parser.add_argument('--granularity', choices=['word', 'char'], default='word',
                        help='字幕粒度：word=按词（jieba 分词，默认）；char=逐字')
    args = parser.parse_args()

    # 延迟导入 srt 模块（依赖 torch 等，且需在仓库根目录）
    try:
        from srt import (char_level_timestamps, word_level_timestamps,
                         build_srt, build_vtt, expand_pinyin_annotations)
    except ImportError:
        print('错误：无法导入 srt 模块，请确认在 CosyVoice 仓库根目录、装好依赖的环境运行')
        return

    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as fh:
            text = fh.read().strip()
    else:
        text = args.text
    if not text:
        print('错误：请通过 --text 或 --text-file 提供已知文本')
        return
    # 移除文本内的换行：TTS 合成的语音是连续文本，逐字对齐时不应含换行符
    text = text.replace('\n', '').replace('\r', '')
    text = ' '.join(text.split())  # 压缩多余空白
    # 剥除发音注解（如 信{xìn} -> 信）：对齐与字幕都应使用干净原文
    text_clean = expand_pinyin_annotations(text)[1]
    if text_clean != text:
        print('检测到发音注解，已剥除用于对齐/字幕: {}'.format(text_clean))
    text = text_clean
    if not os.path.exists(args.audio):
        print('错误：音频不存在 {}'.format(args.audio))
        return

    # 读取采样率
    import torchaudio
    info = torchaudio.info(args.audio)
    sr = info.sample_rate

    print('音频: {}（采样率 {}）'.format(args.audio, sr))
    print('ASR 引擎: {} {}'.format(args.engine, args.model or ''))
    print('已知文本: {}'.format(text))

    # 1) 先拿逐字时间戳
    char_segs = char_level_timestamps(
        text, args.audio, sr, engine=args.engine, model_id=args.model,
    )
    # 2) 按粒度合并：word=jieba 分词，char=逐字
    if args.granularity == 'word':
        seg_list = word_level_timestamps(text, char_segs)
    else:
        seg_list = char_segs
    segments = [{'text': c, 'start': s, 'end': e} for c, s, e in seg_list]

    print('字幕粒度: {}（{} 条）'.format(args.granularity, len(segments)))

    if args.format == 'vtt':
        content = build_vtt(segments)
    else:
        content = build_srt(segments)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print('已导出{}字幕到 {}（{} 条）'.format(args.format, args.out, len(segments)))
    else:
        print(content)


if __name__ == '__main__':
    main()
