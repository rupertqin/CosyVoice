#!/usr/bin/env python3
"""align_srt.py — 对"已合成音频 + 已知文本"做逐词/逐字时间戳字幕（SRT/VTT）。

与 gen.py 解耦：gen.py 只合成音频，本脚本只做 ASR 对齐。
支持两个 ASR 引擎：funasr / mlx-whisper，便于对比效果与速度。

核心保证：字幕里的文字 100% 来自 --text 提供的原文，
ASR 只用于提取声学时间戳并映射回原文，绝不改动文字。

粒度说明：word（词级）本质上由 char（逐字）时间戳经 jieba 分组得到，
因此一次 ASR 即可同时产出两种粒度——用 --out 指定主输出粒度，
--out-char 额外导出逐字版，避免为换粒度重复跑 ASR。

用法示例：
    # 词级（默认）：一次 ASR
    conda run -n cosyvoice python align_srt.py --audio output/audio.wav \
      --text-file output/speech.txt --engine funasr --out output/audio.word.srt

    # 一次 ASR 同时产出词级 + 逐字两份
    conda run -n cosyvoice python align_srt.py --audio output/audio.wav \
      --text-file output/speech.txt --engine funasr \
      --out output/audio.word.srt --out-char output/audio.char.srt

    # 用 mlx-whisper 对齐（可指定模型）
    conda run -n cosyvoice python align_srt.py --audio output/audio.wav \
      --text-file output/speech.txt --engine mlx-whisper \
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
    parser.add_argument('--out-char',
                        help='可选：在同一次 ASR 中额外导出逐字字幕路径'
                             '（word 由逐字时间戳推导，无需重复对齐）')
    parser.add_argument('--format', choices=['srt', 'vtt'], default='srt',
                        help='字幕格式，默认 srt')
    parser.add_argument('--granularity', choices=['word', 'char'], default='word',
                        help='字幕粒度：word=按词（jieba 分词，默认）；char=逐字')
    args = parser.parse_args()

    # 延迟导入 srt 模块（依赖 torch 等，且需在仓库根目录）
    try:
        from srt import (char_level_timestamps, word_level_timestamps,
                         build_srt, build_vtt, expand_pinyin_annotations,
                         _clamp_unaligned_edges, merge_punct_chars)
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

    # 1) 先拿逐字时间戳（return_align=True 记录哪些字被 ASR 真正对齐）
    char_segs = char_level_timestamps(
        text, args.audio, sr, engine=args.engine, model_id=args.model,
        return_align=True,
    )
    # 2) 两种粒度都从同一份逐字时间戳本地推导（不额外跑 ASR）
    #    word = jieba 分词（过滤纯标点、压缩首尾未对齐段）
    #    char = 逐字（同样过滤纯标点、压缩首尾未对齐段）
    word_list = word_level_timestamps(
        text, char_segs, clamp_unaligned_edges=True, merge_punct=True)
    char_list = merge_punct_chars(_clamp_unaligned_edges(char_segs, drop_flag=True))

    def _format(seg_list):
        segs = [{'text': c, 'start': s, 'end': e} for c, s, e in seg_list]
        return (build_vtt(segs) if args.format == 'vtt' else build_srt(segs)), segs

    primary_list = word_list if args.granularity == 'word' else char_list
    content, primary_segs = _format(primary_list)
    print('字幕粒度: {}（{} 条）'.format(args.granularity, len(primary_segs)))

    wrote_file = False
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(content)
        wrote_file = True
        print('已导出{}字幕到 {}（{} 条）'.format(
            args.format, args.out, len(primary_segs)))

    # 额外导出逐字版（一次 ASR 同时拿到词级 + 字级）
    if args.out_char and args.out_char != args.out:
        char_content, char_segs_out = _format(char_list)
        with open(args.out_char, 'w', encoding='utf-8') as fh:
            fh.write(char_content)
        wrote_file = True
        print('已导出{}逐字字幕到 {}（{} 条）'.format(
            args.format, args.out_char, len(char_segs_out)))

    # 未写任何文件时打印到 stdout
    if not wrote_file:
        print(content)


if __name__ == '__main__':
    main()
