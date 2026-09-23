import sys
import os
import argparse
from typing import Any
sys.path.append('third_party/Matcha-TTS')
# 重依赖（torch / hyperpyyaml / onnxruntime 等）延迟到解析参数后再导入，
# 方便在未装依赖的环境里先跑 --help / 参数校验。
# python gen.py --model cosyvoice3 --voice nice_caixukun --text-file output/speech.txt --out output/audio.wav --srt output/audio.srt
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


def main():
    voices = discover_voices()
    if not voices:
        print('错误：未在 {} 的子目录下发现任何 (wav,txt) 对'.format(VOICES_ROOT))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS.keys()), default='cosyvoice2',
                        help='切换模型：cosyvoice2 / cosyvoice3')
    parser.add_argument('--voice', required=True, choices=sorted(voices.keys()),
                        help='参考音色，例如 man_normal / man_surprise / nice_caixukun / nice_guimi 等')
    parser.add_argument('--text', help='外置目标文本（与 --text-file 二选一，优先 --text-file）')
    parser.add_argument('--text-file', help='从文件读取目标文本')
    parser.add_argument('--out', default='output.wav', help='输出 wav 路径')
    parser.add_argument('--srt', help='可选：同步导出的 srt/vtt 字幕路径，例如 output.srt（不传则不导出）')
    parser.add_argument('--srt-format', choices=['srt', 'vtt'], default='srt',
                        help='字幕格式，默认 srt')
    parser.add_argument('--srt-min-length', type=int, default=12,
                        help='每条字幕的最短字符数，相邻过短句会向后合并（默认 12；传 0 表示仅按标点切不合并）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子，固定可复现')
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as fh:
            tts_text = fh.read().strip()
    else:
        tts_text = args.text
    if not tts_text:
        print('错误：请通过 --text 或 --text-file 提供目标文本')
        return

    model_dir, is_chat_prompt = MODELS[args.model]
    prompt_wav, transcript = voices[args.voice]
    prompt_text = (CV3_PROMPT_PREFIX + transcript) if is_chat_prompt else transcript

    print('模型: {} ({})'.format(args.model, model_dir))
    print('参考音色: {} -> {}'.format(args.voice, prompt_wav))
    print('prompt 文本: {}'.format(prompt_text))
    print('目标文本: {}'.format(tts_text))

    AutoModel, set_all_random_seed, torch, torchaudio = _load_deps()
    from srt import CosyVoiceSRT, expand_pinyin_annotations

    # 发音注解支持：信{xìn} -> TTS 用 [x][ìn]（仅 cosyvoice3），字幕保留"信"
    tts_text_in, subtitle_text = expand_pinyin_annotations(tts_text)
    if tts_text_in != tts_text:
        print('检测到发音注解，TTS 文本: {}'.format(tts_text_in))
        print('字幕文本（剥除注解）: {}'.format(subtitle_text))
        if args.model != 'cosyvoice3':
            print('警告：发音注解的拼音 hotfix 仅 cosyvoice3 支持，当前模型 {} 可能不生效'.format(args.model))

    cosyvoice = AutoModel(model_dir=model_dir)
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
            tts_text_in, prompt_text, prompt_wav, stream=False,
            srt_path=args.srt, return_subtitles=args.srt_format,
            subtitle_min_length=args.srt_min_length,
            subtitle_text=subtitle_text,
        )
    else:
        # 不导出字幕：透明委托原生生成器，逐段收集后拼接音频
        chunks = [j['tts_speech'] for j in cv.inference_zero_shot(
            tts_text_in, prompt_text, prompt_wav, stream=False)]
        speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torchaudio.save(args.out, speech, cosyvoice.sample_rate)
    print('已保存到 {}（时长 {:.2f}s）'.format(args.out, speech.shape[1] / cosyvoice.sample_rate))
    if args.srt and subtitle is not None:
        with open(args.srt, 'w', encoding='utf-8') as fh:
            fh.write(subtitle)
        print('已导出逐句字幕到 {}（{} 格式，{} 条）'.format(
            args.srt, args.srt_format,
            subtitle.count('\n') // 3 + (1 if args.srt_format == 'vtt' else 0)))


if __name__ == '__main__':
    main()
