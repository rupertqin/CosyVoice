"""
CosyVoiceSRT — Middleware wrapping CosyVoice/CosyVoice2/CosyVoice3
with optional SRT subtitle generation for synthesized speech.

Usage:
    from cosyvoice.cli.cosyvoice import AutoModel
    from srt import CosyVoiceSRT

    cv = CosyVoiceSRT(AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B'))

    # ── Original API (no SRT, generator-based, fully transparent) ──
    for result in cv.inference_sft('你好', '中文女'):
        torchaudio.save('out.wav', result['tts_speech'], cv.sample_rate)

    # ── With SRT: returns (full_audio_tensor, srt_string) ──
    audio, srt = cv.inference_sft(
        '今天天气真好。我们出去玩吧。晚上吃什么？',
        '中文女',
        return_srt=True,
    )
    torchaudio.save('output.wav', audio, cv.sample_rate)
    with open('output.srt', 'w', encoding='utf-8') as f:
        f.write(srt)

    # ── Auto-save SRT to file ──
    audio, srt = cv.inference_sft('你好', '中文女', srt_path='output.srt')

    # ── All inference methods supported ──
    audio, srt = cv.inference_zero_shot(..., return_srt=True)
    audio, srt = cv.inference_cross_lingual(..., return_srt=True)
    audio, srt = cv.inference_instruct(..., return_srt=True)
    audio, srt = cv.inference_instruct2(..., return_srt=True)
"""

import torch
import re
from typing import Optional, List, Dict, Generator
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3


# ═══════════════════════════════════════════════════════════════
#  Subtitle sentence splitting (independent of synthesis splitting)
# ═══════════════════════════════════════════════════════════════

# 按逗号/句号/问号/感叹号等标点切分（字幕更短），再按 min_length 向后合并，
# 避免相邻短句太碎。默认不拆碎到 12 个中文字符以下。
_SUBTITLE_SENT_END = re.compile(r'(?<=[。？！；.!?;:：…，,])')
_SUBTITLE_MIN_LENGTH = 12
# 允许保留在句尾的标点：只有问号/感叹号
_TRAILING_KEEP = set('？！?!')


def _trim_trailing_punct(s: str) -> str:
    """去掉句尾除问号/感叹号之外的所有标点。

    例如 "今天天气真好。" -> "今天天气真好"；"你说什么？" -> "你说什么？"；
    "太棒了！" -> "太棒了！"；"不会吧。？！" -> "不会吧？！"（保留结尾的问号感叹号）。
    """
    s = s.strip()
    # 找到正文与句尾标点的分界：从末尾往前，连续跳过所有标点，
    # 然后只看被跳过的部分里是否含问号/感叹号并保留它们。
    end = len(s)
    while end > 0 and s[end - 1] in '。？！；.!?;:：…，,、':
        end -= 1
    if end == len(s):
        # 没有句尾标点，原样返回
        return s
    body = s[:end].strip()
    trailing = s[end:]
    # 只保留 trailing 中的问号/感叹号（按原顺序）
    keep = ''.join(ch for ch in trailing if ch in _TRAILING_KEEP)
    return body + keep


# ═══════════════════════════════════════════════════════════════
#  Pinyin pronunciation annotations: 信{xìn}
# ═══════════════════════════════════════════════════════════════
# 注释式发音修正：在字后用花括号标注带调拼音，如 信{xìn}。
# 展开后：
#   - TTS 文本：信{xìn} -> [x][ìn]（CosyVoice3 hotfix 拼音，仅 cosyvoice3 支持）
#   - 字幕文本：信{xìn} -> 信（保持干净原文，供字幕与 ASR 对齐使用）

_PINYIN_INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                    'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
_ANNOTATION_RE = re.compile(r'(.)(\{[a-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+\})')


def _pinyin_to_hotfix(pinyin: str) -> str:
    """把带调拼音音节转成 CosyVoice3 hotfix 语法 [声母][韵母]。

    例如 xìn -> [x][ìn]，shì -> [sh][ì]，ā（零声母）-> [ā]。
    """
    pinyin = pinyin.strip()
    for ini in _PINYIN_INITIALS:
        if pinyin.startswith(ini):
            final = pinyin[len(ini):]
            if final:
                return '[{}][{}]'.format(ini, final)
            return '[{}]'.format(ini)
    # 零声母音节（如 ā、ài）整体作为韵母 token
    return '[{}]'.format(pinyin)


def expand_pinyin_annotations(text: str):
    """展开注释式发音标注 信{xìn}。

    Returns:
        (tts_text, clean_text)
        tts_text: 注解展开为 CosyVoice3 hotfix 拼音（如 [x][ìn]），用于合成；
        clean_text: 注解剥除，保留原字（如 信），用于字幕与对齐。
    """
    def _to_hotfix(m):
        return _pinyin_to_hotfix(m.group(2)[1:-1])

    def _to_clean(m):
        return m.group(1)

    tts_text = _ANNOTATION_RE.sub(_to_hotfix, text)
    clean_text = _ANNOTATION_RE.sub(_to_clean, text)
    return tts_text, clean_text


# ═══════════════════════════════════════════════════════════════
#  Word-level timestamps via external ASR engines
# ═══════════════════════════════════════════════════════════════
# 两个引擎可选：funasr / mlx-whisper。ASR 只用于提取"声学时间戳"，
# 最终 srt 里的文字一律用调用方提供的原文，保证不改字。

_ASR_ENGINE = None   # 缓存已加载的引擎，避免重复加载模型
_ASR_ENGINE_NAME = None


def _load_funasr():
    from funasr import AutoModel as FunASRAutoModel
    # paraformer-zh 是官方短名，原生输出字级时间戳（timestamp 字段）。
    # 注意：用长模型名 speech_paraformer-large_asr_nat-... 时不会返回 timestamp，
    # 必须用 paraformer-zh 才能拿到逐字时间戳。
    # 注意：不能加 punc_model！加标点模型后 text 会变成无空格连续文本，
    # 导致 text.split() 无法逐字对齐 timestamp。去掉标点模型，text 才是
    # 空格分隔的字，timestamp 与之逐一对应。
    return FunASRAutoModel(model="paraformer-zh",
                           vad_model="fsmn-vad",
                           disable_update=True,
                           device="cpu" if not _cuda_available() else "cuda")


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _extract_funasr_words(wav_path, sample_rate):
    """FunASR: 返回字级时间戳 [(char, start_sec, end_sec), ...]"""
    global _ASR_ENGINE, _ASR_ENGINE_NAME
    if _ASR_ENGINE is None or _ASR_ENGINE_NAME != 'funasr':
        _ASR_ENGINE = _load_funasr()
        _ASR_ENGINE_NAME = 'funasr'
    res = _ASR_ENGINE.generate(input=wav_path, batch_size_s=60)
    res = res[0]
    words = []
    # paraformer-zh 的 text 是空格分隔的字，timestamp 与之逐一对应（毫秒）
    text = res.get('text', '')
    ts = res.get('timestamp', []) or []
    chars = text.split()
    if len(ts) == len(chars):
        for c, (s, e) in zip(chars, ts):
            words.append((c, s / 1000.0, e / 1000.0))
    return words


def _load_mlx_whisper():
    import mlx_whisper
    return mlx_whisper


def _extract_mlx_whisper_words(wav_path, sample_rate, model_id='mlx-community/whisper-large-v3-mlx'):
    """mlx-whisper: 返回词级时间戳 [(char, start_sec, end_sec), ...]
    注意：mlx-whisper 是词级（word）时间戳，中文一个词常含多个字，
    这里把词的时长按其字数均分到每个字。
    """
    import mlx_whisper
    res = mlx_whisper.transcribe(wav_path, path_or_hf_repo=model_id, word_timestamps=True)
    words = []
    for seg in res.get('segments', []):
        for w in seg.get('words', []):
            word_text = w['word'].replace(' ', '')
            if not word_text:
                continue
            start, end = w['start'], w['end']
            chars = list(word_text)
            n = len(chars)
            per = (end - start) / n if n else 0
            for i, c in enumerate(chars):
                words.append((c, start + i * per, start + (i + 1) * per))
    return words


def _asr_word_timestamps(wav_path, sample_rate, engine='funasr', model_id=None):
    """统一入口：根据 engine 返回字级时间戳 [(char, start_sec, end_sec), ...]"""
    if engine == 'funasr':
        return _extract_funasr_words(wav_path, sample_rate)
    elif engine == 'mlx-whisper':
        return _extract_mlx_whisper_words(wav_path, sample_rate, model_id or 'mlx-community/whisper-large-v3-mlx')
    else:
        raise ValueError('unknown asr engine: {}'.format(engine))


def char_level_timestamps(text, wav_path, sample_rate, engine='funasr', model_id=None,
                          return_align=False):
    """把 ASR 时间戳映射回给定原文，返回逐字时间。

    保证每个 char 都来自 text（不改字）。使用 difflib 在字符级把 ASR 识别序列
    对齐到原文；ASR 缺失/多余的字通过相邻锚点线性插值补全。

    Args:
        return_align: 为 True 时返回 4 元组 (char, start, end, aligned)，
            aligned 表示该字是否被 ASR 真正对齐（False = 插值补出的）。
    """
    import difflib
    asr_words = _asr_word_timestamps(wav_path, sample_rate, engine, model_id)
    if not asr_words:
        # ASR 无结果，退化为按时长均匀分配
        raise RuntimeError('ASR 未能提取任何字级时间戳')

    # ASR token 可能含多个字符（英文单词/数字等），而 SequenceMatcher 的下标
    # 是按"字符"计的。这里把每个 token 的时长按字符数均分，展平成逐字序列，
    # 保证下标与字符一一对应（否则会出现 IndexError）。
    asr_char_times = []   # [(char, start_sec, end_sec), ...]
    for tok, s, e in asr_words:
        n_chars = len(tok)
        if n_chars == 0:
            continue
        per = (e - s) / n_chars
        for k, ch in enumerate(tok):
            asr_char_times.append((ch, s + k * per, s + (k + 1) * per))

    asr_text = ''.join(c for c, _, _ in asr_char_times)
    target = ''.join(c for c in text)

    sm = difflib.SequenceMatcher(None, target, asr_text, autojunk=False)
    # 建立原文每个字符的时间区间
    n = len(target)
    starts = [None] * n
    ends = [None] * n
    # 通过 opcodes 对齐：只信任 equal 块；其余用插值
    matches = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            # target[i1:i2] 对应 asr_text[j1:j2]
            for k, off in enumerate(range(j1, j2)):
                c, s, e = asr_char_times[off]
                matches.append((i1 + k, s, e))
    # 把所有匹配按 target 位置填入，缺失的用相邻插值
    match_map = {i: (s, e) for i, s, e in matches}
    filled = []
    last_i, last_e = -1, 0.0
    for i in range(n):
        aligned = i in match_map
        if aligned:
            s, e = match_map[i]
        else:
            # 插值：找前后最近的锚点
            prev_e = last_e
            next_s = None
            for k in range(i + 1, n):
                if k in match_map:
                    next_s = match_map[k][0]
                    break
            if next_s is not None:
                s = prev_e
                e = next_s
            else:
                s = prev_e
                e = prev_e + 0.1
        starts[i] = s
        ends[i] = e
        if return_align:
            filled.append((target[i], s, e, aligned))
        else:
            filled.append((target[i], s, e))
        last_i, last_e = i, e
    return filled


def _is_pure_punct(token: str) -> bool:
    """判断 token 是否纯标点（不含汉字/字母/数字）。"""
    for ch in token:
        if ch.isalnum() or '\u4e00' <= ch <= '\u9fff':
            return False
    return bool(token)


def merge_punct_chars(segs):
    """把逐字序列中的纯标点并入前一条（逐字模式用）。

    标点不单独成条：时间并入前一条目；仅问号/感叹号保留文本追加到前一条目。

    Args:
        segs: [(text, start_sec, end_sec), ...]
    Returns:
        处理后的 [(text, start_sec, end_sec), ...]
    """
    out = []
    for c, s, e in segs:
        if _is_pure_punct(c):
            if out:
                pc, ps, pe = out[-1]
                keep = c if c in _TRAILING_KEEP else ''
                out[-1] = (pc + keep, ps, max(pe, e))
            # 前面没有条目时直接跳过（不产生纯标点字幕）
            continue
        out.append((c, s, e))
    return out


def _clamp_unaligned_edges(segs, drop_flag=False):
    """把首尾"未被 ASR 对齐"的字符压缩到贴近锚点的短窗口内。

    首尾未对齐字符的插值时间往往跨越静音（如开头 0~0.92s 的空白），直接显示会让
    字幕覆盖静音区；但直接丢弃又会丢失真实文本（ASR 偶尔漏字）。因此这里把它们
    压缩成锚点旁最多 max_edge 秒的短窗口，既不覆盖长静音、也不丢文本。

    Args:
        segs: (char, start, end, aligned) 列表。
        drop_flag: 为 True 时返回 3 元组（去掉 aligned 标记）。
    """
    n = len(segs)
    if n == 0 or len(segs[0]) != 4:
        return [(s[0], s[1], s[2]) for s in segs] if drop_flag else list(segs)

    first = next((i for i in range(n) if segs[i][3]), None)
    last = next((i for i in range(n - 1, -1, -1) if segs[i][3]), None)
    if first is None or last is None:
        # 没有任何对齐信息，原样返回
        return [(s[0], s[1], s[2]) for s in segs] if drop_flag else list(segs)

    MAX_EDGE, PER_CHAR = 0.6, 0.2
    out = [[c, s, e, a] for c, s, e, a in segs]

    # 头部未对齐 run [0, first)：压缩到第一个锚点前的短窗口
    if first > 0:
        win = min(first * PER_CHAR, MAX_EDGE)
        run_start = max(segs[first][1] - win, 0.0)
        per = win / first
        for k in range(first):
            out[k][1] = run_start + k * per
            out[k][2] = run_start + (k + 1) * per

    # 尾部未对齐 run (last, n)：压缩到最后一个锚点后的短窗口
    tail_n = n - 1 - last
    if tail_n > 0:
        win = min(tail_n * PER_CHAR, MAX_EDGE)
        run_start = segs[last][2]
        per = win / tail_n
        for k in range(tail_n):
            idx = last + 1 + k
            out[idx][1] = run_start + k * per
            out[idx][2] = run_start + (k + 1) * per

    if drop_flag:
        return [(c, s, e) for c, s, e, _ in out]
    return [tuple(x) for x in out]


def word_level_timestamps(text, char_segs, clamp_unaligned_edges=True,
                          merge_punct=True):
    """把逐字时间戳按 jieba 分词合并成词级时间戳。

    Args:
        text: 原文（与 char_segs 字符一一对应）。
        char_segs: 逐字时间戳，3 元组 (char, start, end) 或 4 元组
            (char, start, end, aligned)（由 char_level_timestamps(return_align=True) 提供）。
        clamp_unaligned_edges: 为 True 时把首尾"未被 ASR 对齐"的字符时间压缩到
            锚点旁的短窗口（避免字幕覆盖静音，同时不丢文本）。需 4 元组输入才生效。
        merge_punct: 为 True 时纯标点词条不单独成条：时间并入前一个词，
            仅结尾的问号/感叹号保留文本追加到前词。

    Returns:
        词级时间戳 [(word, start_sec, end_sec), ...]，每个 word 来自原文。
    """
    import jieba
    # 用 jieba 对原文分词；空白 token 会被跳过，但位置照常推进（保证下标不漂移）
    raw_tokens = list(jieba.cut(text))
    if not any(t.strip() for t in raw_tokens):
        return [(s[0], s[1], s[2]) for s in char_segs]

    # 首尾未对齐字符的时间压缩（保留对齐标记，仅调整时间）
    has_align = len(char_segs) > 0 and len(char_segs[0]) == 4
    if has_align and clamp_unaligned_edges:
        char_segs = _clamp_unaligned_edges(char_segs)

    # 逐字时间戳按原文顺序建索引
    segs_by_char = {idx: (s[1], s[2]) for idx, s in enumerate(char_segs)}

    pos = 0
    words = []          # [(word, start, end), ...]
    for token in raw_tokens:
        n = len(token)
        if not token.strip():
            pos += n   # 空白 token 不输出，但位置要推进
            continue
        # 取 token 对应的字区间 [pos, pos+n)
        start = segs_by_char[pos][0] if pos in segs_by_char else 0.0
        end_pos = pos + n - 1
        end = segs_by_char[end_pos][1] if end_pos in segs_by_char else start

        if merge_punct and _is_pure_punct(token):
            # 纯标点并入前一个词：延长时间；仅保留问号/感叹号文本
            if words:
                pw, ps, pe = words[-1]
                keep = ''.join(ch for ch in token if ch in _TRAILING_KEEP)
                words[-1] = (pw + keep, ps, max(pe, end))
            pos += n
            continue

        words.append((token, start, end))
        pos += n

    return words


def split_subtitle_sentences(text, min_length=_SUBTITLE_MIN_LENGTH):
    """按逗号/句号/问号/感叹号等标点切分，再合并过短的相邻句。

    与 CosyVoice 合成用的 60~80 token 长句切分解耦：
    只用于字幕断句，不改变合成句段。

    Args:
        text: 待断句文本。
        min_length: 每条字幕的最短字符数（中文字符数近似）。相邻短句会被
            向后合并，保证每条不低于该长度。传 0 表示不做合并（全按标点切）。

    Returns:
        断句后的字符串列表（非空）。
    """
    # 1) 按标点初步切分
    parts = [p for p in _SUBTITLE_SENT_END.split(text) if p.strip()]
    # 中文全角引号如果孤悬在句尾，归并到前一句
    merged = []
    for p in parts:
        if merged and p in ('"', '"', '”', '「', '」'):
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)

    # 2) 向后合并过短句，保证每条 >= min_length
    if min_length <= 0:
        return [_trim_trailing_punct(p) for p in merged if p.strip()]

    result = []
    buf = ''
    for p in merged:
        buf += p
        if len(buf) >= min_length:
            result.append(buf.strip())
            buf = ''
    # 末尾剩余不足 min_length 的尾巴：并入上一条（若有），否则单独成条
    if buf.strip():
        if result:
            result[-1] = result[-1] + buf.strip()
        else:
            result.append(buf.strip())
    return [_trim_trailing_punct(r) for r in result if r]


def _len_zh(text: str) -> int:
    """粗略估算字符长度，用于按比例分配时长。"""
    return len(text)


# ═══════════════════════════════════════════════════════════════
#  SRT helpers
# ═══════════════════════════════════════════════════════════════

def _fmt_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    # clamp millis to [0, 999]
    millis = max(0, min(999, millis))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(segments: List[Dict]) -> str:
    """Build SRT formatted string from segment list.

    Each segment: {text: str, start: float (seconds), end: float (seconds)}
    """
    lines = []
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        lines.append(
            f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}"
        )
        lines.append(seg['text'])
        lines.append('')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  Asynchronous versions of the helpers (for webui streaming)
# ═══════════════════════════════════════════════════════════════

def _fmt_vtt_time(seconds: float) -> str:
    """VTT timestamp: HH:MM:SS.mmm (WebVTT format)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    millis = max(0, min(999, millis))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def build_vtt(segments: List[Dict]) -> str:
    """Build WebVTT formatted string from segment list."""
    lines = ['WEBVTT', '']
    for idx, seg in enumerate(segments, 1):
        lines.append(
            f"{_fmt_vtt_time(seg['start'])} --> {_fmt_vtt_time(seg['end'])}"
        )
        lines.append(seg['text'])
        lines.append('')
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  Wrapper
# ═══════════════════════════════════════════════════════════════

class CosyVoiceSRT:
    """Middleware wrapper that adds SRT generation to any CosyVoice instance.

    - Without `return_srt` / `srt_path`: behaves **identically** to the
      original CosyVoice (generator yields ``{'tts_speech': tensor}``).
    - With ``return_srt=True``: consumes the whole generator internally,
      returns ``(full_audio_tensor, srt_string)`` (requires ``stream=False``).
    - With ``srt_path='path.srt'``: same as above, also writes the .srt file.
    - Use ``return_subtitles='srt'`` / ``return_subtitles='vtt'`` to choose
      between SRT and WebVTT format.

    Additionally supports ``build_subtitles_only=True`` to produce subtitles
    **without synthesising audio** (uses only the frontend sentence split and
    a simulated placeholder duration). Useful for previewing subtitle timing.

    Also supports ``return_seq`` to return per-sentence audio chunks alongside
    per-sentence SRT entries instead of concatenated audio.
    """

    def __init__(self, cosyvoice):
        self._cv = cosyvoice
        self.sample_rate = getattr(cosyvoice, 'sample_rate', 24000)

    # ── transparent delegation ──

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._cv, name)

    # ── internal runner ──

    def _run_and_build(
        self, method_name: str, tts_text,
        return_srt: bool, srt_path: Optional[str],
        return_subtitles: Optional[str], build_subtitles_only: bool,
        return_seq: bool, subtitle_split: bool = True,
        subtitle_min_length: int = _SUBTITLE_MIN_LENGTH,
        asr_engine: Optional[str] = None,
        asr_model_id: Optional[str] = None,
        subtitle_text: Optional[str] = None,
        *args, **kwargs
    ):
        """Run a single inference method, collect segments, optionally build subtitles.

        Args:
            method_name: e.g. 'inference_sft'
            tts_text: passed to the inference method (string or generator)
            return_srt / srt_path / return_subtitles / build_subtitles_only / return_seq:
                described in the class docstring.
            subtitle_split: when True, further split each synthesis segment into
                sentence-level subtitles (by 。？！，, etc.) and apportion the
                segment's duration proportionally by character count. This only
                affects subtitles, NOT the synthesis prosody.
            subtitle_min_length: minimum character length per subtitle. Adjacent
                short sentences are merged backward so each subtitle is at least
                this long. Pass 0 to disable merging (split purely by punctuation).
            asr_engine: optional ASR engine ('funasr' or 'mlx-whisper') for
                word-level timestamps. When set, synthesised audio is aligned to
                tts_text via ASR and a word-level SRT is produced.
            asr_model_id: optional model id for mlx-whisper (default
                'mlx-community/whisper-large-v3-mlx').
            *args / **kwargs: forwarded to the inference method.

        Returns:
            When subtitles are not requested:
                The original generator.
            When build_subtitles_only is True:
                (None, subtitles_string)
            When return_seq is True:
                A generator yielding (audio_tensor, srt_entry_dict) per sentence.
            Otherwise:
                (full_audio_tensor, subtitles_string)
        """
        word_srt = asr_engine is not None
        want_subtitles = return_srt or bool(srt_path) or return_subtitles is not None or word_srt

        if not want_subtitles and not build_subtitles_only and not return_seq:
            # No SRT requested — delegate to original
            return getattr(self._cv, method_name)(tts_text, *args, **kwargs)

        # ── Pre-split text for SRT ──
        # subtitle_text（若提供）用于字幕断句（如剥除发音注解后的干净原文），
        # 合成仍使用 tts_text（含 hotfix 拼音）。两者标点一致时句段数一一对应。
        text_frontend = kwargs.pop('text_frontend', True)
        split_source = subtitle_text if subtitle_text is not None else tts_text
        if isinstance(split_source, str):
            texts = list(self._cv.frontend.text_normalize(
                split_source, split=True, text_frontend=text_frontend
            ))
        else:
            # tts_text is a generator; we can't pre-split
            texts = None

        # ── Build subtitles only (dry-run, no synthesis) ──
        if build_subtitles_only:
            segments = []
            if texts:
                dummy_duration = 2.0  # placeholder seconds per sentence
                offset = 0.0
                for t in texts:
                    subs = split_subtitle_sentences(t, subtitle_min_length) if subtitle_split else [t]
                    n = max(len(subs), 1)
                    per = dummy_duration / n
                    for s in subs:
                        segments.append({
                            'text': s,
                            'start': offset,
                            'end': offset + per,
                        })
                        offset += per
            return None, self._format_subtitles(segments, return_subtitles or 'srt')

        # ── Run inference and collect segments ──
        # Remove 'return_srt'/'srt_path'/'return_subtitles'/'build_subtitles_only'/'return_seq' from kwargs
        # since they may have been mixed in
        for k in ['return_srt', 'srt_path', 'return_subtitles', 'build_subtitles_only', 'return_seq']:
            kwargs.pop(k, None)

        gen = getattr(self._cv, method_name)(tts_text, *args, **kwargs)

        if return_seq:
            # Yield (audio, srt_entry) per sentence
            return self._yield_seq(gen, texts, subtitle_split, subtitle_min_length)

        # Collect all
        segments = []
        offset = 0.0
        audios = []

        for idx, result in enumerate(gen):
            audios.append(result['tts_speech'])
            duration = result['tts_speech'].shape[1] / self.sample_rate
            seg_text = (texts[idx] if texts and idx < len(texts)
                        else f'[{idx}]')
            if subtitle_split and texts:
                # 把该合成段按句子级标点拆成更短的字幕，并按字数比例分配时长
                subs = split_subtitle_sentences(seg_text, subtitle_min_length)
                if len(subs) > 1:
                    total = max(sum(_len_zh(s) for s in subs), 1)
                    acc = offset
                    for s in subs:
                        dur = duration * _len_zh(s) / total
                        segments.append({
                            'text': s,
                            'start': acc,
                            'end': acc + dur,
                        })
                        acc += dur
                    offset += duration
                    continue
            segments.append({
                'text': seg_text,
                'start': offset,
                'end': offset + duration,
            })
            offset += duration

        full_audio = torch.cat(audios, dim=1) if audios else torch.empty(1, 0)

        subtitle_format = (return_subtitles if return_subtitles is not None
                           else 'srt')

        if word_srt:
            # ── Word-level SRT via ASR alignment ──
            if not isinstance(tts_text, str):
                raise ValueError('word-level SRT requires tts_text to be a string, not a generator')
            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
                tmp_wav = tmpf.name
            try:
                import torchaudio
                torchaudio.save(tmp_wav, full_audio, self.sample_rate)
                word_segs = char_level_timestamps(
                    tts_text, tmp_wav, self.sample_rate,
                    engine=asr_engine, model_id=asr_model_id,
                )
            finally:
                if _os.path.exists(tmp_wav):
                    _os.remove(tmp_wav)
            # 逐字（可合并为词或保持逐字）构建 segment 列表
            segments = [{'text': c, 'start': s, 'end': e} for c, s, e in word_segs]
            subtitles_content = self._format_subtitles(segments, subtitle_format)
            if srt_path:
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(subtitles_content)
            return full_audio, subtitles_content

        subtitles_content = self._format_subtitles(segments, subtitle_format)

        if srt_path:
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(subtitles_content)

        return full_audio, subtitles_content

    # ── per-mode wrappers ──

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0,
                      text_frontend=True,
                      return_srt=False, srt_path=None,
                      return_subtitles: Optional[str] = None,
                      build_subtitles_only: bool = False,
                      return_seq: bool = False,
                      subtitle_split: bool = True,
                      subtitle_min_length: int = _SUBTITLE_MIN_LENGTH,
                      asr_engine: Optional[str] = None,
                      asr_model_id: Optional[str] = None):
        if return_srt or srt_path or return_subtitles or build_subtitles_only or return_seq or asr_engine:
            assert not stream, 'return_srt requires stream=False; use return_seq for streaming access'
        return self._run_and_build(
            'inference_sft', tts_text, return_srt, srt_path,
            return_subtitles, build_subtitles_only, return_seq,
            subtitle_split, subtitle_min_length,
            asr_engine, asr_model_id,
            spk_id, stream=stream, speed=speed, text_frontend=text_frontend
        )

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav,
                            zero_shot_spk_id='', stream=False, speed=1.0,
                            text_frontend=True,
                            return_srt=False, srt_path=None,
                            return_subtitles: Optional[str] = None,
                            build_subtitles_only: bool = False,
                            return_seq: bool = False,
                            subtitle_split: bool = True,
                            subtitle_min_length: int = _SUBTITLE_MIN_LENGTH,
                            asr_engine: Optional[str] = None,
                            asr_model_id: Optional[str] = None,
                            subtitle_text: Optional[str] = None):
        if return_srt or srt_path or return_subtitles or build_subtitles_only or return_seq or asr_engine:
            assert not stream, 'return_srt requires stream=False; use return_seq for streaming access'
        return self._run_and_build(
            'inference_zero_shot', tts_text, return_srt, srt_path,
            return_subtitles, build_subtitles_only, return_seq,
            subtitle_split, subtitle_min_length,
            asr_engine, asr_model_id, subtitle_text,
            prompt_text, prompt_wav,
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream, speed=speed, text_frontend=text_frontend
        )

    def inference_cross_lingual(self, tts_text, prompt_wav,
                                zero_shot_spk_id='', stream=False, speed=1.0,
                                text_frontend=True,
                                return_srt=False, srt_path=None,
                                return_subtitles: Optional[str] = None,
                                build_subtitles_only: bool = False,
                                return_seq: bool = False,
                                subtitle_split: bool = True,
                                subtitle_min_length: int = _SUBTITLE_MIN_LENGTH,
                                asr_engine: Optional[str] = None,
                                asr_model_id: Optional[str] = None):
        if return_srt or srt_path or return_subtitles or build_subtitles_only or return_seq or asr_engine:
            assert not stream, 'return_srt requires stream=False; use return_seq for streaming access'
        return self._run_and_build(
            'inference_cross_lingual', tts_text, return_srt, srt_path,
            return_subtitles, build_subtitles_only, return_seq,
            subtitle_split, subtitle_min_length,
            asr_engine, asr_model_id,
            prompt_wav,
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream, speed=speed, text_frontend=text_frontend
        )

    def inference_instruct(self, tts_text, spk_id, instruct_text,
                           stream=False, speed=1.0, text_frontend=True,
                           return_srt=False, srt_path=None,
                           return_subtitles: Optional[str] = None,
                           build_subtitles_only: bool = False,
                           return_seq: bool = False,
                           subtitle_split: bool = True,
                           subtitle_min_length: int = _SUBTITLE_MIN_LENGTH,
                           asr_engine: Optional[str] = None,
                           asr_model_id: Optional[str] = None):
        if return_srt or srt_path or return_subtitles or build_subtitles_only or return_seq or asr_engine:
            assert not stream, 'return_srt requires stream=False; use return_seq for streaming access'
        return self._run_and_build(
            'inference_instruct', tts_text, return_srt, srt_path,
            return_subtitles, build_subtitles_only, return_seq,
            subtitle_split, subtitle_min_length,
            asr_engine, asr_model_id,
            spk_id, instruct_text,
            stream=stream, speed=speed, text_frontend=text_frontend
        )

    def inference_instruct2(self, tts_text, instruct_text, prompt_wav,
                            zero_shot_spk_id='', stream=False, speed=1.0,
                            text_frontend=True,
                            return_srt=False, srt_path=None,
                            return_subtitles: Optional[str] = None,
                            build_subtitles_only: bool = False,
                            return_seq: bool = False,
                            subtitle_split: bool = True,
                            subtitle_min_length: int = _SUBTITLE_MIN_LENGTH,
                            asr_engine: Optional[str] = None,
                            asr_model_id: Optional[str] = None):
        if return_srt or srt_path or return_subtitles or build_subtitles_only or return_seq or asr_engine:
            assert not stream, 'return_srt requires stream=False; use return_seq for streaming access'
        return self._run_and_build(
            'inference_instruct2', tts_text, return_srt, srt_path,
            return_subtitles, build_subtitles_only, return_seq,
            subtitle_split, subtitle_min_length,
            asr_engine, asr_model_id,
            instruct_text, prompt_wav,
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream, speed=speed, text_frontend=text_frontend
        )

    # ── internal helpers ──

    def _format_subtitles(self, segments: List[Dict], fmt: str) -> str:
        if fmt == 'vtt':
            return build_vtt(segments)
        return build_srt(segments)

    def _yield_seq(self, gen, texts, subtitle_split=True,
                   subtitle_min_length: int = _SUBTITLE_MIN_LENGTH):
        """Yields (audio_chunk, srt_entry_dict) per sentence."""
        offset = 0.0
        counter = 0
        for idx, result in enumerate(gen):
            audio = result['tts_speech']
            duration = audio.shape[1] / self.sample_rate
            seg_text = (texts[idx] if texts and idx < len(texts)
                        else f'[{idx}]')
            if subtitle_split and texts:
                subs = split_subtitle_sentences(seg_text, subtitle_min_length)
                if len(subs) > 1:
                    total = max(sum(_len_zh(s) for s in subs), 1)
                    acc = offset
                    for s in subs:
                        dur = duration * _len_zh(s) / total
                        counter += 1
                        yield (audio, {
                            'index': counter,
                            'text': s,
                            'start': acc,
                            'end': acc + dur,
                            'duration': dur,
                        })
                        acc += dur
                    offset += duration
                    continue
            counter += 1
            entry = {
                'index': counter,
                'text': seg_text,
                'start': offset,
                'end': offset + duration,
                'duration': duration,
            }
            offset += duration
            yield (audio, entry)
