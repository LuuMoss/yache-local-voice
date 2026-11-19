# -*- coding: utf-8 -*-
"""voice.py — model-family scanner + voice registry helpers (stdlib only).

Single source of truth for: which variants/quantizations exist in a models
directory, the built-in speaker list of a CustomVoice model, quant/device
recommendation, and the v2 voice-registry format. Imported by both the
gateway half (dashboard/api.py) and the worker half (worker.py), so it must
stay torch-free and dependency-free.

Design doc: docs live beside the plugin; see design-voice-tts-v2 (section 5).
"""
from __future__ import annotations

# Bilingual user-facing strings: three-mode load like sibling modules; a
# standalone run outside the plugin tree keeps the Chinese original.
try:
    from . import i18n as _i18n                   # package mode
except (ImportError, SystemError):
    try:
        import i18n as _i18n                      # bare-module mode
    except ImportError:
        try:
            import importlib.util as _ilu
            from pathlib import Path as _P
            _sp = _ilu.spec_from_file_location(
                'yache_voice_i18n', str(_P(__file__).with_name('i18n.py')))
            _i18n = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_i18n)         # external mode
        except Exception:
            class _i18n:                          # standalone fallback: zh only
                @staticmethod
                def ui(zh, en):
                    return zh


import json
import os
import re
import struct
from pathlib import Path

# ---------------------------------------------------------------- GGUF header

# Official GGUF type table (gguf.spec): 0..7 primitives, 8 STRING, 9 ARRAY,
# 10..12 primitives, 13..22 typed arrays. Getting 8/9 the right way round is
# everything — misreading STRING as ARRAY desyncs every later offset.
_T_FMT = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f',
          7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
_ARR_ELEM = {13: 0, 14: 1, 15: 2, 16: 3, 17: 4, 18: 5, 19: 6, 20: 7,
             21: 8, 22: 9}   # typed-array type -> element type
_GGUF_STR = 8
_MAX_STR = 8_000_000
_MAX_ARR = 8_000_000


class _Bail(Exception):
    pass


def _rstr(f):
    n = struct.unpack('<Q', f.read(8))[0]
    if n > _MAX_STR:
        raise _Bail('huge string')
    return f.read(n).decode('utf-8', 'replace')


def _rval(f, t, depth=0):
    if t == _GGUF_STR:
        return _rstr(f)
    if t == 9:
        elem = struct.unpack('<I', f.read(4))[0]
    elif t in _ARR_ELEM:
        elem = _ARR_ELEM[t]
    else:
        fmt = _T_FMT.get(t)
        if fmt is None:
            raise _Bail('unknown type %d' % t)
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    # here: a real array — read small runs fully (speaker tables are tiny and
    # we need their real values), skip gigantic token tables by seeking so the
    # header walk stays fast and offset-safe on every file
    n = struct.unpack('<Q', f.read(8))[0]
    if n > _MAX_ARR:
        raise _Bail('huge array')
    if elem in _T_FMT:
        fmt = _T_FMT[elem]
        if n <= 4096:
            sz = struct.calcsize(fmt)
            return [struct.unpack(fmt, f.read(sz))[0] for _ in range(n)]
        f.seek(n * struct.calcsize(fmt), 1)
        return ['?'] * 8
    if depth > 3:
        raise _Bail('deep array')
    if elem == _GGUF_STR:
        return [_rstr(f) for _ in range(n)]
    if elem == 9:
        return [_rval(f, 9, depth + 1) for _ in range(n)]
    raise _Bail('weird elem %d' % elem)


def read_gguf_kvs(path, want=None):
    """Parse GGUF header KV pairs. `want` = set of keys to keep (still walks
    the whole header so offsets stay sane). Returns {} on unreadable file."""
    out = {}
    try:
        with open(path, 'rb') as f:
            if f.read(4) != b'GGUF':
                return {}
            struct.unpack('<I', f.read(4))[0]          # version
            struct.unpack('<Q', f.read(8))[0]          # n_tensors
            try:
                nkv = struct.unpack('<Q', f.read(8))[0]
                for _ in range(nkv):
                    k = _rstr(f)
                    t = struct.unpack('<I', f.read(4))[0]
                    v = _rval(f, t)
                    if want is None or k in want:
                        out[k] = v
                    if want is not None and want <= out.keys():
                        break
            except (_Bail, struct.error, EOFError):
                pass
    except OSError:
        return {}
    return out


# ------------------------------------------------------------------- scanning

# GGUF metadata key written by the Serveurperso converter; authoritative
# variant id (measured on this disk: base=base, custom=custom_voice,
# design=voice_design). Values with underscores, key with dots.
_MODEL_TYPE_KEY = 'qwen3-tts.model_type'
_MODEL_TYPE_TO_VARIANT = {
    'base': 'base',
    'custom_voice': 'customvoice',
    'voice_design': 'voicedesign',
}
VARIANT_ORDER = ('base', 'customvoice', 'voicedesign')

_TOK_RE = re.compile(r'^qwen-tokenizer-12hz-.*\.gguf$', re.I)
# filename pre-filter: any name containing the variant token and a quant tail
_PRE_RE = re.compile(r'(base|customvoice|voicedesign)', re.I)
_QUANT_RE = re.compile(r'[-_](Q[\dA-Za-z_]+|BF16|F16)\.gguf$', re.I)

# Third dimension: the model SIZE, read from the file name. GGUF metadata
# carries no size key (measured), so this is the ONE label the scanner takes
# from the naming convention the converter enforces
# (qwen-talker-<size>-<variant>-<QUANT>.gguf). Names without a size token
# bucket as '' and count as the big size. Without this, 0.6b and 1.7b
# files sharing a quant label collided in the quants list and one size
# was silently invisible.
_SIZE_RE = re.compile(r'^qwen-talker-(\d+(?:\.\d+)?)b-', re.I)
# measured on this 3090: 1.7b Q8 weights ~2.0 GB, 0.6b Q8 ~1.2 GB
_SIZE_VRAM_FACTOR = {'1.7b': 1.0, '0.6b': 0.5}


def _norm_size(fname: str) -> str:
    m = _SIZE_RE.match(str(fname))
    if not m:
        return ''
    try:
        return '%gb' % float(m.group(1))
    except ValueError:
        return ''

_SPEAKER_KEYS = {'qwen3-tts.codec.speaker_names', 'qwen3-tts.codec.speaker_ids',
                 'qwen3-tts.codec.speaker_dialects', _MODEL_TYPE_KEY}


class ScanError(Exception):
    """Human-readable scan failure (surfaced verbatim by the dashboard)."""


def _find_models_dir(root: str):
    """Pick the directory that actually holds the gguf files: root itself or
    root/models (users point at either one)."""
    p = Path(root)
    if not p.exists():
        raise ScanError(_i18n.ui('目录不存在: %s', 'directory does not exist: %s') % root)
    if not p.is_dir():
        raise ScanError(_i18n.ui('不是目录: %s', 'not a directory: %s') % root)
    for cand in (p, p / 'models'):
        try:
            if any(f.suffix.lower() == '.gguf' for f in cand.glob('*.gguf')):
                return cand
        except OSError:
            continue
    raise ScanError(_i18n.ui(
        '目录里没有 .gguf 模型文件（找过 %s 和 %s）。请把 Qwen3-TTS 的 talker/tokenizer gguf 放进 models\\ 子目录。',
        'no .gguf model files in this directory (looked in %s and %s). Put the Qwen3-TTS '
        'talker/tokenizer gguf files into the models\\ subdirectory.') % (p, p / 'models'))


def scan_models_dir(root: str) -> dict:
    """Scan a models directory; returns dict per design doc §5.2:

    {dir, tokenizer: {files, preferred}, variants: [ {variant, model_type,
     quants: [{quant, file, size_mib}], speakers: [{name, langs, ids}],
     dialect_pairs: [...] } ], skipped: [...]}

    Raises ScanError with a human-readable reason on any hard failure.
    """
    mdir = _find_models_dir(root)
    toks = sorted(f.name for f in mdir.glob('*.gguf') if _TOK_RE.match(f.name))
    if not toks:
        raise ScanError(_i18n.ui(
            '缺少分词器 qwen-tokenizer-12hz-*.gguf —— 三种变体共用它，没有它什么音色都造不出来。请把它放进 %s。',
            'the tokenizer qwen-tokenizer-12hz-*.gguf is missing — all three variants share it '
            'and nothing can be voiced without it. Put it into %s.') % mdir)

    variants = {}   # variant -> {model_type, quants: [...]}
    skipped = []
    for f in sorted(mdir.glob('*.gguf')):
        if _TOK_RE.match(f.name):
            continue
        m = _PRE_RE.search(f.name)
        if not m:
            continue
        if not _QUANT_RE.search(f.name):
            skipped.append(f.name + _i18n.ui('（文件名末尾没有量化标记）', ' (no quant tag at the end of the file name)'))
            continue
        kvs = read_gguf_kvs(str(f), want={_MODEL_TYPE_KEY})
        mt = kvs.get(_MODEL_TYPE_KEY)
        if mt not in _MODEL_TYPE_TO_VARIANT:
            # GGUF unreadable (half-downloaded?) or foreign model
            skipped.append(f.name + (_i18n.ui('（读不到模型类型，可能没下完）', ' (model type unreadable; the download may be incomplete)')
                                     if not kvs else
                                     _i18n.ui('（模型类型 %r 不认识）', ' (unknown model type %r)') % mt))
            continue
        variant = _MODEL_TYPE_TO_VARIANT[mt]
        if variant != m.group(1).lower():
            skipped.append(_i18n.ui('%s：文件名写的是 %s，但 GGUF 内部声明是 %s，已按 GGUF 为准',
                                    '%s: the file name says %s but the GGUF header declares %s; trusting the GGUF')
                           % (f.name, m.group(1).lower(), variant))
        qm = _QUANT_RE.search(f.name)
        quant = qm.group(1).upper()
        try:
            size_mib = round(f.stat().st_size / (1024 * 1024))
        except OSError:
            size_mib = 0
        ent = variants.setdefault(variant, {'model_type': mt, 'quants': []})
        size = _norm_size(f.name)
        # dedup on (quant, size): the old quant-only key let a 0.6b and a
        # 1.7b Q4_K_M collapse into ONE row — in mixed directories an entire
        # model size went invisible in the UI, and which build survived was
        # pure readdir luck. The two sizes are different models; keep both.
        if not any(q['quant'] == quant and q.get('model_size', '') == size
                   for q in ent['quants']):
            ent['quants'].append({'quant': quant, 'file': f.name,
                                  'model_size': size,
                                  'size_mib': size_mib})

    if not variants:
        raise ScanError(_i18n.ui(
            '在 %s 里没找到任何 Qwen3-TTS 变体（应类似 '
            'qwen-talker-1.7b-base-Q8_0.gguf / '
            'qwen-talker-1.7b-customvoice-Q4_K_M.gguf / '
            'qwen-talker-1.7b-voicedesign-Q8_0.gguf）。',
            'no Qwen3-TTS variant found under %s (expected names like '
            'qwen-talker-1.7b-base-Q8_0.gguf / '
            'qwen-talker-1.7b-customvoice-Q4_K_M.gguf / '
            'qwen-talker-1.7b-voicedesign-Q8_0.gguf).') % mdir)

    out_variants = []
    for v in VARIANT_ORDER:
        if v not in variants:
            continue
        ent = variants[v]
        # order = size first (big before small; unpinned loads then keep the
        # historical "largest file wins" default), quant quality second
        ent['quants'].sort(key=lambda q: (
            0 if (q.get('model_size') or '1.7b') == '1.7b' else 1,
            ('Q8_0', 'Q4_K_M', 'BF16').index(q['quant'])
            if q['quant'] in ('Q8_0', 'Q4_K_M', 'BF16') else 9))
        item = {'variant': v, 'model_type': ent['model_type'],
                'quants': ent['quants']}
        if v == 'customvoice':
            # built-in voice list lives in the GGUF metadata; read it from the
            # smallest existing quant file of the BIG size (same table in
            # every quant; the big build ships the complete speaker set)
            big = [q for q in ent['quants']
                   if (q.get('model_size') or '1.7b') == '1.7b'] or ent['quants']
            q0 = min(big, key=lambda q: q['size_mib'])
            kvs = read_gguf_kvs(str(mdir / q0['file']), want=_SPEAKER_KEYS)
            names = kvs.get('qwen3-tts.codec.speaker_names') or []
            ids = kvs.get('qwen3-tts.codec.speaker_ids') or []
            dia = kvs.get('qwen3-tts.codec.speaker_dialects') or []
            item['speakers'] = [
                {'name': n,
                 'id': ids[i] if i < len(ids) else None,
                 'dialect': dia[i] if i < len(dia) else ''}
                for i, n in enumerate(names)]
        out_variants.append(item)

    pref = next((t for t in toks
                 if t.lower().endswith('q8_0.gguf')), toks[0])
    return {'dir': str(mdir),
            'tokenizer': {'files': toks, 'preferred': pref},
            'variants': out_variants,
            'skipped': skipped}


# ---------------------------------------------------------------- recommendation

# Approximate VRAM footprint per quant (weights + activations headroom), MiB.
# Measured on this 3090: Q8 weights 2.0 GB -> ~1500 MiB VRAM; Q4 -> ~1100 MiB.
_QUANT_VRAM_MIB = {'Q8_0': 1900, 'Q4_K_M': 1400, 'BF16': 4300}


def recommend(quants, free_mib, has_cuda: bool) -> dict:
    """Return {"quant", "model_size", "device", "reason"} — a suggestion the
    UI pre-selects but the user may override (design doc §5.5). quants = the
    quant list present on disk for the chosen variant: scanner entries
    ({'quant', 'model_size', ...}) or plain quant names (read as the big
    size). Size is part of the call — a 0.6b Q8 costs about
    half the VRAM of a 1.7b Q8, so when the card cannot host the big build
    the honest auto answer is the same tier of the small build."""
    pairs = []  # unique (quant, size); scanner order = big size, then quality
    seen = set()
    for q in quants or []:
        name = str((q.get('quant') if isinstance(q, dict) else q) or '').upper()
        size = (str((q.get('model_size') if isinstance(q, dict) else '')
                    or '').lower() or '1.7b')
        if name and (name, size) not in seen:
            seen.add((name, size))
            pairs.append((name, size))

    def _vram(p):
        return _QUANT_VRAM_MIB.get(p[0], 9999) * _SIZE_VRAM_FACTOR.get(p[1], 1.0)

    fits = [p for p in pairs
            if _vram(p) <= (free_mib if free_mib is not None else 10 ** 9)]

    def _best(cands):
        for p in fits:            # scanner order keeps the big size first
            if p[0] in cands:
                return p
        return None

    def _light(cands):
        c = [p for p in pairs if p[0] in cands]
        return min(c, key=_vram) if c else None

    if not has_cuda or free_mib is None:
        p = _light({'Q4_K_M'}) or _light(set(_QUANT_VRAM_MIB))
        q = p[0] if p else None
        return {'quant': q, 'model_size': (p[1] if p else ''), 'device': 'cpu',
                'reason': _i18n.ui('没检测到可用 GPU，建议 CPU 跑%s', 'no usable GPU detected; CPU run recommended%s') % (
                    _i18n.ui('（%s 档 CPU 实测很流畅）', ' (%s runs smoothly on CPU in our tests)') % q if q else '')}
    if free_mib >= 3800:
        p = _best(('Q8_0', 'BF16', 'Q4_K_M'))
        if p:
            cand, ms = p
            extra = (_i18n.ui('，Q8 最清晰', ', Q8 sounds the clearest') if cand == 'Q8_0' else '')
            return {'quant': cand, 'model_size': ms, 'device': 'cuda',
                    'reason': _i18n.ui('空闲显存 %d MiB 很充裕，建议 GPU + %s%s',
                                       'free VRAM %d MiB is plenty; GPU + %s%s recommended')
                              % (free_mib, cand, extra)}
    if any(n == 'Q4_K_M' for n, _ in pairs):
        p = _best({'Q4_K_M'}) or next((x for x in pairs if x[0] == 'Q4_K_M'), None)
        return {'quant': 'Q4_K_M', 'model_size': (p[1] if p else ''), 'device': 'cuda',
                'reason': _i18n.ui('空闲显存 %d MiB，Q8 会挤爆，建议 GPU + Q4_K_M（约 1.1G，可与聊天模型共存）',
                                    'free VRAM %d MiB is too tight for Q8; GPU + Q4_K_M recommended '
                                    '(~1.1G, coexists with the chat model)') % free_mib}
    return {'quant': (sorted({n for n, _ in pairs})[0] if pairs else None),
            'model_size': '', 'device': 'cpu',
            'reason': _i18n.ui('空闲显存 %d MiB 且该变体没有 Q4 档，建议改 CPU 跑',
                                'free VRAM %d MiB and this variant has no Q4 quant; CPU run recommended')
                      % free_mib}


# ------------------------------------------------------------- voice registry v2

KINDS = ('clone', 'speaker', 'design')
VARIANTS = VARIANT_ORDER


def check_quant(quant, avail):
    """Return the entry from `avail` that matches `quant`
    (case-insensitive), or None when that quantization file is not on
    disk.  Used by the synthesis-config save to fail fast with a
    readable "which quants DO exist" message instead of a worker crash.
    `avail` rows may be quant names or scanner entries ({'quant', ...});
    a matched scanner entry is returned whole so callers can see which
    model size(s) actually carry that tier."""
    q = str(quant or '').strip().upper()
    for a in avail or []:
        if isinstance(a, dict):
            if str(a.get('quant') or '').strip().upper() == q:
                return a
        elif str(a).strip().upper() == q:
            return a
    return None



def normalize_entry(name: str, entry: dict) -> dict:
    """Return an entry with all v2 fields filled (older registry entries
    keep behaving exactly as before: kind=clone + default dir + auto quant/device)."""
    e = dict(entry or {})
    e.setdefault('kind', 'clone')
    e.setdefault('variant', 'base')
    e.setdefault('models_dir', None)     # None -> worker uses default base dir
    e.setdefault('quant', None)          # None -> auto by free VRAM
    e.setdefault('device', None)         # None -> auto (VRAM gate)
    if 'ref_text' not in e and e.get('ref_text_file'):
        try:
            e['ref_text'] = Path(str(e['ref_text_file'])
                                 ).read_text(encoding='utf-8').strip()
        except OSError:
            e['ref_text'] = ''
    return e


def load_registry(path: str) -> dict:
    """{default, voices: {name: normalized entry}}; {} when absent."""
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    voices = {n: normalize_entry(n, v)
              for n, v in (raw.get('voices') or {}).items()}
    return {'default': raw.get('default') or (next(iter(voices), None)),
            'voices': voices, 'raw': raw}


def save_registry(path: str, reg: dict, backup: bool = True) -> None:
    """Atomic write; keeps unknown top-level keys from `raw` untouched.
    Caller must have read the file first (load_registry keeps raw around)."""
    p = Path(path)
    out = dict(reg.get('raw') or {})
    out['default'] = reg.get('default')
    out['voices'] = {n: v for n, v in (reg.get('voices') or {}).items()}
    tmp = p.with_name(p.name + '.tmp')
    if backup and p.exists():
        try:
            bak = p.with_name(p.name + '.bak')
            bak.write_bytes(p.read_bytes())
        except OSError:
            pass
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    os.replace(tmp, p)
