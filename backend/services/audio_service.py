"""Audio analysis: Whisper transcription with segment timestamps, transcript
parsing, and extraction of forensic Who / Where / When facts."""
import re
from pathlib import Path

from flask import current_app

from ..utils import groq_client
from ..utils.logger import get_logger
from ..utils.model_loader import get_device, load_model

log = get_logger(__name__)

LANG_CODES = {"en": "english", "hi": "hindi", "te": "telugu", "fr": "french"}


class AudioEngineUnavailable(RuntimeError):
    """Raised when no speech-to-text engine could be loaded."""


# ------------------------------------------------------------ transcription
def _load_hf_asr():
    def loader():
        from transformers import pipeline

        return pipeline(
            "automatic-speech-recognition",
            model=current_app.config["WHISPER_MODEL"],
            device=0 if get_device() == "cuda" else -1,
            chunk_length_s=30,
        )

    return load_model("whisper-hf", loader)


def _load_whisper_pkg():
    def loader():
        import whisper

        return whisper.load_model("base", device=get_device())

    return load_model("whisper-pkg", loader)


def transcribe_audio(path, language=None):
    """Return (segments, engine). Segments: [{start, end, text}] in seconds."""
    language = LANG_CODES.get((language or "").lower(), language) or None
    pipe = _load_hf_asr()
    if pipe:
        try:
            import librosa

            audio, sr = librosa.load(str(path), sr=16000, mono=True)
            kwargs = {"task": "transcribe"}
            if language:
                kwargs["language"] = language
            out = pipe({"raw": audio, "sampling_rate": sr}, return_timestamps=True, generate_kwargs=kwargs)
            duration = len(audio) / sr
            segments = []
            for ch in out.get("chunks", []):
                start, end = ch["timestamp"]
                start = float(start or 0.0)
                end = float(end if end is not None else min(duration, start + 5))
                text = ch["text"].strip()
                if text:
                    segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})
            if not segments and out.get("text"):
                segments = [{"start": 0.0, "end": round(duration, 2), "text": out["text"].strip()}]
            return segments, "whisper-hf"
        except Exception as exc:
            log.warning("HF Whisper failed: %s", exc)
    model = _load_whisper_pkg()
    if model:
        try:
            res = model.transcribe(str(path), language=(language[:2] if language else None), fp16=False)
            segs = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()} for s in res["segments"] if s["text"].strip()]
            return segs, "whisper-pkg"
        except Exception as exc:
            log.warning("openai-whisper failed: %s", exc)
    raise AudioEngineUnavailable(
        "No speech-to-text engine is available. Install the Whisper dependencies "
        "or upload a text transcript (.txt/.srt/.vtt) instead."
    )


# ------------------------------------------------------ transcript parsing
_TS = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
_SRT_RE = re.compile(rf"{_TS}\s*-->\s*{_TS}")
_LINE_RE = re.compile(rf"^\s*[\[(]?{_TS}[\])]?\s*[-–:]?\s*(.+)$")


def _sec(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + (int(ms.ljust(3, "0")) / 1000 if ms else 0)


def parse_transcript_text(text: str):
    """Parse SRT/VTT cues, '[MM:SS] text' lines, or plain text (4 s per line)."""
    segments = []
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    i = 0
    while i < len(lines):
        m = _SRT_RE.search(lines[i])
        if m:
            g = m.groups()
            start, end = _sec(*g[0:4]), _sec(*g[4:8])
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i].strip())
                i += 1
            if buf:
                segments.append({"start": start, "end": end, "text": " ".join(buf)})
        i += 1
    if segments:
        return segments
    plain_t = 0.0
    for line in lines:
        if not line.strip() or line.strip().upper() == "WEBVTT":
            continue
        m = _LINE_RE.match(line)
        if m:
            start = _sec(*m.groups()[0:4])
            segments.append({"start": start, "end": start + 4.0, "text": m.group(5).strip()})
            plain_t = start + 4.0
        else:
            segments.append({"start": plain_t, "end": plain_t + 4.0, "text": line.strip()})
            plain_t += 4.0
    for a, b in zip(segments, segments[1:]):  # close gaps up to the next cue
        a["end"] = min(a["end"], b["start"]) if b["start"] > a["start"] else a["end"]
    return [{**s, "start": round(s["start"], 2), "end": round(s["end"], 2)} for s in segments]


# --------------------------------------------------------- forensic facts
_PLACE_WORDS = r"parking lot|alley|store|shop|market|station|street|road|house|bank|park|bar|restaurant|building|apartment|highway|bridge|mall|school|office"
_WHO = [
    re.compile(r"(?i:my name is|i am|i'm|this is|officer|detective|witness)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"),
    re.compile(r"\b((?:a|the|two|three|four|one)\s+(?:tall |short |young |old |heavy-?set |thin )?(?:man|men|woman|women|boy|girl|guy|guys|suspects?|person|people)(?:\s+(?:in|wearing|with)\s+(?:a|an|the)?\s*[a-z ]{3,25}?)?)(?=[,.;]|\s+(?:pulled|ran|was|were|and|who|then)\b|$)", re.I),
]
_WHERE = [
    re.compile(r"(?i:\b(?:at|in|near|outside|inside|behind|on|toward|towards)\s+(?:the\s+)?)([A-Z][\w']+(?:\s[A-Z][\w']+){0,3})"),
    re.compile(rf"\b((?:the\s+)?(?:[a-z]+\s)?(?:{_PLACE_WORDS}))\b", re.I),
]
_WHEN = [
    re.compile(r"\b(\d{1,2}[:.]\d{2}\s?(?:am|pm|a\.m\.|p\.m\.)?)", re.I),
    re.compile(r"\b(last night|this morning|yesterday|tonight|this afternoon|around midnight|at night|\d+\s+(?:minutes|hours|days)\s+ago)\b", re.I),
    re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?)\b", re.I),
]


def _dedupe(items, limit=12):
    """Drop repeats, including 'the X' vs 'X' and items contained in a longer one."""
    out, keys = [], []
    for it in items:
        it = re.sub(r"\s+", " ", it).strip(" ,.;")
        key = re.sub(r"^(?:the|a|an|toward|towards)\s+", "", it.lower())
        if len(key) <= 2 or any(key in k or k in key for k in keys):
            continue
        keys.append(key)
        out.append(it)
    return out[:limit]


def extract_facts_regex(text: str) -> dict:
    return {
        "who": _dedupe([m.group(1) for p in _WHO for m in p.finditer(text)]),
        "where": _dedupe([m.group(1) for p in _WHERE for m in p.finditer(text)]),
        "when": _dedupe([m.group(1) for p in _WHEN for m in p.finditer(text)]),
        "engine": "regex",
    }


def extract_facts(text: str) -> dict:
    regex_facts = extract_facts_regex(text)
    reply = groq_client.chat(
        "You extract forensic facts from testimony. Return JSON with keys who, where, when, each a list of short "
        "strings copied or tightly paraphrased from the text. Use only what is stated. Empty list if absent.",
        text[:6000],
        temperature=0.1,
        json_mode=True,
    )
    data = groq_client.parse_json(reply)
    if isinstance(data, dict):
        out = {k: _dedupe([str(v) for v in (data.get(k) or [])]) for k in ("who", "where", "when")}
        if any(out.values()):
            out["engine"] = "groq"
            return out
    return regex_facts


# ------------------------------------------------------------------ driver
def analyze_audio_file(path, kind: str, evidence_id: str, language=None, offset=0.0) -> dict:
    path = Path(path)
    if kind == "transcript":
        segments = parse_transcript_text(path.read_text(encoding="utf-8", errors="replace"))
        engine = "transcript-file"
    else:
        segments, engine = transcribe_audio(path, language)
    offset = float(offset or 0.0)
    for s in segments:
        s["start"] = round(s["start"] + offset, 2)
        s["end"] = round(s["end"] + offset, 2)
    full_text = " ".join(s["text"] for s in segments)
    return {
        "evidence_id": evidence_id,
        "file": path.name,
        "kind": kind,
        "language": language or "auto",
        "segments": segments,
        "full_text": full_text,
        "duration": segments[-1]["end"] if segments else 0.0,
        "facts": extract_facts(full_text) if full_text else {"who": [], "where": [], "when": [], "engine": "none"},
        "engine": engine,
    }
