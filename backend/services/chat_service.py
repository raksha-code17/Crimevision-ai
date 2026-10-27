"""Evidence-grounded Q&A with persistent sessions and MM:SS temporal parsing."""
import re

from ..utils import groq_client
from ..utils.session_store import get_store
from .fusion_service import fmt_ts

MMSS_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
SECS_RE = re.compile(r"\b(\d{1,4})\s*(?:s|sec|secs|second|seconds)\b", re.I)
STOP = set("the a an of to in on at is was were and or for with what who where when how did do does it this that "
           "happened happen there any about from by be been are as his her they their them".split())
TIME_WINDOW = 2.0

SYSTEM_PROMPT = (
    "You are the CrimeVision forensic assistant. Answer ONLY from the EVIDENCE below. Cite timestamps as [MM:SS] "
    "when you use timed evidence. If the evidence does not contain the answer, say exactly: 'The evidence does not "
    "say.' Do not speculate, do not accuse anyone, and keep answers under 120 words."
)


def parse_times(question: str) -> list:
    times = [int(m) * 60 + int(s) for m, s in MMSS_RE.findall(question) if int(s) < 60]
    times += [int(n) for n in SECS_RE.findall(question)]
    return times


def _tokens(text):
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def evidence_chunks(session) -> list:
    """Flatten a processed session into citable chunks."""
    fusion = session.get("fusion") or {}
    chunks = []
    for item in fusion.get("timeline", []):
        chunks.append({"t": item["t"], "end": item.get("end"), "source": item["source"], "text": item["text"]})
    for v in session.get("results", {}).get("video", []):
        if v.get("narrative"):
            chunks.append({"t": None, "source": "video", "text": f"Video {v['evidence_id']} narrative: {v['narrative']}"})
    for im in session.get("results", {}).get("images", []):
        if im.get("tags"):
            chunks.append({"t": None, "source": "image", "text": f"Image {im['evidence_id']} possible indicators: " + ", ".join(f"{t['label']} ({int(t['score'] * 100)}%)" for t in im["tags"])})
    facts = fusion.get("facts") or {}
    for k in ("who", "where", "when"):
        if facts.get(k):
            chunks.append({"t": None, "source": "facts", "text": f"{k.capitalize()}: " + "; ".join(facts[k])})
    if fusion.get("crime_type"):
        c = fusion["crime_type"]
        chunks.append({"t": None, "source": "classification", "text": f"Incident classified as {c['type']}. {c['rationale']}"})
    cons = fusion.get("consistency")
    if cons:
        chunks.append({"t": None, "source": "consistency", "text": f"Consistency score {cons['score']}. {cons['summary']}"})
        for c in cons["claims"]:
            chunks.append({"t": c["t"], "source": "consistency", "text": f"Statement '{c['statement']}' is {c['status']}: {c['evidence']}"})
    for h in (fusion.get("hypotheses") or {}).get("items", []):
        chunks.append({"t": None, "source": "hypothesis", "text": f"{h['title']} ({int(h['confidence'] * 100)}%): {h['narrative']}"})
    return chunks


def retrieve(question: str, chunks: list, limit=8) -> list:
    times = parse_times(question)
    if times:
        def distance(c):
            if c["t"] is None:
                return None
            d = min(abs(c["t"] - t) for t in times)
            if c.get("end") and any(c["t"] <= t < c["end"] for t in times):
                return 0.0  # the moment falls inside this segment
            return d if d <= TIME_WINDOW else None

        hits = [(distance(c), c) for c in chunks]
        hits = sorted(((d, c) for d, c in hits if d is not None), key=lambda p: (p[0], p[1]["t"]))
        if hits:
            return [c for _, c in hits][:limit]
    q = _tokens(question)
    scored = sorted(((len(q & _tokens(c["text"])), c) for c in chunks), key=lambda p: -p[0])
    hits = [c for s, c in scored if s > 0][:limit]
    if not hits and re.search(r"summar|overview|happen|what is this|case", question, re.I):
        hits = [c for c in chunks if c["source"] in ("classification", "facts", "consistency")][:limit]
    return hits


def _deterministic_answer(question, hits, times) -> str:
    if not hits:
        if times:
            return f"No timed evidence exists within {TIME_WINDOW:.0f} seconds of {', '.join(fmt_ts(t) for t in times)}. The evidence does not say."
        return "The evidence does not say."
    lead = f"At {', '.join(fmt_ts(t) for t in times)} the evidence shows:" if times else "From the evidence:"
    lines = [f"[{fmt_ts(c['t'])}] {c['text']}" if c["t"] is not None else c["text"] for c in hits[:6]]
    return lead + "\n" + "\n".join(f"- {l}" for l in lines)


def ask(session_id: str, question: str) -> dict:
    store = get_store()
    session = store.get(session_id)
    if session is None:
        raise KeyError(session_id)
    if not session.get("fusion"):
        answer = {"answer": "No analysis yet. Add evidence and select Run analysis first.", "sources": [], "engine": "none"}
    else:
        chunks = evidence_chunks(session)
        hits = retrieve(question, chunks)
        times = parse_times(question)
        reply = None
        if hits:
            context = "\n".join(f"[{fmt_ts(c['t'])}] ({c['source']}) {c['text']}" if c["t"] is not None else f"({c['source']}) {c['text']}" for c in hits)
            history = [{"role": m["role"], "content": m["content"]} for m in session["chat"][-6:]]
            reply = groq_client.chat(SYSTEM_PROMPT, f"EVIDENCE:\n{context}\n\nQUESTION: {question}", temperature=0.1, max_tokens=350, history=history)
        answer = {
            "answer": reply.strip() if reply else _deterministic_answer(question, hits, times),
            "sources": [{"t": c["t"], "source": c["source"], "text": c["text"][:160]} for c in hits[:6]],
            "engine": "groq" if reply else "deterministic",
        }
    session["chat"].append({"role": "user", "content": question})
    session["chat"].append({"role": "assistant", "content": answer["answer"], "sources": answer["sources"]})
    store.save(session)
    return answer
