"""Multimodal fusion: one chronological axis, cross-modal consistency scoring,
crime-type classification and three AI scenario hypotheses."""
import re
from datetime import datetime, timezone

from flask import current_app

from ..utils import groq_client
from . import video_service

WEAPON_RE = re.compile(r"\b(knife|knives|gun|pistol|revolver|rifle|weapon|blade|machete|bat|stabb?(?:ed|ing)?|shot|shoot(?:ing)?)\b", re.I)
VIOLENCE_RE = re.compile(r"\b(hit|punch(?:ed)?|fight(?:ing)?|attack(?:ed)?|struck|beat(?:ing)?|push(?:ed)?|shov(?:e|ed)|kick(?:ed)?|grabbed|ran|run(?:ning)?|fled|chas(?:e|ed|ing)|scream(?:ed|ing)?)\b", re.I)
COUNT_RE = re.compile(r"\b(two|three|four|five|2|3|4|5)\s+(?:\w+\s+){0,2}(?:men|people|persons|guys|women|suspects|individuals|boys|girls)\b", re.I)
NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}


def fmt_ts(t) -> str:
    if t is None:
        return "undated"
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


def image_weapons(img) -> list:
    """Weapon labels from detector boxes plus classifier / vision-model hints."""
    return sorted({d["label"] for d in img["detections"] if d["category"] == "weapon"} | set(img.get("weapon_hints", [])))


def _frames(results):
    """Per-second video observations plus key-frame observations on one list."""
    points = []
    for v in results.get("video", []):
        points += [dict(f, evidence_id=v["evidence_id"]) for f in v["timeline"]]
        violence = bool(v.get("stats", {}).get("violence_visible"))
        points += [{"t": kf["t"], "persons": kf["people"], "weapons": kf["weapons"], "active": violence, "motion": 0.0, "evidence_id": v["evidence_id"]}
                   for kf in v.get("keyframes", [])]
    return points


def build_timeline(results) -> list:
    items = []
    for v in results.get("video", []):
        for ev in v["events"]:
            items.append({"t": ev["t"], "source": "video", "kind": ev["type"], "text": ev["description"], "severity": ev["severity"], "ref": v["evidence_id"]})
    for a in results.get("audio", []):
        for s in a["segments"]:
            items.append({"t": s["start"], "end": s["end"], "source": "audio", "kind": "speech", "text": s["text"], "severity": "info", "ref": a["evidence_id"]})
    for im in results.get("images", []):
        items.append({"t": im.get("offset"), "source": "image", "kind": "image", "text": im["summary"], "severity": "flag" if image_weapons(im) else "info", "ref": im["evidence_id"]})
    items.sort(key=lambda i: (i["t"] is None, i["t"] or 0, i["source"]))
    for n, item in enumerate(items, 1):
        item["idx"] = n
    return items


def check_consistency(results) -> dict:
    """Corroborate what is *said* against what is *seen*."""
    window = current_app.config["CORROBORATION_WINDOW_SEC"]
    frames, images = _frames(results), results.get("images", [])
    segments = [s for a in results.get("audio", []) for s in a["segments"]]
    has_visual = bool(frames or images)
    weapon_detector = any(x.get("weapons_checked", x.get("engine") == "rt-detr") for x in results.get("video", []) + images)
    claims = []

    def near(t):
        return [f for f in frames if abs(f["t"] - t) <= window]

    for seg in segments:
        t, text = seg["start"], seg["text"]
        if WEAPON_RE.search(text):
            word = WEAPON_RE.search(text).group(1).lower()
            if not has_visual:
                status, why = "unverifiable", "No visual evidence to compare against."
            elif not weapon_detector:
                status, why = "unverifiable", "No weapon detector is active; install the AI models (requirements-ai.txt) to check weapons."
            else:
                hit = [f for f in near(t) if f["weapons"]] or [i for i in images if image_weapons(i)]
                status = "corroborated" if hit else "unsupported"
                why = "Weapon-class object detected near this time." if hit else f"No weapon detected within ±{window:.0f}s."
            claims.append({"t": t, "type": "weapon", "statement": text, "keyword": word, "status": status, "evidence": why})
        if VIOLENCE_RE.search(text):
            word = VIOLENCE_RE.search(text).group(1).lower()
            if not frames:
                status, why = "unverifiable", "No video evidence to compare against."
            else:
                hit = [f for f in near(t) if f["active"]]
                status = "corroborated" if hit else "unsupported"
                why = "Sudden high activity observed nearby." if hit else f"No unusual activity within ±{window:.0f}s."
            claims.append({"t": t, "type": "action", "statement": text, "keyword": word, "status": status, "evidence": why})
        m = COUNT_RE.search(text)
        if m:
            n = NUMBERS[m.group(1).lower()]
            if not has_visual:
                status, why = "unverifiable", "No visual evidence to compare against."
            else:
                peak = max([f["persons"] for f in near(t)] or [f["persons"] for f in frames] or [i["counts"]["persons"] for i in images] or [0])
                status = "corroborated" if peak >= n else "unsupported"
                why = f"Up to {peak} person(s) detected; statement says {n}."
            claims.append({"t": t, "type": "people", "statement": text, "keyword": m.group(0), "status": status, "evidence": why})

    ok = sum(c["status"] == "corroborated" for c in claims)
    bad = sum(c["status"] == "unsupported" for c in claims)
    score = round(100 * ok / (ok + bad)) if (ok + bad) else None
    if score is None:
        summary = "No checkable audio claims were matched against visual evidence." if not claims else "Claims were found but could not be verified with the available visual evidence."
    else:
        summary = f"{ok} of {ok + bad} verifiable statements are backed by visual evidence."
    return {"score": score, "corroborated": ok, "unsupported": bad, "unverifiable": len(claims) - ok - bad, "claims": claims, "summary": summary}


def merge_facts(results) -> dict:
    merged = {"who": [], "where": [], "when": []}
    for a in results.get("audio", []):
        for k in merged:
            for item in a["facts"].get(k, []):
                if item.lower() not in [x.lower() for x in merged[k]]:
                    merged[k].append(item)
    return merged


def _digest(results, crime, facts, consistency) -> str:
    stats = [v["stats"] for v in results.get("video", [])]
    lines = [f"Crime type: {crime['type']} ({crime['rationale']})", f"Video stats: {stats}", f"Facts: {facts}",
             f"Consistency: {consistency['summary']}"]
    lines += [f"Video {v['evidence_id']} narrative: {v['narrative']}" for v in results.get("video", []) if v.get("narrative")]
    lines += [f"Image {i['evidence_id']}: {i['summary']} Indicators: {[t['label'] for t in i.get('tags', [])]}" for i in results.get("images", [])]
    lines += [f"[{fmt_ts(s['start'])}] {s['text']}" for a in results.get("audio", []) for s in a["segments"]][:40]
    return "\n".join(lines)[:7000]


def _fallback_hypotheses(crime, facts, consistency, stats) -> list:
    base = (consistency["score"] if consistency["score"] is not None else 50) / 100
    who = ", ".join(facts["who"][:2]) or "the people involved"
    where = facts["where"][0] if facts["where"] else "the recorded location"
    peak = max([s["peak_persons"] for s in stats] or [0])
    weapons = sorted({w for s in stats for w in s["weapons"]})
    return [
        {"title": f"Primary: {crime['type']}", "confidence": round(min(0.9, 0.35 + 0.5 * base), 2),
         "narrative": f"The evidence is most consistent with a {crime['type'].lower()} involving {who} at {where}. {crime['rationale']}",
         "supporting_evidence": [f"Peak of {peak} person(s) on camera", f"Weapon-class objects: {', '.join(weapons) or 'none detected'}", consistency["summary"]],
         "gaps": ["Intent cannot be established from footage alone."]},
        {"title": "Alternative: mutual altercation", "confidence": round(0.25 + 0.15 * (1 - base), 2),
         "narrative": f"The activity at {where} may be a two-sided dispute rather than a one-sided offence, with no clear aggressor.",
         "supporting_evidence": ["Multiple people with bursts of motion" if peak >= 2 else "Limited visual detail"],
         "gaps": ["Who initiated contact is not determined.", "Audio may not capture the start of the exchange."]},
        {"title": "Alternative: misinterpreted or incomplete evidence", "confidence": round(0.1 + 0.2 * (1 - base), 2),
         "narrative": "Detection limits, camera angle or missing audio may make an ordinary event look like an incident.",
         "supporting_evidence": [f"{consistency['unsupported']} statement(s) lacked visual support"],
         "gaps": ["Obtain additional angles and the original, unprocessed recordings."]},
    ]


def generate_hypotheses(results, crime, facts, consistency) -> dict:
    stats = [v["stats"] for v in results.get("video", [])]
    reply = groq_client.chat(
        "You are a forensic analyst. From ONLY the evidence digest, write exactly 3 distinct hypotheses about what "
        'happened. Return JSON {"hypotheses":[{"title":str,"confidence":0-1,"narrative":str (2-3 sentences),'
        '"supporting_evidence":[str],"gaps":[str]}]}. Never invent facts not in the digest.',
        _digest(results, crime, facts, consistency),
        temperature=0.2, json_mode=True, max_tokens=1200,
    )
    data = groq_client.parse_json(reply)
    if isinstance(data, dict) and isinstance(data.get("hypotheses"), list) and len(data["hypotheses"]) >= 3:
        hyps = []
        for h in data["hypotheses"][:3]:
            hyps.append({"title": str(h.get("title", "Hypothesis")), "confidence": round(float(h.get("confidence", 0.3)), 2),
                         "narrative": str(h.get("narrative", "")), "supporting_evidence": [str(x) for x in h.get("supporting_evidence", [])],
                         "gaps": [str(x) for x in h.get("gaps", [])]})
        return {"items": hyps, "engine": "groq"}
    return {"items": _fallback_hypotheses(crime, facts, consistency, stats), "engine": "rules"}


def fuse(session: dict) -> dict:
    results = session["results"]
    timeline = build_timeline(results)
    facts = merge_facts(results)
    consistency = check_consistency(results)
    transcript = " ".join(a["full_text"] for a in results.get("audio", []))
    stats = {
        "peak_persons": max([v["stats"]["peak_persons"] for v in results.get("video", [])] + [i["counts"]["persons"] for i in results.get("images", [])] + [0]),
        "weapons": sorted({w for v in results.get("video", []) for w in v["stats"]["weapons"]} | {w for i in results.get("images", []) for w in image_weapons(i)}),
        "violence_visible": any(v["stats"].get("violence_visible") for v in results.get("video", [])),
        "active_seconds": sum(v["stats"]["active_seconds"] for v in results.get("video", [])),
        "images": len(results.get("images", [])),
    }
    notes = " ".join(v.get("narrative", "") for v in results.get("video", [])) + " " + " ".join(i["summary"] for i in results.get("images", []))
    crime = video_service.classify_crime(stats, transcript, notes)
    hypotheses = generate_hypotheses(results, crime, facts, consistency)
    times = [i["t"] for i in timeline if i["t"] is not None]
    summary = f"{crime['type']} ({int(crime['confidence'] * 100)}% confidence). {consistency['summary']}"
    return {
        "timeline": timeline,
        "duration": round(max(times), 2) if times else 0,
        "facts": facts,
        "crime_type": crime,
        "consistency": consistency,
        "hypotheses": hypotheses,
        "stats": stats,
        "summary": summary,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
