"""Video analysis for any short clip: robust 1 Hz sampling, batched detection,
motion scoring, key-frame analysis (CLIP + vision model), action narrative,
event generation and crime-type classification."""
from pathlib import Path

import cv2
import numpy as np
from flask import current_app
from PIL import Image

from ..utils import groq_client
from ..utils.logger import get_logger
from . import vision_service

log = get_logger(__name__)

CRIME_TYPES = [
    "Assault", "Armed Assault", "Theft", "Robbery", "Burglary", "Vandalism",
    "Arson", "Traffic Accident", "Suspicious Activity", "No Incident",
]

SEQ_PROMPT = (
    "You are a forensic video analyst assisting an investigator. These {n} frames were sampled in order from one video "
    "at these times (seconds): {times}. Describe ONLY what is visibly happening. Do not identify anyone and do not "
    "guess about anything not visible; write 'unclear' when unsure. Reply with a single JSON object and nothing else: "
    '{{"summary": "at most three sentences describing the actions across the frames", "violence_visible": false, '
    '"frames": [{{"i": 1, "description": "one sentence", "people_count": 0, "weapons": []}}], '
    '"weapons": ["objects that look like weapons"], "actions": ["short phrases, e.g. two people arguing"]}}'
)


def extract_frames(path, max_seconds=None, max_side=None):
    """Sample one frame per second by reading sequentially (robust across codecs).

    Returns ([(t_seconds, BGR ndarray)], meta).
    """
    max_seconds = max_seconds or current_app.config["VIDEO_MAX_SECONDS"]
    max_side = max_side or current_app.config["VIDEO_MAX_SIDE"]
    if not hasattr(cv2, "VideoCapture"):
        raise ValueError("OpenCV is not installed correctly on this server, so video cannot be read. Use opencv-python-headless.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video '{Path(path).name}'. Convert it to MP4 (H.264) and try again.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= (fps or 0) <= 240 else 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames, idx, next_sample = [], 0, 0.0
    while cap.grab():  # grab() skips decoding for frames we do not sample
        if idx >= next_sample - 1e-6:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                scale = max_side / max(h, w)
                if scale < 1:  # keep aspect ratio
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                frames.append((float(len(frames)), frame))
            next_sample += fps
            if len(frames) > max_seconds:
                break
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from '{Path(path).name}'. The file may be corrupt or use an unsupported codec.")
    duration = total / fps if total > 0 else idx / fps
    meta = {"fps": round(fps, 2), "duration": round(max(duration, len(frames) - 1), 2), "width": width, "height": height, "sampled": len(frames)}
    return frames, meta


def _motion_scores(frames):
    scores, prev = [], None
    for _, f in frames:
        gray = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        scores.append(0.0 if prev is None or prev.shape != gray.shape else float(cv2.absdiff(gray, prev).mean() / 255.0))
        prev = gray
    return scores


def pick_keyframes(timeline, k):
    """Choose up to k informative, well-spread frame indices (always includes the first)."""
    n = len(timeline)
    if n <= k:
        return list(range(n))
    score = lambda i: timeline[i]["motion"] * 10 + timeline[i]["persons"] * 0.5 + (3 if timeline[i]["weapons"] else 0)  # noqa: E731
    gap, chosen = max(1, n // (2 * k)), [0]
    for i in sorted(range(1, n), key=lambda i: (-score(i), i)):
        if len(chosen) >= k:
            break
        if all(abs(i - c) >= gap for c in chosen):
            chosen.append(i)
    for i in np.linspace(0, n - 1, k).astype(int):  # top up with evenly spaced frames if needed
        if len(chosen) >= k:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return sorted(chosen)


def describe_sequence(pils, times):
    """One vision-model call over the key frames: narrative, actions, per-frame notes."""
    prompt = SEQ_PROMPT.format(n=len(pils), times=", ".join(f"{t:g}" for t in times))
    data = groq_client.parse_json(groq_client.vision(prompt, pils))
    if not isinstance(data, dict) or not str(data.get("summary", "")).strip():
        return None
    frames = {}
    for item in data.get("frames") or []:
        if isinstance(item, dict):
            frames[vision_service._as_int(item.get("i"))] = {
                "description": " ".join(str(item.get("description", "")).split())[:300],
                "people_count": vision_service._as_int(item.get("people_count")),
                "weapons": vision_service._str_list(item.get("weapons")),
            }
    return {
        "summary": " ".join(str(data["summary"]).split())[:700],
        "violence_visible": bool(data.get("violence_visible")),
        "weapons": vision_service._str_list(data.get("weapons")),
        "actions": vision_service._str_list(data.get("actions")),
        "frames": frames,
    }


def analyze_keyframes(frames, timeline, indices, evidence_id, stem, folder, offset):
    pils = [Image.fromarray(cv2.cvtColor(frames[i][1], cv2.COLOR_BGR2RGB)) for i in indices]
    clips = [vision_service.clip_analyze(p) for p in pils]
    narrative = describe_sequence(pils, [timeline[i]["t"] - offset for i in indices])
    keyframes = []
    for n, (i, pil, clip) in enumerate(zip(indices, pils, clips), start=1):
        entry = timeline[i]
        note = (narrative or {}).get("frames", {}).get(n, {})
        tags = clip["tags"] if clip else []
        weapons = sorted(
            set(entry["weapons"]) | set(vision_service.weapon_hints_from(tags)) | {w.lower() for w in note.get("weapons", [])}
        )
        second = int(round(entry["t"] - offset))
        keyframes.append({
            "t": entry["t"],
            "file": vision_service.draw_annotations(pil, entry["boxes"], folder, f"{evidence_id}_{stem}_t{second:03d}"),
            "tags": tags,
            "scene": clip["scene"]["label"] if clip else None,
            "description": note.get("description", ""),
            "people": max(entry["persons"], note.get("people_count", 0)),
            "weapons": weapons,
        })
    return keyframes, narrative, any(c is not None for c in clips)


def build_events(timeline, keyframes=()):
    """Turn per-second observations and key-frame findings into readable events."""
    events, prev_persons, seen_weapons = [], 0, set()
    for e in timeline:
        t = e["t"]
        for w in [w for w in e["weapons"] if w not in seen_weapons]:
            seen_weapons.add(w)
            events.append({"t": t, "type": "weapon", "description": f"Weapon-class object visible: {w}", "severity": "flag"})
        if e["persons"] != prev_persons:
            events.append({"t": t, "type": "people", "description": f"Person count changed {prev_persons} → {e['persons']}", "severity": "info"})
            prev_persons = e["persons"]
        if e["active"]:
            events.append({"t": t, "type": "motion", "description": f"Sudden high activity (motion {e['motion']:.2f})", "severity": "warn"})
    for kf in keyframes:
        text = kf["description"] or (
            "Possible indicators: " + ", ".join(f"{t['label']} ({int(t['score'] * 100)}%)" for t in kf["tags"]) if kf["tags"] else ""
        )
        if text:
            events.append({"t": kf["t"], "type": "scene", "description": f"Key frame: {text}", "severity": "flag" if kf["weapons"] else "info"})
    return sorted(events, key=lambda e: e["t"])


def classify_crime(stats: dict, transcript: str = "", notes: str = "") -> dict:
    """Classify incident type with Groq (temp ≤ 0.2); deterministic rules otherwise."""
    reply = groq_client.chat(
        "You are a forensic analyst. Classify the incident from the observations only. Return JSON: "
        '{"type": one of ' + str(CRIME_TYPES) + ', "confidence": 0-1, "rationale": "one sentence"}.',
        f"Observations: {stats}\nVisual narrative: {notes[:900]}\nTranscript excerpt: {transcript[:1500]}",
        temperature=0.1, json_mode=True, max_tokens=200,
    )
    data = groq_client.parse_json(reply)
    if isinstance(data, dict) and data.get("type") in CRIME_TYPES:
        return {"type": data["type"], "confidence": float(data.get("confidence", 0.6)), "rationale": data.get("rationale", ""), "engine": "groq"}
    weapons, peak, bursts = stats.get("weapons", []), stats.get("peak_persons", 0), stats.get("active_seconds", 0)
    violence = stats.get("violence_visible", False)
    if weapons and (peak >= 2 or violence):
        out = ("Armed Assault", 0.65, f"Weapon-class object ({weapons[0]}) with {max(peak, 2)} people or violence visible.")
    elif weapons:
        out = ("Suspicious Activity", 0.5, f"Weapon-class object ({weapons[0]}) visible.")
    elif violence and peak >= 1:
        out = ("Assault", 0.5, "The vision model reports violence visible in the footage.")
    elif peak >= 2 and bursts >= 2:
        out = ("Assault", 0.45, f"{peak} people with {bursts} bursts of sudden activity.")
    elif bursts >= 1:
        out = ("Suspicious Activity", 0.4, f"{bursts} burst(s) of sudden activity.")
    else:
        out = ("No Incident", 0.4, "No weapons or sudden activity detected.")
    return {"type": out[0], "confidence": out[1], "rationale": out[2], "engine": "rules"}


def analyze_video(path, evidence_id: str, offset=0.0) -> dict:
    path = Path(path)
    frames, meta = extract_frames(path)
    motion = _motion_scores(frames)
    threshold = max(0.06, float(np.mean(motion) + 2 * np.std(motion))) if len(motion) > 2 else 0.06

    detections, engine = [], "fallback-hog"
    batch = current_app.config["VIDEO_BATCH_SIZE"]
    for i in range(0, len(frames), batch):
        chunk = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for _, f in frames[i : i + batch]]
        dets, engine = vision_service.detect_objects(chunk)
        detections.extend(dets)

    offset = float(offset or 0.0)
    timeline = []
    for (t, _), dets, mo in zip(frames, detections, motion):
        timeline.append({
            "t": round(t + offset, 2),
            "persons": sum(d["category"] == "person" for d in dets),
            "weapons": sorted({d["label"] for d in dets if d["category"] == "weapon"}),
            "motion": round(mo, 4),
            "active": bool(mo >= threshold and mo > 0.02),
            "boxes": [d for d in dets if d["category"] in ("person", "weapon")],
        })

    indices = pick_keyframes(timeline, current_app.config["VIDEO_KEYFRAMES"])
    keyframes, narrative, clip_ok = analyze_keyframes(frames, timeline, indices, evidence_id, path.stem, path.parent, offset)
    events = build_events(timeline, keyframes)

    weapons = sorted({w for e in timeline for w in e["weapons"]} | {w for kf in keyframes for w in kf["weapons"]} | {w.lower() for w in (narrative or {}).get("weapons", [])})
    stats = {
        "duration_sec": meta["duration"],
        "peak_persons": max([e["persons"] for e in timeline] + [kf["people"] for kf in keyframes]),
        "weapons": weapons,
        "active_seconds": sum(e["active"] for e in timeline),
        "violence_visible": bool((narrative or {}).get("violence_visible")),
    }
    return {
        "evidence_id": evidence_id,
        "file": path.name,
        "meta": meta,
        "timeline": timeline,
        "events": events,
        "keyframes": keyframes,
        "narrative": narrative["summary"] if narrative else "",
        "actions": narrative["actions"] if narrative else [],
        "stats": stats,
        "crime_type": classify_crime(stats, notes=narrative["summary"] if narrative else ""),
        "weapons_checked": engine == "rt-detr" or clip_ok or narrative is not None,
        "engines": {"detector": engine, "clip": clip_ok, "vision_model": narrative is not None},
        "engine": engine,
        "offset": offset,
    }
