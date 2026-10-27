"""Image analysis for any crime-scene photo.

Three independent layers, each optional, all combined into one result:
  1. RT-DETR object detection (people, knives, bats, scissors ...)
  2. CLIP zero-shot scene + forensic indicator tags (firearm, blood, broken glass ...)
  3. Groq vision-language model description of what is visibly present

CLIP tags and model descriptions are *indicators*, not proof; the report says so.
"""
from pathlib import Path

import cv2
import numpy as np
from flask import current_app
from PIL import Image, ImageDraw

from ..utils import groq_client
from ..utils.logger import get_logger
from ..utils.model_loader import get_device, load_model

log = get_logger(__name__)

WEAPON_LABELS = {"knife", "baseball bat", "scissors", "gun", "pistol", "rifle", "firearm", "machete", "sword"}
SCENE_PROMPTS = [
    "a street at night", "an indoor room", "a parking lot", "a store interior", "a residential house",
    "an office", "a crowded public place", "a vehicle interior", "a dark alley", "a park or open field",
    "a table or workbench in front of a wall", "a kitchen", "a bedroom", "an outdoor wall or fence",
]
# (label, category, CLIP prompt). Category "none" entries soak up probability for ordinary photos.
FORENSIC_TAGS = [
    ("firearm", "weapon", "a photo of a gun or firearm"),
    ("knife", "weapon", "a photo of a knife or blade"),
    ("blood", "injury", "a photo of blood stains or blood on a surface"),
    ("person on ground", "injury", "a photo of a person lying motionless on the ground"),
    ("broken glass", "damage", "a photo of shattered or broken glass"),
    ("forced entry", "damage", "a photo of a broken or forced door or window"),
    ("ransacked room", "damage", "a photo of a messy ransacked room with scattered items"),
    ("fire damage", "damage", "a photo of fire, smoke or burnt property"),
    ("vehicle collision", "damage", "a photo of a damaged car after a crash"),
    ("graffiti or vandalism", "damage", "a photo of graffiti or vandalised property"),
    ("drugs", "contraband", "a photo of illegal drugs, pills or syringes"),
    ("cash", "contraband", "a photo of a pile of cash money"),
    ("nothing", "none", "an ordinary photo with nothing suspicious"),
    ("nothing", "none", "a normal everyday photograph of people and objects"),
    ("nothing", "none", "a normal everyday photograph of a room or street"),
]
_hog = None


def categorize(label: str) -> str:
    if label == "person":
        return "person"
    return "weapon" if label in WEAPON_LABELS else "object"


def _det(label, score, x1, y1, x2, y2):
    clip = lambda v: round(max(0.0, min(1.0, v)), 4)  # noqa: E731
    return {"label": label, "score": round(float(score), 3), "box": [clip(x1), clip(y1), clip(x2), clip(y2)], "category": categorize(label)}


# ---------------------------------------------------------------- detection
def _load_detector():
    def loader():
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        name = current_app.config["RTDETR_MODEL"]
        processor = RTDetrImageProcessor.from_pretrained(name)
        return processor, RTDetrForObjectDetection.from_pretrained(name).to(get_device()).eval()

    return load_model("rtdetr", loader)


def _hog_detect(pil: Image.Image):
    """People-only fallback using OpenCV's built-in HOG detector (skipped if OpenCV lacks it)."""
    global _hog
    if not hasattr(cv2, "HOGDescriptor"):
        log.warning("This OpenCV build has no HOGDescriptor; people detection skipped")
        return []
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    arr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    scale = min(1.0, 640 / max(w, 1))
    if scale < 1:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)))
    sh, sw = arr.shape[:2]
    rects, weights = _hog.detectMultiScale(arr, winStride=(8, 8), padding=(8, 8), scale=1.05)
    weights = np.array(weights).reshape(-1)
    return [
        _det("person", min(0.99, wt / 2.0), x / sw, y / sh, (x + rw) / sw, (y + rh) / sh)
        for (x, y, rw, rh), wt in zip(rects, weights)
        if wt >= 0.6
    ]


def detect_objects(images, threshold=None):
    """Batch detection. Returns (list_of_detection_lists, engine_name)."""
    threshold = threshold or current_app.config["DETECTION_THRESHOLD"]
    bundle = _load_detector()
    if bundle:
        try:
            import torch

            processor, model = bundle
            rgb = [im.convert("RGB") for im in images]
            inputs = processor(images=rgb, return_tensors="pt").to(get_device())
            with torch.no_grad():
                outputs = model(**inputs)
            sizes = torch.tensor([im.size[::-1] for im in rgb])
            results = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=threshold)
            batch = []
            for im, res in zip(rgb, results):
                w, h = im.size
                dets = []
                for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
                    x1, y1, x2, y2 = [float(v) for v in b.tolist()]
                    dets.append(_det(model.config.id2label[int(l)], s, x1 / w, y1 / h, x2 / w, y2 / h))
                batch.append(dets)
            return batch, "rt-detr"
        except Exception as exc:
            log.warning("RT-DETR inference failed (%s); using HOG fallback", exc)
    results = []
    for im in images:
        try:
            results.append(_hog_detect(im))
        except Exception as exc:  # a broken OpenCV build must not stop the analysis
            log.warning("Fallback people detection failed: %s", exc)
            results.append([])
    return results, "fallback-hog"


# ------------------------------------------------------------- CLIP layer
def tags_from_probs(probs, threshold=0.3):
    """Pure helper: probabilities over FORENSIC_TAGS -> sorted indicator tags."""
    tags = [
        {"label": label, "category": category, "score": round(float(p), 3)}
        for (label, category, _), p in zip(FORENSIC_TAGS, probs)
        if category != "none" and p >= threshold
    ]
    return sorted(tags, key=lambda t: -t["score"])


def weapon_hints_from(tags):
    return sorted({t["label"] for t in tags if t["category"] == "weapon"})


def clip_analyze(pil: Image.Image):
    """Scene + forensic tags from one CLIP pass. Returns None if CLIP is unavailable."""

    def loader():
        from transformers import CLIPModel, CLIPProcessor

        name = current_app.config["CLIP_MODEL"]
        return CLIPProcessor.from_pretrained(name), CLIPModel.from_pretrained(name).to(get_device()).eval()

    bundle = load_model("clip", loader)
    if not bundle:
        return None
    try:
        import torch

        processor, model = bundle
        prompts = SCENE_PROMPTS + [t[2] for t in FORENSIC_TAGS]
        inputs = processor(text=prompts, images=pil.convert("RGB"), return_tensors="pt", padding=True).to(get_device())
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0]
        n = len(SCENE_PROMPTS)
        scene_p, tag_p = logits[:n].softmax(0).tolist(), logits[n:].softmax(0).tolist()
        ranked = sorted(zip(SCENE_PROMPTS, scene_p), key=lambda p: -p[1])
        scene = {"label": ranked[0][0], "score": round(ranked[0][1], 3),
                 "top": [{"label": l, "score": round(s, 3)} for l, s in ranked[:3]], "engine": "clip"}
        return {"scene": scene, "tags": tags_from_probs(tag_p, current_app.config["FORENSIC_TAG_THRESHOLD"])}
    except Exception as exc:
        log.warning("CLIP inference failed: %s", exc)
        return None


def heuristic_scene(pil: Image.Image) -> dict:
    brightness = float(np.array(pil.convert("L")).mean())
    label = "a low-light scene" if brightness < 70 else "a well-lit scene"
    return {"label": label, "score": 0.5, "top": [{"label": label, "score": 0.5}], "engine": "fallback-brightness"}


def classify_scene(pil: Image.Image) -> dict:
    result = clip_analyze(pil)
    return result["scene"] if result else heuristic_scene(pil)


def caption_image(pil: Image.Image):
    """Optional local BLIP/BLIP-2 caption (ENABLE_CAPTIONING=true)."""
    if not current_app.config.get("ENABLE_CAPTIONING"):
        return None

    def loader():
        from transformers import AutoModelForVision2Seq, AutoProcessor

        name = current_app.config["CAPTION_MODEL"]
        return AutoProcessor.from_pretrained(name), AutoModelForVision2Seq.from_pretrained(name).to(get_device()).eval()

    bundle = load_model("caption", loader)
    if not bundle:
        return None
    try:
        processor, model = bundle
        inputs = processor(images=pil.convert("RGB"), return_tensors="pt").to(get_device())
        return processor.batch_decode(model.generate(**inputs, max_new_tokens=40), skip_special_tokens=True)[0].strip()
    except Exception as exc:
        log.warning("Captioning failed: %s", exc)
        return None


# ------------------------------------------------- vision-language (Groq)
VLM_PROMPT = (
    "You are a forensic image analyst assisting an investigator. Describe ONLY what is visibly present in the image. "
    "Do not identify anyone, and do not guess motives or events that are not visible; write 'unclear' when unsure. "
    "Reply with a single JSON object and nothing else, in this shape: "
    '{"description": "two concise sentences", "people_count": 0, "weapons": ["objects that look like weapons"], '
    '"evidence_items": ["notable objects or marks, e.g. stains resembling blood, broken glass, tools"], '
    '"injuries_visible": false, "scene_type": "short phrase", "damage_or_disorder": "short phrase or none"}'
)


def _as_int(v, default=0):
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return default


def _str_list(v, limit=8):
    return [" ".join(str(x).split())[:80] for x in (v if isinstance(v, list) else []) if str(x).strip()][:limit]


def clean_vlm(data):
    if not isinstance(data, dict) or not str(data.get("description", "")).strip():
        return None
    return {
        "description": " ".join(str(data["description"]).split())[:500],
        "people_count": _as_int(data.get("people_count")),
        "weapons": _str_list(data.get("weapons")),
        "evidence_items": _str_list(data.get("evidence_items")),
        "injuries_visible": bool(data.get("injuries_visible")),
        "scene_type": " ".join(str(data.get("scene_type", "")).split())[:80],
        "damage_or_disorder": " ".join(str(data.get("damage_or_disorder", "")).split())[:120],
    }


def vlm_describe(pil: Image.Image):
    return clean_vlm(groq_client.parse_json(groq_client.vision(VLM_PROMPT, [pil])))


# ------------------------------------------------------------- annotations
def draw_annotations(pil: Image.Image, dets, folder, name) -> str:
    """Save a copy with detection boxes under <folder>/_derived. Returns the relative path."""
    im = pil.convert("RGB")
    im.thumbnail((1280, 1280))
    draw, w, h = ImageDraw.Draw(im), im.size[0], im.size[1]
    for b in dets or []:
        x1, y1, x2, y2 = b["box"]
        color = {"person": (242, 183, 5), "weapon": (255, 77, 46)}.get(b["category"], (111, 179, 255))
        draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=color, width=3)
        label = f"{b['label']} {int(b['score'] * 100)}%"
        top = max(0, y1 * h - 14)
        draw.rectangle([x1 * w, top, x1 * w + draw.textlength(label) + 6, top + 14], fill=color)
        draw.text((x1 * w + 3, top + 1), label, fill=(22, 32, 42))
    out = Path(folder) / "_derived"
    out.mkdir(parents=True, exist_ok=True)
    im.save(out / f"{name}.jpg", "JPEG", quality=88)
    return f"_derived/{name}.jpg"


# ------------------------------------------------------------------ summary
def _counts(dets):
    return {
        "persons": sum(d["category"] == "person" for d in dets),
        "weapons": sum(d["category"] == "weapon" for d in dets),
        "objects": sum(d["category"] == "object" for d in dets),
    }


def _tag_text(tags, categories=None):
    return ", ".join(f"{t['label']} ({int(t['score'] * 100)}%)" for t in tags if not categories or t["category"] in categories)


def summarize_image(dets, scene, caption=None, weapons_checked=True, tags=None, vlm=None) -> str:
    tags = tags or []
    if vlm and vlm.get("description"):
        return vlm["description"]
    counts = _counts(dets)
    weapons = sorted({d["label"] for d in dets if d["category"] == "weapon"})
    hints = weapon_hints_from(tags)
    facts = (
        f"Scene: {scene['label']}. Persons: {counts['persons']}. "
        f"Weapons detected: {', '.join(weapons) or ('none' if weapons_checked else 'NOT CHECKED')}. "
        f"Classifier indicators (unverified): {_tag_text(tags) or 'none'}. "
        f"Other objects: {', '.join(sorted({d['label'] for d in dets if d['category'] == 'object'})[:8]) or 'none'}."
        + (f" Caption: {caption}." if caption else "")
    )
    reply = groq_client.chat(
        "You are a forensic image analyst. Using ONLY the facts given, write exactly two concise lines. "
        "Treat classifier indicators as possibilities, not certainties. Do not speculate beyond the facts.",
        facts, temperature=0.1, max_tokens=120,
    )
    if reply:
        return " ".join(reply.strip().splitlines())
    people = f"{counts['persons']} person(s)" if counts["persons"] else "no people"
    if weapons:
        threat = f"weapon-class objects detected ({', '.join(weapons)})"
    elif hints:
        threat = f"the image classifier suggests a possible {', '.join(hints)}"
    elif not weapons_checked:
        threat = "weapon detection unavailable in this mode, so weapons cannot be ruled out"
    else:
        threat = "no weapons detected"
    first = caption.capitalize() if caption else f"The image shows {scene['label'].replace('a ', '', 1)}"
    other = _tag_text(tags, {"injury", "damage", "contraband"})
    return f"{first}. {people.capitalize()} visible; {threat}." + (f" Other possible indicators: {other}." if other else "")


def analyze_image(path, evidence_id: str, offset=None) -> dict:
    path = Path(path)
    pil = Image.open(path).convert("RGB")
    batch, engine = detect_objects([pil])
    dets = batch[0]
    clip = clip_analyze(pil)
    scene = clip["scene"] if clip else heuristic_scene(pil)
    tags = clip["tags"] if clip else []
    caption = caption_image(pil)
    vlm = vlm_describe(pil)
    counts = _counts(dets)
    if vlm:
        counts["persons"] = max(counts["persons"], vlm["people_count"])
    hints = sorted(set(weapon_hints_from(tags)) | {w.lower() for w in (vlm or {}).get("weapons", [])})
    checked = engine == "rt-detr" or clip is not None or vlm is not None
    return {
        "evidence_id": evidence_id,
        "file": path.name,
        "size": list(pil.size),
        "detections": dets,
        "counts": counts,
        "scene": scene,
        "tags": tags,
        "vlm": vlm,
        "weapon_hints": hints,
        "caption": caption,
        "summary": summarize_image(dets, scene, caption, checked, tags, vlm),
        "annotated": draw_annotations(pil, dets, path.parent, f"{evidence_id}_{path.stem}_annotated"),
        "weapons_checked": checked,
        "engines": {"detector": engine, "clip": clip is not None, "vision_model": vlm is not None},
        "engine": engine,
        "offset": offset,
    }
