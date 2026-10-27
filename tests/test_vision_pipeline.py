"""Tests for the image/video understanding layers. The Groq vision call is simulated;
CLIP tag post-processing is tested as a pure function (the model itself needs a download)."""
import json
from unittest import mock

import cv2
import numpy as np

from backend.services import fusion_service, vision_service, video_service
from backend.utils import demo_data, groq_client

IMAGE_JSON = json.dumps({
    "description": "A combat knife is stuck upright in a wooden table in front of a brick wall. No people are visible.",
    "people_count": 0, "weapons": ["Combat knife"], "evidence_items": ["knife embedded in table"],
    "injuries_visible": False, "scene_type": "table against brick wall", "damage_or_disorder": "none",
})
SEQ_JSON = json.dumps({
    "summary": "Two people argue near a shop. One pulls out a knife and pushes the other.",
    "violence_visible": True, "weapons": ["knife"], "actions": ["two people arguing", "one person pushing another"],
    "frames": [{"i": 1, "description": "Two people stand apart.", "people_count": 2, "weapons": []},
               {"i": 3, "description": "One person holds a knife.", "people_count": 2, "weapons": ["knife"]}],
})


def test_tags_from_probs_threshold_and_ordering():
    probs = [0.0] * len(vision_service.FORENSIC_TAGS)
    probs[0], probs[1], probs[2], probs[12] = 0.2, 0.7, 0.35, 0.9  # firearm, knife, blood, "nothing"
    tags = vision_service.tags_from_probs(probs, 0.3)
    assert [t["label"] for t in tags] == ["knife", "blood"]
    assert vision_service.weapon_hints_from(tags) == ["knife"]


def test_clean_vlm_handles_garbage():
    assert vision_service.clean_vlm(None) is None
    assert vision_service.clean_vlm({"description": " "}) is None
    out = vision_service.clean_vlm({"description": "x  y", "people_count": "3", "weapons": "oops"})
    assert out["description"] == "x y" and out["people_count"] == 3 and out["weapons"] == []


def test_parse_json_tolerates_wrapping_text():
    assert groq_client.parse_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}


def test_image_uses_vision_model_and_marks_weapon_hint(app, tmp_path):
    demo_data.make_image(tmp_path / "scene.png")
    with mock.patch.object(groq_client, "vision", return_value=IMAGE_JSON):
        out = vision_service.analyze_image(tmp_path / "scene.png", "E1")
    assert "knife" in out["summary"].lower()
    assert out["weapon_hints"] == ["combat knife"] and out["weapons_checked"] is True
    assert out["engines"]["vision_model"] is True
    assert (tmp_path / out["annotated"]).exists()


def test_image_without_any_model_never_claims_no_weapons(app, tmp_path):
    demo_data.make_image(tmp_path / "scene.png")
    out = vision_service.analyze_image(tmp_path / "scene.png", "E1")
    assert out["weapons_checked"] is False and "cannot be ruled out" in out["summary"]


def test_vision_model_failure_falls_back_cleanly(app, tmp_path):
    demo_data.make_image(tmp_path / "scene.png")
    with mock.patch.object(groq_client, "vision", return_value="not json at all"):
        out = vision_service.analyze_image(tmp_path / "scene.png", "E1")
    assert out["vlm"] is None and out["summary"]


def test_video_keyframes_narrative_and_events(app, tmp_path):
    path = demo_data.make_video(tmp_path / "v")
    with mock.patch.object(groq_client, "vision", return_value=SEQ_JSON):
        out = video_service.analyze_video(path, "E3")
    assert 1 <= len(out["keyframes"]) <= 5
    assert all((tmp_path / kf["file"]).exists() for kf in out["keyframes"])
    assert out["narrative"].startswith("Two people argue")
    assert "knife" in out["stats"]["weapons"] and out["stats"]["violence_visible"] is True
    assert out["weapons_checked"] is True
    assert out["crime_type"]["type"] == "Armed Assault"
    assert any(e["type"] == "scene" for e in out["events"])


def test_pick_keyframes_spread_and_includes_first():
    tl = [{"motion": 0.0, "persons": 0, "weapons": []} for _ in range(30)]
    tl[20]["motion"] = 0.5
    picks = video_service.pick_keyframes(tl, 5)
    assert picks[0] == 0 and 20 in picks and len(picks) == 5 and picks == sorted(picks)


def test_extract_frames_one_per_second_and_aspect_ratio(app, tmp_path):
    path = demo_data.make_video(tmp_path / "v", seconds=6, size=(1280, 720))
    frames, meta = video_service.extract_frames(path, max_side=640)
    assert [t for t, _ in frames][:3] == [0.0, 1.0, 2.0] and len(frames) in (6, 7)
    h, w = frames[0][1].shape[:2]
    assert max(h, w) == 640 and abs(w / h - 16 / 9) < 0.02


def test_corrupt_video_gives_helpful_error(app, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    try:
        video_service.extract_frames(bad)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Convert it to MP4" in str(exc) or "No frames" in str(exc)


def test_keyframe_weapon_hint_corroborates_spoken_weapon(app):
    results = {
        "video": [{"evidence_id": "E1", "engine": "fallback-hog", "weapons_checked": True, "stats": {"violence_visible": False},
                   "timeline": [{"t": 8.0, "persons": 2, "weapons": [], "active": False, "motion": 0}],
                   "keyframes": [{"t": 8.0, "people": 2, "weapons": ["knife"]}]}],
        "images": [], "audio": [{"segments": [{"start": 8.0, "end": 10.0, "text": "He pulled out a knife"}]}],
    }
    assert fusion_service.check_consistency(results)["claims"][0]["status"] == "corroborated"


def test_end_to_end_with_uploaded_image_and_video(client, app):
    import io
    sid = client.post("/api/sessions").get_json()["id"]
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))
    vid = demo_data.make_video(app.config["UPLOAD_FOLDER"] / "clip")
    data = {"session_id": sid, "files": [(io.BytesIO(buf.tobytes()), "photo.jpg"), (open(vid, "rb"), vid.name)]}
    assert client.post("/api/upload", data=data, content_type="multipart/form-data").status_code == 201
    with mock.patch.object(groq_client, "vision", side_effect=[IMAGE_JSON, SEQ_JSON]):
        body = client.post(f"/api/process/{sid}").get_json()
    fusion = body["session"]["fusion"]
    assert body["errors"] == [] and fusion["crime_type"]["type"] == "Armed Assault"
    assert "knife" in fusion["stats"]["weapons"]
    pdf = client.get(f"/api/report/{sid}")
    assert pdf.data[:5] == b"%PDF-" and pdf.data.count(b"/Subtype /Image") >= 2
    assert client.get(f"/api/files/{sid}/" + body["session"]["results"]["images"][0]["annotated"]).status_code == 200
