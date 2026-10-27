from backend.services import audio_service, chat_service, fusion_service, video_service, vision_service
from backend.utils import demo_data
from backend.utils.session_store import SessionStore, valid_id


def test_parse_timestamped_transcript():
    segs = audio_service.parse_transcript_text("[00:05] First line\n[01:10] Second line")
    assert segs[0]["start"] == 5 and segs[1]["start"] == 70
    assert segs[0]["end"] <= 70


def test_parse_srt():
    srt = "1\n00:00:01,000 --> 00:00:03,500\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld\n"
    segs = audio_service.parse_transcript_text(srt)
    assert segs[0] == {"start": 1.0, "end": 3.5, "text": "Hello"}
    assert len(segs) == 2


def test_parse_plain_text_assigns_sequential_times():
    segs = audio_service.parse_transcript_text("one\ntwo")
    assert segs[1]["start"] == 4.0


def test_fact_extraction_who_where_when():
    facts = audio_service.extract_facts_regex(
        "This is Officer Rana. It is 11:42 PM near the Central Market. Two men in red jackets ran toward the parking lot last night."
    )
    assert any("Rana" in w for w in facts["who"])
    assert any("11:42" in w for w in facts["when"]) and "last night" in [w.lower() for w in facts["when"]]
    assert any("Central Market" in w for w in facts["where"])


def test_chat_time_parsing():
    assert chat_service.parse_times("what happened at 00:08 and 1:05?") == [8, 65]
    assert chat_service.parse_times("at 42 seconds") == [42]
    assert chat_service.parse_times("no time here") == []


def test_session_store_roundtrip_and_traversal(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create("x")
    assert valid_id(s["id"]) and (tmp_path / f"session_{s['id']}.json").exists()
    assert store.get(s["id"])["title"] == "x"
    assert store.get("../../etc/passwd") is None


def test_video_pipeline_produces_second_by_second_floats(app, tmp_path):
    path = demo_data.make_video(tmp_path / "v")
    result = video_service.analyze_video(path, "E1")
    times = [e["t"] for e in result["timeline"]]
    assert times[:3] == [0.0, 1.0, 2.0] and all(isinstance(t, float) for t in times)
    assert result["stats"]["active_seconds"] >= 1
    assert any(e["type"] == "motion" and 7 <= e["t"] <= 12 for e in result["events"])
    assert result["crime_type"]["type"] in video_service.CRIME_TYPES


def test_image_analysis_fallback(app, tmp_path):
    demo_data.make_image(tmp_path / "i.png")
    out = vision_service.analyze_image(tmp_path / "i.png", "E1")
    assert out["summary"] and out["scene"]["label"] and out["size"] == [640, 400]


def test_consistency_marks_unverifiable_without_weapon_detector(app):
    results = {
        "video": [{"evidence_id": "E1", "engine": "fallback-hog", "timeline": [{"t": 8.0, "persons": 2, "weapons": [], "active": True, "motion": 0.2}]}],
        "images": [],
        "audio": [{"segments": [{"start": 8.0, "end": 12.0, "text": "He pulled a knife and pushed two men"}]}],
    }
    cons = fusion_service.check_consistency(results)
    by_type = {c["type"]: c["status"] for c in cons["claims"]}
    assert by_type["weapon"] == "unverifiable"
    assert by_type["action"] == "corroborated"
    assert by_type["people"] == "corroborated"
    assert cons["score"] == 100


def test_consistency_flags_unsupported_claims(app):
    results = {
        "video": [{"evidence_id": "E1", "engine": "rt-detr", "timeline": [{"t": 1.0, "persons": 1, "weapons": [], "active": False, "motion": 0}]}],
        "images": [], "audio": [{"segments": [{"start": 1.0, "end": 3.0, "text": "He had a gun"}]}],
    }
    cons = fusion_service.check_consistency(results)
    assert cons["claims"][0]["status"] == "unsupported" and cons["score"] == 0


def test_fallback_summary_does_not_claim_no_weapons(app, tmp_path):
    demo_data.make_image(tmp_path / "i.png")
    out = vision_service.analyze_image(tmp_path / "i.png", "E1")
    assert out["weapons_checked"] is False
    assert "no weapons detected" not in out["summary"]
    assert "cannot be ruled out" in out["summary"]


def test_missing_opencv_people_detector_does_not_crash(app, tmp_path):
    import types
    from unittest import mock

    demo_data.make_image(tmp_path / "i.png")
    with mock.patch.object(vision_service, "cv2", types.SimpleNamespace()):
        out = vision_service.analyze_image(tmp_path / "i.png", "E1")
    assert out["counts"]["persons"] == 0 and out["summary"]
