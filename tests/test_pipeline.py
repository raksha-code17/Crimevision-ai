def _run(client, demo_session):
    res = client.post(f"/api/process/{demo_session['id']}")
    assert res.status_code == 200, res.get_json()
    return res.get_json()


def test_full_pipeline_on_demo_case(client, demo_session):
    body = _run(client, demo_session)
    fusion = body["session"]["fusion"]
    assert body["errors"] == []
    sources = {i["source"] for i in fusion["timeline"]}
    assert sources == {"video", "audio", "image"}
    timed = [i["t"] for i in fusion["timeline"] if i["t"] is not None]
    assert timed == sorted(timed)
    assert len(fusion["hypotheses"]["items"]) == 3
    assert fusion["facts"]["when"] and fusion["facts"]["who"]


def test_separate_endpoints(client, demo_session):
    sid = demo_session["id"]
    assert client.post(f"/api/images/{sid}/analyze").status_code == 200
    assert client.post(f"/api/audio/{sid}/analyze", json={"language": "en"}).status_code == 200
    assert client.post(f"/api/video/{sid}/analyze").status_code == 200
    assert client.get(f"/api/process/{sid}/result").status_code == 200
    assert client.get(f"/api/audio/{sid}/transcript").get_json()[0]["segments"]
    assert client.get(f"/api/video/{sid}/timeline").get_json()[0]["events"]


def test_result_404_before_analysis(client, demo_session):
    assert client.get(f"/api/process/{demo_session['id']}/result").status_code == 404


def test_chat_temporal_query_is_grounded(client, demo_session):
    _run(client, demo_session)
    sid = demo_session["id"]
    body = client.post(f"/api/chat/{sid}", json={"message": "What happened at 00:08?"}).get_json()
    assert body["engine"] == "deterministic"
    assert "knife" in body["answer"].lower()
    assert all(s["t"] is None or abs(s["t"] - 8) <= 3 for s in body["sources"])
    assert len(client.get(f"/api/chat/{sid}/history").get_json()) == 2


def test_chat_declines_when_not_in_evidence(client, demo_session):
    _run(client, demo_session)
    body = client.post(f"/api/chat/{demo_session['id']}", json={"message": "What colour was the getaway car?"}).get_json()
    assert "does not say" in body["answer"].lower()


def test_chat_before_analysis(client, demo_session):
    body = client.post(f"/api/chat/{demo_session['id']}", json={"message": "hi"}).get_json()
    assert "Run analysis" in body["answer"]


def test_pdf_report(client, demo_session):
    _run(client, demo_session)
    res = client.get(f"/api/report/{demo_session['id']}")
    assert res.status_code == 200 and res.data[:5] == b"%PDF-" and len(res.data) > 3000


def test_docx_report(client, demo_session):
    _run(client, demo_session)
    res = client.get(f"/api/report/{demo_session['id']}?format=docx")
    assert res.status_code == 200 and res.data[:2] == b"PK"


def test_report_requires_analysis(client, demo_session):
    assert client.get(f"/api/report/{demo_session['id']}").status_code == 409


def test_served_video_supports_range(client, demo_session):
    name = next(f["name"] for f in demo_session["files"] if f["kind"] == "video")
    res = client.get(f"/api/files/{demo_session['id']}/{name}", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206
