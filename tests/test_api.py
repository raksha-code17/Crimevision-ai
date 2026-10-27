import io


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok" and body["groq"] is False


def test_frontend_is_served(client):
    assert client.get("/").status_code == 200


def test_unknown_api_route_is_json_404(client):
    res = client.get("/api/nope")
    assert res.status_code == 404 and "error" in res.get_json()


def test_session_lifecycle(client):
    sid = client.post("/api/sessions", json={"title": "T"}).get_json()["id"]
    assert client.get(f"/api/sessions/{sid}").get_json()["title"] == "T"
    assert any(s["id"] == sid for s in client.get("/api/sessions").get_json())
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_invalid_session_id_rejected(client):
    assert client.get("/api/sessions/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/sessions/ZZZ").status_code == 404


def test_upload_rejects_bad_types_and_accepts_good(client):
    data = {"files": [(io.BytesIO(b"x"), "evil.exe"), (io.BytesIO(b"[00:01] hello there"), "t.txt")]}
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 201
    assert [f["kind"] for f in body["saved"]] == ["transcript"]
    assert body["rejected"][0]["name"] == "evil.exe"


def test_upload_without_files_is_400(client):
    assert client.post("/api/upload", data={}, content_type="multipart/form-data").status_code == 400


def test_process_requires_evidence(client):
    sid = client.post("/api/sessions").get_json()["id"]
    assert client.post(f"/api/process/{sid}").status_code == 400


def test_chat_requires_message(client, demo_session):
    assert client.post(f"/api/chat/{demo_session['id']}", json={}).status_code == 400


def test_delete_evidence_file_removes_results_and_never_reuses_ids(client, demo_session):
    sid = demo_session["id"]
    client.post(f"/api/process/{sid}")
    image = next(f for f in demo_session["files"] if f["kind"] == "image")
    body = client.delete(f"/api/sessions/{sid}/files/{image['id']}").get_json()
    assert all(f["id"] != image["id"] for f in body["files"])
    assert body["results"]["images"] == []
    assert all(i["source"] != "image" for i in body["fusion"]["timeline"])
    new = client.post("/api/upload", data={"session_id": sid, "files": (io.BytesIO(b"[00:01] hi"), "x.txt")}, content_type="multipart/form-data").get_json()
    assert new["saved"][0]["id"] not in {f["id"] for f in body["files"]} | {image["id"]}
    assert client.delete(f"/api/sessions/{sid}/files/E99").status_code == 404


def test_deleting_last_file_clears_analysis(client):
    sid = client.post("/api/sessions").get_json()["id"]
    up = client.post("/api/upload", data={"session_id": sid, "files": (io.BytesIO(b"[00:01] hello"), "a.txt")}, content_type="multipart/form-data").get_json()
    client.post(f"/api/process/{sid}")
    body = client.delete(f"/api/sessions/{sid}/files/{up['saved'][0]['id']}").get_json()
    assert body["files"] == [] and body["fusion"] is None
