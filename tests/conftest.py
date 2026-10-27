import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import create_app  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": tmp_path / "uploads",
        "SESSION_FOLDER": tmp_path / "sessions",
        "REPORT_FOLDER": tmp_path / "reports",
        "USE_ML_MODELS": False,   # heuristics only: fast, offline
        "GROQ_API_KEY": "",
    })
    with app.app_context():
        yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def demo_session(client):
    res = client.post("/api/demo")
    assert res.status_code == 201
    return res.get_json()
