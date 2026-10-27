"""Disk-backed JSON session persistence (session_<id>.json)."""
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_lock = threading.RLock()


def valid_id(session_id: str) -> bool:
    return bool(session_id and _ID_RE.match(session_id))


class SessionStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id):
        if not valid_id(session_id):
            raise ValueError("Invalid session id")
        return self.folder / f"session_{session_id}.json"

    def create(self, title=None) -> dict:
        sid = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        session = {
            "id": sid,
            "title": title or f"Case {sid[:6]}",
            "created": now,
            "updated": now,
            "files": [],
            "results": {"images": [], "audio": [], "video": []},
            "fusion": None,
            "chat": [],
        }
        self.save(session)
        return session

    def get(self, session_id):
        try:
            path = self._path(session_id)
        except ValueError:
            return None
        with _lock:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

    def save(self, session: dict) -> dict:
        session["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self._path(session["id"])
        tmp = path.with_suffix(".tmp")
        with _lock:
            tmp.write_text(json.dumps(session, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        return session

    def list(self) -> list:
        out = []
        for p in self.folder.glob("session_*.json"):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                out.append(
                    {
                        "id": s["id"],
                        "title": s.get("title"),
                        "updated": s.get("updated"),
                        "files": len(s.get("files", [])),
                    }
                )
            except (OSError, ValueError, KeyError):
                continue
        return sorted(out, key=lambda s: s["updated"] or "", reverse=True)

    def delete(self, session_id) -> bool:
        try:
            path = self._path(session_id)
        except ValueError:
            return False
        with _lock:
            if path.exists():
                path.unlink()
                return True
        return False


def get_store() -> SessionStore:
    return SessionStore(current_app.config["SESSION_FOLDER"])


def require_session(session_id: str) -> dict:
    """Return the session or abort with a JSON 404."""
    from flask import abort

    session = get_store().get(session_id)
    if session is None:
        abort(404, description="Session not found")
    return session
