"""Video frame extraction, detection and timeline API."""
from flask import Blueprint, current_app, jsonify

from ..services import fusion_service, video_service
from ..utils.logger import get_logger
from ..utils.session_store import get_store, require_session

bp = Blueprint("video", __name__, url_prefix="/api/video")
log = get_logger(__name__)


def run_video_analysis(session, language=None) -> list:
    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    session["results"]["video"], errors = [], []
    for f in session["files"]:
        if f["kind"] != "video":
            continue
        try:
            session["results"]["video"].append(video_service.analyze_video(folder / f["name"], f["id"], f.get("offset") or 0.0))
        except Exception as exc:
            log.warning("Video analysis failed for %s: %s", f["name"], exc)
            errors.append({"file": f["name"], "error": str(exc)})
    return errors


@bp.post("/<sid>/analyze")
def analyze(sid):
    session = require_session(sid)
    if not any(f["kind"] == "video" for f in session["files"]):
        return jsonify({"error": "No video files in this session."}), 400
    errors = run_video_analysis(session)
    session["fusion"] = fusion_service.fuse(session)
    get_store().save(session)
    return jsonify({"results": session["results"]["video"], "errors": errors, "fusion": session["fusion"]})


@bp.get("/<sid>/timeline")
def timeline(sid):
    session = require_session(sid)
    return jsonify([{"file": v["file"], "meta": v["meta"], "events": v["events"], "crime_type": v["crime_type"]} for v in session["results"]["video"]])
