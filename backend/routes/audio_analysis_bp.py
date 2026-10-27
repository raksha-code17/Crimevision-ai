"""Audio / transcript analysis API."""
from flask import Blueprint, current_app, jsonify, request

from ..services import audio_service, fusion_service
from ..utils.logger import get_logger
from ..utils.session_store import get_store, require_session

bp = Blueprint("audio", __name__, url_prefix="/api/audio")
log = get_logger(__name__)


def run_audio_analysis(session, language=None) -> list:
    """Analyse audio files and transcripts. Returns a list of per-file errors."""
    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    session["results"]["audio"], errors = [], []
    for f in session["files"]:
        if f["kind"] not in ("audio", "transcript"):
            continue
        try:
            session["results"]["audio"].append(
                audio_service.analyze_audio_file(folder / f["name"], f["kind"], f["id"], language, f.get("offset") or 0.0)
            )
        except Exception as exc:
            log.warning("Audio analysis failed for %s: %s", f["name"], exc)
            errors.append({"file": f["name"], "error": str(exc)})
    return errors


@bp.post("/<sid>/analyze")
def analyze(sid):
    session = require_session(sid)
    language = (request.get_json(silent=True) or {}).get("language")
    if not any(f["kind"] in ("audio", "transcript") for f in session["files"]):
        return jsonify({"error": "No audio or transcript files in this session."}), 400
    errors = run_audio_analysis(session, language)
    session["fusion"] = fusion_service.fuse(session)
    get_store().save(session)
    return jsonify({"results": session["results"]["audio"], "errors": errors, "fusion": session["fusion"]})


@bp.get("/<sid>/transcript")
def transcript(sid):
    session = require_session(sid)
    return jsonify([{"file": a["file"], "segments": a["segments"], "facts": a["facts"]} for a in session["results"]["audio"]])
