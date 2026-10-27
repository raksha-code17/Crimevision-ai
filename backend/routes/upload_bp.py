"""Session management and bulk evidence upload."""
import shutil

from flask import Blueprint, abort, current_app, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from ..services import fusion_service
from ..utils.session_store import get_store, require_session

bp = Blueprint("upload", __name__, url_prefix="/api")


def classify_file(filename: str):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    for kind, exts in current_app.config["ALLOWED_EXTENSIONS"].items():
        if ext in exts:
            return kind
    return None


def unique_name(folder, name):
    stem, dot, ext = name.rpartition(".")
    candidate, n = name, 2
    while (folder / candidate).exists():
        candidate = f"{stem}_{n}.{ext}" if dot else f"{name}_{n}"
        n += 1
    return candidate


def add_file_record(session, name, kind, size, offset=None):
    next_n = max([int(f["id"][1:]) for f in session["files"]] + [0]) + 1  # never reuse an id after a delete
    record = {"id": f"E{next_n}", "name": name, "kind": kind, "size": size, "offset": offset}
    session["files"].append(record)
    return record


def add_evidence(session, filename, data: bytes, offset=None):
    """Save one in-memory file into the session. Returns its record, or None if the type is unsupported."""
    original = secure_filename(filename or "")
    kind = classify_file(original)
    if not original or kind is None:
        return None
    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    folder.mkdir(parents=True, exist_ok=True)
    name = unique_name(folder, original)
    (folder / name).write_bytes(data)
    session["fusion"] = None  # evidence changed: previous analysis is stale
    return add_file_record(session, name, kind, len(data), offset)


def remove_evidence(session, fid) -> bool:
    """Remove one evidence file, its results and derived images; refresh the fused analysis."""
    record = next((f for f in session["files"] if f["id"] == fid), None)
    if record is None:
        return False
    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    (folder / record["name"]).unlink(missing_ok=True)
    for derived in (folder / "_derived").glob(f"{fid}_*"):  # annotated images and key frames
        derived.unlink(missing_ok=True)
    session["files"] = [f for f in session["files"] if f["id"] != fid]
    for kind, items in session["results"].items():
        session["results"][kind] = [r for r in items if r["evidence_id"] != fid]
    session["fusion"] = fusion_service.fuse(session) if any(session["results"].values()) else None
    return True


@bp.post("/sessions")
def create_session():
    title = (request.get_json(silent=True) or {}).get("title")
    return jsonify(get_store().create(title)), 201


@bp.get("/sessions")
def list_sessions():
    return jsonify(get_store().list())


@bp.get("/sessions/<sid>")
def get_session(sid):
    return jsonify(require_session(sid))


@bp.delete("/sessions/<sid>")
def delete_session(sid):
    require_session(sid)
    get_store().delete(sid)
    shutil.rmtree(current_app.config["UPLOAD_FOLDER"] / sid, ignore_errors=True)
    return jsonify({"deleted": sid})


@bp.post("/upload")
def upload():
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files received. Send multipart field 'files'."}), 400
    store = get_store()
    sid = request.form.get("session_id")
    session = require_session(sid) if sid else store.create()
    try:
        offset = float(request.form["offset_seconds"]) if request.form.get("offset_seconds") else None
    except ValueError:
        return jsonify({"error": "offset_seconds must be a number"}), 400

    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    folder.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        original = secure_filename(f.filename or "")
        kind = classify_file(original)
        if not original or kind is None:
            rejected.append({"name": f.filename, "reason": "Unsupported file type"})
            continue
        name = unique_name(folder, original)
        f.save(folder / name)
        record = add_file_record(session, name, kind, (folder / name).stat().st_size, offset)
        saved.append(record)
    if saved:
        session["fusion"] = None  # evidence changed: previous analysis is stale
        store.save(session)
    return jsonify({"session": session, "saved": saved, "rejected": rejected}), (201 if saved else 400)


@bp.get("/files/<sid>/<path:name>")
def serve_file(sid, name):
    require_session(sid)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"] / sid, name, conditional=True)


@bp.delete("/sessions/<sid>/files/<fid>")
def delete_file(sid, fid):
    """Remove one evidence file and refresh the analysis."""
    session = require_session(sid)
    if not remove_evidence(session, fid):
        abort(404, description="Evidence file not found")
    get_store().save(session)
    return jsonify(session)
