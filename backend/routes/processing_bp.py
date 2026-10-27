"""Full multimodal pipeline, image analysis, demo case and report download."""
from flask import Blueprint, current_app, jsonify, request, send_file

from ..services import fusion_service, report_service, vision_service
from ..utils import demo_data
from ..utils.logger import get_logger
from ..utils.session_store import get_store, require_session
from .audio_analysis_bp import run_audio_analysis
from .upload_bp import add_file_record
from .video_analysis_bp import run_video_analysis

bp = Blueprint("processing", __name__, url_prefix="/api")
log = get_logger(__name__)


def run_image_analysis(session, language=None) -> list:
    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    session["results"]["images"], errors = [], []
    for f in session["files"]:
        if f["kind"] != "image":
            continue
        try:
            session["results"]["images"].append(vision_service.analyze_image(folder / f["name"], f["id"], f.get("offset")))
        except Exception as exc:
            log.warning("Image analysis failed for %s: %s", f["name"], exc)
            errors.append({"file": f["name"], "error": str(exc)})
    return errors


@bp.post("/process/<sid>")
def process(sid):
    session = require_session(sid)
    if not session["files"]:
        return jsonify({"error": "Add evidence before running the analysis."}), 400
    language = (request.get_json(silent=True) or {}).get("language")
    errors = []
    for runner in (run_image_analysis, run_audio_analysis, run_video_analysis):
        errors += runner(session, language)
    if not any(session["results"].values()):
        get_store().save(session)
        return jsonify({"error": "None of the files could be analysed.", "errors": errors}), 422
    session["fusion"] = fusion_service.fuse(session)
    get_store().save(session)
    return jsonify({"session": session, "errors": errors})


@bp.post("/images/<sid>/analyze")
def analyze_images(sid):
    session = require_session(sid)
    if not any(f["kind"] == "image" for f in session["files"]):
        return jsonify({"error": "No image files in this session."}), 400
    errors = run_image_analysis(session)
    session["fusion"] = fusion_service.fuse(session)
    get_store().save(session)
    return jsonify({"results": session["results"]["images"], "errors": errors, "fusion": session["fusion"]})


@bp.get("/process/<sid>/result")
def result(sid):
    session = require_session(sid)
    if not session["fusion"]:
        return jsonify({"error": "Analysis has not been run for this session."}), 404
    return jsonify(session["fusion"])


@bp.get("/report/<sid>")
def report(sid):
    session = require_session(sid)
    fmt = request.args.get("format", "pdf").lower()
    try:
        path = report_service.generate_docx(session) if fmt == "docx" else report_service.generate_pdf(session)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    return send_file(path, as_attachment=True, download_name=f"crimevision_report_{sid}.{path.suffix[1:]}")


def create_demo_session() -> dict:
    """Create a ready-to-run synthetic case so the interface can be tried instantly."""
    store = get_store()
    session = store.create("Demo case: Central Market")
    folder = current_app.config["UPLOAD_FOLDER"] / session["id"]
    folder.mkdir(parents=True, exist_ok=True)
    demo_data.make_image(folder / "scene_photo.png")
    (folder / "witness_statement.txt").write_text(demo_data.DEMO_TRANSCRIPT, encoding="utf-8")
    video_path = demo_data.make_video(folder / "cam02_footage")
    for path, kind in ((folder / "scene_photo.png", "image"), (folder / "witness_statement.txt", "transcript"), (video_path, "video")):
        add_file_record(session, path.name, kind, path.stat().st_size, None)
    store.save(session)
    return session


@bp.post("/demo")
def demo():
    return jsonify(create_demo_session()), 201
