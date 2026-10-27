"""CrimeVision AI: Flask application factory and blueprint registration."""
import sys
from pathlib import Path

# Allow `python backend/app.py` as well as `python -m backend.app`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "backend"

from flask import Flask, jsonify, send_from_directory  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from .config import Config  # noqa: E402
from .routes import audio_analysis_bp, chat_bp, processing_bp, upload_bp, video_analysis_bp  # noqa: E402
from .utils import model_loader  # noqa: E402
from .utils.groq_client import cloud_vision_enabled, groq_enabled, last_error  # noqa: E402
from .utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def _enable_cors(app):
    """Flask-CORS for port 3000; a minimal built-in fallback if it is not installed."""
    try:
        from flask_cors import CORS

        CORS(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})
    except ImportError:
        log.warning("flask-cors not installed; using built-in CORS headers")

        @app.after_request
        def add_headers(resp):
            from flask import request

            origin = request.headers.get("Origin")
            if origin in app.config["CORS_ORIGINS"]:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
                resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
            return resp


def create_app(test_config=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    for key in ("UPLOAD_FOLDER", "SESSION_FOLDER", "REPORT_FOLDER"):
        app.config[key] = Path(app.config[key])
        app.config[key].mkdir(parents=True, exist_ok=True)

    _enable_cors(app)
    for module in (upload_bp, processing_bp, audio_analysis_bp, video_analysis_bp, chat_bp):
        app.register_blueprint(module.bp)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "groq": groq_enabled(), "cloud_vision": cloud_vision_enabled(), "cloud_vision_error": last_error(), "models": model_loader.status(), "ml_enabled": app.config["USE_ML_MODELS"]})

    frontend = Path(app.config["FRONTEND_DIR"])

    @app.get("/")
    def index():
        return send_from_directory(frontend, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename):
        if filename.startswith("api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(frontend, filename)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        message = "File too large. Raise MAX_UPLOAD_MB in .env." if exc.code == 413 else exc.description
        return jsonify({"error": message}), exc.code

    @app.errorhandler(Exception)
    def unexpected(exc):
        log.exception("Unhandled error")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host=Config.HOST, port=Config.PORT, debug=False, threaded=True)
