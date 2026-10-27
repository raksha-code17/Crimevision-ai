"""Evidence-grounded chatbot API."""
from flask import Blueprint, jsonify, request

from ..services import chat_service
from ..utils.session_store import require_session

bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@bp.post("/<sid>")
def chat(sid):
    require_session(sid)
    message = ((request.get_json(silent=True) or {}).get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 1000:
        return jsonify({"error": "message must be 1000 characters or fewer"}), 400
    return jsonify(chat_service.ask(sid, message))


@bp.get("/<sid>/history")
def history(sid):
    return jsonify(require_session(sid)["chat"])


@bp.delete("/<sid>/history")
def clear(sid):
    from ..utils.session_store import get_store

    session = require_session(sid)
    session["chat"] = []
    get_store().save(session)
    return jsonify({"cleared": True})
