"""Thin Groq wrapper. Returns None on any failure so callers can fall back."""
import base64
import io
import json
import re

from flask import current_app

from .logger import get_logger

log = get_logger(__name__)
_clients = {}
_last_error = None
_working_model = None


def last_error():
    return _last_error


def cloud_vision_enabled() -> bool:
    return groq_enabled() and bool(current_app.config.get("CLOUD_VISION", True))


def groq_enabled() -> bool:
    return bool(current_app.config.get("GROQ_API_KEY"))


def _client():
    key = current_app.config["GROQ_API_KEY"]
    if key not in _clients:
        from groq import Groq

        _clients[key] = Groq(api_key=key)
    return _clients[key]


def chat(system, user, temperature=None, max_tokens=900, json_mode=False, history=None):
    if not groq_enabled():
        return None
    try:
        messages = [{"role": "system", "content": system}]
        messages += history or []
        messages.append({"role": "user", "content": user})
        kwargs = {
            "model": current_app.config["GROQ_MODEL"],
            "messages": messages,
            "temperature": (
                current_app.config["GROQ_TEMPERATURE"]
                if temperature is None
                else temperature
            ),
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        reply = _client().chat.completions.create(**kwargs)
        return reply.choices[0].message.content
    except Exception as exc:
        global _last_error
        _last_error = str(exc)[:300]
        log.warning("Groq call failed: %s", exc)
        return None


def parse_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                return None
    return None


def _jpeg_b64(pil, max_side=1024) -> str:
    image = pil.convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def vision(prompt, images, max_tokens=1800):
    """Ask a Groq vision model about 1-5 PIL images. Returns text or None."""
    global _last_error, _working_model
    if not cloud_vision_enabled():
        return None
    models = [m.strip() for m in current_app.config["GROQ_VISION_MODELS"].split(",") if m.strip()]
    if _working_model in models:  # remember the model that worked
        models.remove(_working_model)
        models.insert(0, _working_model)
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(im)}"}} for im in images[:5]]
    for model in models:
        try:
            reply = _client().chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}], temperature=0.1, max_tokens=max_tokens
            )
            text = re.sub(r"<think>.*?</think>", "", reply.choices[0].message.content or "", flags=re.S).strip()
            _working_model, _last_error = model, None
            return text
        except Exception as exc:
            _last_error = f"{model}: {str(exc)[:240]}"
            log.warning("Groq vision failed with %s: %s", model, exc)
    return None
