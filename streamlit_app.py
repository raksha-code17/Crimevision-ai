"""CrimeVision AI: Streamlit interface.

Runs the same analysis engine as the Flask dashboard (backend/services), so results are identical.
Start locally with:  streamlit run streamlit_app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="CrimeVision AI", page_icon="🔎", layout="wide")

ON_CLOUD = Path("/mount/src").exists()  # Streamlit Community Cloud
BASE = Path(os.environ.get("CRIMEVISION_DATA", Path(tempfile.gettempdir()) / "crimevision"))
VIDEO_MIME = {"mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime", "avi": "video/x-msvideo", "mkv": "video/x-matroska"}
STATUS_TEXT = {"corroborated": ":green[Corroborated]", "unsupported": ":red[Not supported]", "unverifiable": ":gray[Can't verify]"}
SUGGESTIONS = ["What happened at 00:08?", "Who was involved?", "Where did it take place?", "Summarize the case"]


@st.cache_resource(show_spinner=False)
def get_app():
    from backend.app import create_app

    default_ml = "false" if ON_CLOUD else "true"  # free cloud servers are too small for the AI models
    use_ml = os.environ.get("USE_ML_MODELS", default_ml).strip().lower() in {"1", "true", "yes", "on"}
    return create_app({
        "USE_ML_MODELS": use_ml,
        "UPLOAD_FOLDER": BASE / "uploads",
        "SESSION_FOLDER": BASE / "sessions",
        "REPORT_FOLDER": BASE / "reports",
        "VIDEO_MAX_SECONDS": 60 if ON_CLOUD else 300,
    })


def fmt_size(n):
    return f"{n / 1e6:.1f} MB" if n > 1e6 else f"{max(1, round(n / 1e3))} KB"


def tag_text(tags):
    return ", ".join(f"{t['label']} {int(t['score'] * 100)}%" for t in tags)


def main():
    from backend.routes.audio_analysis_bp import run_audio_analysis
    from backend.routes.processing_bp import create_demo_session, run_image_analysis
    from backend.routes.upload_bp import add_evidence, remove_evidence
    from backend.routes.video_analysis_bp import run_video_analysis
    from backend.services import chat_service, fusion_service, report_service
    from backend.services.fusion_service import fmt_ts
    from backend.utils import model_loader
    from backend.utils.session_store import get_store
    from flask import current_app

    store = get_store()
    ss = st.session_state
    session = store.get(ss.get("sid") or "")
    if session is None:
        session = store.create()
        ss["sid"] = session["id"]
    sid, folder = session["id"], current_app.config["UPLOAD_FOLDER"] / session["id"]
    fusion = session["fusion"]

    # ------------------------------------------------------------- sidebar
    with st.sidebar:
        st.markdown("### 🔎 CrimeVision AI")
        st.caption(f"Case: {session['title']}")
        left, right = st.columns(2)
        if left.button("New case"):
            ss["sid"] = store.create()["id"]
            st.rerun()
        if right.button("Demo case"):
            with st.spinner("Building demo case..."):
                ss["sid"] = create_demo_session()["id"]
            st.rerun()

        exts = [e for group in current_app.config["ALLOWED_EXTENSIONS"].values() for e in sorted(group)]
        uploads = st.file_uploader("Add evidence (photos, video, audio, transcript)", accept_multiple_files=True, type=exts, key=f"up_{ss.get('up_n', 0)}")
        if uploads:
            skipped = [f.name for f in uploads if add_evidence(session, f.name, f.getvalue()) is None]
            store.save(session)
            ss["up_n"] = ss.get("up_n", 0) + 1
            ss["notice"] = f"Skipped unsupported files: {', '.join(skipped)}" if skipped else None
            st.rerun()

        with st.expander("Diagnostics"):
            import platform

            try:
                import cv2

                cv_info = f"OpenCV {cv2.__version__}, people detector: {'yes' if hasattr(cv2, 'HOGDescriptor') else 'MISSING'}"
            except Exception as exc:
                cv_info = f"OpenCV failed to load: {exc}"
            st.code(f"Python {platform.python_version()}\n{cv_info}\nModels loaded: {', '.join(model_loader.status()['loaded']) or 'none'}")

        st.markdown("**Evidence**")
        if not session["files"]:
            st.caption("No evidence yet. Upload files above or load the demo case.")
        for f in session["files"]:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{f['id']}** · {f['name']}  \n{f['kind']} · {fmt_size(f['size'])}")
            if c2.button("✕", key=f"rm_{sid}_{f['id']}", help=f"Remove {f['name']}"):
                remove_evidence(session, f["id"])
                store.save(session)
                st.rerun()

        if st.button("Run analysis", type="primary", disabled=not session["files"]):
            errors = []
            with st.spinner("Analysing evidence. Long videos can take a few minutes..."):
                for runner in (run_image_analysis, run_audio_analysis, run_video_analysis):
                    errors += runner(session, None)
                if any(session["results"].values()):
                    session["fusion"] = fusion_service.fuse(session)
                store.save(session)
            ss["errors"] = errors
            st.rerun()

    # ---------------------------------------------------------------- header
    st.title("CrimeVision AI")
    st.caption("Multimodal forensic analysis: photos, video, audio and transcripts on one timeline. AI-generated leads, not proof.")
    if ss.get("notice"):
        st.warning(ss["notice"])
    for err in ss.get("errors") or []:
        st.error(f"{err['file']}: {err['error']}")
    loaded = model_loader.status()["loaded"]
    if not ({"rtdetr", "clip"} & set(loaded)):
        st.info("Basic detection mode: people and motion are analysed, and weapons are reported as 'not checked'. "
                "Install the AI models (requirements-ai.txt) and run locally for knife detection and forensic tags.")

    tab_ev, tab_tl, tab_find, tab_chat, tab_rep = st.tabs(["Evidence", "Timeline", "Findings", "Ask the evidence", "Report"])

    # -------------------------------------------------------------- evidence
    with tab_ev:
        if not session["files"]:
            st.subheader("No evidence open")
            st.write("Upload photos, video, audio or a transcript in the sidebar, or click **Demo case** to see how it works.")
        else:
            files = session["files"]
            chosen = files[st.selectbox("Open evidence", list(range(len(files))), format_func=lambda n: f"{files[n]['id']} · {files[n]['name']} ({files[n]['kind']})")]
            path, res = folder / chosen["name"], session["results"]
            if chosen["kind"] == "image":
                r = next((i for i in res["images"] if i["evidence_id"] == chosen["id"]), None)
                shown = folder / r["annotated"] if r and (folder / r["annotated"]).exists() else path
                st.image(str(shown), width=640, caption="Boxes mark detected people and weapon-class objects" if r else chosen["name"])
                if r:
                    st.write(r["summary"])
                    if r["tags"]:
                        st.caption("Possible indicators (unverified): " + tag_text(r["tags"]))
                else:
                    st.caption("Run the analysis to see what the AI finds in this image.")
            elif chosen["kind"] == "video":
                r = next((v for v in res["video"] if v["evidence_id"] == chosen["id"]), None)
                st.video(str(path), format=VIDEO_MIME.get(path.suffix[1:].lower(), "video/mp4"), start_time=int(ss.get("jump", 0)))
                if r:
                    st.write(f"{r['meta']['duration']}s · peak {r['stats']['peak_persons']} people · incident: {r['crime_type']['type']}")
                    if r["narrative"]:
                        st.write(r["narrative"])
                    cols = st.columns(max(1, min(5, len(r["keyframes"]))))
                    for col, kf in zip(cols, r["keyframes"]):
                        col.image(str(folder / kf["file"]), caption=f"{fmt_ts(kf['t'])} {kf['description'] or tag_text(kf['tags'])}"[:90])
                else:
                    st.caption("Run the analysis to see detections and key frames.")
            elif chosen["kind"] == "audio":
                st.audio(str(path))
            if chosen["kind"] in ("audio", "transcript"):
                r = next((a for a in res["audio"] if a["evidence_id"] == chosen["id"]), None)
                if r:
                    st.dataframe([{"Time": fmt_ts(s["start"]), "Speech": s["text"]} for s in r["segments"]], hide_index=True)
                else:
                    st.caption("Run the analysis to see the transcript.")

    # -------------------------------------------------------------- timeline
    with tab_tl:
        if not fusion:
            st.info("The combined timeline appears after you run the analysis.")
        else:
            videos = session["results"]["video"]
            if videos:
                frames = sorted((f for v in videos for f in v["timeline"]), key=lambda f: f["t"])
                st.caption("Video activity per second (x-axis: seconds into the video)")
                st.area_chart({"Motion (%)": [round(f["motion"] * 100, 1) for f in frames]})
                st.line_chart({"People": [f["persons"] for f in frames]})
            entries = [i for i in fusion["timeline"] if i["t"] is not None]
            if entries and videos:
                pick = entries[st.selectbox("Jump the video to a moment", list(range(len(entries))),
                                            format_func=lambda n: f"{fmt_ts(entries[n]['t'])} · {entries[n]['source']} · {entries[n]['text'][:70]}")]
                ss["jump"] = int(pick["t"])
                vfile = next(f for f in session["files"] if f["kind"] == "video")
                st.video(str(folder / vfile["name"]), format=VIDEO_MIME.get((folder / vfile["name"]).suffix[1:].lower(), "video/mp4"), start_time=ss["jump"])
            sources = st.multiselect("Show sources", ["video", "audio", "image"], default=["video", "audio", "image"])
            st.dataframe([{"Time": fmt_ts(i["t"]), "Source": i["source"], "Event": i["text"], "Level": i["severity"]}
                          for i in fusion["timeline"] if i["source"] in sources], hide_index=True)

    # -------------------------------------------------------------- findings
    with tab_find:
        if not fusion:
            st.info("Select **Run analysis** in the sidebar to classify the incident and draft scenarios." if session["files"] else "Add evidence to begin.")
        else:
            crime, cons = fusion["crime_type"], fusion["consistency"]
            st.subheader(f"Incident: {crime['type']}")
            st.progress(float(min(1.0, max(0.0, crime["confidence"]))))
            st.caption(f"{int(crime['confidence'] * 100)}% confidence. {crime['rationale']}")

            st.subheader("What the AI saw")
            for im in session["results"]["images"]:
                a, b = st.columns([1, 2])
                shown = folder / im["annotated"]
                if shown.exists():
                    a.image(str(shown), width=260)
                b.markdown(f"**{im['evidence_id']} · {im['file']}**")
                b.write(im["summary"])
                if im["tags"]:
                    b.caption("Possible indicators (unverified): " + tag_text(im["tags"]))
            for v in session["results"]["video"]:
                st.markdown(f"**{v['evidence_id']} · {v['file']}**: {v['narrative'] or 'Detections, motion and key frames are shown for this video.'}")

            st.subheader("Do the statements match the footage?")
            st.metric("Statements backed by visual evidence", f"{cons['score']}%" if cons["score"] is not None else "n/a")
            st.caption(cons["summary"])
            for cl in cons["claims"]:
                st.markdown(f"**{fmt_ts(cl['t'])}** {STATUS_TEXT[cl['status']]} · {cl['type']}: {cl['evidence']}")

            st.subheader("Who, where, when")
            for k in ("who", "where", "when"):
                st.markdown(f"**{k.capitalize()}:** {'; '.join(fusion['facts'][k]) or 'Not stated'}")

            st.subheader("Three scenarios")
            for h in fusion["hypotheses"]["items"]:
                st.markdown(f"**{h['title']}** ({int(h['confidence'] * 100)}%)")
                st.progress(float(min(1.0, max(0.0, h["confidence"]))))
                st.write(h["narrative"])
                with st.expander("Supporting evidence and gaps"):
                    st.write("Supporting: " + ("; ".join(h["supporting_evidence"]) or "none listed"))
                    st.write("Gaps: " + ("; ".join(h["gaps"]) or "none listed"))

    # ------------------------------------------------------------------ chat
    with tab_chat:
        if not fusion:
            st.info("Run the analysis first, then ask questions about what was found.")
        else:
            st.caption("Answers come only from the analysed evidence. Times use MM:SS.")
            for m in session["chat"]:
                with st.chat_message("user" if m["role"] == "user" else "assistant"):
                    st.write(m["content"])
            question = st.chat_input("Ask about the evidence, e.g. What happened at 00:08?")
            if not session["chat"]:
                cols = st.columns(len(SUGGESTIONS))
                for col, text in zip(cols, SUGGESTIONS):
                    if col.button(text, key=f"sg_{text}"):
                        question = text
            if question:
                with st.spinner("Checking the evidence..."):
                    chat_service.ask(sid, question)
                st.rerun()

    # ---------------------------------------------------------------- report
    with tab_rep:
        if not fusion:
            st.info("Run the analysis to enable the report.")
        else:
            st.write("The report includes annotated images, key frames, the timeline, facts, statement checks and scenarios.")
            if st.button("Generate report files"):
                with st.spinner("Building report..."):
                    ss["reports"] = {"sid": sid, "pdf": report_service.generate_pdf(store.get(sid)).read_bytes(),
                                     "docx": report_service.generate_docx(store.get(sid)).read_bytes()}
            ready = ss.get("reports")
            if ready and ready["sid"] == sid:
                st.download_button("Download PDF report", ready["pdf"], file_name=f"crimevision_report_{sid}.pdf", mime="application/pdf")
                st.download_button("Download Word report", ready["docx"], file_name=f"crimevision_report_{sid}.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


with get_app().app_context():
    main()
