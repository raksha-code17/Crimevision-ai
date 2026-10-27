import { api, el, emit, fmtTime, on } from "../chat_handler.js";

const COLORS = { person: "#f2b705", weapon: "#ff4d2e", object: "#6fb3ff" };

export function mountViewport(root) {
  let session = null, selected = null, pendingSeek = null, media = null, canvas = null, frames = [], cues = [];

  const file = () => session?.files.find((f) => f.id === selected);

  function drawBoxes(boxes, aspectW, aspectH) {
    if (!canvas || !media) return;
    const w = (canvas.width = media.clientWidth), h = (canvas.height = media.clientHeight);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 2; ctx.font = "600 12px Archivo, sans-serif";
    for (const b of boxes || []) {
      const [x1, y1, x2, y2] = b.box, color = COLORS[b.category] || COLORS.object;
      ctx.strokeStyle = color; ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
      const label = `${b.label} ${Math.round(b.score * 100)}%`, tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color; ctx.fillRect(x1 * w, Math.max(0, y1 * h - 16), tw, 16);
      ctx.fillStyle = "#16202a"; ctx.fillText(label, x1 * w + 4, Math.max(12, y1 * h - 4));
    }
  }

  function frameAt(t) {
    let best = null;
    for (const f of frames) { if (f.t <= t + 0.25) best = f; else break; }
    return best;
  }

  function renderVideo(f) {
    const result = session.results.video.find((v) => v.evidence_id === f.id);
    frames = result?.timeline || [];
    const video = el("video", { class: "media", controls: true, preload: "metadata", src: api.fileUrl(session.id, f.name) });
    canvas = el("canvas", { class: "overlay" });
    media = video;
    const sync = () => { drawBoxes(frameAt(video.currentTime)?.boxes); emit("time", video.currentTime); };
    video.addEventListener("timeupdate", sync);
    video.addEventListener("seeked", sync);
    video.addEventListener("loadedmetadata", () => { if (pendingSeek != null) { video.currentTime = pendingSeek; pendingSeek = null; } sync(); });
    video.addEventListener("error", () => {
      const stage = root.querySelector(".stage");
      if (!stage) return;
      stage.append(el("div", { class: "kfs" }, el("p", { style: { color: "#cfd8dd", gridColumn: "1 / -1", margin: 0 } }, "This browser can’t play this video format, so key frames are shown instead."),
        ...(result?.keyframes || []).map((kf) => el("img", { class: "kf", alt: kf.description || "key frame", title: fmtTime(kf.t), src: api.fileUrl(session.id, kf.file) }))));
    });
    new ResizeObserver(() => sync()).observe(video);
    return el("div", { class: "media-wrap" }, video, canvas);
  }

  function renderImage(f) {
    const r = session.results.images.find((i) => i.evidence_id === f.id);
    const img = el("img", { class: "media", alt: r?.summary || f.name, src: api.fileUrl(session.id, f.name) });
    canvas = el("canvas", { class: "overlay" });
    media = img;
    const draw = () => drawBoxes(r?.detections);
    img.addEventListener("load", draw);
    new ResizeObserver(draw).observe(img);
    return el("div", { class: "media-wrap" }, img, canvas);
  }

  function renderCues(f, withAudio) {
    const r = session.results.audio.find((a) => a.evidence_id === f.id);
    cues = r?.segments || [];
    const list = el("div", { class: "transcript" });
    if (!cues.length) list.append(el("p", { class: "muted" }, "Run the analysis to see the transcript."));
    let audio = null;
    if (withAudio) { audio = el("audio", { controls: true, src: api.fileUrl(session.id, f.name), style: { width: "100%", marginBottom: "10px" } }); list.append(audio); }
    cues.forEach((c, i) => list.append(el("button", { class: "cue", type: "button", "data-i": i, onclick: () => { if (audio) audio.currentTime = c.start; emit("seek", c.start); } }, el("time", {}, fmtTime(c.start)), el("span", { class: "serif" }, c.text))));
    if (audio) audio.addEventListener("timeupdate", () => list.querySelectorAll(".cue").forEach((n) => n.classList.toggle("active", cues[+n.dataset.i].start <= audio.currentTime && audio.currentTime < cues[+n.dataset.i].end)));
    return list;
  }

  function render() {
    media = canvas = null;
    const f = file();
    if (!f) {
      root.replaceChildren(el("div", { class: "empty" }, el("h3", {}, "No evidence open"), el("p", {}, "Add photos, video, audio or a transcript, or load the demo case to see how the workspace fits together.")));
      return;
    }
    const head = el("div", { class: "panel-head" }, el("span", { class: "marker" }, f.id), el("h2", {}, f.name),
      el("div", { class: "legend" }, el("span", {}, el("i", { style: { color: COLORS.person } }), "Person"), el("span", {}, el("i", { style: { color: COLORS.weapon } }), "Weapon")));
    let body;
    if (f.kind === "video") body = el("div", { class: "stage" }, renderVideo(f));
    else if (f.kind === "image") {
      const r = session.results.images.find((i) => i.evidence_id === f.id);
      body = el("div", { class: "imgpane" }, el("div", { class: "stage" }, renderImage(f)),
        el("div", { class: "obs" }, el("p", { class: "serif" }, r ? r.summary : "Run the analysis to see what the AI finds in this image."),
          r?.tags?.length ? el("div", { class: "tagrow" }, r.tags.map((t) => el("span", { class: `tag ${t.category}`, title: "Classifier indicator, unverified" }, `${t.label} ${Math.round(t.score * 100)}%`))) : null));
    }
    else body = renderCues(f, f.kind === "audio");
    root.replaceChildren(head, body);
  }

  const firstVideo = () => session?.files.find((f) => f.kind === "video");

  on("session", (s) => {
    const prev = selected;
    session = s;
    if (!s.files.some((f) => f.id === prev)) { selected = (firstVideo() || s.files[0])?.id ?? null; emit("select", selected); }
    render();
  });
  on("select", (id) => { if (id !== selected || !media) { selected = id; render(); } });
  on("seek", (t) => {
    if (media?.tagName === "VIDEO") { media.currentTime = t; return; }
    const v = firstVideo();
    if (v) { pendingSeek = t; selected = v.id; emit("select", v.id); render(); }
  });
  render();
}
