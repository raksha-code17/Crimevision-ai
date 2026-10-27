import { el, emit, fmtTime, on, timelineModel } from "../chat_handler.js";

export function mountTimeline(root) {
  let session = null, model = null, chosen = null, playhead = null;

  const pct = (t) => `${(t / model.duration) * 100}%`;

  function ruler() {
    const step = model.duration <= 30 ? 5 : model.duration <= 120 ? 15 : 60;
    const track = el("div", { class: "track", style: { background: "none", cursor: "default", height: "16px" } });
    for (let t = 0; t <= model.duration; t += step) track.append(el("span", { class: "tick", style: { left: pct(t) } }, fmtTime(t)));
    return el("div", { class: "ruler" }, el("span"), track);
  }

  function seekFromClick(e, track) {
    const r = track.getBoundingClientRect();
    emit("seek", Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * model.duration);
  }

  function lane(kind, label, build) {
    const track = el("div", { class: "track", onclick: (e) => { if (e.target === track || e.target.closest("svg")) seekFromClick(e, track); } });
    build(track);
    return el("div", { class: `lane ${kind}` }, el("span", { class: "lane-name" }, label), track);
  }

  function choose(item) { chosen = item; renderDetail(); }

  let detailEl = null;
  function renderDetail() {
    if (!detailEl) return;
    detailEl.replaceChildren(...(chosen
      ? [el("time", {}, fmtTime(chosen.t)), el("span", { class: "muted" }, chosen.source), el("span", { class: "serif" }, chosen.text)]
      : [el("span", { class: "muted" }, "Select an event or speech segment to read it. Click empty track space to jump the video.")]));
  }

  function render() {
    root.replaceChildren(el("div", { class: "panel-head" }, el("h2", {}, "Timeline"), el("span", { class: "muted" }, model ? `${session.fusion.timeline.length} entries · ${fmtTime(model.duration)}` : "")));
    if (!model) { root.append(el("div", { class: "empty" }, el("p", {}, "The combined timeline appears after you run the analysis."))); return; }

    const body = el("div", { class: "tl-body" });
    body.append(ruler());

    body.append(lane("video", "Video", (track) => {
      if (model.activity.length > 1) {
        const peak = Math.max(0.05, ...model.activity.map((a) => a.motion));
        const pts = model.activity.map((a) => `${(a.t / model.duration) * 1000},${40 - (a.motion / peak) * 36}`).join(" ");
        track.innerHTML = `<svg class="activity" viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true"><polyline points="0,40 ${pts} 1000,40" fill="rgba(44,92,143,.22)" stroke="#2c5c8f" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
      }
      for (const it of model.lanes.video) track.append(el("button", { class: `pin ${it.severity}`, type: "button", title: `${fmtTime(it.t)} ${it.text}`, style: { left: pct(it.t) }, onclick: () => { choose(it); emit("seek", it.t); } }));
    }));

    body.append(lane("audio", "Speech", (track) => {
      for (const it of model.lanes.audio) track.append(el("button", { class: "seg", type: "button", title: `${fmtTime(it.t)} ${it.text}`, style: { left: pct(it.t), width: `${Math.max(0.6, (((it.end ?? it.t + 2) - it.t) / model.duration) * 100)}%` }, onclick: () => { choose(it); emit("seek", it.t); } }));
    }));

    const stills = [...model.lanes.image, ...model.undated.filter((i) => i.source === "image")];
    if (stills.length) {
      body.append(el("div", { class: "lane image" }, el("span", { class: "lane-name" }, "Photos"),
        el("div", { class: "chips" }, stills.map((it) => el("button", { class: "chip", type: "button", onclick: () => { choose(it); emit("select", it.ref); if (it.t != null) emit("seek", it.t); } }, el("span", { class: "marker sm" }, it.ref), it.t == null ? "undated" : fmtTime(it.t))))));
    }

    playhead = el("div", { class: "playhead", style: { left: "0" } });
    detailEl = el("div", { class: "tl-detail" });
    root.append(body, detailEl);
    renderDetail();
    positionPlayhead(0);
  }

  function positionPlayhead(t) {
    const track = root.querySelector(".lane .track");
    const body = root.querySelector(".tl-body");
    if (!track || !body || !model) return;
    if (!playhead.isConnected) body.append(playhead);
    const left = track.offsetLeft + (Math.min(t, model.duration) / model.duration) * track.offsetWidth;
    playhead.style.left = `${left}px`;
  }

  on("session", (s) => { session = s; model = timelineModel(s); chosen = null; render(); });
  on("time", positionPlayhead);
  window.addEventListener("resize", () => positionPlayhead(0));
  render();
}
