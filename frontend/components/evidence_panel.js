import { el, on, emit } from "../chat_handler.js";

const KIND = { image: "Photo", video: "Video", audio: "Audio", transcript: "Transcript" };
const size = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

function detail(file, results) {
  if (file.kind === "image") {
    const r = results.images.find((i) => i.evidence_id === file.id);
    if (!r) return "";
    const w = r.counts.weapons ? `${r.counts.weapons} weapon(s)` : r.weapon_hints?.length ? `possible ${r.weapon_hints.join(", ")}` : r.weapons_checked === false ? "weapons not checked" : "no weapons";
    return `${r.counts.persons} people · ${w}`;
  }
  if (file.kind === "video") {
    const r = results.video.find((v) => v.evidence_id === file.id);
    return r ? `${r.meta.duration}s · peak ${r.stats.peak_persons} people${r.stats.weapons.length ? ` · ${r.stats.weapons.join(", ")}` : ""}` : "";
  }
  const r = results.audio.find((a) => a.evidence_id === file.id);
  return r ? `${r.segments.length} segments` : "";
}

export function mountEvidencePanel(root) {
  let session = null, selected = null;

  function render() {
    root.replaceChildren(
      el("div", { class: "panel-head" }, el("h2", {}, "Evidence"), el("span", { class: "muted" }, session ? `${session.files.length} item(s)` : "")),
    );
    const pick = el("input", { type: "file", multiple: true, hidden: true, accept: "image/*,audio/*,video/*,.txt,.srt,.vtt", onchange: (e) => emit("files", [...e.target.files]) });
    const zone = el("button", { class: "dropzone", type: "button", onclick: () => pick.click() },
      el("strong", {}, "Drop files here"), "Photos, video, audio or a transcript (.txt, .srt, .vtt)");
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); emit("files", [...e.dataTransfer.files]); });

    const list = el("ul", { class: "evlist" });
    for (const f of session?.files || []) {
      const info = detail(f, session.results);
      list.append(el("li", { class: "evwrap" }, el("button", { class: `evitem kind-${f.kind}`, type: "button", "aria-current": String(f.id === selected), onclick: () => emit("select", f.id) },
        el("span", { class: "marker" }, f.id),
        el("span", {}, el("div", { class: "name" }, f.name), el("div", { class: "meta" }, el("span", { class: "kindtag" }, KIND[f.kind]), ` · ${size(f.size)}`, info ? ` · ${info}` : ""))), el("button", { class: "evremove", type: "button", title: `Remove ${f.name}`, "aria-label": `Remove ${f.name}`, onclick: () => emit("remove", f) }, "×")));
    }
    root.append(pick, zone, el("div", { class: "scroll" }, list));
  }

  on("session", (s) => { session = s; render(); });
  on("select", (id) => { selected = id; render(); });
  render();
}
