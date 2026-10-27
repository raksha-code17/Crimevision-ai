import { api, el, emit, fmtTime, on } from "../chat_handler.js";

const LABEL = { corroborated: "Corroborated", unsupported: "Not supported", unverifiable: "Can’t verify" };

export function mountFindings(root) {
  const tagRow = (tags) => (tags?.length ? el("div", { class: "tagrow" }, tags.map((t) => el("span", { class: `tag ${t.category}`, title: "Classifier indicator, unverified" }, `${t.label} ${Math.round(t.score * 100)}%`))) : null);

  function seen(session) {
    const { images, video } = session.results;
    if (!images.length && !video.length) return null;
    return el("div", { class: "sec" }, el("h3", {}, "What the AI saw"),
      ...images.map((im) => el("div", { class: "seen" }, el("div", { class: "seen-head" }, el("span", { class: "marker sm" }, im.evidence_id), el("span", { class: "muted" }, im.file)),
        el("p", { class: "serif" }, im.summary), tagRow(im.tags),
        im.vlm?.evidence_items?.length ? el("p", { class: "muted" }, `Items noted: ${im.vlm.evidence_items.join("; ")}`) : null)),
      ...video.map((v) => el("div", { class: "seen" }, el("div", { class: "seen-head" }, el("span", { class: "marker sm" }, v.evidence_id), el("span", { class: "muted" }, v.file)),
        v.narrative ? el("p", { class: "serif" }, v.narrative) : el("p", { class: "muted" }, "Detections, motion and key frames are shown for this video."),
        v.actions?.length ? el("p", { class: "muted" }, `Actions: ${v.actions.join("; ")}`) : null,
        el("div", { class: "kfs" }, (v.keyframes || []).map((kf) => el("button", { class: "kfbtn", type: "button", title: kf.description || "Jump to this moment", onclick: () => emit("seek", kf.t) },
          el("img", { class: "kf", alt: kf.description || "key frame", src: api.fileUrl(session.id, kf.file) }), el("span", { class: "timebtn" }, fmtTime(kf.t))))))));
  }

  function render(session) {
    const f = session?.fusion;
    if (!f) {
      root.replaceChildren(el("div", { class: "empty" }, el("h3", {}, "No findings yet"),
        el("p", {}, session?.files.length ? "Select Run analysis to classify the incident, check statements against the footage and draft scenarios." : "Add evidence to begin.")));
      return;
    }
    const c = f.crime_type, cons = f.consistency;
    const timeBtn = (t) => el("button", { class: "timebtn", type: "button", onclick: () => emit("seek", t) }, fmtTime(t));

    root.replaceChildren(
      el("div", { class: "sec" }, el("h3", {}, "Incident classification"),
        el("p", { class: "verdict" }, c.type),
        el("div", { class: "bar", role: "img", "aria-label": `Confidence ${Math.round(c.confidence * 100)} percent` }, el("span", { style: { width: `${c.confidence * 100}%` } })),
        el("p", { class: "muted" }, `${Math.round(c.confidence * 100)}% confidence. ${c.rationale}`)),

      seen(session),

      el("div", { class: "sec" }, el("h3", {}, "Do the statements match the footage?"),
        el("div", { class: "score" }, el("b", {}, cons.score == null ? "n/a" : `${cons.score}%`), el("span", { class: "muted" }, cons.summary)),
        ...cons.claims.map((cl) => el("div", { class: "claim" }, timeBtn(cl.t),
          el("div", {}, el("span", { class: `status-tag status-${cl.status}` }, LABEL[cl.status]), ` · ${cl.type}`, el("div", { class: "muted" }, cl.evidence))))),

      el("div", { class: "sec" }, el("h3", {}, "Who, where, when"),
        el("dl", { class: "facts" }, ...["who", "where", "when"].flatMap((k) => [el("dt", {}, k[0].toUpperCase() + k.slice(1)), el("dd", {}, f.facts[k].join("; ") || "Not stated")]))),

      el("div", { class: "sec" }, el("h3", {}, "Three scenarios"),
        ...f.hypotheses.items.map((h) => el("div", { class: "hyp" },
          el("div", { class: "hyp-top" }, el("span", {}, h.title), el("span", {}, `${Math.round(h.confidence * 100)}%`)),
          el("div", { class: "bar" }, el("span", { style: { width: `${h.confidence * 100}%` } })),
          el("p", { class: "serif" }, h.narrative),
          h.supporting_evidence.length ? el("details", {}, el("summary", {}, "Supporting evidence"), el("ul", {}, h.supporting_evidence.map((x) => el("li", {}, x)))) : null,
          h.gaps.length ? el("details", {}, el("summary", {}, "Gaps"), el("ul", {}, h.gaps.map((x) => el("li", {}, x)))) : null))),
    );
  }
  on("session", render);
  render(null);
}
