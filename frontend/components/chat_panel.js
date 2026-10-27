import { api, el, emit, fmtTime, on } from "../chat_handler.js";

const SUGGESTIONS = ["What happened at 00:08?", "Who was involved?", "Where did it take place?", "Summarize the case"];

export function mountChat(root) {
  let session = null;
  const log = el("div", { class: "messages", "aria-live": "polite" });
  const input = el("input", { type: "text", placeholder: "Ask about the evidence, e.g. what happened at 00:08?", maxlength: 1000, "aria-label": "Question" });
  const send = el("button", { class: "btn primary", type: "submit" }, "Ask");
  const form = el("form", { class: "chatform" }, input, send);
  root.replaceChildren(log, form);

  const sourceChips = (sources) => el("div", { class: "sources" }, sources.filter((s, i, all) => s.t != null && all.findIndex((x) => Math.round(x.t) === Math.round(s.t)) === i).map((s) => el("button", { class: "timebtn", type: "button", title: s.text, onclick: () => emit("seek", s.t) }, fmtTime(s.t))));
  const addUser = (text) => log.append(el("div", { class: "msg user" }, text));
  const addBot = (text, sources = []) => {
    const m = el("div", { class: "msg bot" }, el("div", { class: "body serif" }, text), sources.length ? sourceChips(sources) : null);
    log.append(m); log.scrollTop = log.scrollHeight; return m;
  };

  function renderHistory() {
    log.replaceChildren();
    const hist = session?.chat || [];
    if (!hist.length) {
      log.append(el("p", { class: "muted" }, session?.fusion ? "Answers come only from the analysed evidence. Try one of these:" : "Run the analysis first, then ask questions about what was found."));
      if (session?.fusion) log.append(el("div", { class: "suggest" }, SUGGESTIONS.map((q) => el("button", { class: "chip", type: "button", style: { padding: "5px 11px" }, onclick: () => ask(q) }, q))));
    }
    hist.forEach((m) => (m.role === "user" ? addUser(m.content) : addBot(m.content, m.sources || [])));
    input.disabled = send.disabled = !session?.fusion;
    log.scrollTop = log.scrollHeight;
  }

  async function ask(text) {
    if (!session?.fusion || !text.trim()) return;
    log.querySelector(".suggest")?.parentElement && !session.chat.length && log.replaceChildren();
    addUser(text); input.value = ""; send.disabled = true;
    const pending = el("div", { class: "msg bot typing" }, "Checking the evidence…");
    log.append(pending); log.scrollTop = log.scrollHeight;
    try {
      const res = await api.chat(session.id, text);
      pending.remove(); addBot(res.answer, res.sources);
      session.chat.push({ role: "user", content: text }, { role: "assistant", content: res.answer, sources: res.sources });
    } catch (err) {
      pending.remove(); addBot(err.message);
    } finally { send.disabled = false; input.focus(); }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  on("session", (s) => { session = s; renderHistory(); });
  renderHistory();
}
