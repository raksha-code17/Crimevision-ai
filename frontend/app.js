import { api, el, emit, on } from "./chat_handler.js";
import { mountChat } from "./components/chat_panel.js";
import { mountEvidencePanel } from "./components/evidence_panel.js";
import { mountFindings } from "./components/findings_panel.js";
import { mountTimeline } from "./components/timeline.js";
import { mountViewport } from "./components/viewport.js";

const $ = (id) => document.getElementById(id);
const KEY = "crimevision.session";
let session = null;

function toast(message, kind = "info", ms = 6000) {
  const t = el("div", { class: `toast ${kind}` }, message);
  $("toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

function setSession(s) {
  session = s;
  try { localStorage.setItem(KEY, s.id); } catch { /* storage may be blocked */ }
  $("caseName").textContent = s.title;
  const has = s.files.length > 0;
  $("runBtn").disabled = !has;
  $("runBtn").textContent = s.fusion ? "Re-run analysis" : "Run analysis";
  for (const [id, fmt] of [["reportPdf", "pdf"], ["reportDocx", "docx"]]) {
    const a = $(id);
    a.classList.toggle("disabled", !s.fusion);
    a.setAttribute("aria-disabled", String(!s.fusion));
    a.href = s.fusion ? api.reportUrl(s.id, fmt) : "#";
  }
  emit("session", s);
}

async function busy(btn, label, work) {
  const original = btn.textContent;
  btn.classList.add("busy"); btn.disabled = true; btn.textContent = label;
  try { return await work(); }
  catch (err) { toast(err.message, "error", 9000); }
  finally { btn.classList.remove("busy"); btn.textContent = original; if (session) $("runBtn").disabled = !session.files.length; btn.disabled = false; }
}

async function refreshStatus() {
  try {
    const h = await api.health();
    const detector = h.models.loaded.includes("rtdetr");
    $("status").replaceChildren(
      el("span", { class: `pill ${detector ? "ok" : "warn"}`, title: Object.values(h.models.failed).join(" | ") || "" }, detector ? `Vision models on ${h.models.device}` : h.ml_enabled ? "Vision models load on first run" : "Basic detection only"),
      h.groq
        ? el("span", { class: "pill ok", title: h.cloud_vision_error || "A Groq key is set in .env" }, h.cloud_vision ? "Groq text + vision on" : "Groq text on")
        : el("span", { class: "pill", title: "No images or text leave this computer" }, "Runs locally on this computer"),
    );
  } catch { $("status").replaceChildren(el("span", { class: "pill warn" }, "Backend offline")); }
}

on("remove", async (f) => {
  if (!confirm(`Remove ${f.name}? Its analysis results will be removed too.`)) return;
  try { setSession(await api.deleteFile(session.id, f.id)); toast(`Removed ${f.name}.`); }
  catch (err) { toast(err.message, "error"); }
});
$("newBtn").addEventListener("click", () => busy($("newBtn"), "Creating…", async () => setSession(await api.createSession())));

on("files", (files) => upload(files));
async function upload(files) {
  if (!files.length) return;
  await busy($("addBtn"), "Uploading…", async () => {
    const res = await api.upload(session?.id, files);
    setSession(res.session);
    toast(`Added ${res.saved.length} file(s).`);
    res.rejected.forEach((r) => toast(`${r.name}: ${r.reason}`, "error"));
  });
}

$("addBtn").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => { upload([...e.target.files]); e.target.value = ""; });

$("demoBtn").addEventListener("click", () => busy($("demoBtn"), "Building demo…", async () => {
  setSession(await api.demo());
  toast("Demo case loaded. Select Run analysis.");
}));

$("runBtn").addEventListener("click", () => busy($("runBtn"), "Analysing…", async () => {
  toast("Analysing. Long videos can take a few minutes.", "info", 3500);
  const res = await api.process(session.id);
  setSession(res.session);
  res.errors.forEach((e) => toast(`${e.file}: ${e.error}`, "error", 12000));
  toast("Analysis complete.");
  refreshStatus();
}));

// Tabs
const tabs = { findings: [$("tab-findings"), $("findingsPanel")], chat: [$("tab-chat"), $("chatPanel")] };
for (const [name, [tab]] of Object.entries(tabs)) {
  tab.addEventListener("click", () => {
    for (const [n, [t, p]] of Object.entries(tabs)) { t.setAttribute("aria-selected", String(n === name)); p.hidden = n !== name; }
  });
}

mountEvidencePanel($("evidencePanel"));
mountViewport($("viewportPanel"));
mountTimeline($("timelinePanel"));
mountFindings($("findingsPanel"));
mountChat($("chatPanel"));
refreshStatus();

// Restore the last case after a refresh
try {
  const sid = localStorage.getItem(KEY);
  if (sid) api.getSession(sid).then(setSession).catch(() => localStorage.removeItem(KEY));
} catch { /* ignore */ }
