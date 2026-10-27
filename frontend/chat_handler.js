// API connector for chat + timeline syncing, plus the tiny event bus the dashboard components share.
const devServer = location.port === "3000"; // React/Vite-style dev server → talk to Flask on :5000
export const API_BASE = window.CRIMEVISION_API || (devServer ? "http://localhost:5000" : "");

export async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch {
    throw new Error("Cannot reach the server. Start the backend and try again.");
  }
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const body = isJson ? await res.json() : null;
  if (!res.ok) throw new Error(body?.error || `Request failed (${res.status})`);
  return body;
}
const jsonOpts = (method, data) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(data ?? {}) });

export const api = {
  health: () => request("/api/health"),
  createSession: () => request("/api/sessions", jsonOpts("POST")),
  deleteFile: (sid, fid) => request(`/api/sessions/${sid}/files/${fid}`, { method: "DELETE" }),
  demo: () => request("/api/demo", jsonOpts("POST")),
  getSession: (sid) => request(`/api/sessions/${sid}`),
  upload(sid, files, offset) {
    const fd = new FormData();
    if (sid) fd.append("session_id", sid);
    for (const f of files) fd.append("files", f);
    if (offset) fd.append("offset_seconds", offset);
    return request("/api/upload", { method: "POST", body: fd });
  },
  process: (sid, language) => request(`/api/process/${sid}`, jsonOpts("POST", { language })),
  chat: (sid, message) => request(`/api/chat/${sid}`, jsonOpts("POST", { message })),
  reportUrl: (sid, fmt = "pdf") => `${API_BASE}/api/report/${sid}?format=${fmt}`,
  fileUrl: (sid, name) => `${API_BASE}/api/files/${sid}/${encodeURIComponent(name)}`,
};

// ---- event bus: "session", "select", "seek", "time", "files" ----
const bus = new EventTarget();
export const emit = (type, detail) => bus.dispatchEvent(new CustomEvent(type, { detail }));
export const on = (type, fn) => bus.addEventListener(type, (e) => fn(e.detail));

// ---- helpers ----
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) node.append(c.nodeType ? c : document.createTextNode(String(c)));
  return node;
}

export const fmtTime = (t) => {
  if (t == null) return "--:--";
  const s = Math.max(0, Math.round(t));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

// Syncs backend fusion output into a shape the timeline can draw.
export function timelineModel(session) {
  const fusion = session?.fusion;
  if (!fusion) return null;
  const lanes = { video: [], audio: [], image: [] };
  const undated = [];
  for (const item of fusion.timeline) (item.t == null ? undated : lanes[item.source]).push(item);
  const activity = (session.results.video || []).flatMap((v) => v.timeline).sort((a, b) => a.t - b.t);
  const ends = [fusion.duration, ...activity.map((a) => a.t), ...lanes.audio.map((a) => a.end ?? a.t)];
  return { lanes, undated, activity, duration: Math.max(1, ...ends) };
}
