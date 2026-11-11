"use strict";
// Liver surface zone annotation: fully static, serverless.
// No SAM, no GPU, no server: images come from images.zip (JSZip), edges are
// computed in-browser (Sobel) for polygon snapping, masks are written as
// 8-bit RGBA PNGs (R = G = B = class id, A = 255) to a folder you pick
// (File System Access API) or downloaded as a fallback.

const CFG = {
  num: 7,
  names: ["background", "ventral_right", "dorsal_right", "dorsal_central", "ventral_left", "dorsal_left", "caudate_lobe"],
  jp: ["background", "ventral right", "dorsal right", "dorsal central", "ventral left", "dorsal left", "caudate lobe"],
  palette: [[0, 0, 0], [230, 25, 75], [0, 130, 200], [245, 130, 48], [60, 180, 75], [145, 30, 180], [50, 200, 200]],
};

const S = {
  zip: null, names: [], doneSet: new Set(),
  masksDir: null, savedDir: null,
  name: null, W: 0, H: 0, imageBitmap: null,
  maskData: null, preview: null, edges: null,
  tool: "polygon", curClass: 1, brush: 12, snap: true,
  showEdges: false, snapR: 22, edgeFrac: 0.14, edgeCanvas: null,
  zoom: 1, panX: 0, panY: 0,
  poly: [], mouseImg: null, tracing: false, traceRaw: [],
  painting: false, erasing: false, lastPt: null,
  spaceDown: false, panning: false, panStart: null,
};

const el = (id) => document.getElementById(id);
const cv = { image: el("c-image"), mask: el("c-mask"), preview: el("c-preview"), overlay: el("c-overlay") };
const ctx = {};
for (const k in cv) ctx[k] = cv[k].getContext("2d", { willReadFrequently: true });
const status = (m) => { el("status").textContent = m; };

// tiny IndexedDB key-value (persists the chosen masks-folder handle across launches)
function _idb() { return new Promise((res, rej) => { const r = indexedDB.open("liver_anno_kv", 1); r.onupgradeneeded = () => r.result.createObjectStore("kv"); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); }); }
async function idbGet(k) { const db = await _idb(); return new Promise((res, rej) => { const g = db.transaction("kv").objectStore("kv").get(k); g.onsuccess = () => res(g.result); g.onerror = () => rej(g.error); }); }
async function idbSet(k, v) { const db = await _idb(); return new Promise((res, rej) => { const tx = db.transaction("kv", "readwrite"); tx.objectStore("kv").put(v, k); tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); }); }

// ---------- init ----------
function init() {
  const cl = el("classes");
  for (let c = 0; c < CFG.num; c++) {
    const [r, g, b] = CFG.palette[c];
    const lab = document.createElement("label");
    lab.className = "class-radio";
    lab.innerHTML = `<input type="radio" name="cls" value="${c}" ${c === 1 ? "checked" : ""}>` +
      `<span class="swatch" style="background:rgb(${r},${g},${b})"></span>${c}: ${CFG.jp[c]}`;
    lab.querySelector("input").onchange = () => setClass(c);
    cl.appendChild(lab);
  }
  document.querySelectorAll('input[name=tool]').forEach((r) => r.onchange = () => { S.tool = r.value; clearPreviewState(); });
  el("brush-slider").oninput = (e) => { S.brush = +e.target.value; el("brush-val").textContent = S.brush + "px"; };
  el("snap-toggle").onclick = () => { S.snap = !S.snap; el("snap-toggle").classList.toggle("active", S.snap); el("snap-toggle").textContent = "Edge snap: " + (S.snap ? "ON" : "OFF"); };
  el("show-edges").onclick = () => { S.showEdges = !S.showEdges; el("show-edges").classList.toggle("active", S.showEdges); el("show-edges").textContent = "Show edges: " + (S.showEdges ? "ON" : "OFF"); redrawOverlay(); };
  el("snap-radius").oninput = (e) => { S.snapR = +e.target.value; el("snap-r-val").textContent = S.snapR; };
  el("edge-sens").oninput = (e) => { S.edgeFrac = +e.target.value / 100; el("sens-val").textContent = e.target.value + "%"; if (S.imageBitmap) { computeEdges(); redrawOverlay(); status("Edges recomputed"); } };
  el("pick-masks").onclick = pickMasksFolder;
  el("commit").onclick = handleEnter;
  el("clear-class").onclick = clearCurrentClass;
  el("clear-all").onclick = clearAll;
  el("fit").onclick = fitView;
  el("save").onclick = saveMask;
  el("zip-input").onchange = async (ev) => { const f = ev.target.files[0]; if (f) await openZip(await f.arrayBuffer()); };
  if (!window.showDirectoryPicker) el("masks-status").textContent = "This browser cannot write to a folder; masks are downloaded (use Chrome or Edge)";
  bindCanvas(); bindKeys();
  loadZip();
  restoreMasksFolder();
}

// ---------- data: images.zip ----------
async function loadZip() {
  try {
    const resp = await fetch("images.zip");
    if (!resp.ok) throw new Error("no fetch");
    status("Loading images.zip…");
    await openZip(await resp.arrayBuffer());
  } catch (e) {
    el("zip-status").textContent = "Select images.zip ↓";
    el("zip-input").style.display = "block";
    status("Select images.zip (page opened as file://)");
  }
}
async function openZip(buf) {
  S.zip = await JSZip.loadAsync(buf);
  const man = S.zip.file("manifest.json");
  if (man) S.names = JSON.parse(await man.async("string"));
  else S.names = Object.keys(S.zip.files).filter((n) => n.startsWith("images/") && !S.zip.files[n].dir).map((n) => n.slice(7));
  S.names.sort();
  el("zip-status").textContent = `images.zip: ${S.names.length} images`;
  buildFileList();
  status("Select an image");
}

function buildFileList() {
  const list = el("file-list"); list.innerHTML = "";
  let done = 0;
  for (const name of S.names) {
    const stem = name.replace(/\.jpg$/, "");
    const isDone = S.doneSet.has(stem);
    if (isDone) done++;
    const d = document.createElement("div");
    d.className = "file-item" + (isDone ? " done" : "");
    d.textContent = (isDone ? "✓ " : "") + name;
    d.dataset.name = name;
    d.onclick = () => { document.querySelectorAll(".file-item.sel").forEach((e) => e.classList.remove("sel")); d.classList.add("sel"); selectImage(name); };
    list.appendChild(d);
  }
  el("progress").textContent = `${done} / ${S.names.length}`;
}
function markDone(name) {
  const it = [...document.querySelectorAll(".file-item")].find((e) => e.dataset.name === name);
  if (it && !it.classList.contains("done")) { it.classList.add("done"); it.textContent = "✓ " + name; }
}

async function selectImage(name) {
  const entry = S.zip.file("images/" + name);
  if (!entry) return;
  const bmp = await createImageBitmap(await entry.async("blob"));
  S.name = name; S.W = bmp.width; S.H = bmp.height; S.imageBitmap = bmp;
  for (const k in cv) { cv[k].width = S.W; cv[k].height = S.H; }
  ctx.image.drawImage(bmp, 0, 0);
  S.maskData = new Uint8Array(S.W * S.H); S.preview = null; clearPreviewState();
  status("Computing edges…");
  computeEdges();
  await loadExistingMask(name);
  redrawMask(); fitView();
  status("Opened: " + name);
}

// ---------- masks folder (File System Access, remembered across launches) ----------
async function connectMasks(h) {
  S.masksDir = h;
  el("masks-status").textContent = "Mask folder: " + h.name + " (direct write)";
  el("masks-status").classList.remove("warn");
  el("pick-masks").textContent = "Change mask folder";
  await refreshDoneMarkers();
}
async function restoreMasksFolder() {
  // On launch: reconnect to the previously chosen folder automatically.
  if (!window.showDirectoryPicker || !window.indexedDB) return;
  try {
    const h = await idbGet("masksDir");
    if (!h) return;
    S.savedDir = h;
    const p = await h.queryPermission({ mode: "readwrite" });
    if (p === "granted") { await connectMasks(h); status("Reconnected mask folder: " + h.name); }
    else el("pick-masks").textContent = "Reconnect to previous mask folder: " + h.name; // one click to restore
  } catch (e) { /* ignore */ }
}
async function pickMasksFolder() {
  if (!window.showDirectoryPicker) { alert("Writing to a folder needs Chrome or Edge. In this browser, masks are downloaded."); return; }
  try {
    let h = null;
    if (S.savedDir && !S.masksDir) {           // reconnect to remembered folder first (1 click)
      if (await S.savedDir.requestPermission({ mode: "readwrite" }) === "granted") h = S.savedDir;
    }
    if (!h) h = await window.showDirectoryPicker({ mode: "readwrite" });
    S.savedDir = h;
    await idbSet("masksDir", h);
    await connectMasks(h);
    status("Mask folder set: " + h.name);
  } catch (e) { /* cancelled */ }
}
async function refreshDoneMarkers() {
  if (!S.masksDir) return;
  const done = new Set();
  for await (const [n, h] of S.masksDir.entries()) if (n.endsWith(".png")) done.add(n.slice(0, -4));
  S.doneSet = done;
  buildFileList();
}
async function loadExistingMask(name) {
  if (!S.masksDir) return;
  const stem = name.replace(/\.jpg$/, "");
  try {
    const fh = await S.masksDir.getFileHandle(stem + ".png");
    const bmp = await createImageBitmap(await fh.getFile());
    const off = document.createElement("canvas"); off.width = S.W; off.height = S.H;
    const o = off.getContext("2d", { willReadFrequently: true }); o.drawImage(bmp, 0, 0);
    const px = o.getImageData(0, 0, S.W, S.H).data;
    for (let i = 0; i < S.W * S.H; i++) S.maskData[i] = px[i * 4];
    status("Loaded existing mask: " + name);
  } catch (e) { /* none yet */ }
}

// ---------- edge detection (blur → Sobel → percentile threshold) ----------
function computeEdges() {
  const W = S.W, H = S.H;
  const off = document.createElement("canvas"); off.width = W; off.height = H;
  const o = off.getContext("2d", { willReadFrequently: true }); o.drawImage(S.imageBitmap, 0, 0);
  const d = o.getImageData(0, 0, W, H).data;
  let gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) gray[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  // 3x3 box blur to suppress specular/vessel speckle before gradients
  const g2 = new Float32Array(W * H);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    g2[i] = (gray[i - W - 1] + gray[i - W] + gray[i - W + 1] + gray[i - 1] + gray[i] + gray[i + 1] + gray[i + W - 1] + gray[i + W] + gray[i + W + 1]) / 9;
  }
  gray = g2;
  const mag = new Float32Array(W * H); let mx = 0;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    const gx = -gray[i - W - 1] - 2 * gray[i - 1] - gray[i + W - 1] + gray[i - W + 1] + 2 * gray[i + 1] + gray[i + W + 1];
    const gy = -gray[i - W - 1] - 2 * gray[i - W] - gray[i - W + 1] + gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1];
    const m = Math.sqrt(gx * gx + gy * gy); mag[i] = m; if (m > mx) mx = m;
  }
  // keep the strongest edgeFrac of gradients
  const bins = 256, hist = new Int32Array(bins), scale = (bins - 1) / (mx || 1);
  for (let i = 0; i < mag.length; i++) hist[(mag[i] * scale) | 0]++;
  const want = mag.length * S.edgeFrac; let acc = 0, b = bins - 1;
  for (; b > 0; b--) { acc += hist[b]; if (acc >= want) break; }
  const T = b / scale, e = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) e[i] = mag[i] >= T ? 1 : 0;
  S.edges = e;
  // cache a cyan overlay for visualisation
  const ec = document.createElement("canvas"); ec.width = W; ec.height = H;
  const eo = ec.getContext("2d"); const eimg = eo.createImageData(W, H), ed = eimg.data;
  for (let i = 0; i < W * H; i++) if (e[i]) { const j = i * 4; ed[j] = 0; ed[j + 1] = 255; ed[j + 2] = 255; ed[j + 3] = 200; }
  eo.putImageData(eimg, 0, 0); S.edgeCanvas = ec;
}
function snapPoint(x, y, r) {
  if (!S.snap || !S.edges) return { x, y };
  r = r || S.snapR;
  const cx = Math.round(x), cy = Math.round(y); let best = null, bd = r * r + 1;
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    const px = cx + dx, py = cy + dy;
    if (px < 0 || py < 0 || px >= S.W || py >= S.H) continue;
    if (S.edges[py * S.W + px]) { const dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = { x: px, y: py }; } }
  }
  return best || { x, y };
}
function resampleSnap(raw) {
  if (raw.length < 2) return [];
  let acc = 0; const out = []; const step = 3;
  for (let i = 1; i < raw.length; i++) {
    let ax = raw[i - 1].x, ay = raw[i - 1].y, bx = raw[i].x, by = raw[i].y;
    const seg = Math.hypot(bx - ax, by - ay);
    let t = 0;
    while (acc + (seg - t) >= step) {
      t += step - acc; acc = 0;
      const f = t / (seg || 1);
      out.push(snapPoint(ax + (bx - ax) * f, ay + (by - ay) * f));
    }
    acc += seg - t;
  }
  return out;
}

// ---------- rendering ----------
function redrawMask() {
  const img = ctx.mask.createImageData(S.W, S.H), d = img.data, m = S.maskData;
  for (let i = 0; i < m.length; i++) { const c = m[i]; if (!c) continue; const p = CFG.palette[c], o = i * 4; d[o] = p[0]; d[o + 1] = p[1]; d[o + 2] = p[2]; d[o + 3] = 200; }
  ctx.mask.putImageData(img, 0, 0);
}
function redrawPreview() {
  ctx.preview.clearRect(0, 0, S.W, S.H);
  if (!S.preview) return;
  const img = ctx.preview.createImageData(S.W, S.H), d = img.data, p = CFG.palette[S.curClass];
  for (let i = 0; i < S.preview.length; i++) if (S.preview[i]) { const o = i * 4; d[o] = p[0]; d[o + 1] = p[1]; d[o + 2] = p[2]; d[o + 3] = 230; }
  ctx.preview.putImageData(img, 0, 0);
}
function redrawOverlay() {
  const c = ctx.overlay; c.clearRect(0, 0, S.W, S.H);
  if (S.showEdges && S.edgeCanvas) c.drawImage(S.edgeCanvas, 0, 0);
  if (S.tool === "polygon" && S.poly.length) {
    c.strokeStyle = "#ffff00"; c.lineWidth = 2;
    c.beginPath(); c.moveTo(S.poly[0].x, S.poly[0].y);
    for (let i = 1; i < S.poly.length; i++) c.lineTo(S.poly[i].x, S.poly[i].y);
    c.stroke();
    if (S.mouseImg) { const l = S.poly[S.poly.length - 1]; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(l.x, l.y); c.lineTo(S.mouseImg.x, S.mouseImg.y); c.stroke(); c.setLineDash([]); }
    for (const v of S.poly) { c.beginPath(); c.arc(v.x, v.y, 4, 0, 7); c.fillStyle = "#ffff00"; c.fill(); c.lineWidth = 1; c.strokeStyle = "#000"; c.stroke(); }
  }
  if (S.tool === "brush" && S.mouseImg) { c.beginPath(); c.arc(S.mouseImg.x, S.mouseImg.y, S.brush, 0, 7); c.strokeStyle = "#fff"; c.lineWidth = 1; c.stroke(); }
}
function rasterizePolygon(poly) {
  const off = document.createElement("canvas"); off.width = S.W; off.height = S.H;
  const o = off.getContext("2d", { willReadFrequently: true });
  o.fillStyle = "#fff"; o.beginPath(); o.moveTo(poly[0].x, poly[0].y);
  for (let i = 1; i < poly.length; i++) o.lineTo(poly[i].x, poly[i].y);
  o.closePath(); o.fill();
  const px = o.getImageData(0, 0, S.W, S.H).data, m = new Uint8Array(S.W * S.H);
  for (let i = 0; i < m.length; i++) m[i] = px[i * 4] > 127 ? 1 : 0;
  return m;
}
function closePolygon() {
  if (S.poly.length < 3) { status("At least 3 vertices are needed"); return; }
  S.preview = rasterizePolygon(S.poly); S.poly = [];
  redrawPreview(); redrawOverlay();
  status("Polygon preview. Press Enter to commit");
}

// ---------- view ----------
function applyTransform() { el("world").style.transform = `translate(${S.panX}px,${S.panY}px) scale(${S.zoom})`; }
function fitView() {
  if (!S.W) return;
  const vp = el("viewport").getBoundingClientRect();
  S.zoom = Math.min(vp.width / S.W, vp.height / S.H) * 0.98;
  S.panX = (vp.width - S.W * S.zoom) / 2; S.panY = (vp.height - S.H * S.zoom) / 2;
  applyTransform();
}
function toImg(ev) { const r = cv.image.getBoundingClientRect(); return { x: (ev.clientX - r.left) / (r.width / S.W), y: (ev.clientY - r.top) / (r.height / S.H) }; }
function inBounds(p) { return p.x >= 0 && p.y >= 0 && p.x < S.W && p.y < S.H; }

// ---------- editing ----------
function handleEnter() { if (S.tool === "polygon" && S.poly.length >= 3 && !S.preview) closePolygon(); else commitPreview(); }
function commitPreview() {
  if (!S.preview) return;
  for (let i = 0; i < S.maskData.length; i++) if (S.preview[i]) S.maskData[i] = S.curClass;
  clearPreviewState(); redrawMask(); status("Committed");
}
function clearPreviewState() { S.preview = null; S.poly = []; S.tracing = false; S.traceRaw = []; S.lastPt = null; redrawPreview(); redrawOverlay(); }
function setClass(c) { if (S.preview) commitPreview(); S.curClass = c; const r = document.querySelector(`input[name=cls][value="${c}"]`); if (r) r.checked = true; status(`class ${c}: ${CFG.jp[c]}`); }
function clearCurrentClass() { if (!S.maskData) return; for (let i = 0; i < S.maskData.length; i++) if (S.maskData[i] === S.curClass) S.maskData[i] = 0; redrawMask(); status(`class ${S.curClass} cleared`); }
function clearAll() { if (!S.maskData || !confirm("Clear the whole mask of this image?")) return; S.maskData.fill(0); clearPreviewState(); redrawMask(); status("All cleared"); }
function paintAt(x, y) {
  const val = S.erasing ? 0 : S.curClass, r = S.brush;
  const stamp = (cx, cy) => {
    cx = Math.round(cx); cy = Math.round(cy);
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy > r * r) continue;
      const px = cx + dx, py = cy + dy;
      if (px < 0 || py < 0 || px >= S.W || py >= S.H) continue;
      S.maskData[py * S.W + px] = val;
    }
  };
  if (S.lastPt) { const dx = x - S.lastPt.x, dy = y - S.lastPt.y, n = Math.max(1, Math.ceil(Math.hypot(dx, dy) / (r / 2))); for (let i = 1; i <= n; i++) stamp(S.lastPt.x + dx * i / n, S.lastPt.y + dy * i / n); }
  else stamp(x, y);
  S.lastPt = { x, y }; redrawMask();
}

// ---------- save ----------
async function saveMask() {
  if (!S.name || !S.maskData) return;
  if (S.preview) commitPreview();
  const off = document.createElement("canvas"); off.width = S.W; off.height = S.H;
  const o = off.getContext("2d");
  const img = o.createImageData(S.W, S.H), d = img.data;
  for (let i = 0; i < S.maskData.length; i++) { const v = S.maskData[i], j = i * 4; d[j] = d[j + 1] = d[j + 2] = v; d[j + 3] = 255; }
  o.putImageData(img, 0, 0);
  const stem = S.name.replace(/\.jpg$/, "");
  const blob = await new Promise((res) => off.toBlob(res, "image/png"));
  if (S.masksDir) {
    const fh = await S.masksDir.getFileHandle(stem + ".png", { create: true });
    const w = await fh.createWritable(); await w.write(blob); await w.close();
    S.doneSet.add(stem); markDone(S.name); status("Saved: " + stem + ".png");
  } else {
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = stem + ".png"; a.click();
    markDone(S.name); status("Downloaded: " + stem + ".png (move it to your masks folder)");
  }
}

// ---------- input ----------
function bindCanvas() {
  const vp = el("viewport");
  vp.addEventListener("contextmenu", (e) => e.preventDefault());
  vp.addEventListener("dblclick", (e) => { if (S.tool === "polygon") { e.preventDefault(); closePolygon(); } });
  vp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.25 : 0.8, r = vp.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    S.panX = mx - (mx - S.panX) * f; S.panY = my - (my - S.panY) * f; S.zoom *= f; applyTransform();
  }, { passive: false });
  vp.addEventListener("mousedown", (e) => {
    if (S.spaceDown || e.button === 1) { S.panning = true; S.panStart = { x: e.clientX - S.panX, y: e.clientY - S.panY }; return; }
    const p = toImg(e); if (!inBounds(p) || !S.imageBitmap) return;
    if (S.tool === "polygon") {
      if (e.button === 0) { S.poly.push(snapPoint(p.x, p.y)); S.tracing = true; S.traceRaw = [p]; }
      else S.poly.pop();
      redrawOverlay();
    } else if (S.tool === "brush") {
      if (e.button === 0) { S.painting = true; S.erasing = false; } else { S.painting = true; S.erasing = true; }
      S.lastPt = null; paintAt(p.x, p.y);
    }
  });
  vp.addEventListener("mousemove", (e) => {
    if (S.panning) { S.panX = e.clientX - S.panStart.x; S.panY = e.clientY - S.panStart.y; applyTransform(); return; }
    const p = toImg(e); S.mouseImg = p;
    if (S.tool === "polygon") { if (S.tracing && (e.buttons & 1)) S.traceRaw.push(p); redrawOverlay(); }
    else if (S.tool === "brush") { if (S.painting && inBounds(p)) paintAt(p.x, p.y); redrawOverlay(); }
  });
  window.addEventListener("mouseup", () => {
    if (S.panning) { S.panning = false; return; }
    if (S.tool === "polygon" && S.tracing) {
      S.tracing = false;
      if (S.traceRaw.length > 3) { for (const q of resampleSnap(S.traceRaw)) S.poly.push(q); redrawOverlay(); }
      S.traceRaw = [];
    }
    S.painting = false; S.erasing = false; S.lastPt = null;
  });
}
function bindKeys() {
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") { S.spaceDown = true; el("viewport").style.cursor = "grab"; e.preventDefault(); return; }
    if (e.ctrlKey && e.shiftKey && (e.key === "C" || e.key === "c")) { clearAll(); e.preventDefault(); return; }
    if (e.ctrlKey && (e.key === "s" || e.key === "S")) { saveMask(); e.preventDefault(); return; }
    if (e.shiftKey && (e.key === "C" || e.key === "c")) { clearCurrentClass(); return; }
    if (e.key === "Enter") { handleEnter(); return; }
    if (e.key === "f" || e.key === "F") { fitView(); return; }
    if (e.key === "c" || e.key === "C") { clearPreviewState(); return; }
    if (e.key === "[") { S.brush = Math.max(2, S.brush - 2); el("brush-slider").value = S.brush; el("brush-val").textContent = S.brush + "px"; return; }
    if (e.key === "]") { S.brush = Math.min(80, S.brush + 2); el("brush-slider").value = S.brush; el("brush-val").textContent = S.brush + "px"; return; }
    if (/^[0-6]$/.test(e.key)) setClass(+e.key);
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") { S.spaceDown = false; el("viewport").style.cursor = "crosshair"; } });
}

init();
