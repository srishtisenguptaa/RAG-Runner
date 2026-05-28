// Auto-detect environment:
// - file:// or localhost → talk directly to FastAPI on port 8000
// - nginx/docker (port 80) → use relative /api proxy
const _proto = window.location.protocol;
const _host  = window.location.hostname;
const _port  = window.location.port;
const API = (_proto === "file:" || _host === "localhost" || _host === "127.0.0.1")
  ? "http://127.0.0.1:8000"   // local dev
  : "/api";                   // docker / production (nginx proxy)

let ragMode   = "standard";
let isLoading = false;

// ── Session ID ──────────────────────────────────────────────────────────────
let SESSION_ID = localStorage.getItem("rag_session_id");
if (SESSION_ID) fetch(`${API}/history/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});
SESSION_ID = crypto.randomUUID();
localStorage.setItem("rag_session_id", SESSION_ID);
fetch(`${API}/history/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});

// ── File Queue ───────────────────────────────────────────────────────────────
let fileQueue = new Map(); // filename → File

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatWindow       = document.getElementById("chat-window");
const emptyState       = document.getElementById("empty-state");
const chatInput        = document.getElementById("chat-input");
const sendBtn          = document.getElementById("send-btn");
const fileInput        = document.getElementById("file-input");
const dropZone         = document.getElementById("drop-zone");
const fileQueueEl      = document.getElementById("file-queue");
const indexBtn         = document.getElementById("index-btn");
const indexStatus      = document.getElementById("index-status");
const radioGroup       = document.getElementById("rag-mode-group");
const themeCheck       = document.getElementById("theme-checkbox");
const themeLabel       = document.getElementById("theme-label");
const optionalBadge    = document.getElementById("optional-badge");
const uploadSection    = document.getElementById("upload-section");
const emptyText        = document.getElementById("empty-text");
const indexedFilesList = document.getElementById("indexed-files-list");
const fileCountBadge   = document.getElementById("file-count-badge");

// ── Accepted extensions ───────────────────────────────────────────────────────
const ALLOWED_EXTS = [".pdf", ".xlsx", ".xls", ".xlsm", ".csv"];

function getFileKind(name) {
  const n = name.toLowerCase();
  if (n.endsWith(".pdf"))  return "pdf";
  if (n.endsWith(".csv"))  return "csv";
  return "excel"; // xlsx / xls / xlsm
}

function fileKindIcon(kind) {
  return kind === "pdf" ? "📄" : kind === "csv" ? "📋" : "📊";
}

// ── Theme ─────────────────────────────────────────────────────────────────────
themeCheck.addEventListener("change", () => {
  const light = themeCheck.checked;
  document.documentElement.setAttribute("data-theme", light ? "light" : "dark");
  themeLabel.textContent = light ? "Light" : "Dark";
});

// ── Tooltip positioning ───────────────────────────────────────────────────────
document.querySelectorAll(".tooltip-wrap").forEach(wrap => {
  const icon = wrap.querySelector(".tooltip-icon");
  const box  = wrap.querySelector(".tooltip-box");
  wrap.addEventListener("mouseenter", () => {
    const r = icon.getBoundingClientRect();
    box.style.top = (r.top + r.height / 2) + "px";
    box.style.left = (r.right + 10) + "px";
    box.style.transform = "translateY(-50%)";
  });
});

// ── RAG Mode ──────────────────────────────────────────────────────────────────
function applyModeUI() {
  const isArch = ragMode === "architect";
  optionalBadge.classList.toggle("visible", isArch);
  uploadSection.classList.toggle("faded", isArch);
  emptyText.textContent = isArch
    ? "Ask anything — Architect uses Web Search. Upload files for document-specific answers."
    : "Upload PDF or Excel files and ask a question to get started.";
}

radioGroup.querySelectorAll(".radio-option").forEach(opt => {
  opt.addEventListener("click", () => {
    ragMode = opt.dataset.value;
    radioGroup.querySelectorAll(".radio-option").forEach(o => o.classList.remove("selected"));
    opt.classList.add("selected");
    opt.querySelector("input").checked = true;
    applyModeUI();
  });
});

// ── File Selection ────────────────────────────────────────────────────────────
dropZone.addEventListener("click",    () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave",  () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragover");
  addFilesToQueue([...e.dataTransfer.files]);
});
fileInput.addEventListener("change", () => { addFilesToQueue([...fileInput.files]); fileInput.value = ""; });

function addFilesToQueue(files) {
  let added = 0;
  files.forEach(f => {
    const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
    if (!ALLOWED_EXTS.includes(ext)) {
      showStatus(`❌ "${f.name}" not supported (PDF, XLSX, XLS, XLSM, CSV only).`, "error");
      return;
    }
    if (!fileQueue.has(f.name)) { fileQueue.set(f.name, f); added++; }
  });
  if (added > 0) renderQueue();
}

function renderQueue() {
  fileQueueEl.innerHTML = "";
  fileQueue.forEach((file, name) => {
    const kind = getFileKind(name);
    const icon = fileKindIcon(kind);
    const row  = document.createElement("div"); row.className = "queue-item";
    row.innerHTML = `
      <span class="queue-icon">${icon}</span>
      <span class="queue-name" title="${name}">${name}</span>
      <span class="queue-ext ${kind}">${kind.toUpperCase()}</span>
      <button class="queue-remove" data-name="${name}" title="Remove">✕</button>`;
    fileQueueEl.appendChild(row);
  });
  fileQueueEl.querySelectorAll(".queue-remove").forEach(btn =>
    btn.addEventListener("click", () => { fileQueue.delete(btn.dataset.name); renderQueue(); })
  );
  indexBtn.disabled = fileQueue.size === 0;
  indexStatus.style.display = "none";
}

// ── Index Files ───────────────────────────────────────────────────────────────
indexBtn.addEventListener("click", async () => {
  if (fileQueue.size === 0) return;
  indexBtn.disabled = true;
  indexBtn.innerHTML = '<span class="spinner"></span> Processing…';
  indexStatus.style.display = "none";

  let successCount = 0, errors = [], lastFiles = [];

  for (const [name, file] of fileQueue.entries()) {
    try {
      const form = new FormData();
      form.append("file", file, name);
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (res.ok) {
        const data = await res.json();
        successCount++;
        lastFiles = data.indexed_files ?? [];
      } else {
        const err = await res.json().catch(() => ({}));
        errors.push(`${name}: ${err.detail ?? "upload failed"}`);
      }
    } catch (e) {
      errors.push(`${name}: ${e.message}`);
    }
  }

  fileQueue.clear();
  renderQueue();

  if (successCount > 0) {
    showStatus(
      `✅ ${successCount} file(s) indexed!` + (errors.length ? ` (${errors.length} failed)` : ""),
      errors.length ? "warn" : "success"
    );
    renderIndexedFiles(lastFiles);
  } else {
    showStatus("❌ All uploads failed. Is the backend running?", "error");
  }
  indexBtn.innerHTML = "Index Selected Files";
  indexBtn.disabled  = true;
});

function showStatus(msg, type) {
  indexStatus.textContent = msg; indexStatus.className = type; indexStatus.style.display = "block";
}

// ── Indexed Files Panel ───────────────────────────────────────────────────────
function renderIndexedFiles(files) {
  fileCountBadge.textContent = files.length;
  if (!files || files.length === 0) {
    indexedFilesList.innerHTML = '<div class="no-files-msg">No files indexed yet.</div>';
    return;
  }
  indexedFilesList.innerHTML = "";
  files.forEach(f => {
    const kind   = f.type === "pdf" ? "pdf" : f.type === "csv" ? "csv" : "excel";
    const icon   = fileKindIcon(kind);
    const sheets = f.sheets && f.sheets.length > 0 ? ` · ${f.sheets.length} sheet(s)` : "";
    const row    = document.createElement("div"); row.className = "indexed-file-row";
    row.innerHTML = `
      <span class="indexed-icon">${icon}</span>
      <div class="indexed-info">
        <span class="indexed-name" title="${f.filename}">${f.filename}</span>
        <span class="indexed-meta">${f.chunks} chunks · ${f.type.toUpperCase()}${sheets}</span>
      </div>
      <button class="indexed-remove" data-filename="${f.filename}" title="Remove from index">🗑️</button>`;
    indexedFilesList.appendChild(row);
  });
  indexedFilesList.querySelectorAll(".indexed-remove").forEach(btn => {
    btn.addEventListener("click", async () => {
      try {
        const res = await fetch(`${API}/files/${encodeURIComponent(btn.dataset.filename)}`, { method: "DELETE" });
        if (res.ok) renderIndexedFiles((await res.json()).indexed_files ?? []);
      } catch (e) {
        showStatus(`❌ Could not remove: ${e.message}`, "error");
      }
    });
  });
}

// Load on startup
fetch(`${API}/files`).then(r => r.ok ? r.json() : []).then(renderIndexedFiles).catch(() => {});

// ── Chat Input ────────────────────────────────────────────────────────────────
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!isLoading) sendMessage(); }
});
sendBtn.addEventListener("click", () => { if (!isLoading) sendMessage(); });

// ── Send Message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  isLoading = true; sendBtn.disabled = true;
  chatInput.value = ""; chatInput.style.height = "auto";
  appendMessage("user", text);
  const thinkingId = appendThinking();
  try {
    const res = await fetch(`${API}/chat?mode=${ragMode}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, session_id: SESSION_ID })
    });
    removeThinking(thinkingId);
    if (res.ok) {
      const data = await res.json();
      appendAssistantMessage(
        data.answer ?? "(No answer)",
        `🔍 Source: ${data.sources ?? "N/A"} | 🧠 Engine: ${data.mode ?? ragMode}`,
        data.sources_detail ?? [], data.followups ?? []
      );
    } else {
      appendMessage("assistant", "⚠️ Backend error. Make sure files are uploaded and FastAPI is running.");
    }
  } catch (e) {
    removeThinking(thinkingId);
    appendMessage("assistant", `⚠️ Connection failed: ${e.message}`);
  } finally {
    isLoading = false; sendBtn.disabled = false;
  }
}

// ── Message Rendering ─────────────────────────────────────────────────────────
function appendMessage(role, text) {
  hideEmpty();
  const isUser = role === "user";
  const wrap   = document.createElement("div"); wrap.className = `message ${role}`;
  const avatar = document.createElement("div"); avatar.className = `msg-avatar ${role}`; avatar.textContent = isUser ? "👤" : "🤖";
  const body   = document.createElement("div"); body.className = "msg-body";
  const name   = document.createElement("div"); name.className = "msg-name"; name.textContent = isUser ? "user" : "assistant";
  const bubble = document.createElement("div"); bubble.className = "msg-text"; bubble.textContent = text;
  body.appendChild(name); body.appendChild(bubble);
  if (isUser) { wrap.appendChild(body); wrap.appendChild(avatar); wrap.style.flexDirection = "row-reverse"; }
  else { wrap.appendChild(avatar); wrap.appendChild(body); }
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendAssistantMessage(text, caption, sourcesDetail, followups) {
  hideEmpty();
  const wrap   = document.createElement("div"); wrap.className = "message assistant";
  const avatar = document.createElement("div"); avatar.className = "msg-avatar assistant"; avatar.textContent = "🤖";
  const body   = document.createElement("div"); body.className = "msg-body";
  const name   = document.createElement("div"); name.className = "msg-name"; name.textContent = "assistant";
  const bubble = document.createElement("div"); bubble.className = "msg-text"; bubble.textContent = text;
  const cap    = document.createElement("div"); cap.className = "msg-caption"; cap.textContent = caption;
  body.appendChild(name); body.appendChild(bubble); body.appendChild(cap);

  if (sourcesDetail && sourcesDetail.length > 0) {
    const srcBlock = document.createElement("div"); srcBlock.className = "sources-block";
    const toggle   = document.createElement("div"); toggle.className = "sources-toggle";
    toggle.innerHTML = `<span class="chevron">▶</span> View ${sourcesDetail.length} Source Chunk${sourcesDetail.length > 1 ? "s" : ""}`;
    const cards    = document.createElement("div"); cards.className = "sources-cards";
    const colours  = ["#4b9eff", "#2dc97a", "#f5a623"];

    sourcesDetail.forEach((src, i) => {
      const card = document.createElement("div"); card.className = "source-card";
      card.style.borderLeftColor = colours[i] ?? "#4b9eff";

      const isExcel = src.file_type === "excel" || src.file_type === "csv";
      const icon    = src.file_type === "pdf" ? "📄" : src.file_type === "csv" ? "📋" : "📊";
      const locInfo = isExcel
        ? (src.sheets ? `Sheets: ${src.sheets}` : "Spreadsheet")
        : (src.page !== "N/A" ? `Page ${src.page}` : "N/A");

      card.innerHTML = `
        <div class="source-card-header">
          <span class="source-page">${icon} ${locInfo}</span>
          <span class="source-score">Score: ${src.score}</span>
        </div>
        <div class="source-file-tag">${src.source_file ?? ""}</div>
        <div class="source-snippet">"${src.snippet}${src.snippet.length >= 220 ? "…" : ""}"</div>`;
      cards.appendChild(card);
    });

    toggle.addEventListener("click", () => {
      const open = cards.classList.toggle("visible");
      toggle.classList.toggle("open", open);
      toggle.querySelector(".chevron").textContent = open ? "▼" : "▶";
    });
    srcBlock.appendChild(toggle); srcBlock.appendChild(cards); body.appendChild(srcBlock);
  }

  if (followups && followups.length > 0) {
    const fuBlock = document.createElement("div"); fuBlock.className = "followup-block";
    const fuLabel = document.createElement("div"); fuLabel.className = "followup-label"; fuLabel.textContent = "💡 You might also ask";
    const chips   = document.createElement("div"); chips.className = "followup-chips";
    followups.forEach(q => {
      const chip = document.createElement("button"); chip.className = "chip"; chip.textContent = q;
      chip.addEventListener("click", () => { chatInput.value = q; chatInput.dispatchEvent(new Event("input")); chatInput.focus(); });
      chips.appendChild(chip);
    });
    fuBlock.appendChild(fuLabel); fuBlock.appendChild(chips); body.appendChild(fuBlock);
  }

  wrap.appendChild(avatar); wrap.appendChild(body);
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

let thinkingCounter = 0;
function appendThinking() {
  hideEmpty();
  const id   = "thinking-" + (++thinkingCounter);
  const wrap = document.createElement("div"); wrap.className = "message assistant"; wrap.id = id;
  const av   = document.createElement("div"); av.className = "msg-avatar assistant"; av.textContent = "🤖";
  const body = document.createElement("div"); body.className = "msg-body";
  const nm   = document.createElement("div"); nm.className = "msg-name"; nm.textContent = "assistant";
  const row  = document.createElement("div"); row.className = "thinking-row";
  row.innerHTML = `<span class="spinner"></span> Running ${ragMode} logic…`;
  body.appendChild(nm); body.appendChild(row);
  wrap.appendChild(av); wrap.appendChild(body);
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return id;
}

function removeThinking(id) { const el = document.getElementById(id); if (el) el.remove(); }
function hideEmpty()        { if (emptyState) emptyState.style.display = "none"; }
