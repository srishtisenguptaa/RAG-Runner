const API = "http://localhost:8000";
let selectedFile = null;
let ragMode = "standard";
let isLoading = false;

// ── Session ID: unique per page-load (refresh = new session = cleared memory) ──
let SESSION_ID = localStorage.getItem("rag_session_id");

if (SESSION_ID) {
  // If we have an old ID, tell the backend to wipe it before we start fresh
  fetch(`${API}/history/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});
}

// Generate a brand new ID for this new session and save it
SESSION_ID = crypto.randomUUID();
localStorage.setItem("rag_session_id", SESSION_ID);

// Tell the backend to wipe any stale history for this session slot
fetch(`${API}/history/${SESSION_ID}`, { method: "DELETE" }).catch(() => {});

/* ── Tooltip positioning (fixed, so it escapes sidebar overflow) ── */
document.querySelectorAll(".tooltip-wrap").forEach(wrap => {
  const icon = wrap.querySelector(".tooltip-icon");
  const box  = wrap.querySelector(".tooltip-box");
  wrap.addEventListener("mouseenter", () => {
    const r = icon.getBoundingClientRect();
    box.style.top  = (r.top + r.height / 2) + "px";
    box.style.left = (r.right + 10) + "px";
    box.style.transform = "translateY(-50%)";
  });
});

const chatWindow   = document.getElementById("chat-window");
const emptyState   = document.getElementById("empty-state");
const chatInput    = document.getElementById("chat-input");
const sendBtn      = document.getElementById("send-btn");
const fileInput    = document.getElementById("file-input");
const dropZone     = document.getElementById("drop-zone");
const fileNameEl   = document.getElementById("file-name");
const indexBtn     = document.getElementById("index-btn");
const indexStatus  = document.getElementById("index-status");
const radioGroup   = document.getElementById("rag-mode-group");
const themeCheck   = document.getElementById("theme-checkbox");
const themeLabel   = document.getElementById("theme-label");

/* ── Theme Toggle ── */
themeCheck.addEventListener("change", () => {
  const isLight = themeCheck.checked;
  document.documentElement.setAttribute("data-theme", isLight ? "light" : "dark");
  themeLabel.textContent = isLight ? "Light" : "Dark";
});

const optionalBadge  = document.getElementById("optional-badge");
const uploadSection  = document.getElementById("upload-section");
const emptyText      = document.getElementById("empty-text");

function applyModeUI() {
  const isArchitect = ragMode === "architect";
  optionalBadge.classList.toggle("visible", isArchitect);
  uploadSection.classList.toggle("faded", isArchitect);
  emptyText.textContent = isArchitect
    ? "Ask anything — Architect uses Web Search. Upload a PDF for document-specific answers."
    : "Upload a PDF and ask a question to get started.";
}

/* ── RAG Mode Toggle ── */
radioGroup.querySelectorAll(".radio-option").forEach(opt => {
  opt.addEventListener("click", () => {
    ragMode = opt.dataset.value;
    radioGroup.querySelectorAll(".radio-option").forEach(o => o.classList.remove("selected"));
    opt.classList.add("selected");
    opt.querySelector("input").checked = true;
    applyModeUI();
  });
});

/* ── File Upload ── */
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f && f.type === "application/pdf") setFile(f);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  fileNameEl.textContent = "📄 " + f.name;
  fileNameEl.style.display = "block";
  indexBtn.disabled = false;
  indexStatus.style.display = "none";
}

/* ── Index Document ── */
indexBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  indexBtn.disabled = true;
  indexBtn.innerHTML = '<span class="spinner"></span> Processing…';
  indexStatus.style.display = "none";
  try {
    const form = new FormData();
    form.append("file", selectedFile, selectedFile.name);
    const res = await fetch(`${API}/upload`, { method: "POST", body: form });
    if (res.ok) {
      const data = await res.json();
      showStatus(`✅ Indexed into ${data.chunks_created ?? "?"} semantic chunks!`, "success");
    } else {
      showStatus("❌ Upload failed. Is the backend running?", "error");
    }
  } catch (e) {
    showStatus("❌ Connection failed: " + e.message, "error");
  } finally {
    indexBtn.innerHTML = "Index Document";
    indexBtn.disabled = false;
  }
});

function showStatus(msg, type) {
  indexStatus.textContent = msg;
  indexStatus.className = type;
  indexStatus.style.display = "block";
}

/* ── Chat Input ── */
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
});
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!isLoading) sendMessage(); }
});
sendBtn.addEventListener("click", () => { if (!isLoading) sendMessage(); });

/* ── Send Message ── */
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  isLoading = true;
  sendBtn.disabled = true;
  chatInput.value = "";
  chatInput.style.height = "auto";
  appendMessage("user", text);
  const thinkingId = appendThinking();
  try {
    const res = await fetch(`${API}/chat?mode=${ragMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, session_id: SESSION_ID })
    });
    removeThinking(thinkingId);
    if (res.ok) {
      const data = await res.json();
      appendAssistantMessage(
        data.answer ?? "(No answer)",
        `🔍 Source: ${data.sources ?? "N/A"} | 🧠 Engine: ${data.mode ?? ragMode}`,
        data.sources_detail ?? [],
        data.followups ?? []
      );
    } else {
      appendMessage("assistant", "⚠️ Backend error. Make sure the PDF is uploaded and FastAPI is running.");
    }
  } catch (e) {
    removeThinking(thinkingId);
    appendMessage("assistant", `⚠️ Connection failed: ${e.message}`);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
  }
}

function appendMessage(role, text) {
  hideEmpty();
  const isUser = role === "user";
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = `msg-avatar ${role}`;
  avatar.textContent = isUser ? "👤" : "🤖";
  const body = document.createElement("div"); body.className = "msg-body";
  const name = document.createElement("div"); name.className = "msg-name"; name.textContent = isUser ? "user" : "assistant";
  const bubble = document.createElement("div"); bubble.className = "msg-text"; bubble.textContent = text;
  body.appendChild(name); body.appendChild(bubble);
  if (isUser) { wrap.appendChild(body); wrap.appendChild(avatar); wrap.style.flexDirection = "row-reverse"; }
  else { wrap.appendChild(avatar); wrap.appendChild(body); }
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendAssistantMessage(text, caption, sourcesDetail, followups) {
  hideEmpty();
  const wrap = document.createElement("div"); wrap.className = "message assistant";
  const avatar = document.createElement("div"); avatar.className = "msg-avatar assistant"; avatar.textContent = "🤖";
  const body = document.createElement("div"); body.className = "msg-body";
  const name = document.createElement("div"); name.className = "msg-name"; name.textContent = "assistant";
  const bubble = document.createElement("div"); bubble.className = "msg-text"; bubble.textContent = text;
  const cap = document.createElement("div"); cap.className = "msg-caption"; cap.textContent = caption;
  body.appendChild(name); body.appendChild(bubble); body.appendChild(cap);

  if (sourcesDetail && sourcesDetail.length > 0) {
    const srcBlock = document.createElement("div"); srcBlock.className = "sources-block";
    const toggle = document.createElement("div"); toggle.className = "sources-toggle";
    toggle.innerHTML = `<span class="chevron">▶</span> View ${sourcesDetail.length} Source Chunk${sourcesDetail.length > 1 ? "s" : ""}`;
    const cards = document.createElement("div"); cards.className = "sources-cards";
    sourcesDetail.forEach((src, i) => {
      const card = document.createElement("div"); card.className = "source-card";
      const colours = ["#4b9eff", "#2dc97a", "#f5a623"];
      card.style.borderLeftColor = colours[i] ?? "#4b9eff";
      card.innerHTML = `
        <div class="source-card-header">
          <span class="source-page">📄 Page ${src.page}</span>
          <span class="source-score">Score: ${src.score}</span>
        </div>
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
    const chips = document.createElement("div"); chips.className = "followup-chips";
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
  const id = "thinking-" + (++thinkingCounter);
  const wrap = document.createElement("div"); wrap.className = "message assistant"; wrap.id = id;
  const avatar = document.createElement("div"); avatar.className = "msg-avatar assistant"; avatar.textContent = "🤖";
  const body = document.createElement("div"); body.className = "msg-body";
  const name = document.createElement("div"); name.className = "msg-name"; name.textContent = "assistant";
  const row = document.createElement("div"); row.className = "thinking-row";
  row.innerHTML = `<span class="spinner"></span> Running ${ragMode} logic…`;
  body.appendChild(name); body.appendChild(row);
  wrap.appendChild(avatar); wrap.appendChild(body);
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return id;
}

function removeThinking(id) { const el = document.getElementById(id); if (el) el.remove(); }
function hideEmpty() { if (emptyState) emptyState.style.display = "none"; }
