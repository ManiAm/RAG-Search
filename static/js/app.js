function switchTab(tab) {
  document.querySelectorAll(".tab").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(div => div.classList.remove("visible"));
  document.querySelector(`#tab-${tab}`).classList.add("visible");
  document.querySelector(`.tab-bar .tab:nth-child(${tab === "llm" ? 1 : 2})`).classList.add("active");
}

function clearResponse(targetId) {
  document.getElementById(targetId).innerHTML = "";
}

function appendToResponse(targetId, markdown) {
  const box = document.getElementById(targetId);
  const html = marked.parse(markdown);
  box.innerHTML = html;
  MathJax.typesetPromise();
  box.scrollTop = box.scrollHeight;
}

async function streamAndRender(res, targetId) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  let interim = "";

  const box = document.getElementById(targetId);
  box.innerHTML = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    fullText += chunk;
    interim += chunk;
    box.textContent = interim;
    box.scrollTop = box.scrollHeight;
  }

  appendToResponse(targetId, fullText);
}

async function loadModels() {
  const llmRes = await fetch("/api/v1/llm/models");
  const ragRes = await fetch("/api/v1/rag/embeddings");
  const llms = await llmRes.json();
  const embs = await ragRes.json();

  const llmDropdowns = [document.getElementById("llmModel"), document.getElementById("llmModelRag")];
  llmDropdowns.forEach(sel => {
    sel.innerHTML = "";
    llms.models.forEach((m, idx) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.text = m;
      if (idx === 0 || m === "llama3.1:8b") opt.selected = true; // set default
      sel.appendChild(opt);
    });
  });

  const embedSel = document.getElementById("embedModel");
  embedSel.innerHTML = "";
  embs.models.forEach((m, idx) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.text = m;
    if (idx === 0 || m === "bge-large-en-v1.5") opt.selected = true; // set default
    embedSel.appendChild(opt);
  });
}

async function sendLlm() {
  const targetId = "responseBoxLlm";
  clearResponse(targetId);

  const payload = {
    question: document.getElementById("questionLlm").value,
    llm_model: document.getElementById("llmModel").value,
    session_id: document.getElementById("sessionIdLlm").value,
    context: document.getElementById("contextLlm").value
  };
  const stream = document.getElementById("streamLlm").checked;
  const url = stream ? "/api/v1/llm/chat-stream" : "/api/v1/llm/chat";

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (stream) {
    await streamAndRender(res, targetId);
  } else {
    const data = await res.json();
    appendToResponse(targetId, data.answer);
  }
}

async function sendRag() {
  const targetId = "responseBoxRag";
  clearResponse(targetId);

  const payload = {
    question: document.getElementById("questionRag").value,
    llm_model: document.getElementById("llmModelRag").value,
    embed_model: document.getElementById("embedModel").value,
    session_id: document.getElementById("sessionIdRag").value,
    instructions: document.getElementById("instructionsRag").value
  };
  const stream = document.getElementById("streamRag").checked;
  const url = stream ? "/api/v1/rag/chat-stream" : "/api/v1/rag/chat";

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (stream) {
    await streamAndRender(res, targetId);
  } else {
    const data = await res.json();
    appendToResponse(targetId, data.answer);
  }
}

async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  const embedModel = document.getElementById("embedModel").value;

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`/api/v1/rag/upload?embed_model=${encodeURIComponent(embedModel)}`, {
    method: "POST",
    body: formData
  });

  const json = await res.json();
  alert("Upload: " + JSON.stringify(json));
}

async function uploadPaste() {
  const text = document.getElementById("pastedDoc").value;
  const embedModel = document.getElementById("embedModel").value;

  const res = await fetch("/api/v1/rag/paste", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, embed_model: embedModel })
  });

  const json = await res.json();
  alert("Pasted text: " + JSON.stringify(json));
}

window.onload = () => {
  loadModels();
  switchTab("llm");
};
