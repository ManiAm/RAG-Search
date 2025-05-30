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

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errorText = await res.text();
      alert(`Error ${res.status}: ${errorText}`);
      return;
    }

    if (stream) {
      await streamAndRender(res, targetId);
    } else {
      const data = await res.json();
      appendToResponse(targetId, data.answer);
    }
  } catch (err) {
    alert("Failed to reach server: " + err.message);
  }
}

async function sendRag() {
  const targetId = "responseBoxRag";
  clearResponse(targetId);

  const payload = {
    question: document.getElementById("questionRag").value,
    llm_model: document.getElementById("llmModelRag").value,
    embed_model: document.getElementById("embedModel").value,
    collection_name: document.getElementById("collectionNameRag").value,
    session_id: document.getElementById("sessionIdRag").value,
    instructions: document.getElementById("instructionsRag").value
  };

  const stream = document.getElementById("streamRag").checked;
  const url = stream ? "/api/v1/rag/chat-stream" : "/api/v1/rag/chat";

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errorText = await res.text();
      alert(`Error ${res.status}: ${errorText}`);
      return;
    }

    if (stream) {
      await streamAndRender(res, targetId);
    } else {
      const data = await res.json();
      appendToResponse(targetId, data.answer);
    }
  } catch (err) {
    alert("Failed to reach server: " + err.message);
  }
}

async function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const embedModel = document.getElementById("embedModel").value;
  const collectionName = document.getElementById("collectionNameRag").value;

  if (!fileInput.files.length) {
    alert("Please select a file to upload.");
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  const query = new URLSearchParams({
    embed_model: embedModel,
    collection_name: collectionName
  });

  try {
    const res = await fetch(`/api/v1/rag/upload?${query.toString()}`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Server error: ${res.status} - ${errText}`);
    }

    const json = await res.json();
    alert("Upload successful:\n" + JSON.stringify(json, null, 2));
  } catch (err) {
    alert("Upload failed:\n" + err.message);
  }
}

async function uploadPaste() {
  const text = document.getElementById("pastedDoc").value.trim();
  const embedModel = document.getElementById("embedModel").value.trim();
  const collectionName = document.getElementById("collectionNameRag").value.trim();

  try {
    const res = await fetch("/api/v1/rag/paste", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        embed_model: embedModel,
        collection_name: collectionName
      })
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Server error ${res.status}: ${errorText}`);
    }

    const json = await res.json();
    alert("Pasted text:\n" + JSON.stringify(json, null, 2));
  } catch (err) {
    alert("Upload failed:\n" + err.message);
  }
}

window.onload = () => {
  loadModels();
  switchTab("llm");
};
