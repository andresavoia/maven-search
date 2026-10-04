const queryInput = document.querySelector("#query");
const statusNode = document.querySelector("#status");
const player = document.querySelector("#player");
let videoId = "";

function stamp(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function seek(seconds) {
  const start = Math.max(0, Math.floor(seconds));
  player.src = `https://www.youtube.com/embed/${videoId}?start=${start}&autoplay=1`;
}

function locatorLabel(hit) {
  if (hit.page) return `Paper p.${hit.page}`;
  if (hit.slide) return `Slide ${hit.slide}`;
  return `${stamp(hit.start)}–${stamp(hit.end)}`;
}

function renderHits(node, hits) {
  node.innerHTML = "";
  for (const hit of hits) {
    const block = document.createElement("article");
    block.className = "hit";
    const title = hit.title ? `${hit.title}` : hit.text.slice(0, 90);
    block.innerHTML = `<strong>${stamp(hit.start)}–${stamp(hit.end)} · ${hit.score}</strong><p></p>`;
    block.querySelector("p").textContent = title === hit.text.slice(0, 90) ? hit.text : `${title} — ${hit.text}`;
    block.addEventListener("click", () => seek(hit.start));
    node.appendChild(block);
  }
}

async function ask(query) {
  statusNode.textContent = "Retrieving from both indexes…";
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    statusNode.textContent = payload.detail || "Request failed.";
    return;
  }
  const payload = await response.json();
  document.querySelector("#baseline-answer").textContent = payload.baseline.answer;
  document.querySelector("#moment-answer").textContent = payload.moment.answer;
  renderHits(document.querySelector("#baseline-hits"), payload.baseline.hits);
  renderHits(document.querySelector("#moment-hits"), payload.moment.hits);
  document.querySelector("#cross-answer").textContent = payload.cross.answer;
  renderCrossHits(document.querySelector("#cross-hits"), payload.cross.hits);
  if (payload.moment.hits[0]) seek(payload.moment.hits[0].start);
  const kinds = [...new Set(payload.cross.hits.map((hit) => (hit.page ? "paper" : hit.slide ? "deck" : "video")))];
  statusNode.textContent = `Baseline: ${payload.baseline.mode}. Moment RAG: ${payload.moment.mode}. Cross-source: ${kinds.join(", ") || "none"}.`;
}

function renderCrossHits(node, hits) {
  node.innerHTML = "";
  for (const hit of hits) {
    const block = document.createElement("article");
    block.className = "hit";
    block.innerHTML = `<strong>${locatorLabel(hit)} · ${hit.score}</strong><p></p>`;
    block.querySelector("p").textContent = hit.text;
    block.addEventListener("click", () => {
      if (!hit.page && !hit.slide) seek(hit.locator ? hit.locator.start_ms / 1000 : hit.start);
    });
    node.appendChild(block);
  }
}

document.querySelector("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  ask(queryInput.value.trim());
});

async function boot() {
  const response = await fetch("/api/video");
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    statusNode.textContent = payload.detail || `Could not load the index (${response.status}).`;
    return;
  }
  const payload = await response.json();
  videoId = payload.video.id;
  player.src = `https://www.youtube.com/embed/${videoId}`;
  document.querySelector("#why").textContent = `${payload.video.title} — ${payload.video.author}. ${payload.video.why}`;
  document.querySelector("#meta").textContent =
    `${payload.counts.chunks} semantic chunks · ${payload.counts.moments} moments · ${payload.counts.documents} paper and slide passages · ${Math.round(payload.video.duration / 60)} min`;
  const chips = document.querySelector("#chips");
  for (const sample of payload.samples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = sample;
    button.addEventListener("click", () => {
      queryInput.value = sample;
      ask(sample);
    });
    chips.appendChild(button);
  }
}

boot();
