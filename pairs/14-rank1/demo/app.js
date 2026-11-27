// rank1 demo: three ranked lists linked by the same abstract, and rank1's reasoning on the selected one
const SHOWN = 10;
const STOP = new Set("a an the of in on to for and or is are was were be been by with as at from that this these those it its may can not no into than then which who whom whose has have had do does did more most less their there".split(" "));
const $ = (id) => document.getElementById(id);
const cols = { bm25: document.querySelector('[data-col="bm25"]'), ce: document.querySelector('[data-col="ce"]'), r1: document.querySelector('[data-col="r1"]') };
const nodes = { bm25: new Map(), ce: new Map(), r1: new Map() };

const S = { state: null, q: null, docs: new Map(), order: [], mode: "r1", sel: null, hot: null,
            live: false, running: false, thinking: null, judged: 0, follow: true, es: null,
            run: null, timer: null };
const PER_DOC = 4.5;  // seconds per abstract with reasoning, rank1-1.5b 4-bit on a macbook air (our speed test)
const TOKENS = 300;   // typical length of a reasoning

const fmt = (x, d) => (x === undefined || x === null ? "" : Number(x).toFixed(d));
const escapeHtml = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function stable(docs, key) {
  return docs.map((d, i) => [d, i]).sort((a, b) => key(b[0]) - key(a[0]) || a[1] - b[1]).map((x) => x[0].id);
}
function orderOf(col) {
  const docs = S.order.map((id) => S.docs.get(id));
  if (col === "bm25") return S.order.slice();
  if (col === "ce") return stable(docs, (d) => d.ce);
  if (S.mode === "r2") return stable(docs, (d) => d.r2 ?? -1);
  const judged = docs.filter((d) => d.r1 !== undefined);
  const waiting = docs.filter((d) => d.r1 === undefined);
  return stable(judged, (d) => d.r1).concat(stable(waiting, (d) => d.r2 ?? -1));
}
const rankIn = (col, id) => orderOf(col).indexOf(id) + 1;

// rows
function row(col, d, rank) {
  let li = nodes[col].get(d.id);
  if (!li) {
    li = document.createElement("li");
    li.className = "doc";
    li.innerHTML = '<span class="rank"></span><span class="t"></span><span class="s"></span>';
    li.addEventListener("click", () => select(d.id, true));
    li.addEventListener("mouseenter", () => hover(d.id));
    li.addEventListener("mouseleave", () => hover(null));
    nodes[col].set(d.id, li);
  }
  li.dataset.id = d.id;
  li.querySelector(".rank").textContent = rank;
  li.querySelector(".t").textContent = d.title;
  li.title = d.title;
  li.classList.toggle("labelled", d.labelled);
  li.classList.toggle("sel", d.id === S.sel);
  li.classList.toggle("hot", d.id === S.hot);
  li.classList.remove("waiting", "thinking");
  const s = li.querySelector(".s");
  if (col === "bm25") s.textContent = fmt(d.bm25, 1);
  else if (col === "ce") s.textContent = fmt(d.ce, 2);
  else {
    const pending = S.mode === "r1" && d.r1 === undefined;
    const p = S.mode === "r2" ? d.r2 : pending ? d.r2 : d.r1;
    if (pending) li.classList.add(d.id === S.thinking ? "thinking" : "waiting");
    const text = pending && d.id === S.thinking ? "…" : p === undefined ? "" : fmt(p, 2);
    s.innerHTML = `<span>${text}</span><span class="pbar"><i style="width:${(p ?? 0) * 100}%"></i></span>`;
  }
  return li;
}

function renderCol(col, animate = true) {
  const ol = cols[col].querySelector("ol");
  const before = new Map();
  if (animate) ol.querySelectorAll(".doc").forEach((li) => before.set(li.dataset.id, li.getBoundingClientRect().top));
  const ids = orderOf(col);
  const items = ids.slice(0, SHOWN).map((id, i) => row(col, S.docs.get(id), i + 1));
  const far = ids.slice(SHOWN).filter((id) => S.docs.get(id).labelled);
  if (far.length) {
    const gap = document.createElement("li");
    gap.className = "gap";
    gap.textContent = "· · ·";
    items.push(gap, ...far.map((id) => row(col, S.docs.get(id), ids.indexOf(id) + 1)));
  }
  ol.replaceChildren(...items);
  if (!animate) return;
  items.forEach((li) => {
    const top = before.get(li.dataset.id);
    const dy = top === undefined ? 0 : top - li.getBoundingClientRect().top;
    if (!dy) return;
    li.style.transition = "none";
    li.style.transform = `translateY(${dy}px)`;
    li.getBoundingClientRect();
    li.style.transition = "transform .6s cubic-bezier(.2,.75,.2,1)";
    li.style.transform = "";
  });
  animateLinks(700);
}

function renderAll() {
  cols.r1.classList.toggle("at-once", S.mode === "r2");
  ["bm25", "ce", "r1"].forEach((c) => renderCol(c, false));
  document.querySelectorAll(".col .doc").forEach((li) => {
    li.style.setProperty("--i", [...li.parentNode.children].indexOf(li));
    li.classList.add("fresh");
    li.addEventListener("animationend", () => li.classList.remove("fresh"), { once: true });
  });
  journey();
  judge(true);
  requestAnimationFrame(() => {
    $("links").classList.add("in");
    drawLinks();
    setTimeout(() => $("links").classList.remove("in"), 1200);
  });
}

// a short fade between two results
function swap(then) {
  const results = document.querySelector(".results");
  if (document.body.classList.contains("start")) { document.body.classList.remove("start"); then(); return; }
  results.classList.add("swap");
  setTimeout(() => { then(); results.classList.remove("swap"); }, 220);
}

// lines joining the same abstract across the lists
function drawLinks() {
  const box = $("board").getBoundingClientRect();
  const paths = [];
  [["bm25", "ce"], ["ce", "r1"]].forEach(([a, b]) => {
    nodes[a].forEach((la, id) => {
      const lb = nodes[b].get(id);
      if (!la.isConnected || !lb || !lb.isConnected) return;
      const ra = la.getBoundingClientRect(), rb = lb.getBoundingClientRect();
      const x1 = ra.right - box.left, y1 = ra.top + ra.height / 2 - box.top;
      const x2 = rb.left - box.left, y2 = rb.top + rb.height / 2 - box.top;
      const xm = (x1 + x2) / 2;
      const d = S.docs.get(id);
      const cls = d && d.labelled ? "labelled" : id === S.sel ? "sel" : id === S.hot ? "hot" : "";
      paths.push([cls, `<path pathLength="1" class="${cls}" d="M${x1},${y1} C${xm},${y1} ${xm},${y2} ${x2},${y2}"/>`]);
    });
  });
  const weight = { "": 0, hot: 1, sel: 2, labelled: 3 };
  $("links").innerHTML = paths.sort((p, q) => weight[p[0]] - weight[q[0]]).map((p) => p[1]).join("");
}
function animateLinks(ms) {
  const end = performance.now() + ms;
  const step = () => { drawLinks(); if (performance.now() < end) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

function hover(id) {
  S.hot = id;
  Object.values(nodes).forEach((m) => m.forEach((li, k) => li.classList.toggle("hot", k === id)));
  drawLinks();
}
function select(id, byUser = false) {
  S.sel = id;
  if (byUser && S.running && id !== S.thinking) { S.follow = false; $("follow").hidden = false; }
  Object.values(nodes).forEach((m) => m.forEach((li, k) => li.classList.toggle("sel", k === id)));
  drawLinks();
  judge(true);
}

// where the relevant abstract sits in each list
function journey() {
  const el = $("journey");
  const labelled = S.order.filter((id) => S.docs.get(id).labelled);
  if (!labelled.length) {
    el.innerHTML = S.q.known
      ? `<span>The relevant abstract is not in the BM25 top ${S.order.length}: no reranker can bring it back.</span>`
      : "<span>No abstract is labelled relevant for this claim: which list would you rather read?</span>";
    return;
  }
  const id = labelled.sort((a, b) => rankIn("r1", a) - rankIn("r1", b))[0];
  const pending = S.mode === "r1" && S.docs.get(id).r1 === undefined;
  el.innerHTML = `<span class="lead">Relevant abstract</span>
    <span class="step bm25">BM25 <b>${rankIn("bm25", id)}</b></span><span class="arrow">→</span>
    <span class="step ce">Cross-encoder <b>${rankIn("ce", id)}</b></span><span class="arrow">→</span>
    <span class="step r1">Rank1${S.mode === "r2" ? " (no reasoning)" : ""} <b>${pending ? "…" : rankIn("r1", id)}</b></span>`;
}

// the panel: what rank1 read, what it thought, what it answered
function contentWords(text) {
  const seen = new Set();
  return (text.match(/[A-Za-z0-9][A-Za-z0-9-]*/g) || [])
    .filter((w) => w.length > 2 && !STOP.has(w.toLowerCase()) && !seen.has(w.toLowerCase()) && seen.add(w.toLowerCase()));
}
const stem = (w) => w.slice(0, Math.max(4, w.length - 3));
function decision(text) {
  // the last sentence is where rank1 decides
  const t = text.trim();
  const cut = t.slice(0, -1).search(/[.!?]\s+[^.!?]*[.!?]?$/);
  if (cut < 0) return escapeHtml(t);
  return escapeHtml(t.slice(0, cut + 1)) + " <strong>" + escapeHtml(t.slice(cut + 1).trim()) + "</strong>";
}
function verdictTag(p) {
  if (p === undefined) return "&lt;/think&gt;";
  const v = p >= 0.5 ? "true" : "false";
  return `&lt;/think&gt; <b class="${v}">${v}</b>`;
}
function meter(k, p) {
  $("m" + k).style.width = p === undefined ? "0" : `${p * 100}%`;
  $("v" + k).textContent = p === undefined ? "–" : fmt(p, 2);
}

function judge(reset = false) {
  const d = S.docs.get(S.sel);
  if (!d) return;
  $("doc-title").textContent = d.title;
  const words = contentWords(S.q.claim);
  const hit = words.filter((w) => new RegExp(`\\b${stem(w)}`, "i").test(d.text));
  const miss = words.filter((w) => !hit.includes(w));
  $("missing").innerHTML = miss.length ? `Never mentions <b>${miss.map(escapeHtml).join(", ")}</b>` : "";
  let html = escapeHtml(d.text);
  if (hit.length) html = html.replace(new RegExp(`\\b(${hit.map(stem).join("|")})[\\w-]*`, "gi"), "<mark>$&</mark>");
  $("doc-text").innerHTML = html;

  const thought = $("thought"), body = $("reasoning"), tag = $("verdict-tag");
  thought.classList.toggle("at-once", S.mode === "r2");
  if (S.mode === "r2") {
    body.textContent = "No reasoning: the block is closed at once, and Rank1 answers in one forward pass.";
    tag.innerHTML = verdictTag(d.r2);
  } else if (d.r1 === undefined) {
    body.innerHTML = d.id === S.thinking ? escapeHtml(d.reasoning || "") + '<span class="caret"></span>'
      : `<span class="wait">Waiting: Rank1 reads the abstracts in BM25 order, this one is ${S.order.indexOf(d.id) + 1} of ${S.order.length}.</span>`;
    tag.innerHTML = d.id === S.thinking ? "" : "&lt;/think&gt;";
  } else {
    body.innerHTML = decision(d.reasoning || "");
    tag.innerHTML = verdictTag(d.r1);
  }
  if (reset) thought.scrollTop = 0;
  else if (d.id === S.thinking) thought.scrollTop = thought.scrollHeight;
  meter("1", d.r1);
  meter("2", d.r2);
}

function begin(claim, live) {
  $("claim").value = claim;
  S.live = live; S.thinking = null; S.follow = true; S.judged = 0;
  $("follow").hidden = true;
  cols.r1.classList.toggle("running", live);
  document.querySelector(".strip").classList.toggle("running", live);
  Object.values(nodes).forEach((m) => m.clear());
  $("model").textContent = live ? `${S.state.model.replace("rank1-", "").toUpperCase()}, live` : "3B";
}

// hero and failure: precomputed
async function loadQuery(key) {
  stopLive();
  const q = await (await fetch(`/api/query/${key}`)).json();
  document.querySelectorAll(".examples button").forEach((b) => b.classList.toggle("on", b.dataset.key === key));
  $("claim").value = q.claim;
  swap(() => {
    begin(q.claim, false);
    S.q = { claim: q.claim, known: true };
    S.running = false;
    S.docs = new Map(q.docs.map((d) => [d.id, d]));
    S.order = q.docs.map((d) => d.id);
    S.sel = stable(q.docs.filter((d) => d.labelled), (d) => d.r1)[0] || orderOf("r1")[0];
    renderAll();
  });
}

// any claim: live on this laptop
function stopLive() {
  if (S.es) { S.es.close(); S.es = null; }
  clearInterval(S.timer);
}
// the bar that shows how far the live run is, and how long it should still take
function runbar() {
  const r = S.run;
  if (!r) return;
  const n = S.order.length;
  const segs = $("segments");
  if (segs.children.length !== n) segs.replaceChildren(...Array.from({ length: n }, () => document.createElement("i")));
  [...segs.children].forEach((seg, k) => {
    const d = S.docs.get(S.order[k]);
    const now = d && d.id === S.thinking;
    const f = d && d.r1 !== undefined ? 100 : now ? Math.min(95, ((d.reasoning || "").split(/\s+/).length / TOKENS) * 130) : 0;
    seg.classList.toggle("now", now);
    seg.style.setProperty("--f", `${f}%`);
  });
  const done = r.seconds.length;
  const per = done ? r.seconds.reduce((a, b) => a + b, 0) / done : PER_DOC;
  const current = S.thinking ? (performance.now() - r.started) / 1000 : 0;
  const left = n ? Math.max(1, Math.round(per * (n - done) - current + (r.phase === "quick" ? PER_DOC / 2 : 0))) : 0;
  $("run-what").textContent = r.phase === "search" ? "BM25 and the cross-encoder are ranking the abstracts"
    : r.phase === "quick" ? `Rank1 answers all ${n} abstracts without reasoning`
    : `Rank1 is reasoning on abstract ${Math.min(done + 1, n)} of ${n}`;
  $("run-left").textContent = r.phase === "search" ? "" : `about ${left} s left`;
}

function runLive(claim) {
  if (!claim || S.state.live !== "ready") return;
  stopLive();
  document.querySelectorAll(".examples button").forEach((b) => b.classList.remove("on"));
  swap(() => {
    begin(claim, true);
    S.mode = "r1";
    modeButtons();
    S.q = { claim, known: false };
    S.running = true;
    S.docs = new Map(); S.order = [];
    S.run = { phase: "search", seconds: [], started: 0 };
    ["bm25", "ce", "r1"].forEach((c) => cols[c].querySelector("ol").replaceChildren());
    Object.values(nodes).forEach((m) => m.clear());
    $("links").innerHTML = "";
    $("segments").replaceChildren();
    runbar();
    S.timer = setInterval(runbar, 400);
    S.es = new EventSource(`/api/live?q=${encodeURIComponent(claim)}`);
    S.es.onmessage = (m) => onLive(JSON.parse(m.data));
    S.es.onerror = () => finish();
  });
}
function finish(message) {
  S.running = false;
  S.thinking = null;
  stopLive();
  clearInterval(S.timer);
  cols.r1.classList.remove("running");
  document.querySelector(".strip").classList.remove("running");
  $("follow").hidden = true;
  journey();
  if (message) $("journey").textContent = `Live model error: ${message}`;
}
function onLive(e) {
  if (e.type === "candidates") {
    S.q.known = e.known;
    S.docs = new Map(e.docs.map((d) => [d.id, d]));
    S.order = e.docs.map((d) => d.id);
    S.sel = S.order[0];
    S.run.phase = "quick";
    renderAll();
    runbar();
  } else if (e.type === "r2") {
    Object.entries(e.scores).forEach(([id, p]) => { S.docs.get(id).r2 = p; });
    S.run.phase = "reason";
    renderCol("r1");
    judge();
    runbar();
  } else if (e.type === "start") {
    S.thinking = e.id;
    S.run.started = performance.now();
    S.docs.get(e.id).reasoning = "";
    if (S.follow) select(e.id);
    renderCol("r1", false);
    drawLinks();
    judge();
  } else if (e.type === "text") {
    S.docs.get(e.id).reasoning += e.text;
    if (S.sel === e.id) judge();
  } else if (e.type === "verdict") {
    Object.assign(S.docs.get(e.id), { r1: e.r1 });
    S.thinking = null;
    S.judged += 1;
    S.run.seconds.push(e.seconds);
    runbar();
    renderCol("r1");
    journey();
    judge();
  } else if (e.type === "done" || e.type === "error") {
    finish(e.type === "error" ? e.message : "");
  }
}

// controls
function modeButtons() {
  document.querySelectorAll("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === S.mode));
}
$("mode").addEventListener("click", (ev) => {
  const m = ev.target.dataset.mode;
  if (!m || m === S.mode || !S.q) return;
  S.mode = m;
  modeButtons();
  cols.r1.classList.toggle("at-once", S.mode === "r2");
  renderCol("r1");
  journey();
  judge();
});
$("follow").addEventListener("click", () => { S.follow = true; $("follow").hidden = true; if (S.thinking) select(S.thinking); });
$("ask").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const claim = $("claim").value.trim();
  if (S.state.live === "loading") { S.pending = claim; liveStatus(); return; }
  runLive(claim);
});
window.addEventListener("resize", drawLinks);

function liveStatus() {
  const st = S.state.live, name = S.state.model.replace("rank1-", "Rank1-").replace(/b$/, "B");
  const el = $("status");
  el.classList.toggle("ready", st === "ready");
  el.classList.toggle("off", st.startsWith("unavailable"));
  el.querySelector("span").textContent = st === "ready" ? `${name} ready`
    : st === "loading" ? (S.pending ? `loading ${name}: your claim runs as soon as it is ready` : `loading ${name}`) : "live Rank1 unavailable";
  el.title = st;
  $("run").disabled = st.startsWith("unavailable");
  if (st === "ready" && S.pending) { const claim = S.pending; S.pending = null; runLive(claim); }
}

async function init() {
  S.state = await (await fetch("/api/state")).json();
  S.state.queries.forEach((q) => {
    const b = document.createElement("button");
    b.dataset.key = q.key;
    b.innerHTML = `<b>${q.label}</b><span>${escapeHtml(q.claim)}</span>`;
    b.addEventListener("click", () => loadQuery(q.key));
    $("examples").append(b);
  });
  liveStatus();
  const poll = setInterval(async () => {
    S.state.live = (await (await fetch("/api/state")).json()).live;
    liveStatus();
    if (S.state.live !== "loading") clearInterval(poll);
  }, 2000);
  $("claim").focus();
}
init();
