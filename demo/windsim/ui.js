// Wind & Steel: browser front end (2D canvas). All rules live in core.js (window.WindSteel).
(function () {
  "use strict";
  const W = window.WindSteel, KEY = "windsteel-save-v1";
  let S = load() || W.newGame((Date.now() % 100000) + 1);
  const UI = { tab: "farm", sel: null, auto: 0, timer: null, design: null, plotB: null, test: null, drag: null };
  const $ = (id) => document.getElementById(id), css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v, d = 0) => (v === undefined || v === null || !isFinite(v) ? "–" : Number(v).toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d }));
  function load() { try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); return s && s.v === 1 ? s : null; } catch (e) { return null; } }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) { /* storage unavailable */ } }
  function toast(msg, kind = "info") { const el = document.createElement("div"); el.className = kind; el.textContent = msg; $("toast").prepend(el); setTimeout(() => el.remove(), 5000); while ($("toast").children.length > 4) $("toast").lastChild.remove(); }
  function act(err) { if (err) toast(err, "warn"); save(); render(); }

  // ---------- HUD ----------
  function hud() {
    const w = S.weather, td = S.today || { energy: 0, revenue: 0 }, live = S.turbines.filter((t) => t.status !== "lost"), mw = live.filter((t) => t.status === "running").reduce((a, t) => a + W.TURBINES[t.design.turbine].P, 0);
    $("hud").innerHTML = `<h1>Wind &amp; Steel</h1>` +
      chip("Day", `${S.day + 1} · ${W.season(S.day)}`) + chip("Credits", fmt(S.cash)) + chip("Reputation", `${Math.round(100 * (S.rep || 1))} %`) + chip("Research", fmt(S.rp, 0) + " RP") +
      chip("Farm", `${live.length} turbines · ${mw} MW on`) + chip("Yesterday", `${fmt(td.energy)} MWh · +${fmt(td.revenue)}`) +
      chip("Today", `${w.storm ? '<span class="storm">STORM</span> ' : ""}wind ${fmt(w.U, 0)} m/s · waves ${fmt(w.Hs, 1)} m`) +
      `<div class="fc" title="Forecast for the next three days">${S.forecast.map((f, k) => `<div>${["Tmrw", "+2", "+3"][k]}<br>${f.storm ? '<span class="storm">storm</span>' : `${fmt(f.U * (1 - f.spread), 0)}–${fmt(f.U * (1 + f.spread), 0)} m/s`}<br>${fmt(f.Hs * (1 - f.spread), 1)}–${fmt(f.Hs * (1 + f.spread), 1)} m</div>`).join("")}</div>` +
      `<span class="spacer"></span><button class="primary" id="next">Next day ▸</button><button id="auto">${UI.auto ? "❚❚ Pause" : "▶ Auto"}</button><button id="newg" title="Start again">New game</button>`;
    $("next").onclick = () => step(); $("auto").onclick = () => { UI.auto = UI.auto ? 0 : 1; clearInterval(UI.timer); if (UI.auto) UI.timer = setInterval(step, 900); hud(); };
    $("newg").onclick = () => { if (confirm("Start a new game? Your farm will be lost.")) { S = W.newGame((Date.now() % 100000) + 1); UI.sel = null; UI.design = null; save(); render(); } };
  }
  const chip = (a, b) => `<div class="chip"><small>${a}</small><b>${b}</b></div>`;
  function step() {
    const ev = W.advanceDay(S); save();
    ev.forEach((e) => { const t = S.turbines.find((x) => x.id === e.tid); if (e.type === "lost") toast(`${t.name} lost: ${e.why}`, "bad"); else if (e.type === "failure") toast(`${t.name}: ${W.COMPS.find((c) => c.k === e.comp).name} failed`, "bad"); else if (e.type === "broke") toast("You are deep in debt. Sell power, cut staff, or start again.", "bad"); });
    if (S.weather.storm && UI.auto) { UI.auto = 0; clearInterval(UI.timer); toast("Storm today. Auto-play paused.", "warn"); }
    S.log.slice(0, 3).filter((l) => l.day === S.day - 1 && l.kind === "good").forEach((l) => toast(l.msg, "good"));
    render();
  }

  // ---------- tabs ----------
  const TABS = [["farm", "Farm"], ["build", "Builder"], ["research", "Research"], ["team", "Team"], ["log", "Log"]];
  function render() {
    hud();
    $("tabs").innerHTML = TABS.map(([k, l]) => `<button class="${UI.tab === k ? "on" : ""}" data-t="${k}">${l}</button>`).join("");
    $("tabs").querySelectorAll("[data-t]").forEach((b) => (b.onclick = () => { UI.tab = b.dataset.t; UI.test = null; render(); }));
    ({ farm: farmScreen, build: buildScreen, research: researchScreen, team: teamScreen, log: logScreen })[UI.tab]();
  }

  // ---------- farm ----------
  let farmTF = null;
  function farmScreen() {
    $("screen").innerHTML = `<div class="grid2"><div><canvas id="fc"></canvas><p class="help">Click a plot. Deeper plots further out have stronger wind (and bigger storms). Wear is time-compressed: a day of wear stands for about a month.</p></div><div id="side"></div></div>`;
    const cv = $("fc"); cv.onclick = (e) => { const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; const p = S.plots.find((p) => { const b = farmTF && farmTF.box(p); return b && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h; }); if (p) { UI.sel = p.id; side(); } };
    side();
  }
  function farmDraw(now) {
    const cv = $("fc"); if (!cv) return; const W2 = cv.clientWidth, H = Math.round(Math.min(560, W2 * 0.66)), dpr = Math.min(2, devicePixelRatio || 1);
    if (cv.width !== Math.round(W2 * dpr)) { cv.width = Math.round(W2 * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + "px"; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const gr = g.createLinearGradient(0, 0, W2, 0); gr.addColorStop(0, "#5cc0d6"); gr.addColorStop(1, "#1d6f8c"); g.fillStyle = gr; g.fillRect(0, 0, W2, H);
    // coast and moving wave marks
    g.fillStyle = "#e8d29c"; g.beginPath(); g.moveTo(0, 0); g.lineTo(46, 0); for (let y = 0; y <= H; y += 20) g.lineTo(40 + 8 * Math.sin(y / 37), y); g.lineTo(0, H); g.fill();
    g.strokeStyle = "rgba(255,255,255,0.25)"; g.lineWidth = 1.5; const tt = now / 1000, Us = S.weather.U;
    for (let k = 0; k < 40; k++) { const x = (k * 97 + tt * (8 + Us)) % (W2 + 60) - 30, y = (k * 53) % H; g.beginPath(); g.moveTo(x, y); g.quadraticCurveTo(x + 8, y - 3, x + 16, y); g.stroke(); }
    const pad = 60, gw = (W2 - pad - 20) / 4, gh = (H - 30) / 3;
    farmTF = { box: (p) => ({ x: pad + p.i * gw + 6, y: 15 + p.j * gh + 6, w: gw - 12, h: gh - 12 }) };
    S.plots.forEach((p) => {
      const b = farmTF.box(p), t = S.turbines.find((x) => x.plot === p.id && x.status !== "lost"), lost = S.turbines.find((x) => x.plot === p.id && x.status === "lost");
      g.fillStyle = p.owned ? "rgba(255,255,255,0.18)" : "rgba(10,30,45,0.25)"; g.strokeStyle = UI.sel === p.id ? "#ffd166" : p.owned ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.25)"; g.lineWidth = UI.sel === p.id ? 3 : 1.5;
      g.beginPath(); g.roundRect(b.x, b.y, b.w, b.h, 12); g.fill(); g.stroke();
      g.fillStyle = "rgba(255,255,255,0.92)"; g.font = "600 11px Inter"; g.fillText(`Plot ${p.id + 1} · ${p.depth} m · ${p.soil}`, b.x + 8, b.y + 16);
      if (!p.owned) { g.font = "600 12px Inter"; g.fillText(`Lease ${fmt(p.lease)}`, b.x + 8, b.y + b.h - 10); return; }
      if (p.surveyed) { g.font = "10px Inter"; g.fillText("surveyed", b.x + b.w - 56, b.y + 16); }
      const cx = b.x + b.w / 2, base = b.y + b.h - 12, top = b.y + 34, rr = Math.min(b.h - 50, b.w * 0.42) * 0.45;
      if (t) {
        const tb = W.TURBINES[t.design.turbine], run = t.status === "running";
        g.strokeStyle = "#f3f6f8"; g.lineWidth = 3 + tb.P / 6; g.beginPath(); g.moveTo(cx, base); g.lineTo(cx, top + rr * 0.1); g.stroke();
        g.fillStyle = "#f3f6f8"; g.fillRect(cx - 6, top + rr * 0.1 - 4, 12, 7);
        const ang = run ? tt * (0.6 + Math.min(1.5, S.weather.U / 8)) : t.status === "building" ? 0 : 0.3;
        if (t.status !== "building" || t.buildLeft < 2) { g.lineWidth = 2.5; for (let k = 0; k < 3; k++) { const a = ang + k * 2.094; g.beginPath(); g.moveTo(cx, top + rr * 0.1); g.lineTo(cx + Math.cos(a) * rr, top + rr * 0.1 + Math.sin(a) * rr); g.stroke(); } }
        const col = t.status === "running" ? css("--good") : t.status === "building" ? css("--warn") : css("--bad");
        g.fillStyle = col; g.beginPath(); g.arc(b.x + b.w - 14, b.y + b.h - 14, 7, 0, 7); g.fill();
        g.fillStyle = "rgba(255,255,255,0.92)"; g.font = "600 11px Inter"; g.fillText(`${t.name} · ${tb.name}`, b.x + 8, b.y + b.h - 10);
        if (t.status === "building") { g.font = "10px Inter"; g.fillText(`building · ${t.buildLeft} days`, b.x + 8, b.y + 30); }
        const warn = Object.values(t.known).some((k) => k.v !== undefined && k.v < 0.35) || t.known.crack; if (warn) { g.fillStyle = css("--warn"); g.font = "700 16px Inter"; g.fillText("!", b.x + b.w - 30, b.y + b.h - 8); }
      } else if (lost) { g.strokeStyle = "rgba(60,60,60,0.8)"; g.lineWidth = 4; g.beginPath(); g.moveTo(cx - 20, base); g.lineTo(cx + 26, base - 18); g.stroke(); g.fillStyle = "rgba(255,255,255,0.9)"; g.font = "11px Inter"; g.fillText("wreck cleared · free to rebuild", b.x + 8, b.y + b.h - 10); }
      else { g.fillStyle = "rgba(255,255,255,0.75)"; g.font = "12px Inter"; g.fillText("Empty: design a turbine", b.x + 8, b.y + b.h - 10); }
    });
    if (S.weather.storm) { g.fillStyle = "rgba(20,30,45,0.25)"; g.fillRect(0, 0, W2, H); g.strokeStyle = "rgba(255,255,255,0.35)"; for (let k = 0; k < 60; k++) { const x = (k * 37 + tt * 300) % W2, y = (k * 71 + tt * 500) % H; g.beginPath(); g.moveTo(x, y); g.lineTo(x - 6, y + 14); g.stroke(); } }
  }
  function side() {
    const el = $("side"); if (!el) return; const p = UI.sel !== null ? S.plots[UI.sel] : null;
    if (!p) { el.innerHTML = `<div class="card"><h3>Your farm</h3><p class="small">Every day the wind turns your turbines and you are paid ${S.price} credit per MWh. Build turbines in the Builder, research better technology, and keep them running: inspect, repair, and watch the weather.</p></div>` + summaryCard(); return; }
    const t = S.turbines.find((x) => x.plot === p.id && x.status !== "lost"), sg = W.sigmas(S, p);
    let h = `<div class="card"><h3>Plot ${p.id + 1}</h3><div class="small">${p.depth} m deep · ${p.soil} · wind ×${p.wind.toFixed(2)}<br>Soil strength known to ±${Math.round(164 * sg.soil)} % ${p.surveyed ? "(surveyed)" : p.surveying ? "(survey under way)" : ""}</div><div class="row" style="margin-top:8px">`;
    if (!p.owned) h += `<button class="primary" data-a="lease">Lease for ${fmt(p.lease)}</button>`;
    else { if (!t) h += `<button class="primary" data-a="design">Design a turbine here</button>`; if (!p.surveyed && !p.surveying) h += `<button data-a="survey">Geotech survey (${150 + 3 * p.depth})</button>`; }
    h += `</div></div>`;
    if (t) h += turbineCard(t);
    el.innerHTML = h;
    const on = (a, fn) => el.querySelectorAll(`[data-a="${a}"]`).forEach((b) => (b.onclick = () => fn(b)));
    on("lease", () => act(W.buyPlot(S, p.id))); on("survey", () => act(W.survey(S, p.id)));
    on("design", () => { UI.plotB = p.id; UI.design = null; UI.tab = "build"; render(); });
    on("insp", (b) => act(W.inspect(S, t.id, b.dataset.m))); on("rep", (b) => act(W.repair(S, t.id, b.dataset.c))); on("sens", () => act(W.fitSensors(S, t.id))); on("decom", () => { if (confirm(`Take ${t.name} down for scrap?`)) act(W.decommission(S, t.id)); });
  }
  function summaryCard() { const st = S.stats; return `<div class="card"><h3>So far</h3><div class="small">Energy ${fmt(st.energy / 1000, 1)} GWh · earned ${fmt(st.revenue)} · spent ${fmt(st.spent)}<br>Breakdowns ${st.failures} · turbines lost ${st.lost}</div></div>`; }
  function turbineCard(t) {
    const tb = W.TURBINES[t.design.turbine], jobs = S.jobs.filter((j) => j.tid === t.id), stat = { running: ["good", "Running"], building: ["warn", "Building"], down: ["bad", "Stopped"] }[t.status];
    let h = `<div class="card"><h3>${t.name} · ${tb.name} <span class="pill ${stat[0]}">${stat[1]}</span></h3><div class="small">${t.down ? `${W.COMPS.find((c) => c.k === t.down.comp).name} failed on day ${t.down.since + 1}. ` : ""}Energy ${fmt(t.energy)} MWh · today ${fmt(t.today || 0)} MWh</div>`;
    h += `<div style="margin-top:6px">` + W.COMPS.map((c) => healthRow(c.name, t.known[c.k], t.down && t.down.comp === c.k)).join("") + `</div>`;
    h += `<div class="small" style="margin-top:4px">Structure: design fatigue life ${fmt(t.predicted.life)} years (estimate)${t.known.fatigue ? ` · twin says ${Math.round(100 * Math.min(1, t.known.fatigue.v))} % used` : ""}${t.known.crack ? ' · <b style="color:var(--bad)">weld crack found</b>' : ""}${t.sensors ? " · sensors fitted" : ""}</div>`;
    if (t.lastInspect) h += `<div class="small muted">Last inspection: day ${t.lastInspect.day + 1} (${W.INSPECT[t.lastInspect.method].name.toLowerCase()})</div>`;
    if (jobs.length) h += `<div class="small" style="margin-top:4px">Booked: ${jobs.map((j) => j.type === "repair" ? `${j.comp === "crack" ? "weld repair" : W.COMPS.find((c) => c.k === j.comp).name.toLowerCase()}${j.wait ? ` (parts in ${j.wait} d)` : ""}` : j.type === "inspect" ? W.INSPECT[j.method].name.toLowerCase() : "sensor fit").join(", ")}</div>`;
    h += `</div><div class="card"><h3>Send a crew</h3><div class="small muted">Vessels: ${W.vessels(S).n}, working in waves up to ${W.vessels(S).limit} m.</div><div class="row" style="margin-top:6px">`;
    Object.entries(W.INSPECT).forEach(([k, m]) => { const ok = !m.need || S.done.includes(m.need); h += `<button data-a="insp" data-m="${k}" ${ok ? "" : "disabled"} title="Finds ${Math.round(100 * m.detect)} % of worn parts, ±${Math.round(100 * m.sd)} %; structure ${Math.round(100 * m.structure)} %">${m.name} · ${m.cost}</button>`; });
    if (S.done.includes("sensors") && !t.sensors) h += `<button data-a="sens">Fit sensors · 200</button>`;
    h += `</div><div class="row" style="margin-top:8px">`;
    W.COMPS.forEach((c) => { const failed = t.down && t.down.comp === c.k; h += `<button data-a="rep" data-c="${c.k}" class="${failed ? "primary" : ""}">${failed ? "Replace" : "Overhaul"} ${c.name.toLowerCase()} · ${W.repairCost(t, c.k, failed)}</button>`; });
    if (t.known.crack) h += `<button class="primary" data-a="rep" data-c="crack">Repair weld crack · ${W.repairCost(t, "crack", false)}</button>`;
    if (t.status !== "building") h += `<button data-a="decom" title="Scrap it (15 % back) to build a bigger turbine here">Decommission for repowering · +${Math.round(0.15 * t.cost)}</button>`;
    return h + `</div><p class="help">Planned work costs a third of a repair after failure and needs no spare-part wait.</p></div>`;
  }
  function healthRow(name, k, failed) {
    if (failed) return `<div class="hb"><span>${name}</span><div class="bar"><div class="band" style="left:0;width:2%;background:var(--bad);opacity:1"></div></div><span class="pill bad">failed</span></div>`;
    if (!k) return `<div class="hb"><span>${name}</span><div class="bar"></div><span class="muted small">unknown</span></div>`;
    const age = S.day - k.day, sd = Math.min(0.5, k.sd + 0.004 * age), v = Math.max(0, k.v - 0.003 * age), lo = Math.max(0, v - 1.64 * sd), hi = Math.min(1, v + 1.64 * sd), col = v < 0.35 ? "var(--bad)" : v < 0.6 ? "var(--warn)" : "var(--good)";
    return `<div class="hb"><span>${name}</span><div class="bar"><div class="band" style="left:${100 * lo}%;width:${100 * (hi - lo)}%;background:${col}"></div><div class="est" style="left:${100 * v}%"></div></div><span class="small">${Math.round(100 * v)} % ±${Math.round(164 * sd)}</span></div>`;
  }

  // ---------- builder ----------
  function freePlots() { return S.plots.filter((p) => p.owned && !S.turbines.some((t) => t.plot === p.id && t.status !== "lost")); }
  function buildScreen() {
    const fp = freePlots();
    if (UI.plotB === null || !fp.some((p) => p.id === UI.plotB)) UI.plotB = fp.length ? fp[0].id : null;
    if (UI.plotB === null) { $("screen").innerHTML = `<div class="card"><h3>No free plot</h3><p>Lease a plot on the Farm screen to build another turbine.</p></div>`; return; }
    const p = S.plots[UI.plotB];
    if (!UI.design || UI.design.plot !== p.id) { const tk = S.unlockedTurbines[S.unlockedTurbines.length - 1]; UI.design = { plot: p.id, d: W.starterDesign(S, p.id, tk) }; }
    const d = UI.design.d, sg = W.sigmas(S, p), L = W.LIMITS;
    const sl = (k, lab, unit, st) => `<label class="f"><span>${lab}</span><b class="num" id="v-${k}">${d[k]} ${unit}</b><input type="range" data-k="${k}" min="${L[k][0]}" max="${L[k][1]}" step="${st}" value="${d[k]}"></label>`;
    $("screen").innerHTML = `<div class="grid3"><div>
      <div class="card"><h3>Plot</h3><select id="bplot">${fp.map((q) => `<option value="${q.id}"${q.id === p.id ? " selected" : ""}>Plot ${q.id + 1} · ${q.depth} m · ${q.soil}</option>`).join("")}</select>
        <div class="small" style="margin-top:6px">Design storm waves ≈ ${fmt(p.H50 * Math.exp(p.eps.w * sg.wave), 1)} m (±${Math.round(164 * sg.wave)} %)<br>Soil strength ±${Math.round(164 * sg.soil)} % ${p.surveyed ? "· surveyed" : `<button id="bsurvey" style="padding:2px 8px">Survey</button>`}<br>Analysis model error ±${Math.round(164 * sg.model)} %</div></div>
      <div class="card"><h3>Turbine and steel</h3><select id="bturb">${S.unlockedTurbines.map((k) => `<option value="${k}"${k === d.turbine ? " selected" : ""}>${W.TURBINES[k].name}</option>`).join("")}</select>
        <select id="bsteel">${Object.keys(W.STEELS).filter((k) => k === "S355" || S.done.includes(k)).map((k) => `<option${k === d.steel ? " selected" : ""}>${k}</option>`).join("")}</select>
        <button id="bstart" style="margin-top:6px">Start from scratch</button></div>
      <div class="card"><h3>Walls</h3>${sl("tt", "Tower wall", "mm", 1)}${sl("tp", "Monopile wall", "mm", 1)}<p class="help">Drag the orange handles on the drawing for diameters and how deep the pile goes.</p></div>
    </div><div><canvas id="bc"></canvas><div id="bres"></div></div><div id="bchecks"></div></div>`;
    $("bplot").onchange = (e) => { UI.plotB = +e.target.value; UI.design = null; UI.test = null; render(); };
    $("bturb").onchange = (e) => { UI.design = { plot: p.id, d: Object.assign(W.starterDesign(S, p.id, e.target.value), { steel: d.steel }) }; UI.test = null; render(); };
    $("bsteel").onchange = (e) => { d.steel = e.target.value; UI.test = null; checksPanel(); };
    $("bstart").onclick = () => { UI.design = null; UI.test = null; render(); };
    if ($("bsurvey")) $("bsurvey").onclick = () => act(W.survey(S, p.id));
    $("screen").querySelectorAll("[data-k]").forEach((x) => (x.oninput = () => { d[x.dataset.k] = +x.value; $("v-" + x.dataset.k).textContent = `${x.value} mm`; UI.test = null; checksPanel(); }));
    const cv = $("bc");
    cv.onpointerdown = (e) => { const hdl = bHandles(), r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; const h = hdl.find((q) => Math.hypot(q.x - x, q.y - y) < 16); if (h) { UI.drag = h.k; cv.setPointerCapture(e.pointerId); UI.test = null; } };
    cv.onpointermove = (e) => { const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; if (!UI.drag) { cv.style.cursor = bHandles().some((q) => Math.hypot(q.x - x, q.y - y) < 16) ? "grab" : "default"; return; }
      const tf = UI.btf, k = UI.drag;
      if (k === "Le") d.Le = clamp(Math.round(-(tf.z(y)) - p.depth), L.Le);
      else d[k] = clamp(+(2 * Math.abs(x - tf.cx) / tf.xs).toFixed(2), L[k]);
      if (k === "Dt") d.Dt = Math.min(d.Dt, d.Db); if (k === "Db") d.Dt = Math.min(d.Dt, d.Db);
      checksPanel(); };
    cv.onpointerup = () => { UI.drag = null; };
    checksPanel();
  }
  const clamp = (v, r) => Math.max(r[0], Math.min(r[1], v));
  function bHandles() {
    const tf = UI.btf; if (!tf) return []; const d = UI.design.d, p = S.plots[UI.plotB], g = tf.g;
    return [{ k: "Dt", x: tf.cx + d.Dt / 2 * tf.xs, y: tf.y(g.hub - 3) }, { k: "Db", x: tf.cx + d.Db / 2 * tf.xs, y: tf.y(g.tp) }, { k: "Dp", x: tf.cx + d.Dp / 2 * tf.xs, y: tf.y(-p.depth) }, { k: "Le", x: tf.cx, y: tf.y(-p.depth - d.Le) }];
  }
  function checksPanel() {
    const d = UI.design.d, p = S.plots[UI.plotB], ev = W.evaluate(S, p.id, d), C = ev.checks; UI.ev = ev;
    const bar = (b, lo, hi, kind) => {   // drawn on a log-ish scale between lo and hi
      const X = (v) => Math.max(0, Math.min(100, 100 * (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))));
      const col = kind === "util" ? (b.hi <= 1 ? "var(--good)" : b.est <= 1 ? "var(--warn)" : "var(--bad)") : kind === "life" ? (b.lo >= 25 ? "var(--good)" : b.est >= 25 ? "var(--warn)" : "var(--bad)") : (b.lo >= b.w.lo && b.hi <= b.w.hi ? "var(--good)" : b.est >= b.w.lo && b.est <= b.w.hi ? "var(--warn)" : "var(--bad)");
      let h = `<div class="bar">`;
      if (kind === "freq") h += `<div class="win" style="left:${X(b.w.lo)}%;width:${X(b.w.hi) - X(b.w.lo)}%"></div>`;
      h += `<div class="band" style="left:${X(b.lo)}%;width:${Math.max(1, X(b.hi) - X(b.lo))}%;background:${col}"></div><div class="est" style="left:${X(b.est)}%"></div>`;
      if (kind === "util") h += `<div class="lim" style="left:${X(1)}%"></div>`; if (kind === "life") h += `<div class="lim" style="left:${X(25)}%"></div>`;
      return h + `</div>`;
    };
    const ok = (b, kind) => kind === "util" ? (b.hi <= 1 ? "good" : b.est <= 1 ? "warn" : "bad") : kind === "life" ? (b.lo >= 25 ? "good" : b.est >= 25 ? "warn" : "bad") : (b.lo >= b.w.lo && b.hi <= b.w.hi ? "good" : b.est >= b.w.lo && b.est <= b.w.hi ? "warn" : "bad");
    const word = { good: "safe at P95", warn: "ok on average", bad: "fails" };
    $("bchecks").innerHTML = `<div class="card"><h3>Checks</h3>` +
      `<div class="chk"><div class="top"><span>Strength (${C.strength.at})</span><span class="pill ${ok(C.strength, "util")}">${word[ok(C.strength, "util")]}</span></div>${bar(C.strength, 0.3, 2.5, "util")}<div class="help">Storm stress ÷ steel strength: ${fmt(C.strength.est, 2)} (${fmt(C.strength.lo, 2)}–${fmt(C.strength.hi, 2)}); must stay under 1</div></div>` +
      `<div class="chk"><div class="top"><span>Soil holding</span><span class="pill ${ok(C.holding, "util")}">${word[ok(C.holding, "util")]}</span></div>${bar(C.holding, 0.3, 2.5, "util")}<div class="help">Overturning ÷ soil capacity: ${fmt(C.holding.est, 2)} (${fmt(C.holding.lo, 2)}–${fmt(C.holding.hi, 2)})</div></div>` +
      `<div class="chk"><div class="top"><span>Natural frequency</span><span class="pill ${ok(C.freq, "freq")}">${{ good: "in the window", warn: "near a band", bad: "resonance" }[ok(C.freq, "freq")]}</span></div>${bar(C.freq, 0.08, 0.35, "freq")}<div class="help">${fmt(C.freq.est, 3)} Hz (${fmt(C.freq.lo, 3)}–${fmt(C.freq.hi, 3)}); keep it between the rotor's 1P and 3P: ${fmt(C.freq.w.lo, 3)}–${fmt(C.freq.w.hi, 3)} Hz. Stiffer (fatter, deeper) raises it.</div></div>` +
      `<div class="chk"><div class="top"><span>Fatigue life (${C.life.at})</span><span class="pill ${ok(C.life, "life")}">${{ good: "25 y at P95", warn: "25 y on average", bad: "too short" }[ok(C.life, "life")]}</span></div>${bar(C.life, 4, 200, "life")}<div class="help">${fmt(C.life.est)} years (${fmt(C.life.lo)}–${fmt(C.life.hi)}); needs 25</div></div>` +
      `</div><div class="card"><h3>${fmt(ev.cost)} credits</h3><div class="small">${fmt(ev.mass)} t of steel · turbine ${fmt(W.TURBINES[d.turbine].price)} · energy about ${fmt(W.dayEnergy(W.TURBINES[d.turbine], 9.5 * S.plots[UI.plotB].wind))} MWh a day</div>` +
      `<div class="row" style="margin-top:8px"><button data-test="p50">Storm test (P50)</button><button data-test="p90">Storm test (P90)</button></div>` +
      `<div class="row" style="margin-top:8px"><button class="go" id="bbuild" ${S.cash < ev.cost ? "disabled" : ""}>Build for ${fmt(ev.cost)}</button></div>` +
      `<p class="help">P50 tests the middle of each band; P90 the unlucky end. The real soil, storm and model error stay hidden until the sea tests your turbine.</p></div>`;
    $("bchecks").querySelectorAll("[data-test]").forEach((b) => (b.onclick = () => { UI.test = { level: b.dataset.test, t0: performance.now(), res: W.stormTest(S, UI.plotB, d, b.dataset.test) }; $("bres").innerHTML = ""; }));
    $("bbuild").onclick = () => { const err = W.build(S, UI.plotB, d); if (!err) { toast("Construction started.", "good"); UI.design = null; UI.test = null; UI.tab = "farm"; } act(err); };
  }
  function buildDraw(now) {
    const cv = $("bc"); if (!cv || !UI.design) return; const p = S.plots[UI.plotB], d = UI.design.d, ev = UI.ev || W.evaluate(S, p.id, d), g0 = ev.raw.g, tb = W.TURBINES[d.turbine];
    const W2 = cv.clientWidth, H = Math.round(Math.min(640, Math.max(440, W2 * 1.0))), dpr = Math.min(2, devicePixelRatio || 1);
    if (cv.width !== Math.round(W2 * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W2 * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + "px"; }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ztop = g0.hub + tb.D / 2 + 6, zbot = -p.depth - W.LIMITS.Le[1] - 4, sy = (H - 20) / (ztop - zbot), xs = sy * 5, cx = W2 * 0.42;
    const y = (z) => 10 + (ztop - z) * sy, z = (py) => ztop - (py - 10) / sy; UI.btf = { cx, xs, y, z, g: g0 };
    // test animation state
    const T = UI.test, tt = T ? (now - T.t0) / 1000 : 0, fail = T && !T.res.ok ? T.res.fails[0] : null, storm = T ? Math.min(1, tt / 1.2) : 0;
    // sky, sea, seabed
    const sk = g.createLinearGradient(0, 0, 0, y(0)); sk.addColorStop(0, T ? "#8aa3b5" : css("--sky1")); sk.addColorStop(1, T ? "#c6d3dc" : css("--sky2")); g.fillStyle = sk; g.fillRect(0, 0, W2, y(0));
    g.fillStyle = "#3aa7c1"; g.fillRect(0, y(0), W2, y(-p.depth) - y(0));
    const amp = (T ? 4 + 6 * storm : 2) * sy * 0.6; g.fillStyle = "#3aa7c1"; g.beginPath(); g.moveTo(0, y(0)); for (let x = 0; x <= W2; x += 8) g.lineTo(x, y(0) - amp * Math.sin(x / 40 - now / (T ? 260 : 700))); g.lineTo(W2, y(-p.depth)); g.lineTo(0, y(-p.depth)); g.fill();
    g.fillStyle = "#e8c98f"; g.fillRect(0, y(-p.depth), W2, Math.max(4, 10 * sy)); g.fillStyle = p.soil === "clay" ? "#8e7f73" : "#c9a76c"; g.fillRect(0, y(-p.depth) + 10 * sy, W2, H); g.fillStyle = p.soil === "clay" ? "#6f625a" : "#a8875a"; g.fillRect(0, y(-p.depth - 25), W2, H);
    g.fillStyle = "rgba(0,0,0,0.45)"; g.font = "11px Inter"; g.fillText(`${p.soil}, soil strength ±${Math.round(164 * ev.sg.soil)} %`, 8, y(-p.depth) + 24); g.fillText(`${p.depth} m water`, 8, y(-p.depth / 2));
    // structure geometry (z from pile toe to tower top), with the storm sway or failure motion
    const zToe = -p.depth - d.Le, zf = -p.depth - 0.4 * d.Le, zTop = g0.hub - 3;
    let sway = 0, rot = null;
    if (T) { const util = Math.max(0.2, Math.min(1.5, T.res.strength)); sway = (fail && fail.mode === "resonance" ? (0.5 + Math.min(4, tt) * 1.5) * Math.sin(tt * 18) : Math.sin(tt * 3.1) * storm) * util * 3 * xs / 5;
      if (fail && tt > 2.2) { const k = Math.min(1, (tt - 2.2) / 1.6), e = k * k; if (fail.mode === "buckle-tower") rot = { z: g0.tp, a: e * 1.25 }; else if (fail.mode === "buckle-pile") rot = { z: -p.depth, a: e * 1.1 }; else if (fail.mode === "overturn") rot = { z: zf, a: e * 0.35 }; } }
    const pos = (zz) => { let x = cx + sway * Math.max(0, (zz - zf) / (zTop - zf)) ** 2, yy = y(zz); if (rot && zz > rot.z) { const px = cx + sway * Math.max(0, (rot.z - zf) / (zTop - zf)) ** 2, py = y(rot.z), dx = x - px, dy = yy - py, c = Math.cos(rot.a), s = Math.sin(rot.a); x = px + dx * c - dy * s; yy = py + dx * s + dy * c; } return [x, yy]; };
    const Dat = (zz) => zz < g0.tp ? d.Dp : d.Db + (d.Dt - d.Db) * (zz - g0.tp) / (zTop - g0.tp);
    const util = (zz) => { const r = ev.raw; if (zz >= g0.tp) return r.uTb * (zTop - zz) / (zTop - g0.tp); if (zz >= -p.depth) return r.uTb + (r.uMud - r.uTb) * (g0.tp - zz) / (g0.tp + p.depth); return r.uMud * Math.max(0, (zz - zToe) / (d.Le)); };
    const colU = (u) => { const k = Math.max(0, Math.min(1, u)); return `hsl(${130 - 130 * k}, 70%, ${55 - 10 * k}%)`; };
    const N = 70;
    for (let i = 0; i < N; i++) {
      const za = zToe + (zTop - zToe) * i / N, zb = zToe + (zTop - zToe) * (i + 1) / N, [xa, ya] = pos(za), [xb, yb] = pos(zb), ra = Dat(za) / 2 * xs, rb = Dat(zb) / 2 * xs;
      g.fillStyle = T ? colU(util((za + zb) / 2) * (T.level === "p90" ? 1.15 : 1)) : za < -p.depth ? "#8b98a5" : za < g0.tp ? "#f2c14e" : "#eef1f4";
      g.beginPath(); g.moveTo(xa - ra, ya); g.lineTo(xb - rb, yb); g.lineTo(xb + rb, yb); g.lineTo(xa + ra, ya); g.closePath(); g.fill();
    }
    // walls (inner line), weld and crack marks
    g.strokeStyle = "rgba(0,0,0,0.35)"; g.lineWidth = 1;
    [[zToe, g0.tp, d.tp], [g0.tp, zTop, d.tt]].forEach(([a, b, tw]) => { g.beginPath(); for (let i = 0; i <= 20; i++) { const zz = a + (b - a) * i / 20, [xx, yy] = pos(zz), r = Math.max(1, Dat(zz) / 2 * xs - tw / 1000 * xs * 12); i ? g.lineTo(xx + r, yy) : g.moveTo(xx + r, yy); } g.stroke(); });
    // nacelle and rotor (blades drawn edge-on at true length)
    const [nx, ny] = pos(zTop + 3);
    g.save(); g.translate(nx, ny); if (rot) g.rotate(rot.a); g.fillStyle = "#eef1f4"; g.strokeStyle = "rgba(0,0,0,0.3)"; g.fillRect(-8, -6, 30, 12); g.strokeRect(-8, -6, 30, 12);
    const spin = now / 1000 * (T && !fail ? 0.5 : 0.25); g.strokeStyle = "#f6f8fa"; g.lineWidth = 3; for (let k = 0; k < 3; k++) { const a = spin + k * 2.094; g.beginPath(); g.moveTo(-10, 0); g.lineTo(-10 + Math.cos(a) * 3, Math.sin(a) * tb.D / 2 * sy); g.stroke(); }
    g.restore();
    if (T && fail && fail.mode === "fatigue" && tt > 2) { const zz = ev.checks.life.at === "mudline weld" ? -p.depth : g0.tp, [xx, yy] = pos(zz); g.strokeStyle = css("--bad"); g.lineWidth = 2; for (let k = 0; k < 4; k++) { g.beginPath(); g.moveTo(xx - Dat(zz) / 2 * xs + k * 6, yy - 4); g.lineTo(xx - Dat(zz) / 2 * xs + k * 6 + 4, yy + 4); g.stroke(); } }
    if (T && fail && fail.mode === "overturn" && tt > 2.2) { const [xx, yy] = pos(-p.depth); g.fillStyle = "#a8875a"; g.beginPath(); g.ellipse(xx + 20, yy, 26 * Math.min(1, tt - 2.2), 8 * Math.min(1, tt - 2.2), 0, Math.PI, 0); g.fill(); }
    // handles
    if (!T) bHandles().forEach((h) => { g.fillStyle = css("--accent"); g.strokeStyle = "#fff"; g.lineWidth = 2; g.beginPath(); g.arc(h.x, h.y, 8, 0, 7); g.fill(); g.stroke();
      g.fillStyle = css("--ink"); g.font = "600 11px Inter"; const lab = { Dt: `top Ø ${d.Dt} m`, Db: `base Ø ${d.Db} m`, Dp: `pile Ø ${d.Dp} m`, Le: `${d.Le} m in the soil` }[h.k]; g.fillText(lab, h.x + 12, h.y + 4); });
    // result banner when the test ends
    if (T && tt > (fail ? 4.2 : 3.2) && !T.shown) { T.shown = true; const r = T.res;
      $("bres").innerHTML = r.ok ? `<div class="banner ok">Survived the design storm at ${T.level.toUpperCase()}</div>` :
        `<div class="banner fail">${{ "buckle-tower": "The tower buckled at its base", "buckle-pile": "The pile buckled at the seabed", overturn: "The pile pulled over in the soil", resonance: "Resonance: the turbine shook itself apart", fatigue: `Fatigue: the ${ev.checks.life.at} cracks after about ${fmt(r.life)} years` }[r.fails[0].mode]} (at ${T.level.toUpperCase()})</div>` + (r.fails.length > 1 ? `<p class="help" style="text-align:center">Also failing: ${r.fails.slice(1).map((f) => ({ strength: "strength", holding: "soil holding", freq: "natural frequency (resonance)", life: `fatigue (${fmt(r.life)} years)` })[f.check]).join(", ")}</p>` : ""); }
    if (T && tt > 7) { UI.test = null; }
  }

  // ---------- research and team ----------
  function researchScreen() {
    const br = [...new Set(W.RESEARCH.map((r) => r.branch))];
    $("screen").innerHTML = `<div class="row" style="justify-content:space-between"><h2 style="margin:0">Research</h2><span class="muted">${fmt(S.rp, 0)} research points · engineers add more every day</span></div><div class="tree" style="margin-top:10px">` +
      br.map((b) => `<div><h3>${b}</h3>${W.RESEARCH.filter((r) => r.branch === b).map((r) => { const done = S.done.includes(r.id), av = !done && r.req.every((q) => S.done.includes(q));
        return `<div class="node ${done ? "done" : av ? "avail" : "locked"}"><b>${r.name}</b><div class="small muted">${r.fx || ""}${r.req.length ? `<br>Needs: ${r.req.map((q) => W.RESEARCH.find((x) => x.id === q).name).join(", ")}` : ""}</div>${done ? '<span class="pill good">done</span>' : `<button data-r="${r.id}" ${av && S.rp >= r.rp && S.cash >= r.cost ? "" : "disabled"}>${r.rp} RP · ${fmt(r.cost)}</button>`}</div>`; }).join("")}</div>`).join("") + `</div>`;
    $("screen").querySelectorAll("[data-r]").forEach((b) => (b.onclick = () => act(W.research(S, b.dataset.r))));
  }
  function teamScreen() {
    const pay = S.staff.reduce((a, x) => a + W.ROLES[x.role].pay, 0), rp = S.staff.reduce((a, x) => a + W.ROLES[x.role].rp, 0);
    $("screen").innerHTML = `<div class="grid2"><div><h2 style="margin-top:0">Your team</h2><p class="muted">${S.staff.length} of 8 desks · pay ${pay} credits a day · ${rp.toFixed(1)} research points a day</p>` +
      S.staff.map((x) => `<div class="card row" style="justify-content:space-between"><span><b>${W.ROLES[x.role].name}</b><br><span class="small muted">${W.ROLES[x.role].fx} · ${W.ROLES[x.role].rp} RP/day · pay ${W.ROLES[x.role].pay}/day</span></span><button data-f="${x.id}">Let go</button></div>`).join("") +
      `</div><div><h2 style="margin-top:0">Hire (150 credits)</h2>` + Object.entries(W.ROLES).map(([k, r]) => `<div class="card"><b>${r.name}</b><div class="small muted">${r.fx} · ${r.rp} RP/day · pay ${r.pay}/day</div><button data-h="${k}" style="margin-top:6px">Hire</button></div>`).join("") + `</div></div>`;
    $("screen").querySelectorAll("[data-h]").forEach((b) => (b.onclick = () => act(W.hire(S, b.dataset.h))));
    $("screen").querySelectorAll("[data-f]").forEach((b) => (b.onclick = () => act(W.fire(S, +b.dataset.f))));
  }
  function logScreen() { $("screen").innerHTML = `<h2 style="margin-top:0">Log</h2><div class="log">${S.log.map((l) => `<div class="${l.kind}"><span class="muted">Day ${l.day + 1}</span> · ${l.msg}</div>`).join("")}</div>`; }

  // ---------- loop ----------
  function frame(now) { if (UI.tab === "farm") farmDraw(now); else if (UI.tab === "build") buildDraw(now); requestAnimationFrame(frame); }
  render(); requestAnimationFrame(frame);
  window.__ws = { get state() { return S; }, step, UI };
})();
