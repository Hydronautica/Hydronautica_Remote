// Offshore Wind Developer: simulation core. Pure state and rules, no DOM or rendering, so it can be
// tested headlessly and ported (for example to an Unreal plugin) without the web front end.
// Money in EUR million, time in days, power in MW, energy in MWh, lengths in km unless stated.
(function (root) {
  "use strict";

  // ---------- tuning ----------
  const CFG = {
    start: Date.UTC(2027, 0, 1) / 86400000,       // day number of 1 Jan 2027
    startCash: 210,
    overdraft: -15,
    map: { W: 12, H: 9, nx: 60, ny: 45, portDist: 38 },
    turbine: { name: "15 MW class", P: 15, rotorD: 236, hub: 150, Ur: 10.6, ci: 3, co: 25, Ct: 0.78 },
    cost: {
      turbine: 15.5,                // supply, per turbine
      foundation0: 6.2,             // monopile and transition piece at 30 m water, firm sand
      foundationPerM: 0.24,         // per metre deeper than 30 m
      arrayCable: 1.3,
      exportSystem: 185,            // offshore substation and export cable, up to 25 turbines
      geophys: 3, geotech: 8, consent: 4,
      leaseRent: 0.6,               // per year after award
      insurance: 0.008,             // fraction of capital cost per year
      omFixed: 0.045,               // per turbine per year: onshore base, spares, management
      monitoring: 0.025,            // StormTwin monitoring, per turbine per year
      ctvDay: 0.0045, sovDay: 0.038,
      jackupMob: 2.6, jackupDay: 0.19, rockMob: 0.9, rockPer: 0.45,
      fdnVesselDay: 0.21, wtgVesselDay: 0.26, vesselMob: 3.2,
    },
    durations: { geophys: 35, geotech: 70, consent: 240, exportBuild: 150, fdnDays: 1.6, wtgDays: 1.3, jackupLead: [32, 62], rockLead: [20, 40] },
    access: { ctv: 1.5, sov: 2.8, jackupHs: 2.0, jackupU: 12, installHs: 2.0, installU: 12 },
    debt: { cfd: 0.75, merchant: 0.6, rate: 0.06, years: 18, graceMonths: 12 },
  };

  // components: life in years at average load with annual service, replacement and up-tower repair costs, which vessel replaces it
  const COMPS = [
    { k: "blades", name: "Blades", life: 16, load: "wind", replace: 1.4, repair: 0.12, major: true, hz: 0.0006 },
    { k: "gearbox", name: "Gearbox", life: 13, load: "wind", replace: 1.9, repair: 0.35, major: true, hz: 0.0005 },
    { k: "mainbearing", name: "Main bearing", life: 17, load: "wind", replace: 0.9, repair: 0.22, major: true, hz: 0.0003 },
    { k: "generator", name: "Generator", life: 15, load: "wind", replace: 1.1, repair: 0.18, major: true, hz: 0.0004 },
    { k: "converter", name: "Power converter", life: 6, load: "wind", replace: 0.16, repair: 0.04, major: false, hz: 0.0025 },
    { k: "pitch", name: "Pitch system", life: 7, load: "wind", replace: 0.09, repair: 0.025, major: false, hz: 0.002 },
    { k: "bolts", name: "Tower-base flange bolts", life: 9, load: "wind", replace: 0.06, repair: 0.02, major: false, hz: 0.0006 },
    { k: "scour", name: "Scour protection", life: 11, load: "sea", replace: 0, repair: 0, major: false, rock: true, hz: 0.0004 },
  ];
  const COMP = Object.fromEntries(COMPS.map((c) => [c.k, c]));

  // ---------- random numbers (seeded, so a game can be replayed) ----------
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  const gauss = (r) => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  // ---------- the lease area ----------
  function makeMap(r) {
    const { W, H, nx, ny } = CFG.map, depth = new Float32Array(nx * ny), soil = new Uint8Array(nx * ny), excl = new Uint8Array(nx * ny);
    const mean = 34 + 8 * r(), th = r() * Math.PI, ph = [r() * 6, r() * 6, r() * 6, r() * 6], slope = 0.8 + r() * 0.8;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * W / nx, y = (j + 0.5) * H / ny, a = (x - W / 2) * Math.cos(th) + (y - H / 2) * Math.sin(th), b = -(x - W / 2) * Math.sin(th) + (y - H / 2) * Math.cos(th);
      const bank = Math.sin(2 * Math.PI * a / 2.6 + 0.5 * Math.sin(b * 0.8 + ph[0])), n = 0.4 * Math.sin(1.7 * x + ph[1]) * Math.sin(1.3 * y + ph[2]) + 0.25 * Math.sin(3.1 * x - 2.3 * y + ph[3]);
      depth[j * nx + i] = clamp(mean - 6 * bank + 3 * n + slope * (x - W / 2), 18, 70);
      const sv = Math.sin(0.55 * x + ph[2]) + Math.cos(0.7 * y + ph[1]) + 0.6 * n;
      soil[j * nx + i] = sv > 0.9 ? 2 : sv > -0.2 ? 1 : 0;
    }
    // a cable corridor and a wreck the layout must avoid
    const cy = H * (0.25 + 0.5 * r()), wx = W * (0.3 + 0.4 * r()), wy = H * (0.2 + 0.6 * r());
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * W / nx, y = (j + 0.5) * H / ny;
      if (Math.abs(y - cy - 0.15 * (x - W / 2)) < 0.35 || Math.hypot(x - wx, y - wy) < 0.45) excl[j * nx + i] = 1;
    }
    // what the player sees before surveys: a coarse chart (smoothed, offset) and no soil information
    const chart = new Float32Array(nx * ny), off = (r() - 0.5) * 6;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { let s = 0, n = 0; for (let b = -4; b <= 4; b++) for (let a = -4; a <= 4; a++) { const ii = i + a, jj = j + b; if (ii >= 0 && jj >= 0 && ii < nx && jj < ny) { s += depth[jj * nx + ii]; n++; } } chart[j * nx + i] = s / n + off; }
    return { W, H, nx, ny, depth, soil, excl, chart, cable: cy };
  }
  const SOILS = [{ name: "Dense sand", f: 1.0 }, { name: "Sand over stiff clay", f: 1.14 }, { name: "Soft clay over sand", f: 1.34 }];
  function sample(m, arr, x, y) {
    const fx = clamp(x / m.W * m.nx - 0.5, 0, m.nx - 1.001), fy = clamp(y / m.H * m.ny - 0.5, 0, m.ny - 1.001), i = Math.floor(fx), j = Math.floor(fy), a = fx - i, b = fy - j, n = m.nx;
    return arr[j * n + i] * (1 - a) * (1 - b) + arr[j * n + i + 1] * a * (1 - b) + arr[(j + 1) * n + i] * (1 - a) * b + arr[(j + 1) * n + i + 1] * a * b;
  }
  const cell = (m, x, y) => clamp(Math.floor(y / m.H * m.ny), 0, m.ny - 1) * m.nx + clamp(Math.floor(x / m.W * m.nx), 0, m.nx - 1);

  // ---------- weather: daily wind and sea with seasons, persistence and storms ----------
  function weatherStep(s) {
    const r = s.r, doy = (s.day % 365.25) / 365.25, season = Math.cos(2 * Math.PI * (doy - 0.03));   // +1 mid-winter
    const target = 7.7 + 1.8 * season;
    s.w.a = 0.8 * s.w.a + 0.6 * gauss(r);
    let U10 = Math.max(0.5, target + 2.6 * s.w.a);
    if (s.w.storm > 0) { U10 = Math.max(U10, 20 + 4 * r()); s.w.storm--; }
    else if (r() < 0.006 * (1 + season)) { s.w.storm = 1 + Math.floor(r() * 3); s.log.push({ t: s.day, lvl: "warn", text: "Storm in the forecast: high seas for the next few days." }); }
    const Hs = Math.max(0.3, 0.2 + 0.0105 * U10 * U10 + 0.5 * (0.5 + 0.5 * season) + 0.25 * gauss(r) * 0.6);
    s.w.U10 = U10; s.w.U = U10 * 1.36; s.w.Hs = Hs; s.w.HsMax = Hs * (1.15 + 0.15 * r()); s.w.dir = (240 + 50 * gauss(r) * 0.6 + 360) % 360;
  }

  // ---------- power ----------
  const powerAt = (U) => { const t = CFG.turbine; return U < t.ci || U > t.co ? 0 : t.P * Math.min(1, ((U - t.ci) / (t.Ur - t.ci)) ** 3); };
  function dailyEnergyFactor(U) {        // mean power over a day of gusty wind around the daily mean, as a fraction of rated
    const q = [[-1.5, 0.13], [-0.75, 0.24], [0, 0.26], [0.75, 0.24], [1.5, 0.13]]; let p = 0;
    for (const [z, w] of q) p += w * powerAt(Math.max(0, U * (1 + 0.22 * z)));
    return p / CFG.turbine.P;
  }
  // wake factor per turbine: energy-weighted over 12 sectors around the prevailing south-westerly, Gaussian wakes
  function wakeFactors(ts) {
    const D = CFG.turbine.rotorD / 1000, k = 0.04, ct = CFG.turbine.Ct, out = ts.map(() => 0), wsum = [];
    for (let s = 0; s < 12; s++) wsum.push(Math.exp(1.1 * Math.cos((s * 30 - 240) * Math.PI / 180)));
    const tot = wsum.reduce((a, b) => a + b, 0);
    for (let s = 0; s < 12; s++) {
      const to = (s * 30 + 180) * Math.PI / 180, fx = Math.sin(to), fy = -Math.cos(to);
      ts.forEach((a, i) => { let sum = 0;
        ts.forEach((b) => { if (a === b) return; const dx = a.x - b.x, dy = a.y - b.y, xd = dx * fx + dy * fy; if (xd <= 0.05) return; const rr = Math.abs(dx * fy - dy * fx);
          const sg = (k * xd / D + 0.2 * Math.sqrt(0.5 * (1 + Math.sqrt(1 - ct)) / Math.sqrt(1 - ct))) * D, c = 1 - Math.sqrt(Math.max(0, 1 - ct / (8 * (sg / D) ** 2)));
          sum += (0.7 * c * Math.exp(-rr * rr / (2 * sg * sg))) ** 2; });
        out[i] += wsum[s] / tot * Math.min(0.6, Math.sqrt(sum)); });
    }
    return out.map((d) => 1 - 1.6 * d);   // wakes matter below rated only: about half of the cube law
  }

  // ---------- turbines and costs ----------
  function foundationCost(s, x, y, known) {
    const m = s.map, d = known.geophys ? sample(m, m.depth, x, y) : sample(m, m.chart, x, y), sf = known.geotech ? SOILS[m.soil[cell(m, x, y)]].f : 1.15;
    return { cost: (CFG.cost.foundation0 + CFG.cost.foundationPerM * Math.max(0, d - 30)) * sf * (d > 55 ? 1.25 : 1), depth: d, unc: (known.geophys ? 0.05 : 0.18) + (known.geotech ? 0.04 : 0.2) };
  }
  function capexEstimate(s) {
    const c = CFG.cost, n = s.turbines.length; let f = 0, unc = 0;
    s.turbines.forEach((t) => { const fc = foundationCost(s, t.x, t.y, s.known); f += fc.cost; unc += fc.cost * fc.unc; });
    const install = n * (CFG.durations.fdnDays * c.fdnVesselDay * 1.6 + CFG.durations.wtgDays * c.wtgVesselDay * 1.6) + 2 * c.vesselMob;
    const total = n * (c.turbine + c.arrayCable) + f + install + (n ? c.exportSystem : 0);
    return { total, foundations: f, install, unc: unc + 0.08 * install };
  }
  function yieldEstimate(s) {           // P50 annual energy (MWh) from a long-term wind distribution, with wakes
    const wf = wakeFactors(s.turbines); let cf = 0, n = 0;
    for (let U10 = 0.5; U10 < 32; U10 += 0.5) { const A = 8.7, k = 2.1, p = k / A * (U10 / A) ** (k - 1) * Math.exp(-((U10 / A) ** k)) * 0.5; cf += p * dailyEnergyFactor(U10 * 1.36); n += p; }
    cf /= n;
    return { cf, energy: wf.reduce((a, f) => a + f * cf * CFG.turbine.P * 8766, 0), wake: 1 - wf.reduce((a, b) => a + b, 0) / Math.max(1, wf.length), wf };
  }

  // ---------- game state ----------
  function newGame(seed = 7) {
    const r = rng(seed);
    const s = {
      seed, r, day: CFG.start, cash: CFG.startCash, debt: 0, facility: 0, debtService: 0, map: makeMap(r),
      stage: "lease",                           // lease -> develop -> build -> operate
      known: { geophys: false, geotech: false }, jobs: [],     // surveys and consent in progress
      consent: "none", lease: { won: false, rivals: [6 + 10 * r(), 6 + 10 * r()], round: 1 },
      cfd: { state: "none", strike: 0, clearing: 77 + 10 * r() }, price: 68,
      turbines: [], nextId: 1, exportProgress: 0, build: null, cod: null, capexSpent: 0,
      ctv: 0, sov: false, monitoring: false, jackup: null, rock: null, tasks: [], nextTask: 1,
      w: { a: 0, storm: 0, U10: 8, U: 11, Hs: 1.4, HsMax: 1.6, dir: 240 },
      ledger: [], year: null, history: [], log: [{ t: CFG.start, lvl: "info", text: "Leasing round open: bid for the option on this seabed lease." }],
      over: null,
    };
    weatherStep(s); s.year = blankYear(s);
    return s;
  }
  const blankYear = (s) => ({ y: yearOf(s.day), energy: 0, revenue: 0, opex: 0, capex: 0, debt: 0, failures: 0, prevent: 0, upHours: 0, possibleHours: 0, lostMWh: 0 });
  const yearOf = (d) => new Date(d * 86400000).getUTCFullYear();
  const dateStr = (d) => new Date(d * 86400000).toISOString().slice(0, 10);

  function pay(s, amt, what, kind = "opex") {
    if (!amt) return;
    if (kind === "capex" && s.facility > s.debt) { const draw = Math.min(amt * (s.cfd.state === "won" ? CFG.debt.cfd : CFG.debt.merchant), s.facility - s.debt); s.debt += draw; s.cash += draw; }
    s.cash -= amt; s.year[kind] += amt; if (kind === "capex") s.capexSpent += amt;
    s.ledger.push({ t: s.day, amt: -amt, what, kind }); if (s.ledger.length > 400) s.ledger.shift();
  }
  const say = (s, text, lvl = "info") => { s.log.push({ t: s.day, lvl, text }); if (s.log.length > 200) s.log.shift(); };

  // ---------- commands ----------
  const ok = (msg) => ({ ok: true, msg }), no = (msg) => ({ ok: false, msg });
  const CMD = {
    bidLease(s, bid) {
      if (s.stage !== "lease") return no("The lease is already decided.");
      bid = +bid; if (!(bid > 0) || bid > s.cash - 20) return no("Bid between €1 M and what you can afford, keeping €20 M for development.");
      const best = Math.max(...s.lease.rivals);
      if (bid > best) { pay(s, bid, "Lease option fee", "capex"); s.lease.won = true; s.stage = "develop"; say(s, `Lease won with €${bid.toFixed(1)} M (next best bid €${best.toFixed(1)} M). Survey the site and apply for consent.`, "good"); return ok("Lease won."); }
      s.lease.round++; s.lease.rivals = s.lease.rivals.map(() => 5 + 11 * s.r()); s.map = makeMap(s.r);
      say(s, `Outbid: a rival paid €${best.toFixed(1)} M. A new lease area is on offer in the next round.`, "warn"); return no("Outbid. A new area is on offer.");
    },
    survey(s, kind) {
      if (!s.lease.won) return no("Win the lease first.");
      if (s.known[kind] || s.jobs.some((j) => j.kind === kind)) return no("Already done or under way.");
      pay(s, CFG.cost[kind], kind === "geophys" ? "Geophysical survey" : "Geotechnical survey", "capex");
      s.jobs.push({ kind, done: s.day + CFG.durations[kind] }); say(s, `${kind === "geophys" ? "Geophysical survey vessel mobilised: bathymetry in" : "Geotechnical campaign started: soil cores and CPTs in"} about ${CFG.durations[kind]} days.`); return ok("Started.");
    },
    consent(s) {
      if (!s.lease.won) return no("Win the lease first.");
      if (s.consent !== "none") return no("Consent already applied for.");
      pay(s, CFG.cost.consent, "Consent application and environmental studies", "capex"); s.consent = "pending"; s.jobs.push({ kind: "consent", done: s.day + CFG.durations.consent });
      say(s, "Consent application submitted; a decision is due in about 8 months."); return ok("Applied.");
    },
    place(s, x, y) {
      if (!s.lease.won) return no("Win the lease first.");
      if (s.stage === "operate" || (s.build && s.build.fdnIdx >= s.turbines.length)) return no("Construction is complete.");
      const m = s.map; if (x < 0.2 || y < 0.2 || x > m.W - 0.2 || y > m.H - 0.2) return no("Inside the lease boundary, please.");
      if (m.excl[cell(m, x, y)]) return no("That is inside an exclusion zone (cable corridor or wreck).");
      const D = CFG.turbine.rotorD / 1000, near = s.turbines.reduce((a, t) => Math.min(a, Math.hypot(t.x - x, t.y - y)), 9);
      if (near < 3.5 * D) return no(`Too close to another turbine: keep at least 3.5 rotor diameters (${(3.5 * D * 1000).toFixed(0)} m).`);
      if (s.turbines.length >= 25) return no("The export system takes 25 turbines at most.");
      const t = { id: s.nextId++, name: "T" + String(s.nextId - 1).padStart(2, "0"), x, y, state: "planned", fdnCost: 0, health: {}, est: {}, sig: {}, seen: {}, wake: 1, down: null, energy: 0 };
      COMPS.forEach((c) => { t.health[c.k] = 0.9 + 0.1 * s.r(); t.est[c.k] = t.health[c.k]; t.sig[c.k] = 0.1; t.seen[c.k] = s.day; });
      s.turbines.push(t); refreshWakes(s); return ok(t.name + " planned.");
    },
    remove(s, id) {
      const t = s.turbines.find((q) => q.id === id); if (!t) return no("No such turbine.");
      if (t.state !== "planned") return no("Already under construction.");
      s.turbines = s.turbines.filter((q) => q !== t); refreshWakes(s); return ok("Removed.");
    },
    bidCfd(s, strike) {
      if (s.cfd.state === "won" || s.cfd.state === "merchant") return no("The route to market is already settled.");
      if (s.cfd.state === "waiting") return no("Wait for the next allocation round.");
      if (!s.turbines.length) return no("Plan the layout first; the auction needs a capacity.");
      strike = +strike; if (!(strike > 30 && strike < 150)) return no("Bid a strike price between €30 and €150 per MWh.");
      if (strike <= s.cfd.clearing) { s.cfd.state = "won"; s.cfd.strike = strike; financialClose(s); say(s, `Contract for difference won at €${strike.toFixed(0)}/MWh for 15 years (round cleared at €${s.cfd.clearing.toFixed(0)}). Debt financing closed.`, "good"); return ok("CfD won."); }
      s.cfd.state = "waiting"; s.cfd.next = s.day + 180; say(s, `CfD bid at €${strike.toFixed(0)}/MWh was above the clearing price of €${s.cfd.clearing.toFixed(0)}. Next round in 6 months, or sell at market prices.`, "warn"); return no("Lost the auction.");
    },
    merchant(s) {
      if (s.cfd.state === "won" || s.cfd.state === "merchant") return no("Already settled.");
      if (!s.turbines.length) return no("Plan the layout first.");
      s.cfd.state = "merchant"; financialClose(s); say(s, "Going merchant: power sells at the market price. Lenders take a smaller share of the cost.", "warn"); return ok("Merchant.");
    },
    construct(s) {
      if (s.build) return no("Construction is under way.");
      if (s.consent !== "granted") return no("Consent is needed before construction.");
      if (s.cfd.state !== "won" && s.cfd.state !== "merchant") return no("Settle the route to market (CfD or merchant) to raise the debt first.");
      if (!s.turbines.length) return no("Plan at least one turbine.");
      const eq = capexEstimate(s).total * (1 - (s.cfd.state === "won" ? CFG.debt.cfd : CFG.debt.merchant)) * 1.05;
      if (eq > s.cash) return no(`Lenders want to see about €${eq.toFixed(0)} M of equity (your share of the cost plus 5 % contingency); you have €${s.cash.toFixed(0)} M. Plan fewer turbines${s.cfd.state === "merchant" ? " or win a CfD" : ""}.`);
      s.stage = "build"; s.build = { exportDone: false, exportDays: 0, fdnIdx: 0, wtgIdx: 0, fdnWork: 0, wtgWork: 0, fdnVessel: false, wtgVessel: false };
      s.turbines.sort((a, b) => a.x - b.x || a.y - b.y);
      say(s, "Construction started: offshore substation and export cable first, then foundations and turbines as the weather allows.", "good"); return ok("Started.");
    },
    ctv(s, n) { s.ctv = clamp(Math.round(n), 0, 6); return ok(`${s.ctv} crew transfer vessel(s) on charter.`); },
    sov(s, on) { s.sov = !!on; say(s, s.sov ? "Service operation vessel chartered: crews stay offshore and work in seas up to 2.8 m." : "Service operation vessel released."); return ok("Done."); },
    monitoring(s, on) { s.monitoring = !!on; say(s, s.monitoring ? "StormTwin monitoring switched on: component health estimates update weekly with alerts." : "Monitoring switched off: health is known only from services and inspections."); return ok("Done."); },
    task(s, id, comp, kind) {        // kind: repair (up-tower, crew) | replace | inspect | service
      const t = s.turbines.find((q) => q.id === id); if (!t || t.state !== "operating") return no("Only operating turbines can be worked on.");
      if (s.tasks.some((q) => q.tid === id && q.comp === comp && q.kind === kind)) return no("Already queued.");
      const c = COMP[comp];
      if (kind === "repair" && c && t.health[comp] < 0.2) return no(`${c.name} is too far gone for an up-tower repair; it needs replacing.`);
      const task = { id: s.nextTask++, tid: id, comp, kind, made: s.day };
      if (kind === "inspect") Object.assign(task, { vessel: "crew", work: 1, parts: 0.012, label: "Inspection (rope access, drone, bolt check)" });
      else if (kind === "service") Object.assign(task, { vessel: "crew", work: 2, parts: 0.03, label: "Annual service" });
      else if (c.rock) Object.assign(task, { vessel: "rock", work: 1, parts: CFG.cost.rockPer, label: "Rock placement around the foundation" });
      else if (kind === "repair") Object.assign(task, { vessel: "crew", work: c.major ? 3 : 1, parts: c.repair, label: `Up-tower repair: ${c.name.toLowerCase()}` });
      else Object.assign(task, { vessel: c.major ? "jackup" : "crew", work: c.major ? 3 : 1, parts: c.replace, label: `Replace ${c.name.toLowerCase()}` });
      s.tasks.push(task); return ok("Queued: " + task.label);
    },
    cancel(s, taskId) { s.tasks = s.tasks.filter((q) => q.id !== taskId); return ok("Cancelled."); },
    prioritise(s, taskId) { const i = s.tasks.findIndex((q) => q.id === taskId); if (i > 0) s.tasks.unshift(s.tasks.splice(i, 1)[0]); return ok("Moved to the front."); },
    jackup(s) {
      if (s.jackup) return no("A jack-up campaign is already booked.");
      if (!s.tasks.some((q) => q.vessel === "jackup")) return no("Queue a major replacement first.");
      const [a, b] = CFG.durations.jackupLead, lead = Math.round(a + (b - a) * s.r());
      s.jackup = { arrive: s.day + lead, onsite: false, days: 0 }; pay(s, CFG.cost.jackupMob, "Jack-up vessel mobilisation");
      say(s, `Jack-up vessel booked: on site around ${dateStr(s.day + lead)}. Every major replacement queued by then goes in the same campaign.`); return ok("Booked.");
    },
    rockdump(s) {
      if (s.rock) return no("Already booked.");
      if (!s.tasks.some((q) => q.vessel === "rock")) return no("Queue a scour repair first.");
      const [a, b] = CFG.durations.rockLead, lead = Math.round(a + (b - a) * s.r());
      s.rock = { arrive: s.day + lead }; pay(s, CFG.cost.rockMob, "Rock placement vessel mobilisation");
      say(s, `Rock placement vessel booked for around ${dateStr(s.day + lead)}.`); return ok("Booked.");
    },
  };
  function command(s, name, ...args) { if (s.over) return no("The game is over."); const f = CMD[name]; return f ? f(s, ...args) : no("Unknown command " + name); }
  function refreshWakes(s) { const wf = wakeFactors(s.turbines); s.turbines.forEach((t, i) => { t.wake = wf[i]; }); }
  function financialClose(s) {
    const est = capexEstimate(s).total, share = s.cfd.state === "won" ? CFG.debt.cfd : CFG.debt.merchant;
    s.facility = est * share; s.closeDay = s.day;
  }

  // ---------- the daily step ----------
  function step(s) {
    if (s.over) return;
    s.day += 1; weatherStep(s);
    const W = s.w, c = CFG.cost;
    if (yearOf(s.day) !== s.year.y) { s.history.push(s.year); s.year = blankYear(s); say(s, `Happy new year: ${yearOf(s.day)} begins.`); }
    // development jobs
    s.jobs = s.jobs.filter((j) => {
      if (s.day < j.done) return true;
      if (j.kind === "consent") { s.consent = "granted"; say(s, "Consent granted. You may build once the route to market is settled.", "good"); }
      else { s.known[j.kind] = true; say(s, j.kind === "geophys" ? "Bathymetry survey complete: depths are now known to within half a metre." : "Geotechnical results in: soil zones are mapped and foundation costs firm up.", "good"); }
      return false;
    });
    if (s.cfd.state === "waiting" && s.day >= s.cfd.next) { s.cfd.state = "none"; s.cfd.clearing = 74 + 14 * s.r(); say(s, "A new contract-for-difference allocation round is open."); }
    // market price (merchant) drifts with the season
    s.price = clamp(0.97 * s.price + 0.03 * (70 + 12 * Math.cos(2 * Math.PI * ((s.day % 365.25) / 365.25))) + 3 * gauss(s.r), 20, 160);
    if (s.lease.won) pay(s, c.leaseRent / 365.25, "Lease rent");
    if (s.build) buildStep(s);
    if (s.turbines.some((t) => t.state === "operating")) operateStep(s);
    // finance: interest monthly, repayment after the grace period
    if (new Date(s.day * 86400000).getUTCDate() === 1 && s.debt > 0) {
      const r = CFG.debt.rate / 12, interest = s.debt * r;
      let principal = 0;
      if (s.closeDay && s.day - s.closeDay > CFG.debt.graceMonths * 30.4) {
        if (!s.annuity) { const n = CFG.debt.years * 12; s.annuity = s.debt * r / (1 - Math.pow(1 + r, -n)); }
        principal = Math.min(s.debt, Math.max(0, s.annuity - interest));
      }
      s.debt -= principal; s.cash -= interest + principal; s.year.debt += interest + principal;
      s.ledger.push({ t: s.day, amt: -(interest + principal), what: "Debt service", kind: "debt" });
    }
    if (s.cash < CFG.overdraft) { s.over = { lost: true, why: "The cash ran out and the lenders took over the project." }; say(s, s.over.why, "bad"); }
  }

  function buildStep(s) {
    const b = s.build, W = s.w, c = CFG.cost, A = CFG.access, D = CFG.durations, n = s.turbines.length;
    if (!b.exportDone) {
      if (b.exportDays === 0) pay(s, c.exportSystem * 0.4, "Offshore substation and export cable: first payment", "capex");
      if (W.HsMax < 2.5) b.exportDays++;
      if (b.exportDays >= D.exportBuild) { b.exportDone = true; pay(s, c.exportSystem * 0.6, "Offshore substation and export cable: completion", "capex"); say(s, "Offshore substation and export cable energised.", "good"); }
      else if (b.exportDays === 1) say(s, "Cable lay and substation installation under way.");
    }
    // foundation campaign runs alongside the export system
    if (b.fdnIdx < n) {
      if (!b.fdnVessel) { b.fdnVessel = true; pay(s, c.vesselMob, "Foundation installation vessel mobilisation", "capex"); say(s, "Heavy-lift vessel on site for the monopiles."); }
      pay(s, c.fdnVesselDay, "Foundation vessel day rate", "capex");
      if (W.HsMax < A.installHs) {
        b.fdnWork++;
        const t = s.turbines[b.fdnIdx];
        if (b.fdnWork >= D.fdnDays * (1 + 0.15 * s.r())) {
          b.fdnWork = 0; const fc = foundationCost(s, t.x, t.y, { geophys: true, geotech: true });
          t.fdnCost = fc.cost; t.depth = fc.depth; t.soil = SOILS[s.map.soil[cell(s.map, t.x, t.y)]].name;
          pay(s, fc.cost + c.arrayCable, `${t.name}: monopile, transition piece and array cable`, "capex"); t.state = "foundation"; b.fdnIdx++;
          if (b.fdnIdx === n) { b.fdnVessel = false; say(s, "All foundations installed; the heavy-lift vessel leaves.", "good"); }
        }
      }
    }
    // turbines follow the foundations
    if (b.wtgIdx < b.fdnIdx) {
      if (!b.wtgVessel) { b.wtgVessel = true; pay(s, c.vesselMob, "Turbine installation vessel mobilisation", "capex"); say(s, "Turbine installation jack-up on site."); }
      pay(s, c.wtgVesselDay, "Turbine installation vessel day rate", "capex");
      if (W.HsMax < A.installHs && W.U10 < A.installU) {
        b.wtgWork++;
        if (b.wtgWork >= D.wtgDays * (1 + 0.15 * s.r())) {
          b.wtgWork = 0; const t = s.turbines[b.wtgIdx]; pay(s, c.turbine, `${t.name}: turbine supply and installation`, "capex");
          t.state = b.exportDone ? "operating" : "installed"; t.since = s.day; b.wtgIdx++;
          if (t.state === "operating") say(s, `${t.name} exports first power.`, "good");
        }
      }
    } else if (b.wtgVessel && b.fdnIdx === n) { b.wtgVessel = false; }
    if (b.exportDone) s.turbines.forEach((t) => { if (t.state === "installed") { t.state = "operating"; t.since = s.day; } });
    if (b.exportDone && b.wtgIdx === n) { s.build = null; s.stage = "operate"; s.cod = s.day; say(s, `Commercial operation: all ${n} turbines exporting. The farm is yours to run.`, "good"); }
  }

  function operateStep(s) {
    const W = s.w, c = CFG.cost, A = CFG.access, ops = s.turbines.filter((t) => t.state === "operating");
    const price = s.cfd.state === "won" ? s.cfd.strike : s.price, f = dailyEnergyFactor(W.U);
    // production, wear and failures
    ops.forEach((t) => {
      s.year.possibleHours += 24;
      const pot = f * CFG.turbine.P * 24 * t.wake;
      if (!t.down) { t.energy += pot; s.year.energy += pot; s.year.revenue += pot * price / 1e6; s.cash += pot * price / 1e6; s.year.upHours += 24; }
      else s.year.lostMWh += pot;
      const windLoad = f > 0 ? (0.6 + 0.8 * (W.U / 11) ** 2) * (1 + 0.5 * (1 - t.wake)) : 0.35, seaLoad = (W.Hs / 1.6) ** 2;
      COMPS.forEach((cp) => {
        const load = cp.load === "sea" ? seaLoad : windLoad, svc = s.day - (t.lastService || t.since) > 400 ? 1.5 : 1;
        const dh = load * svc * (1 / (cp.life * 365.25)) * Math.exp(0.35 * gauss(s.r));
        t.health[cp.k] = Math.max(0, t.health[cp.k] - dh);
        if (t.down) return;
        const hz = cp.hz * Math.exp(5.5 * (1 - t.health[cp.k])) / Math.exp(5.5) * 8 + (t.health[cp.k] <= 0 ? 1 : 0);
        if (s.r() < hz) failure(s, t, cp);
      });
      // what the operator knows: monitoring updates weekly, otherwise the estimate goes stale and its band widens
      COMPS.forEach((cp) => {
        if (s.monitoring && (s.day - t.seen[cp.k] >= 7)) observe(s, t, cp.k, 0.07);
        else t.sig[cp.k] = Math.min(0.35, t.sig[cp.k] + (s.monitoring ? 0 : 0.0006));
      });
      if (s.monitoring) COMPS.forEach((cp) => { const lo = t.est[cp.k] - 1.645 * t.sig[cp.k];
        if (lo < 0.3 && !t["alert_" + cp.k] && !t.down) { t["alert_" + cp.k] = true; say(s, `StormTwin alert: ${t.name} ${cp.name.toLowerCase()} health could be as low as ${(100 * Math.max(0, lo)).toFixed(0)} %. Plan a repair.`, "warn"); } });
    });
    // costs that run every day
    pay(s, ops.length * (c.omFixed + (s.monitoring ? c.monitoring : 0)) / 365.25 + s.ctv * c.ctvDay + (s.sov ? c.sovDay : 0), "Operations: base, vessels, monitoring");
    pay(s, s.capexSpent * c.insurance / 365.25, "Insurance");
    // annual services fall due
    ops.forEach((t) => { if (s.day - (t.lastService || t.since) > 365 && !s.tasks.some((q) => q.tid === t.id && q.kind === "service")) { CMD.task(s, t.id, null, "service"); } });
    // crews work when the sea allows: CTVs up to 1.5 m, the SOV up to 2.8 m
    let crew = (W.HsMax < A.ctv ? s.ctv : 0) + (s.sov && W.HsMax < A.sov ? 3 : 0);
    s.crewToday = []; s.accessToday = { ctv: W.HsMax < A.ctv, sov: W.HsMax < A.sov };
    const order = s.tasks.filter((q) => q.vessel === "crew").sort((a, b) => (isDown(s, b) - isDown(s, a)) || (s.tasks.indexOf(a) - s.tasks.indexOf(b)));
    for (const q of order) { if (crew <= 0) break; const use = Math.min(crew, q.work - (q.done || 0)); q.done = (q.done || 0) + use; crew -= use; s.crewToday.push(q.tid); if (q.done >= q.work) finish(s, q); }
    // jack-up campaign
    if (s.jackup) {
      const J = s.jackup;
      if (!J.onsite && s.day >= J.arrive) { J.onsite = true; say(s, "Jack-up vessel on site."); }
      if (J.onsite) {
        pay(s, c.jackupDay, "Jack-up vessel day rate");
        const q = s.tasks.find((x) => x.vessel === "jackup");
        if (!q) { s.jackup = null; say(s, "Jack-up campaign complete; the vessel leaves.", "good"); }
        else if (W.HsMax < A.jackupHs && W.U10 < A.jackupU) { q.done = (q.done || 0) + 1; s.jackupAt = q.tid; if (q.done >= q.work) finish(s, q); }
      }
    }
    if (s.rock && s.day >= s.rock.arrive) {
      s.tasks.filter((q) => q.vessel === "rock").forEach((q) => finish(s, q)); s.rock = null; say(s, "Rock placement complete.", "good");
    }
  }
  const isDown = (s, q) => { const t = s.turbines.find((x) => x.id === q.tid); return t && t.down ? 1 : 0; };
  function failure(s, t, cp) {
    t.down = { comp: cp.k, since: s.day }; s.year.failures++;
    say(s, `${t.name} stopped: ${cp.name.toLowerCase()} failure.${cp.major ? " A jack-up vessel is needed to replace it." : cp.rock ? " Rock placement is needed before restart." : ""}`, "bad");
    CMD.task(s, t.id, cp.k, "replace");
  }
  function observe(s, t, k, sig) { t.sig[k] = sig; t.est[k] = clamp(t.health[k] + sig * gauss(s.r), 0, 1); t.seen[k] = s.day; }
  function finish(s, q) {
    const t = s.turbines.find((x) => x.id === q.tid); s.tasks = s.tasks.filter((x) => x !== q); if (!t) return;
    pay(s, q.parts, `${t.name}: ${q.label.toLowerCase()}`);
    if (q.kind === "inspect") { COMPS.forEach((cp) => observe(s, t, cp.k, 0.03)); say(s, `${t.name} inspected: component health known closely for now.`); return; }
    if (q.kind === "service") { t.lastService = s.day; COMPS.forEach((cp) => { if (!cp.rock) { t.health[cp.k] = Math.min(1, t.health[cp.k] + 0.02); observe(s, t, cp.k, 0.1); } }); return; }
    const cp = COMP[q.comp];
    t.health[q.comp] = q.kind === "repair" ? Math.min(0.92, t.health[q.comp] + 0.45) : 0.98; observe(s, t, q.comp, 0.04); t["alert_" + q.comp] = false;
    if (q.kind === "repair") s.year.prevent++;
    if (t.down && t.down.comp === q.comp) { say(s, `${t.name} back in service after ${Math.round(s.day - t.down.since)} days: ${cp.name.toLowerCase()} ${q.kind === "repair" ? "repaired" : "replaced"}.`, "good"); t.down = null; }
    else say(s, `${t.name}: ${cp.name.toLowerCase()} ${q.kind === "repair" ? "repaired up-tower before it failed" : "replaced"}.`, "good");
  }

  // ---------- views for the front end ----------
  function summary(s) {
    const ops = s.turbines.filter((t) => t.state === "operating"), Y = s.year;
    return {
      date: dateStr(s.day), day: s.day, stage: s.stage, cash: s.cash, debt: s.debt, facility: s.facility,
      capacity: s.turbines.length * CFG.turbine.P, operating: ops.length, down: ops.filter((t) => t.down).length,
      powerNow: ops.reduce((a, t) => a + (t.down ? 0 : dailyEnergyFactor(s.w.U) * CFG.turbine.P * t.wake), 0),
      availability: Y.possibleHours ? Y.upHours / Y.possibleHours : null, year: Y, weather: s.w, price: s.cfd.state === "won" ? s.cfd.strike : s.price,
      capex: capexEstimate(s), yield: s.turbines.length ? yieldEstimate(s) : null,
    };
  }

  const API = { CFG, COMPS, COMP, SOILS, newGame, step, command, summary, sample, cell, foundationCost, capexEstimate, yieldEstimate, powerAt, dailyEnergyFactor, dateStr, yearOf };
  if (typeof module !== "undefined" && module.exports) module.exports = API; else root.WindSim = API;
})(typeof window !== "undefined" ? window : globalThis);
