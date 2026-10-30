// Wind & Steel: simulation core. Pure state and rules (no DOM), deterministic from a seed, so it can be tested
// headlessly (balance_test.js) and ported later. Money in credits (1 credit per MWh sold), lengths in m, masses in t,
// forces in MN, moments in MN·m, stresses in MPa. A game day is a real day for energy; wear and fatigue are
// time-compressed (WEAR) so a 25-year design life plays out in about 300 days.
(function (root) {
  "use strict";
  const G = 9.81, E = 2.1e5, RHO_S = 7.85, WEAR = 30, YEAR = 120;   // E in MN/m²; steel t/m³; days per game year (four 30-day seasons)

  // ---------- catalogue ----------
  const TURBINES = {
    t6: { name: "6 MW", P: 6, D: 154, mRNA: 360, rpm: [5, 10], Ur: 12, thrust: 0.95, price: 900, cutIn: 3, cutOut: 25 },
    t10: { name: "10 MW", P: 10, D: 190, mRNA: 560, rpm: [5, 9], Ur: 11.5, thrust: 1.5, price: 1500, cutIn: 3, cutOut: 25 },
    t15: { name: "15 MW", P: 15, D: 240, mRNA: 850, rpm: [4.5, 7.6], Ur: 10.6, thrust: 2.4, price: 2300, cutIn: 3, cutOut: 25 },
    t20: { name: "20 MW", P: 20, D: 275, mRNA: 1150, rpm: [4, 6.6], Ur: 10.5, thrust: 3.1, price: 3300, cutIn: 3, cutOut: 25 },
  };
  const STEELS = { S355: { name: "S355", fy: 355, cost: 1.0 }, S420: { name: "S420", fy: 420, cost: 1.12 }, S460: { name: "S460", fy: 460, cost: 1.2 } };
  const COMPS = [
    { k: "blades", name: "Blades", life: 300, repair: 300, fix: 3, lead: 6 },
    { k: "drive", name: "Drivetrain", life: 240, repair: 480, fix: 3, lead: 12 },
    { k: "elec", name: "Electrical", life: 140, repair: 120, fix: 1, lead: 3 },
  ];
  const INSPECT = {
    visual: { name: "Visual (rope access)", cost: 12, detect: 0.45, sd: 0.25, comps: ["blades", "elec"], structure: 0.15 },
    drone: { name: "Drone survey", cost: 30, detect: 0.85, sd: 0.12, comps: ["blades"], structure: 0.3, need: "drones" },
    rov: { name: "ROV and ultrasonic", cost: 55, detect: 0.85, sd: 0.1, comps: [], structure: 0.85, need: "rov" },
    full: { name: "Full service inspection", cost: 80, detect: 0.8, sd: 0.12, comps: ["blades", "drive", "elec"], structure: 0.5, need: "drones" },
  };
  // research: id -> { name, branch, rp, cost, req[], effect text }
  const RESEARCH = [
    { id: "t10", name: "10 MW turbine", branch: "Turbines", rp: 40, cost: 800, req: [], fx: "Bigger rotor, more energy per foundation" },
    { id: "t15", name: "15 MW turbine", branch: "Turbines", rp: 120, cost: 2500, req: ["t10"], fx: "" },
    { id: "t20", name: "20 MW turbine", branch: "Turbines", rp: 260, cost: 6000, req: ["t15", "S420"], fx: "" },
    { id: "S420", name: "S420 steel", branch: "Materials", rp: 50, cost: 600, req: [], fx: "18 % stronger steel" },
    { id: "S460", name: "S460 steel", branch: "Materials", rp: 140, cost: 1800, req: ["S420"], fx: "30 % stronger than S355" },
    { id: "weld", name: "Improved weld detailing", branch: "Materials", rp: 70, cost: 900, req: [], fx: "Fatigue life ×1.6" },
    { id: "ipc", name: "Individual pitch control", branch: "Control", rp: 80, cost: 1200, req: [], fx: "Wind fatigue loads −20 %, blade wear −25 %" },
    { id: "damper", name: "Tower damper", branch: "Control", rp: 110, cost: 1500, req: ["ipc"], fx: "Resonance penalties halved, fatigue −10 %" },
    { id: "steer", name: "Wake steering", branch: "Control", rp: 130, cost: 1600, req: ["ipc"], fx: "Wake losses −40 %" },
    { id: "sheet", name: "Spreadsheet design tool", branch: "Software", rp: 25, cost: 300, req: [], fx: "Model error ±20 % → ±14 %" },
    { id: "fe", name: "Beam finite elements", branch: "Software", rp: 70, cost: 900, req: ["sheet"], fx: "Model error → ±9 %" },
    { id: "td", name: "Time-domain simulation", branch: "Software", rp: 150, cost: 2200, req: ["fe"], fx: "Model error → ±6 %, fatigue band narrower" },
    { id: "twin", name: "Digital twin", branch: "Software", rp: 260, cost: 4000, req: ["td", "sensors"], fx: "Model error → ±4 %; sensors read fatigue directly" },
    { id: "buoy", name: "Wave buoy campaign", branch: "Software", rp: 45, cost: 700, req: [], fx: "Storm-load band narrower" },
    { id: "drones", name: "Inspection drones", branch: "Inspection", rp: 30, cost: 400, req: [], fx: "Drone and full-service inspections" },
    { id: "rov", name: "ROV and ultrasonic testing", branch: "Inspection", rp: 70, cost: 900, req: ["drones"], fx: "Finds weld cracks under water" },
    { id: "sensors", name: "Structural-health sensors", branch: "Inspection", rp: 140, cost: 1800, req: ["rov"], fx: "Fit sensors: continuous health and crack alarms" },
    { id: "ctv2", name: "Second crew vessel", branch: "Vessels", rp: 40, cost: 1200, req: [], fx: "Two jobs a day" },
    { id: "sov", name: "Service operation vessel", branch: "Vessels", rp: 160, cost: 4000, req: ["ctv2"], fx: "Works in waves up to 2.5 m" },
  ];
  const ROLES = {
    structural: { name: "Structural engineer", pay: 6, rp: 1.0, fx: "Model error ×0.8" },
    geotech: { name: "Geotechnical engineer", pay: 6, rp: 0.8, fx: "Soil band ×0.7" },
    metocean: { name: "Metocean engineer", pay: 6, rp: 0.8, fx: "Storm and fatigue bands ×0.6" },
    data: { name: "Data scientist", pay: 8, rp: 1.4, fx: "Inspections read health ±40 % tighter" },
    junior: { name: "Junior engineer", pay: 3, rp: 0.6, fx: "Research only" },
  };
  const PRICE = 1.0, OM_DAY = 2.0, STEEL_CREDIT = { pile: 0.5, tower: 0.6 };

  // ---------- random numbers (state-held, so saves continue identically) ----------
  function rnd(s) { let t = (s.rng = (s.rng + 0x6D2B79F5) >>> 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
  const nrm = (s) => Math.sqrt(-2 * Math.log(rnd(s) + 1e-12)) * Math.cos(2 * Math.PI * rnd(s));

  // ---------- engineering (simplified StormTwin) ----------
  function geometry(tb, plot, d) {
    const hub = tb.D / 2 + 30, tp = 15, depth = plot.depth, Lp = depth + tp, Lt = hub - tp, H = depth + hub;
    const mTower = Math.PI * 0.5 * (d.Db + d.Dt) * d.tt / 1000 * Lt * RHO_S, mPile = Math.PI * d.Dp * d.tp / 1000 * (Lp + d.Le) * RHO_S;
    return { hub, tp, depth, Lp, Lt, H, mTower, mPile };
  }
  // first natural frequency: stepped cantilever with a rotational soil spring, Rayleigh-style tip flexibility
  function frequency(tb, plot, d, s, g) {
    const N = 60; let flex = 0;
    for (let i = 0; i < N; i++) {
      const z = (i + 0.5) / N * g.H, D = z < g.Lp ? d.Dp : d.Db + (d.Dt - d.Db) * (z - g.Lp) / g.Lt, t = (z < g.Lp ? d.tp : d.tt) / 1000;
      flex += (g.H - z) ** 2 / (E * Math.PI / 8 * D ** 3 * t) * g.H / N;
    }
    const Kr = 1.5 * s * d.Dp * d.Le ** 3 * 1.0;          // MN·m/rad
    flex = (flex + g.H * g.H / Kr) * 1.15;                  // 1.15: lateral give of the pile and the soil
    const m = tb.mRNA + 0.24 * g.mTower + 0.1 * g.mPile;
    return Math.sqrt(1000 / (flex * m)) / (2 * Math.PI);
  }
  function bandOf(tb) { return { lo: tb.rpm[1] / 60 * 1.1, hi: 3 * tb.rpm[0] / 60 * 0.9, p1: [tb.rpm[0] / 60, tb.rpm[1] / 60], p3: [3 * tb.rpm[0] / 60, 3 * tb.rpm[1] / 60] }; }
  function buckling(D, tmm, fy) { const r = D / (tmm / 1000); return Math.max(0.35, Math.min(1, 1.25 - 0.0026 * r * Math.sqrt(fy / 235))); }
  // the four checks as utilisation-type numbers (strength, holding: load/capacity; resonance: distance into the band; fatigue: life)
  function checks(st, tb, plot, d, p) {   // p: { s, H50, sev, tech } the parameters to evaluate with
    const g = geometry(tb, plot, d), fy = STEELS[d.steel].fy, Hmax = Math.min(1.86 * p.H50, 0.78 * plot.depth);
    const Fw = 0.0066 * d.Dp ** 2 * Hmax, Fwind = 1.3 * tb.thrust;
    const Mmud = 1.35 * (Fwind * (g.hub + g.depth) + Fw * 0.7 * g.depth), Hmud = 1.35 * (Fwind + Fw), Mtb = 1.35 * Fwind * (g.hub - g.tp) * 1.05;
    const Wp = Math.PI / 4 * d.Dp ** 2 * d.tp / 1000, Wt = Math.PI / 4 * d.Db ** 2 * d.tt / 1000;
    const sMud = Mmud / Wp, sTb = Mtb / Wt, cMud = buckling(d.Dp, d.tp, fy) * fy / 1.1, cTb = buckling(d.Db, d.tt, fy) * fy / 1.1;
    const strength = Math.max(sMud / cMud, sTb / cTb), strengthAt = sMud / cMud > sTb / cTb ? "mudline" : "tower base";
    const e = Mmud / Hmud, Mcap = 0.5 * 0.01 * 1.25 * p.s * d.Dp * d.Le ** 3 * e / (e + d.Le) / 1.25;
    const holding = Mmud / Mcap;
    const f1 = frequency(tb, plot, d, p.s, g), w = bandOf(tb);
    // fatigue: wind part (rated thrust cycles) and wave part (fatigue sea, dynamically amplified near the wave period)
    const tech = p.tech, ctrl = (tech.ipc ? 0.8 : 1) * (tech.damper ? 0.9 : 1);
    const r = (1 / 7.5) / f1, daf = 1 / Math.sqrt((1 - r * r) ** 2 + (2 * 0.06 * r) ** 2);
    const inBand = (f1 >= w.p1[0] && f1 <= w.p1[1] * 1.05) || (f1 >= w.p3[0] * 0.95 && f1 <= w.p3[1]);
    const res = inBand ? (tech.damper ? 1.3 : 1.6) : 1;
    const MfWind = 0.33 * tb.thrust * ctrl * res, MfWave = 0.0042 * d.Dp ** 2 * 1.6 * plot.Hsm * Math.min(daf, 4) * p.sev;
    const dsMud = 1.3 * (MfWind * (g.hub + g.depth) + MfWave * 0.7 * g.depth) / Wp, dsTb = 1.3 * MfWind * (g.hub - g.tp) / Wt;
    const thick = (t) => Math.pow(Math.max(1, t / 25), 0.2);
    const lifeOf = (ds, t) => 25 * Math.pow(50 / (ds * thick(t)), 4) * (tech.weld ? 1.6 : 1);
    const life = Math.min(lifeOf(dsMud, d.tp), lifeOf(dsTb, d.tt)), lifeAt = lifeOf(dsMud, d.tp) < lifeOf(dsTb, d.tt) ? "mudline weld" : "tower base weld";
    const freqPos = f1 < w.lo ? (f1 - w.lo) / (w.hi - w.lo) : f1 > w.hi ? (f1 - w.hi) / (w.hi - w.lo) : 0;
    return { strength, strengthAt, holding, f1, w, inBand, freqPos, life, lifeAt, Mmud, mass: g.mTower + g.mPile, g, sMud, sTb, dsMud, dsTb, uMud: sMud / cMud, uTb: sTb / cTb };
  }
  function tech(st) { const t = {}; st.done.forEach((k) => { t[k] = true; }); return t; }
  function software(st) { return st.done.includes("twin") ? 0.04 : st.done.includes("td") ? 0.06 : st.done.includes("fe") ? 0.09 : st.done.includes("sheet") ? 0.14 : 0.2; }
  function staffCount(st, role) { return st.staff.filter((x) => x.role === role).length; }
  function sigmas(st, plot) {
    const sm = software(st) * Math.pow(0.8, Math.min(2, staffCount(st, "structural")));
    const soil = (plot.surveyed ? 0.12 : 0.3) * Math.pow(0.7, Math.min(2, staffCount(st, "geotech")));
    const met = Math.pow(0.6, Math.min(1, staffCount(st, "metocean")));
    return { model: sm, soil, wave: (st.done.includes("buoy") ? 0.06 : 0.14) * met, sev: (st.done.includes("twin") ? 0.12 : st.done.includes("td") ? 0.2 : 0.32) * met };
  }
  // what the player sees: estimate and 90 % band for every check, from biased parameters and the tools' model error
  function evaluate(st, plotId, d) {
    const plot = st.plots[plotId], tb = TURBINES[d.turbine], sg = sigmas(st, plot), tk = tech(st);
    const est = { s: plot.s * Math.exp(plot.eps.s * sg.soil), H50: plot.H50 * Math.exp(plot.eps.w * sg.wave), sev: plot.sev * Math.exp(plot.eps.f * sg.sev), tech: tk };
    const c = checks(st, tb, plot, d, est), k = 1.645;
    const bias = (key) => Math.exp(plot.eps.m[key] * sg.model);
    // band half-widths in log terms: parameter sensitivity (finite differences) combined with the model error
    const sens = (fn, key, sig) => { const up = checks(st, tb, plot, d, Object.assign({}, est, { [key]: est[key] * Math.exp(sig) })); return Math.abs(Math.log(fn(up) / fn(c))); };
    const sStr = Math.hypot(sens((x) => x.strength, "H50", sg.wave), sg.model), sHold = Math.hypot(sens((x) => x.holding, "s", sg.soil), sens((x) => x.holding, "H50", sg.wave), sg.model);
    const sF = Math.hypot(sens((x) => x.f1, "s", sg.soil), sg.model * 0.35), sL = Math.hypot(sens((x) => x.life, "sev", sg.sev), sg.model * 2.5);
    const band = (v, sig) => ({ est: v, lo: v * Math.exp(-k * sig), hi: v * Math.exp(k * sig), sig });
    const out = {
      strength: Object.assign(band(c.strength * bias("str"), sStr), { label: "Strength", limit: 1, at: c.strengthAt, kind: "util" }),
      holding: Object.assign(band(c.holding * bias("hold"), sHold), { label: "Soil holding", limit: 1, kind: "util" }),
      freq: Object.assign(band(c.f1 * bias("freq") ** 0.35, sF), { label: "Natural frequency", lim: [c.w.lo, c.w.hi], w: c.w, kind: "freq" }),
      life: Object.assign(band(c.life / bias("life") ** 2.5, sL), { label: "Fatigue life", limit: 25, at: c.lifeAt, kind: "life" }),
    };
    const cost = costOf(st, plot, d, c);
    return { checks: out, cost, mass: c.mass, raw: c, sg, est };
  }
  function costOf(st, plot, d, c) {
    const tb = TURBINES[d.turbine], sc = STEELS[d.steel].cost;
    return Math.round(tb.price + sc * (STEEL_CREDIT.pile * c.g.mPile + STEEL_CREDIT.tower * c.g.mTower) + 250 + 6 * plot.depth);
  }
  // design storm test at a confidence level: "p50" uses the estimates, "p90" the unfavourable end of the one-sided 90 % band
  function stormTest(st, plotId, d, level) {
    const ev = evaluate(st, plotId, d), z = level === "p90" ? 1.2816 : 0, C = ev.checks;
    const val = (b) => b.est * Math.exp(z * b.sig), lifeV = C.life.est * Math.exp(-z * C.life.sig);
    const fv = C.freq.est, fLo = fv * Math.exp(-z * C.freq.sig), fHi = fv * Math.exp(z * C.freq.sig);
    const fails = [];
    if (val(C.strength) > 1) fails.push({ mode: C.strength.at === "tower base" ? "buckle-tower" : "buckle-pile", check: "strength", util: val(C.strength) });
    if (val(C.holding) > 1) fails.push({ mode: "overturn", check: "holding", util: val(C.holding) });
    if (fLo < C.freq.w.lo || fHi > C.freq.w.hi) fails.push({ mode: "resonance", check: "freq", util: fLo < C.freq.w.lo ? C.freq.w.lo / fLo : fHi / C.freq.w.hi });
    if (lifeV < 25) fails.push({ mode: "fatigue", check: "life", util: 25 / lifeV });
    fails.sort((a, b) => (a.check === "life") - (b.check === "life") || b.util - a.util);   // what breaks in the storm first, fatigue last
    return { ok: !fails.length, fails, strength: val(C.strength), holding: val(C.holding), f: [fLo, fHi], life: lifeV, ev };
  }
  // the truth: the same checks with the plot's real parameters and no model error
  function truth(st, plotId, d) { const plot = st.plots[plotId]; return checks(st, TURBINES[d.turbine], plot, d, { s: plot.s, H50: plot.H50, sev: plot.sev, tech: tech(st) }); }

  // ---------- new game ----------
  function newGame(seed = 1) {
    const st = { v: 1, seed, rng: seed >>> 0 || 1, day: 0, cash: 3500, rp: 0, done: [], staff: [{ role: "junior", id: 1 }], nextId: 2,
      plots: [], turbines: [], jobs: [], log: [], weather: null, price: PRICE, stats: { energy: 0, revenue: 0, spent: 0, failures: 0, lost: 0 }, unlockedTurbines: ["t6"] };
    // 12 plots in a 4 × 3 grid: deeper and windier offshore (columns), soils and storms vary
    for (let j = 0; j < 3; j++) for (let i = 0; i < 4; i++) {
      const depth = Math.round(18 + 9 * i + 4 * rnd(st) + 2 * j), clay = rnd(st) < 0.35;
      const p = { id: st.plots.length, i, j, depth, soil: clay ? "clay" : "sand", s: Math.exp(0.28 * nrm(st)) * (clay ? 0.85 : 1.0), H50: (6.5 + 0.09 * depth) * Math.exp(0.1 * nrm(st)), Hsm: 1.2 + 0.012 * depth,
        sev: Math.exp(0.2 * nrm(st)), wind: 1 + 0.035 * i + 0.01 * j, lease: Math.round(400 + 260 * i + 120 * j + 60 * rnd(st)), owned: i === 0 && j === 1, surveyed: false,
        eps: { s: nrm(st), w: nrm(st), f: nrm(st), m: { str: nrm(st), hold: nrm(st), freq: nrm(st), life: nrm(st) } } };
      st.plots.push(p);
    }
    st.weather = weatherNext(st, null);
    st.forecast = forecast(st);
    say(st, "Welcome. You own one plot. Design your first turbine in the Builder.");
    return st;
  }
  function say(st, msg, kind = "info") { st.log.unshift({ day: st.day, msg, kind }); if (st.log.length > 80) st.log.length = 80; }
  const season = (day) => ["Winter", "Spring", "Summer", "Autumn"][Math.floor((day % YEAR) / 30)];

  // ---------- weather ----------
  function weatherNext(st, prev) {
    const sIdx = Math.floor((st.day % YEAR) / 30), A = [11, 9.5, 7.5, 10][sIdx], stormP = [0.05, 0.02, 0.006, 0.035][sIdx];
    const z = prev ? 0.6 * prev.z + 0.8 * nrm(st) : nrm(st), U = Math.max(0.5, A * 0.89 * Math.exp(0.32 * z - 0.05));
    let Hs = Math.max(0.3, 0.3 + 0.014 * U * U + 0.25 * nrm(st)), storm = null;
    if (rnd(st) < stormP) { const r = Math.exp(-0.35 + 0.17 * nrm(st)); storm = { r }; Hs = Math.max(Hs, 7.5 * r); }   // r = 1: the design storm
    return { z, U: storm ? Math.max(U, 22 * storm.r) : U, Hs, storm, season: season(st.day) };
  }
  function forecast(st) {   // next three days: the weather process run forward from today, shown with spreading bands
    const save = st.rng, out = []; let w = st.weather;
    for (let k = 1; k <= 3; k++) { st.day += k; w = weatherNext(st, w); st.day -= k; out.push({ U: w.U, Hs: w.Hs, storm: !!w.storm, spread: 0.15 * k }); }
    st.rng = save; return out;
  }
  function powerAt(tb, U) { if (U < tb.cutIn || U > tb.cutOut) return 0; return tb.P * Math.min(1, Math.pow((U - tb.cutIn) / (tb.Ur - tb.cutIn), 2.4)); }
  function dayEnergy(tb, Umean) { let e = 0; const q = [-1.5, -0.75, 0, 0.75, 1.5], w = [0.07, 0.24, 0.38, 0.24, 0.07]; q.forEach((x, k) => { e += w[k] * powerAt(tb, Math.max(0, Umean * (1 + 0.25 * x))); }); return e * 24; }

  // ---------- actions ----------
  function canAfford(st, c) { return st.cash >= c; }
  function spend(st, c) { st.cash -= c; st.stats.spent += c; }
  function buyPlot(st, id) { const p = st.plots[id]; if (p.owned) return "Already yours."; if (!canAfford(st, p.lease)) return "Not enough credits."; spend(st, p.lease); p.owned = true; say(st, `Leased plot ${id + 1} (${p.depth} m, ${p.soil}).`); return null; }
  function survey(st, id) { const p = st.plots[id], cost = 150 + 3 * p.depth; if (p.surveyed || p.surveying) return "Already surveyed."; if (!canAfford(st, cost)) return "Not enough credits.";
    spend(st, cost); p.surveying = 3; say(st, `Geotechnical survey of plot ${id + 1} under way (3 days).`); return null; }
  function build(st, plotId, d) {
    const p = st.plots[plotId], ev = evaluate(st, plotId, d);
    if (!p.owned) return "Lease the plot first."; if (st.turbines.some((t) => t.plot === plotId && t.status !== "lost")) return "There is already a turbine here.";
    if (!st.unlockedTurbines.includes(d.turbine)) return "That turbine is not unlocked."; if (d.steel !== "S355" && !st.done.includes(d.steel)) return "That steel is not unlocked.";
    if (!canAfford(st, ev.cost)) return "Not enough credits.";
    spend(st, ev.cost);
    const tr = truth(st, plotId, d), crackAt = Math.exp(0.3 * nrm(st));
    st.turbines.push({ id: st.nextId++, plot: plotId, design: JSON.parse(JSON.stringify(d)), cost: ev.cost, status: "building", buildLeft: 3, true: { strength: tr.strength, holding: tr.holding, f1: tr.f1, inBand: tr.inBand, life: tr.life },
      predicted: { strength: ev.checks.strength.est, holding: ev.checks.holding.est, f1: ev.checks.freq.est, life: ev.checks.life.est },
      health: { blades: 1, drive: 1, elec: 1 }, known: {}, fatigue: 0, crackAt, crack: 0, sensors: false, down: null, daysUp: 0, energy: 0, name: "T" + String(st.turbines.length + 1).padStart(2, "0") });
    say(st, `Construction started on plot ${plotId + 1}: ${TURBINES[d.turbine].name}, ${Math.round(ev.mass)} t of steel, ${ev.cost} credits.`);
    return null;
  }
  // repower: take a turbine down for scrap (15 % of what it cost) to free the plot for a bigger machine
  function decommission(st, tid) {
    const t = st.turbines.find((x) => x.id === tid); if (!t || t.status === "lost" || t.status === "building") return "Nothing to take down.";
    const back = Math.round(0.15 * t.cost); st.cash += back; t.status = "lost"; t.lostWhy = "decommissioned"; st.jobs = st.jobs.filter((j) => j.tid !== tid);
    say(st, `${t.name} decommissioned for repowering; ${back} credits of scrap.`); return null;
  }
  function queue(st, job) { st.jobs.push(job); }
  function inspect(st, tid, method) {
    const t = st.turbines.find((x) => x.id === tid), m = INSPECT[method]; if (!t || t.status === "lost") return "No turbine.";
    if (m.need && !st.done.includes(m.need)) return "Not unlocked."; if (!canAfford(st, m.cost)) return "Not enough credits.";
    spend(st, m.cost); queue(st, { type: "inspect", tid, method, wait: 0 }); return null;
  }
  function fitSensors(st, tid) { const t = st.turbines.find((x) => x.id === tid); if (!st.done.includes("sensors")) return "Not unlocked."; if (t.sensors) return "Already fitted."; if (!canAfford(st, 200)) return "Not enough credits."; spend(st, 200); queue(st, { type: "sensors", tid, wait: 0 }); return null; }
  function repair(st, tid, comp) {
    const t = st.turbines.find((x) => x.id === tid); if (!t || t.status === "lost") return "No turbine.";
    const c = comp === "crack" ? { repair: 600, lead: 3, fix: 2 } : COMPS.find((x) => x.k === comp), failed = t.down && t.down.comp === comp;
    const cost = repairCost(t, comp, failed); if (!canAfford(st, cost)) return "Not enough credits.";
    if (st.jobs.some((j) => j.tid === tid && j.type === "repair" && j.comp === comp)) return "Already booked.";
    spend(st, cost); queue(st, { type: "repair", tid, comp, wait: failed ? c.lead : comp === "crack" ? 2 : 1, planned: !failed }); return null;
  }
  // repairs scale with the machine: a 15 MW drivetrain costs more than a 6 MW one; planned work is a third of the price
  function repairCost(t, comp, failed) { const c = comp === "crack" ? { repair: 600 } : COMPS.find((x) => x.k === comp); return Math.round(c.repair * Math.pow(TURBINES[t.design.turbine].P / 6, 0.8) * (failed ? 1 : 0.35)); }
  function research(st, id) {
    const r = RESEARCH.find((x) => x.id === id); if (!r || st.done.includes(id)) return "Done already."; if (!r.req.every((q) => st.done.includes(q))) return "Research its prerequisites first.";
    if (st.rp < r.rp) return "Not enough research points."; if (!canAfford(st, r.cost)) return "Not enough credits.";
    st.rp -= r.rp; spend(st, r.cost); st.done.push(id); if (TURBINES[id]) st.unlockedTurbines.push(id); say(st, `Unlocked: ${r.name}.`, "good"); st.forecast = forecast(st); return null;
  }
  function hire(st, role) { if (!ROLES[role]) return "Unknown role."; if (st.staff.length >= 8) return "The office is full."; if (!canAfford(st, 150)) return "Not enough credits for the hiring cost.";
    spend(st, 150); st.staff.push({ role, id: st.nextId++ }); say(st, `Hired a ${ROLES[role].name.toLowerCase()}.`); return null; }
  function fire(st, id) { st.staff = st.staff.filter((x) => x.id !== id); return null; }
  const vessels = (st) => ({ n: st.done.includes("ctv2") ? 2 : 1, limit: st.done.includes("sov") ? 2.5 : 1.5 });

  // ---------- one day ----------
  function advanceDay(st) {
    const ev = [], w = st.weather, tk = tech(st), live = st.turbines.filter((t) => t.status !== "lost");
    // surveys
    st.plots.forEach((p) => { if (p.surveying) { p.surveying--; if (!p.surveying) { p.surveyed = true; say(st, `Survey of plot ${p.id + 1} done: the soil band is much narrower now.`, "good"); } } });
    // construction (needs waves under 2 m)
    live.filter((t) => t.status === "building").forEach((t) => { if (w.Hs < 2.0) t.buildLeft--; if (t.buildLeft <= 0) { t.status = "running"; say(st, `${t.name} is generating.`, "good"); } });
    // energy and money
    const running = live.filter((t) => t.status === "running"), n = running.length, wake = n > 1 ? Math.min(0.14, 0.03 * Math.sqrt(n - 1)) * (tk.steer ? 0.6 : 1) : 0;
    let energy = 0;
    running.forEach((t) => { const tb = TURBINES[t.design.turbine], e = dayEnergy(tb, w.U * st.plots[t.plot].wind) * (1 - wake); t.energy += e; t.today = e; energy += e; t.daysUp++; });
    st.rep = Math.min(1, (st.rep || 1) + 0.0007);
    const revenue = energy * st.price * st.rep, pay = st.staff.reduce((a, x) => a + ROLES[x.role].pay, 0), om = live.filter((t) => t.status !== "building").reduce((a, t) => a + OM_DAY * TURBINES[t.design.turbine].P / 6, 0);
    st.cash += revenue - pay - om; st.stats.energy += energy; st.stats.revenue += revenue; st.today = { energy, revenue, pay, om, wake };
    st.rp += st.staff.reduce((a, x) => a + ROLES[x.role].rp, 0);
    // wear, fatigue, cracks
    const rough = Math.max(0.4, Math.min(3, (w.U / 9) ** 2 * 0.6 + w.Hs / 3));
    live.filter((t) => t.status !== "building").forEach((t) => {
      const res = t.true.inBand ? (tk.damper ? 1.25 : 1.5) : 1, op = t.status === "running";
      COMPS.forEach((c) => { if (t.down && t.down.comp === c.k) return; const rate = (1 / c.life) * rough * (op ? 1 : 0.3) * (c.k === "blades" && tk.ipc ? 0.75 : 1) * (c.k !== "elec" ? res : 1) * Math.exp(0.35 * nrm(st));
        t.health[c.k] = Math.max(0, t.health[c.k] - rate);
        if (t.health[c.k] <= 0 && !t.down) { t.down = { comp: c.k, since: st.day }; t.status = "down"; st.stats.failures++; say(st, `${t.name}: ${c.name.toLowerCase()} failed. The turbine has stopped.`, "bad"); ev.push({ type: "failure", tid: t.id, comp: c.k }); } });
      t.fatigue += WEAR * rough / (Math.max(0.5, t.true.life) * 365);
      if (!t.crack && t.fatigue >= t.crackAt) { t.crack = 1; ev.push({ type: "crack", tid: t.id }); if (t.sensors) say(st, `${t.name}: the sensors picked up a weld crack starting.`, "bad"); }
      if (t.crack) { t.crack++; if (t.crack > 45) lose(st, t, "the cracked weld gave way", ev); }
    });
    // storms load the real structures
    if (w.storm) {
      say(st, `Storm: waves ${w.Hs.toFixed(1)} m, wind ${w.U.toFixed(0)} m/s (${Math.round(100 * w.storm.r)} % of the design storm).`, "warn");
      live.filter((t) => t.status !== "lost").forEach((t) => {
        const k = Math.pow(w.storm.r, 1.25) * (t.status === "building" ? 0.6 : 1);
        if (t.true.strength * k > 1) lose(st, t, t.true.strength * k > t.true.holding * k ? "the structure buckled in the storm" : "the structure buckled in the storm", ev);
        else if (t.true.holding * k > 1) lose(st, t, "the pile pulled over in the storm", ev);
        else { COMPS.forEach((c) => { if (t.status !== "down") t.health[c.k] = Math.max(0, t.health[c.k] - 0.04 * w.storm.r * rnd(st)); }); }
      });
    }
    // vessel jobs: one per vessel per day, inside the weather window
    const V = vessels(st); let slots = V.n;
    st.jobs.forEach((j) => { if (j.wait > 0) j.wait--; });
    // vessels go to stopped turbines first, then cracks, then planned overhauls, then inspections
    const pri = (j) => { const t = st.turbines.find((x) => x.id === j.tid); return j.type === "repair" ? (t && t.down && t.down.comp === j.comp ? 0 : j.comp === "crack" ? 1 : 2) : 3; };
    st.jobs.sort((a, b) => pri(a) - pri(b));
    const left = [];
    for (const j of st.jobs) {
      const t = st.turbines.find((x) => x.id === j.tid);
      if (!t || t.status === "lost") continue;
      if (j.wait > 0 || slots <= 0 || w.Hs > V.limit) { left.push(j); continue; }
      slots--; doJob(st, j, t, ev);
    }
    st.jobs = left;
    if (w.Hs > V.limit && st.jobs.some((j) => j.wait <= 0)) say(st, `Waves ${w.Hs.toFixed(1)} m: vessels stay in port (limit ${V.limit} m).`, "warn");
    // sensors report continuously
    st.turbines.forEach((t) => { if (t.sensors && t.status !== "lost") { const sd = 0.06 * (staffCount(st, "data") ? 0.6 : 1); COMPS.forEach((c) => { if (c.k !== "blades") t.known[c.k] = { v: clamp01(t.health[c.k] + sd * nrm(st)), sd, day: st.day }; });
      if (st.done.includes("twin")) t.known.fatigue = { v: t.fatigue / Math.max(0.3, t.crackAt), day: st.day }; } });
    st.day++; st.weather = weatherNext(st, w); st.forecast = forecast(st);
    if (st.cash < -500) ev.push({ type: "broke" });
    return ev;
  }
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  function doJob(st, j, t, ev) {
    if (j.type === "inspect") {
      const m = INSPECT[j.method], sd = m.sd * (staffCount(st, "data") ? 0.6 : 1), found = [];
      m.comps.forEach((k) => { if (rnd(st) < m.detect || t.health[k] < 0.25) { t.known[k] = { v: clamp01(t.health[k] + sd * nrm(st)), sd, day: st.day }; if (t.health[k] < 0.35) found.push(COMPS.find((c) => c.k === k).name.toLowerCase()); } });
      if (t.crack && rnd(st) < m.structure) { t.known.crack = { day: st.day }; found.push("a weld crack"); }
      t.lastInspect = { day: st.day, method: j.method };
      say(st, `${t.name}: ${m.name.toLowerCase()} done${found.length ? `, found ${found.join(" and ")}` : ", nothing serious found"}.`, found.length ? "warn" : "info");
    } else if (j.type === "sensors") { t.sensors = true; say(st, `${t.name}: structural-health sensors fitted.`, "good"); }
    else if (j.type === "repair") {
      if (j.comp === "crack") { if (t.crack) { t.crack = 0; t.fatigue *= 0.5; t.crackAt *= 1.05; delete t.known.crack; say(st, `${t.name}: weld crack repaired.`, "good"); } else say(st, `${t.name}: divers found no crack; the weld was fine.`); return; }
      t.health[j.comp] = 1; t.known[j.comp] = { v: 1, sd: 0.02, day: st.day };
      if (t.down && t.down.comp === j.comp) { t.down = null; t.status = "running"; say(st, `${t.name}: ${COMPS.find((c) => c.k === j.comp).name.toLowerCase()} replaced; back online.`, "good"); }
      else say(st, `${t.name}: planned ${COMPS.find((c) => c.k === j.comp).name.toLowerCase()} overhaul done.`, "good");
    }
  }
  function lose(st, t, why, ev) { st.rep = Math.max(0.5, (st.rep || 1) - 0.1); const c = Math.max(300, Math.round(0.35 * t.cost)); t.status = "lost"; t.lostWhy = why; st.stats.lost++; st.cash -= c; st.stats.spent += c; say(st, `${t.name} lost: ${why}. Clearing the wreck costs ${c} credits, and the buyer cuts your price for a while (reputation ${Math.round(100 * st.rep)} %).`, "bad"); ev.push({ type: "lost", tid: t.id, why }); }

  // a reasonable starting design for a plot and turbine (deliberately not optimised)
  function starterDesign(st, plotId, turbine) {
    const tb = TURBINES[turbine], k = tb.D / 154, p = st.plots[plotId];
    return { turbine, steel: "S355", Db: +(5.6 * k).toFixed(1), Dt: +(3.8 * k).toFixed(1), tt: Math.round(30 * k), Dp: +(6.2 * k + 0.02 * p.depth).toFixed(1), tp: Math.round(55 * k + 0.4 * p.depth), Le: Math.round(4.2 * (6.2 * k)) };
  }
  const LIMITS = { Db: [3, 11], Dt: [2.5, 8], tt: [15, 110], Dp: [4, 13], tp: [30, 160], Le: [10, 70] };

  const api = { TURBINES, STEELS, COMPS, INSPECT, RESEARCH, ROLES, LIMITS, YEAR, newGame, repairCost, evaluate, stormTest, truth, build, buyPlot, survey, inspect, fitSensors, repair, decommission, research, hire, fire, advanceDay, starterDesign, vessels, season, powerAt, dayEnergy, sigmas };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.WindSteel = api;
})(typeof window !== "undefined" ? window : this);
