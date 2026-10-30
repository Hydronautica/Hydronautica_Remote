// Headless balance check: two bots play a year (360 days) on several seeds.
//   careful:  designs to pass the P90 storm test, inspects every 25 days, overhauls worn parts, fixes cracks
//   reckless: designs to pass at P50 only and repairs only what has broken
// node demo/windsim/balance_test.js
const W = require("./core.js");

function size(st, pid, tk, level) {           // grow the design until the storm test passes (what a player does by hand)
  const d = W.starterDesign(st, pid, tk), L = W.LIMITS;
  for (let it = 0; it < 600; it++) {
    const t = W.stormTest(st, pid, d, level); if (t.ok) return d;
    const f = t.fails[0];
    if (f.check === "strength") { if (t.ev.checks.strength.at === "tower base") { d.tt = Math.min(L.tt[1], d.tt + 2); d.Db = Math.min(L.Db[1], +(d.Db + 0.05).toFixed(2)); } else d.tp = Math.min(L.tp[1], d.tp + 3); }
    else if (f.check === "holding") d.Le = Math.min(L.Le[1], d.Le + 1);
    else if (f.check === "freq") { const C = t.ev.checks.freq; if (t.f[0] < C.w.lo) { d.Dp = Math.min(L.Dp[1], +(d.Dp + 0.1).toFixed(2)); d.Db = Math.min(L.Db[1], +(d.Db + 0.1).toFixed(2)); } else { d.Dp = Math.max(L.Dp[0], +(d.Dp - 0.1).toFixed(2)); d.Db = Math.max(L.Db[0], +(d.Db - 0.1).toFixed(2)); } }
    else if (f.check === "life") { d.tt = Math.min(L.tt[1], d.tt + 2); d.tp = Math.min(L.tp[1], d.tp + 2); }
  }
  return null;
}

function play(seed, style, days = 360) {
  const st = W.newGame(seed), level = style === "careful" ? "p90" : "p50";
  const order = ["sheet", "drones", "t10", "S420", "fe", "ctv2", "weld", "ipc", "rov", "buoy", "t15", "td", "sensors", "steer", "damper", "S460", "sov", "twin", "t20"];
  for (let day = 0; day < days; day++) {
    // design and build on free plots, biggest unlocked turbine first
    st.plots.filter((p) => p.owned && !st.turbines.some((t) => t.plot === p.id && t.status !== "lost")).forEach((p) => {
      if (style === "careful" && !p.surveyed && !p.surveying && st.cash > 600) W.survey(st, p.id);
      if (style === "careful" && p.surveying) return;
      for (const tk of [...st.unlockedTurbines].reverse()) { const d = size(st, p.id, tk, level); if (!d) continue; if (st.done.includes("S420")) d.steel = st.done.includes("S460") ? "S460" : "S420"; const ok = size(st, p.id, tk, level) && d; if (ok && W.evaluate(st, p.id, d).cost < st.cash - 300) { W.build(st, p.id, d); break; } }
    });
    // grow: lease the cheapest plot when rich
    const free = st.plots.filter((p) => !p.owned).sort((a, b) => a.lease - b.lease)[0];
    if (free && st.cash > free.lease + 3000) W.buyPlot(st, free.id);
    // research and staff
    const next = order.find((id) => !st.done.includes(id)); if (next) W.research(st, next);
    if (st.staff.length < 2 + Math.floor(st.turbines.length / 2) && st.cash > 2500) W.hire(st, ["structural", "geotech", "metocean", "data", "junior"][st.staff.length % 5]);
    // maintenance
    st.turbines.filter((t) => t.status !== "lost").forEach((t) => {
      if (t.down && !st.jobs.some((j) => j.tid === t.id && j.comp === t.down.comp)) W.repair(st, t.id, t.down.comp);
      if (style !== "careful") return;
      if ((!t.lastInspect || st.day - t.lastInspect.day > 25) && !st.jobs.some((j) => j.tid === t.id && j.type === "inspect")) W.inspect(st, t.id, st.done.includes("drones") ? "full" : "visual");
      Object.entries(t.known).forEach(([k, v]) => { if (k !== "crack" && k !== "fatigue" && v.v < 0.35 && !st.jobs.some((j) => j.tid === t.id && j.comp === k)) W.repair(st, t.id, k); });
      if (t.known.crack && !st.jobs.some((j) => j.tid === t.id && j.comp === "crack")) W.repair(st, t.id, "crack");
      if (st.done.includes("rov") && (!t.lastRov || st.day - t.lastRov > 60) && t.fatigue > 0.4) { W.inspect(st, t.id, "rov"); t.lastRov = st.day; }
      if (st.done.includes("sensors") && !t.sensors && st.cash > 1500) W.fitSensors(st, t.id);
      const best = W.TURBINES[st.unlockedTurbines[st.unlockedTurbines.length - 1]];
      if (t.status === "running" && W.TURBINES[t.design.turbine].P < best.P && st.cash > 3 * best.price + 6000) W.decommission(st, t.id);   // repower
    });
    W.advanceDay(st);
  }
  const live = st.turbines.filter((t) => t.status !== "lost");
  return { seed, style, cash: Math.round(st.cash), turbines: live.length, MW: live.reduce((a, t) => a + W.TURBINES[t.design.turbine].P, 0), lost: st.stats.lost, failures: st.stats.failures,
    energyGWh: +(st.stats.energy / 1000).toFixed(1), research: st.done.length };
}

const rows = [];
for (const style of ["careful", "reckless"]) for (const seed of [1, 2, 3, 4, 5]) rows.push(play(seed, style, 720));
console.table(rows);
const avg = (style, k) => rows.filter((r) => r.style === style).reduce((a, r) => a + r[k], 0) / 5;
console.log(`careful: cash ${avg("careful", "cash").toFixed(0)}, ${avg("careful", "MW").toFixed(0)} MW, lost ${avg("careful", "lost")}, failures ${avg("careful", "failures")}`);
console.log(`reckless: cash ${avg("reckless", "cash").toFixed(0)}, ${avg("reckless", "MW").toFixed(0)} MW, lost ${avg("reckless", "lost")}, failures ${avg("reckless", "failures")}`);
module.exports = { play, size };
