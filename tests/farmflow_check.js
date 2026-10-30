// Physics checks of demo/stormtwin/farmflow.js; prints one JSON object (read by tests/test_farmflow.py)
const FF = require("../demo/stormtwin/farmflow.js");
const ty = { D: 240, P: 15, Cp: 0.48, rpm: 7.56, rpmMin: 5, bladeMass: 65000, hubMass: 190000, cutIn: 3, cutOut: 25 };
const row = (n, sp) => Array.from({ length: n }, (_, i) => ({ name: "T" + (i + 1), x: i * sp * ty.D, y: 0, ty }));
const out = {};
// time-averaged hub-height profile across the wake at x diameters downstream of the first turbine (frames are in the wind frame)
function profile(r, xd) {
  const fm = r.frame, i = Math.round(((r.pos[0].s + xd * ty.D) - fm.sMin) / (fm.sMax - fm.sMin) * fm.FW - 0.5), prof = new Float64Array(fm.FH);
  r.frames.forEach((f) => { for (let j = 0; j < fm.FH; j++) prof[j] += f.f[j * fm.FW + i] / r.frames.length; });
  const n = Array.from({ length: fm.FH }, (_, j) => fm.nMin + (j + 0.5) / fm.FH * (fm.nMax - fm.nMin));
  return { n, def: Array.from(prof, (v) => 1 - v), dn: (fm.nMax - fm.nMin) / fm.FH };
}
// 1) static wake vs the analytic Gaussian (no turbulence, no meandering)
let r = FF.run({ turbines: row(1, 7), U: 8, dir: 270, TI: 1e-4, T: 400, meander: false, seed: 1, frameN: [240, 160] });
const ct = r.stats[0].Th * 1e6 / (0.5 * 1.225 * Math.PI * 120 * 120 * r.stats[0].u ** 2);
let p = profile(r, 7), j0 = p.def.indexOf(Math.max(...p.def));
out.static = { ct, centre: p.def[j0], nAtMax: p.n[j0], integral: p.def.reduce((a, b) => a + b, 0) * p.dn };
// 2) meandering with 8 % turbulence: same farm, time-averaged
r = FF.run({ turbines: row(1, 7), U: 8, dir: 270, TI: 0.08, T: 1800, meander: true, seed: 2, frameN: [240, 160], frameDt: 5 });
const rs = FF.run({ turbines: row(1, 7), U: 8, dir: 270, TI: 0.08, T: 1800, meander: false, seed: 2, frameN: [240, 160], frameDt: 5 });
const pm = profile(r, 7), ps = profile(rs, 7);
// wander of the wake centre at 7 D, from the wake planes' centres (frames keep every second plane)
const s7 = r.pos[0].s + 7 * ty.D;
const cen = r.frames.map((f) => { const wc = f.wc[0]; let k = 0; while (k < wc.length - 1 && wc[k + 1][0] >= s7) k++; const a = wc[k], b = wc[Math.min(k + 1, wc.length - 1)], fr = a[0] === b[0] ? 0 : (a[0] - s7) / (a[0] - b[0]); return a[1] + fr * (b[1] - a[1]); });
const cm = cen.reduce((a, b) => a + b, 0) / cen.length, wander = Math.sqrt(cen.reduce((a, b) => a + (b - cm) ** 2, 0) / cen.length) / ty.D;
out.meander = { centre: Math.max(...pm.def), integral: pm.def.reduce((a, b) => a + b, 0) * pm.dn, centreStatic: Math.max(...ps.def), integralStatic: ps.def.reduce((a, b) => a + b, 0) * ps.dn, wander };
// 3) rows at 7D
for (const TI of [0.06, 0.10]) { r = FF.run({ turbines: row(5, 7), U: 8, dir: 270, TI, T: 1200, seed: 3 }); out["row" + TI] = { ratios: r.stats.map((s) => s.P / r.stats[0].P), eff: r.farm.eff, frontLoss: r.stats[0].loss }; }
// 4) yaw deflection
r = FF.run({ turbines: row(1, 7), U: 8, dir: 270, TI: 1e-4, T: 400, meander: false, seed: 1, yaw: [25], frameN: [240, 160] });
p = profile(r, 7); j0 = p.def.indexOf(Math.max(...p.def)); out.yaw = { nAtMax: p.n[j0] / ty.D };
// 5) pre-filled statistics runs agree with full spin-up (aligned 12-turbine grid, average of 3 seeds)
{ const grid = []; for (let j = 0; j < 3; j++) for (let i = 0; i < 4; i++) grid.push({ name: "g", x: i * 1800, y: -j * 1600, ty });
  let a = 0, b = 0; for (let seed = 1; seed <= 3; seed++) { a += FF.run({ turbines: grid, U: 9, dir: 270, TI: 0.077, T: 240, frames: false, seed }).farm.loss / 3; b += FF.run({ turbines: grid, U: 9, dir: 270, TI: 0.077, T: 240, frames: false, prefill: true, seed }).farm.loss / 3; }
  out.prefill = { spin: a, prefill: b }; }
// 6) speed
const t0 = Date.now(); const tb = []; for (let j = 0; j < 3; j++) for (let i = 0; i < 4; i++) tb.push({ name: "x", x: i * 7 * 240, y: j * 5 * 240, ty });
FF.run({ turbines: tb, U: 9, dir: 260, TI: 0.07, T: 600, seed: 2 }); out.ms12x600 = Date.now() - t0;
console.log(JSON.stringify(out));
