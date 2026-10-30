// FarmFlow: a dynamic farm-flow simulator in the spirit of FAST.Farm's dynamic wake meandering (DWM), on StormTwin's engine.
//
// Ambient wind: a 2-D field of u' and v' at hub height (spectral synthesis with a von Karman-type spectrum, integral length
// L), frozen and carried downwind at the mean speed (Taylor). Scales finer than the grid are added at each rotor as a
// rotor-effective random signal, so every turbine sees the ambient turbulence intensity in full.
//
// Turbines: reduced-order dynamic models with the single-turbine engine's aerodynamics and control (Heier Cp curve scaled to
// the type's peak Cp, thrust from momentum theory with Glauert's correction, variable-speed torque law, gain-scheduled pitch
// PI), driven by the rotor-averaged inflow and yaw (thrust ~ cos^2, power from the normal velocity).
//
// Wakes: each turbine releases passive wake planes every half rotor diameter of travel. A plane keeps the thrust coefficient
// and the effective turbulence of its turbine at release; its deficit is the Gaussian of Bastankhah & Porte-Agel (2014),
// sigma = k* x + eps D with k* = 0.3837 I + 0.003678 (Niayifar & Porte-Agel 2016), capped at the momentum-theory deficit in
// the near wake. Planes are carried downstream at the local wind speed less half their deficit and moved sideways by the
// large-scale crosswind averaged over a disc of 1.9 wake radii around them (wake meandering, as in FAST.Farm), plus the
// yaw-induced deflection of Jimenez et al. (2010). Deficits from different turbines add in quadrature; the wake-added
// turbulence of Crespo & Hernandez (1996) comes from the dominant upstream wake (Niayifar & Porte-Agel 2016) and is felt by
// downstream turbines as extra inflow turbulence.
//
// Every turbine also has a wake-free twin driven by the same ambient wind, so wake losses are measured, not estimated.
(function (root) {
  "use strict";
  const RHO = 1.225, D2R = Math.PI / 180;
  const heier = (lam, b) => { const li = 1 / (1 / (lam + 0.08 * b) - 0.035 / (b ** 3 + 1)); return Math.max(0, 0.5176 * (116 / li - 0.4 * b - 5) * Math.exp(-21 / li) + 0.0068 * lam); };
  const HEIER_MAX = (() => { let m = 0, l0 = 8; for (let l = 3; l < 14; l += 0.02) { const c = heier(l, 0); if (c > m) { m = c; l0 = l; } } return { cp: m, lam: l0 }; })();
  function ctFromCp(cp, lam) {
    const c = Math.min(cp, 0.5925); let a = 0.1;
    for (let i = 0; i < 30; i++) { const f = 4 * a * (1 - a) ** 2 - c, df = 4 * (1 - a) * (1 - 3 * a); a = Math.min(0.333, Math.max(0, a - f / (df || 1))); }
    let ct = 1.3 * 4 * a * (1 - a); if (lam > 1) ct += 0.05 * Math.max(0, lam - HEIER_MAX.lam) / 4; return Math.min(1.1, ct);
  }
  function rng(seed) { let s = (seed * 2654435761) >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; }; }
  function gauss(r) { return () => Math.sqrt(-2 * Math.log(r())) * Math.cos(2 * Math.PI * r()); }
  function fft(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) { const a = (inv ? 2 : -2) * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a), hl = len >> 1;
      for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < hl; k++) { const p = i + k, q = p + hl, tr = re[q] * cr - im[q] * ci, ti = re[q] * ci + im[q] * cr; re[q] = re[p] - tr; im[q] = im[p] - ti; re[p] += tr; im[p] += ti; const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr; } } }
    if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  const pow2 = (n) => { let p = 1; while (p < n) p <<= 1; return p; };

  // 2-D Gaussian random field with spectrum ~ (1 + (kL)^2)^(-4/3) (one-dimensional cuts fall off as k^-5/3), unit variance
  function field2d(ns, nn, h, L, g) {
    const N = ns * nn, re = new Float64Array(N), im = new Float64Array(N);
    for (let j = 0; j < nn; j++) for (let i = 0; i < ns; i++) {
      const ki = 2 * Math.PI * (i < ns / 2 ? i : i - ns) / (ns * h), kj = 2 * Math.PI * (j < nn / 2 ? j : j - nn) / (nn * h), k2 = ki * ki + kj * kj;
      const a = (i === 0 && j === 0) ? 0 : Math.pow(1 + k2 * L * L, -2 / 3);   // amplitude = sqrt of the spectrum
      re[j * ns + i] = a * g(); im[j * ns + i] = a * g();
    }
    // inverse FFT rows then columns
    const rr = new Float64Array(ns), ri = new Float64Array(ns);
    for (let j = 0; j < nn; j++) { for (let i = 0; i < ns; i++) { rr[i] = re[j * ns + i]; ri[i] = im[j * ns + i]; } fft(rr, ri, true); for (let i = 0; i < ns; i++) { re[j * ns + i] = rr[i]; im[j * ns + i] = ri[i]; } }
    const cr = new Float64Array(nn), ci = new Float64Array(nn);
    for (let i = 0; i < ns; i++) { for (let j = 0; j < nn; j++) { cr[j] = re[j * ns + i]; ci[j] = im[j * ns + i]; } fft(cr, ci, true); for (let j = 0; j < nn; j++) re[j * ns + i] = cr[j]; }
    let s2 = 0; for (let k = 0; k < N; k++) s2 += re[k] * re[k];
    const sc = 1 / Math.sqrt(s2 / N), out = new Float32Array(N); for (let k = 0; k < N; k++) out[k] = re[k] * sc;
    return out;
  }

  function addedTI(ct, I0, x) {                     // Crespo & Hernandez (1996); x in rotor diameters
    if (x <= 0.1) return 0;
    const a = 0.5 * (1 - Math.sqrt(1 - Math.min(0.95, ct)));
    return 0.73 * Math.pow(a, 0.8325) * Math.pow(Math.max(I0, 0.01), 0.0325) * Math.pow(Math.max(x, 2), -0.32);
  }

  function run(cfg, onProg) {
    const c = Object.assign({ U: 10, dir: 270, TI: 0.08, T: 1200, dt: 1, dtT: 0.2, seed: 1, L: 0, meander: true, addTI: true, superpose: "rss",
      outDt: 2, frameDt: 10, frameN: [160, 80], spin: true, yaw: null, Cm: 1.9 }, cfg);
    const tb = c.turbines, n = tb.length, U = c.U, r = rng(c.seed), g = gauss(r);
    const Dmax = Math.max(...tb.map((t) => t.ty.D));
    const L = c.L || 340;                            // integral length of u (m), IEC 61400-1 (8.1 x 42 m)
    // wind-aligned frame: s downstream, n to the left of the wind
    const to = (c.dir + 180) * D2R, es = [Math.sin(to), Math.cos(to)], en = [-es[1], es[0]];   // x east, y north
    const P = tb.map((t) => ({ s: t.x * es[0] + t.y * es[1], n: t.x * en[0] + t.y * en[1] }));
    const s0 = Math.min(...P.map((p) => p.s)), s1 = Math.max(...P.map((p) => p.s)), n0 = Math.min(...P.map((p) => p.n)), n1 = Math.max(...P.map((p) => p.n));
    const sMin = s0 - 3 * Dmax, sMax = s1 + 12 * Dmax, nMin = n0 - 6 * Dmax, nMax = n1 + 6 * Dmax;
    // ambient field: periodic downwind, long enough for the run; resolution about a third of a rotor diameter
    const h = Math.max(25, Math.min(80, Dmax / 3));
    const ns = pow2(Math.max(64, Math.ceil(Math.min(40000, sMax - sMin + U * c.T * 0.5) / h))), nn = pow2(Math.max(32, Math.ceil((nMax - nMin) / h)));
    const sigU = c.TI * U, sigV = 0.8 * sigU;
    const fu = field2d(ns, nn, h, L, g), fv = field2d(ns, nn, h, L, g);   // crosswind with the same length scale (offshore Mann-type fields)
    // the grid resolves scales above ~2h; the rest is added at each rotor with the remaining variance
    const kc = Math.PI / (2 * h), resolved = 1 - Math.pow(1 + kc * kc * L * L, -1 / 3);
    const sigHF = sigU * Math.sqrt(Math.max(0.05, 1 - resolved)), sRes = sigU * Math.sqrt(Math.min(0.95, resolved));
    const LS = ns * h, LN = nn * h;
    const samp = (F, s, nn2, t) => {                // bilinear, periodic; the field is frozen and moves downwind at U
      const x = ((s - sMin - U * t) % LS + LS) % LS / h, y = ((nn2 - nMin) % LN + LN) % LN / h;
      const i0 = Math.floor(x), j0 = Math.floor(y), fx = x - i0, fy = y - j0, i1 = (i0 + 1) % ns, j1 = (j0 + 1) % nn;
      return (F[j0 * ns + i0] * (1 - fx) + F[j0 * ns + i1] * fx) * (1 - fy) + (F[j1 * ns + i0] * (1 - fx) + F[j1 * ns + i1] * fx) * fy;
    };
    const uAmb = (s, nn2, t) => U + sRes * samp(fu, s, nn2, t), vAmb = (s, nn2, t) => sigV * Math.sqrt(Math.min(0.95, resolved)) * samp(fv, s, nn2, t);

    // ----- turbine models -----
    const yaw = c.yaw || tb.map(() => 0);
    const mk = (t, i, wake) => {
      const ty = t.ty, R = ty.D / 2, A = Math.PI * R * R, Pr = ty.P * 1e6, eta = 0.955, Omr = ty.rpm * Math.PI / 30, Ommin = (ty.rpmMin || 4) * Math.PI / 30;
      const cpS = ty.Cp / HEIER_MAX.cp, lamOpt = HEIER_MAX.lam, Kg = 0.5 * RHO * A * R ** 3 * ty.Cp / lamOpt ** 3, J = 3 * ty.bladeMass * R * R * 0.11 + (ty.hubMass || 0) * 4;
      return { i, wake, ty, R, A, Pr, eta, Omr, Ommin, cpS, lamOpt, Kg, J, Om: Math.min(Omr, Math.max(Ommin, lamOpt * U / R)), beta: 0, I: 0, Qg: 0,
        u: U, ct: 0.7, P: 0, Th: 0, tiEff: c.TI, dI: 0, parked: false };
    };
    const M = tb.map((t, i) => mk(t, i, true)), F0 = tb.map((t, i) => mk(t, i, false));
    // rotor-effective small-scale turbulence (unresolved ambient, wake-added), first-order processes shared by a turbine and its
    // wake-free twin so that the loss comparison sees the same gusts
    const noise = tb.map(() => ({ hf: 0, ad: 0 }));
    const stepNoise = (i, dtT) => { const tau = Math.max(2, 0.5 * tb[i].ty.D / Math.max(U, 1)), ar = Math.exp(-dtT / tau), sq = Math.sqrt(1 - ar * ar), z = noise[i]; z.hf = ar * z.hf + sq * g(); z.ad = ar * z.ad + sq * g(); };
    const stepT = (m, Uin, gam, dtT, tiExtra) => {
      const z = noise[m.i], u = Math.max(0, Uin + sigHF * 0.75 * z.hf + tiExtra * Uin * z.ad);      // 0.75: rotor averaging of the small scales
      m.u = u;
      const ty = m.ty, un = u * Math.pow(Math.cos(gam * D2R), 1.88 / 3);   // yawed power ~ cos^1.88 (field measurements), thrust ~ cos^2
      if (un < ty.cutIn || u > ty.cutOut) {        // parked or idling
        m.parked = true; m.Om = Math.max(0.02, m.Om * 0.98); m.beta = Math.min(90, m.beta + 8 * dtT); m.ct = 0.05; m.P = 0;
        m.Th = 0.5 * RHO * m.A * m.ct * u * u; return;
      }
      if (m.parked) { m.parked = false; m.beta = 0; m.I = 0; m.Om = Math.max(m.Ommin, m.lamOpt * un / m.R * 0.8); }
      const lam = m.Om * m.R / Math.max(un, 0.1), cp = m.cpS * heier(Math.max(0.5, lam), m.beta), ct = ctFromCp(cp, lam);
      const Qa = 0.5 * RHO * m.A * cp * un ** 3 / Math.max(m.Om, 0.05);
      const Qrated = m.Pr / (m.eta * m.Omr);
      m.Qg = m.Om >= m.Omr * 0.99 || m.beta > 0.5 ? Qrated : Math.min(Qrated, m.Kg * m.Om * m.Om);
      m.Om = Math.max(m.Ommin * 0.5, m.Om + (Qa - m.Qg) / m.J * dtT);
      const e = m.Om - m.Omr, gs = 1 / (1 + m.beta / 6);
      m.I = Math.min(1.5, Math.max(0, m.I + 0.3 * e * gs * dtT));
      const cmd = Math.max(0, Math.min(90, (2 * e * gs + m.I) / D2R)); m.beta += Math.max(-8 * dtT, Math.min(8 * dtT, cmd - m.beta));
      m.ct = Math.min(1.1, ct) * Math.cos(gam * D2R) ** 2; m.P = Math.min(m.Pr, m.Qg * m.Om * m.eta); m.Th = 0.5 * RHO * m.A * ct * un * un;
    };

    // ----- wake planes -----
    const planes = tb.map(() => []);                  // per turbine, oldest first (furthest downstream)
    const sigmaOf = (p, x) => { const b = (1 + Math.sqrt(1 - Math.min(0.95, p.ct))) / (2 * Math.sqrt(1 - Math.min(0.95, p.ct))), k = 0.3837 * p.ti + 0.003678; return (k * x / p.D + 0.2 * Math.sqrt(b)) * p.D; };
    const ampOf = (p, sg) => { const cmax = 1 - Math.sqrt(1 - Math.min(0.95, p.ct)), q = 1 - p.ct / (8 * (sg / p.D) ** 2); return Math.min(cmax, q > 0 ? 1 - Math.sqrt(q) : cmax); };
    const emit = (i, m) => planes[i].push({ s: P[i].s, n: P[i].n, se: P[i].s, ct: Math.max(0.02, m.ct), ti: m.tiEff, D: m.ty.D, yaw: yaw[i], x: 0 });
    // wake state of turbine j at downstream position s: centre, width, amplitude (interpolated between its two nearest planes)
    function wakeAt(j, s) {
      const pl = planes[j]; if (!pl.length || s <= P[j].s - 0.2 * pl[0].D) return null;
      let lo = 0, hi = pl.length - 1;              // planes are ordered from far (low index) to near (high index): s decreasing
      if (s < pl[hi].s) { const p = pl[hi], x = Math.max(0, s - p.se); return { n: p.n, sg: sigmaOf(p, x), p, x }; }   // between the rotor and the newest plane
      if (s > pl[0].s) return null;                  // beyond the oldest plane
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pl[mid].s >= s) lo = mid; else hi = mid; }
      const a = pl[lo], b = pl[hi], f = (a.s - s) / Math.max(1e-6, a.s - b.s), x = (1 - f) * a.x + f * b.x;
      const p = { ct: (1 - f) * a.ct + f * b.ct, ti: (1 - f) * a.ti + f * b.ti, D: a.D, yaw: a.yaw };
      return { n: (1 - f) * a.n + f * b.n, sg: sigmaOf(p, x), p, x };
    }
    // every wake crossing the line of constant s: centre, width, amplitude (looked up once per rotor or flow-map column)
    function wakesAtS(s, skip) {
      const L2 = [];
      for (let j = 0; j < n; j++) { if (j === skip) continue; const w = wakeAt(j, s); if (w) { w.amp = ampOf(w.p, w.sg); w.k2 = 1 / (2 * w.sg * w.sg); L2.push(w); } }
      return L2;
    }
    function deficitFrom(W, nn2, dz) {                // combined fractional deficit at a point from the wakes W
      let d2 = 0, dl = 0, dm = 0;
      for (const w of W) { const d = w.amp * Math.exp(-((nn2 - w.n) ** 2 + dz * dz) * w.k2); d2 += d * d; dl += d; if (d > dm) dm = d; }
      return c.superpose === "linear" ? Math.min(0.95, dl) : c.superpose === "max" ? dm : Math.min(0.95, Math.sqrt(d2));
    }
    function addedFrom(W, nn2) {                       // wake-added turbulence from the dominant upstream wake at a point
      let a = 0; if (!c.addTI) return 0;
      for (const w of W) a = Math.max(a, addedTI(w.p.ct, w.p.ti, w.x / w.p.D) * Math.exp(-((nn2 - w.n) ** 2) * w.k2 / 1.69));
      return a;
    }
    // rotor average over the disc: centre, 6 points at 0.45 R and 12 at 0.85 R, weighted by the areas they stand for; the ambient
    // field varies across (not up), the wakes are axisymmetric about hub height
    const RP = [[0, 0, 0.0625]];
    for (let q = 0; q < 6; q++) RP.push([0.45 * Math.cos(q * Math.PI / 3), 0.45 * Math.sin(q * Math.PI / 3), 0.3375 / 6]);
    for (let q = 0; q < 12; q++) RP.push([0.85 * Math.cos(q * Math.PI / 6 + 0.26), 0.85 * Math.sin(q * Math.PI / 6 + 0.26), 0.6 / 12]);
    function rotorInflow(i, t, wakes) {
      const R = tb[i].ty.D / 2, W = wakes ? wakesAtS(P[i].s, i) : null; let u = 0;
      for (const [qy, qz, w] of RP) { const nn2 = P[i].n + qy * R, ua = uAmb(P[i].s, nn2, t); u += w * ua * (W && W.length ? 1 - deficitFrom(W, nn2, qz * R) : 1); }
      return [u, W ? addedFrom(W, P[i].n) : 0];
    }
    function stepPlane(p, t, dt) {
          const sg = sigmaOf(p, p.x), amp = ampOf(p, sg), rm = c.Cm * sg;
          let ua = 0, va = 0; const K = [[0, 0], [rm, 0], [-rm, 0], [0, rm], [0, -rm]];
          for (const [ds, dn] of K) { ua += uAmb(p.s + ds, p.n + dn, t); if (c.meander) va += vAmb(p.s + ds, p.n + dn, t); }
          ua /= K.length; va /= K.length;
          // yaw-induced deflection (Jimenez): skew angle decaying with distance
          let vy = 0; if (p.yaw) { const k = 0.3837 * p.ti + 0.003678, xi0 = 0.3 * p.yaw * D2R / Math.cos(p.yaw * D2R) * (1 - Math.sqrt(Math.max(0, 1 - p.ct))); vy = ua * Math.tan(xi0 * (p.D / (p.D + 2 * k * p.x)) ** 2); }
          const uc = ua * (1 - 0.5 * amp);
          p.s += uc * dt; p.n += (va + vy) * dt; p.x += uc * dt;
    }
    function movePlanes(t, dt) {
      for (let j = 0; j < n; j++) {
        const pl = planes[j];
        for (const p of pl) stepPlane(p, t, dt);
        while (pl.length && pl[0].s > sMax) pl.shift();
      }
    }

    // ----- time loop -----
    const sub = Math.max(1, Math.round(c.dt / c.dtT)), dtT = c.dt / sub;
    // spin-up: long enough for the wakes to cross the farm (all of the domain when frames are wanted)
    const spinT = c.spinT !== undefined ? c.spinT : c.prefill ? 60 : c.spin ? Math.min(2400, ((c.frames === false ? s1 - s0 + 4 * Dmax : sMax - sMin)) / Math.max(0.7 * U, 2)) : 0;
    const dEmit = 0.5 * Dmax / Math.max(U, 2);
    const nOut = Math.floor(c.T / c.outDt) + 1, out = { t: new Float32Array(nOut), Pfarm: new Float32Array(nOut), Pfree: new Float32Array(nOut) };
    const per = tb.map(() => ({ u: new Float32Array(nOut), P: new Float32Array(nOut), Th: new Float32Array(nOut), ti: new Float32Array(nOut), def: new Float32Array(nOut), rpm: new Float32Array(nOut), pitch: new Float32Array(nOut), Pfree: new Float32Array(nOut), ufree: new Float32Array(nOut) }));
    const [FW, FH] = c.frameN, frames = [], fs = [sMin, sMax], fn = [nMin, nMax];
    let lastEmit = -1e9, k = 0, kOut = 0;
    if (c.prefill) {                                 // statistics runs: warm the controllers up in free wind, then lay every wake out straight
      for (let q = 0; q < Math.round(60 / dtT); q++) for (let i = 0; i < n; i++) { stepNoise(i, dtT); stepT(M[i], U, yaw[i], dtT, 0); stepT(F0[i], U, 0, dtT, 0); }
      // each plane is released at the time it would have left its rotor and carried through the frozen field up to t = 0,
      // so the wakes start out meandered
      for (let i = 0; i < n; i++) for (let te = -Math.ceil((sMax - P[i].s) / (0.85 * U) / dEmit) * dEmit; te < 0; te += dEmit) {
        const p = { s: P[i].s, n: P[i].n, se: P[i].s, ct: Math.max(0.02, M[i].ct), ti: c.TI, D: tb[i].ty.D, yaw: yaw[i], x: 0 };
        for (let tt = te; tt < 0; tt += 2) stepPlane(p, tt, Math.min(2, -tt));
        if (p.s <= sMax) planes[i].push(p);
      }
      // waked turbines shed wakes with their raised turbulence (faster recovery): two passes so deeper rows see the change
      for (let it = 0; it < 2; it++) for (let i = 0; i < n; i++) { const dI = rotorInflow(i, 0, true)[1], ti = Math.sqrt(c.TI * c.TI + dI * dI); M[i].tiEff = ti; planes[i].forEach((p) => { p.ti = ti; }); }
    }
    const nSteps = Math.round((spinT + c.T) / c.dt);
    for (let st = 0; st <= nSteps; st++) {
      const t = st * c.dt - spinT, tf = t + spinT;  // tf: time for the frozen field (starts at 0)
      // inflow and turbine dynamics
      for (let i = 0; i < n; i++) {
        const [uw, dI] = rotorInflow(i, tf, true), [uf] = rotorInflow(i, tf, false);
        M[i].dI = dI; M[i].tiEff = Math.sqrt(c.TI * c.TI + dI * dI);
        for (let q = 0; q < sub; q++) { stepNoise(i, dtT); stepT(M[i], uw, yaw[i], dtT, c.addTI ? dI : 0); stepT(F0[i], uf, 0, dtT, 0); }
      }
      if (tf - lastEmit >= dEmit - 1e-9) { for (let i = 0; i < n; i++) emit(i, M[i]); lastEmit = tf; }
      movePlanes(tf, c.dt);
      if (t >= kOut * c.outDt - 1e-6 && kOut < nOut) {
        out.t[kOut] = t; let pf = 0, pr = 0;
        for (let i = 0; i < n; i++) { const m = M[i], o = per[i]; o.u[kOut] = m.u; o.P[kOut] = m.P / 1e6; o.Th[kOut] = m.Th / 1e6; o.ti[kOut] = m.tiEff; o.rpm[kOut] = m.Om * 30 / Math.PI; o.pitch[kOut] = m.beta; o.Pfree[kOut] = F0[i].P / 1e6; o.ufree[kOut] = F0[i].u; pf += m.P; pr += F0[i].P; }
        out.Pfarm[kOut] = pf / 1e6; out.Pfree[kOut] = pr / 1e6; kOut++;
      }
      if (c.frames !== false && t >= frames.length * c.frameDt - 1e-6 && frames.length * c.frameDt <= c.T + 1e-6) {
        const fr = new Float32Array(FW * FH);
        for (let ii = 0; ii < FW; ii++) { const sx = sMin + (ii + 0.5) / FW * (sMax - sMin), W = wakesAtS(sx, -1);
          for (let jj = 0; jj < FH; jj++) { const nn2 = nMin + (jj + 0.5) / FH * (nMax - nMin); fr[jj * FW + ii] = uAmb(sx, nn2, tf) * (1 - (W.length ? deficitFrom(W, nn2, 0) : 0)) / U; } }
        frames.push({ t, f: fr, wc: planes.map((pl) => pl.filter((_, q) => q % 2 === 0).map((p) => [p.s, p.n])) });
      }
      if (onProg && st % 50 === 0) onProg(st / nSteps);
    }
    out.n = kOut;
    // statistics
    const mean = (a) => { let s2 = 0; for (let q = 0; q < kOut; q++) s2 += a[q]; return s2 / Math.max(1, kOut); };
    const std = (a) => { const m = mean(a); let s2 = 0; for (let q = 0; q < kOut; q++) s2 += (a[q] - m) ** 2; return Math.sqrt(s2 / Math.max(1, kOut)); };
    const stats = tb.map((t, i) => { const o = per[i], Pm = mean(o.P), Pf = mean(o.Pfree), um = mean(o.u);
      return { name: t.name, P: Pm, Pfree: Pf, loss: Pf > 0 ? 1 - Pm / Pf : 0, u: um, ufree: mean(o.ufree), TIrotor: std(o.u) / Math.max(um, 0.1), TIeff: mean(o.ti), Th: mean(o.Th), ThStd: std(o.Th) }; });
    const Pm = mean(out.Pfarm), Pf = mean(out.Pfree);
    return { cfg: c, frame: { es, en, sMin, sMax, nMin, nMax, FW, FH }, pos: P, out, per, frames, stats, farm: { P: Pm, Pfree: Pf, loss: Pf > 0 ? 1 - Pm / Pf : 0, eff: Pf > 0 ? Pm / Pf : 1 },
      grid: { h, ns, nn, L, resolved }, sigma: { u: sigU, hf: sigHF } };
  }

  const api = { run, heier, ctFromCp, addedTI, HEIER_MAX };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.FarmFlow = api;
})(typeof window !== "undefined" ? window : this);
