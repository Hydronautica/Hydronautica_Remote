// Offshore Wind Developer: rendering and interface. All rules live in sim.js (WindSim); this file draws the
// state, turns clicks into commands and runs the clock.
(function () {
  "use strict";
  const S = window.WindSim, $ = (id) => document.getElementById(id), css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v, d = 1) => (v === null || v === undefined || !isFinite(v) ? "–" : Number(v).toFixed(d));
  const eur = (m) => (Math.abs(m) >= 100 ? "€" + fmt(m, 0) + " M" : "€" + fmt(m, 1) + " M");
  let G = S.newGame(Math.floor(Math.random() * 1e6) + 1), speed = 0, placing = false, sel = null, lastYearShown = null, finalShown = false;
  const cmd = (name, ...a) => { const r = S.command(G, name, ...a); if (!r.ok && r.msg) toast(r.msg); dirty = true; return r; };

  // ---------- turbine meshes, in metres: shared by the farm view and the builder ----------
  // Monopile and tower are lathed from the type's diameter stations (tapers and cones show as built);
  // blades are lofted from a root cylinder into a cambered airfoil with chord, thickness and twist
  // varying along the span, with pre-bend, cone and shaft tilt.
  const TM = (() => {
    const cache = new Map();
    const smooth = (a, b, x) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };
    function bladeGeo(R, rHub) {
      const key = R.toFixed(1) + ":" + rHub.toFixed(2); if (cache.has(key)) return cache.get(key);
      const L = R - rHub, NP = 30, dRoot = Math.max(0.034 * R, 2.4), cMax = 0.068 * R;
      const span = []; for (let i = 0; i <= 40; i++) { const u = i / 40; span.push(u < 0.3 ? u * u / 0.3 * 0.5 + u * 0.5 : u); }   // denser at the root
      const pos = [], idx = [];
      span.forEach((s) => {
        const rnd = 1 - smooth(0.015, 0.2, s);                                   // 1: circular root, 0: airfoil
        let c = s < 0.2 ? dRoot + (cMax - dRoot) * smooth(0.02, 0.2, s) : cMax * (1 - 0.8 * Math.pow((s - 0.2) / 0.8, 1.05));
        if (s > 0.965) c *= 0.4 + 0.6 * (1 - s) / 0.035;
        const tc = s < 0.2 ? 0.36 : 0.15 + 0.21 * Math.exp(-(s - 0.2) / 0.16);
        const tw = (s < 0.18 ? 13 : 13 * Math.pow((1 - s) / 0.82, 2)) * Math.PI / 180, ct = Math.cos(tw), st = Math.sin(tw);
        const px = 0.5 * rnd + 0.33 * (1 - rnd), bend = 0.035 * L * s * s;
        for (let j = 0; j < NP; j++) {
          const u = 2 * Math.PI * j / NP, x = 0.5 * (1 + Math.cos(u)), up = j <= NP / 2 ? 1 : -1;
          const yt = 5 * tc * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
          const yc = x < 0.4 ? 0.03 / 0.16 * (0.8 * x - x * x) : 0.03 / 0.36 * (0.2 + 0.8 * x - x * x);
          const ya = yc + up * yt, yr = 0.5 * Math.sin(u);
          const cx = (x - px) * c, cz = (rnd * yr + (1 - rnd) * ya) * c;
          pos.push(cx * ct - cz * st, rHub + s * L, cx * st + cz * ct + bend);
        }
      });
      const ns = span.length;
      for (let i = 0; i < ns - 1; i++) for (let j = 0; j < NP; j++) { const a = i * NP + j, b = i * NP + (j + 1) % NP, c2 = a + NP, d = b + NP; idx.push(a, c2, b, b, c2, d); }
      const tip = pos.length / 3; let sx = 0, sz = 0; for (let j = 0; j < NP; j++) { sx += pos[((ns - 1) * NP + j) * 3]; sz += pos[((ns - 1) * NP + j) * 3 + 2]; }
      pos.push(sx / NP, R + 0.05, sz / NP); for (let j = 0; j < NP; j++) idx.push((ns - 1) * NP + j, tip, (ns - 1) * NP + (j + 1) % NP);
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
      cache.set(key, g); return g;
    }
    function lathe(zs, rOf, seg = 40) { return new THREE.LatheGeometry(zs.map((z) => new THREE.Vector2(Math.max(0.05, rOf(z)), z)), seg); }
    function roundRect(w, h, r) {
      const s = new THREE.Shape(), x = -w / 2, y = -h / 2;
      s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r); s.lineTo(x + w, y + h - r); s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      s.lineTo(x + r, y + h); s.quadraticCurveTo(x, y + h, x, y + h - r); s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y); return s;
    }
    // m: turbine description (metaFor); mats: {white, blade, grey, yellow, dark}; zBot: lowest elevation drawn
    function build(m, mats, zBot) {
      const g = new THREE.Group(), Dz = m.Dz, zi = m.z_interface, zt = m.z_hub - 5, R = m.rotor_D / 2, Dt = Dz(zt);
      const zsP = []; for (let i = 0; i <= 40; i++) zsP.push(zBot + (zi - zBot) * i / 40);
      const pile = new THREE.Mesh(lathe(zsP, (z) => Dz(Math.min(z, zi - 0.01)) / 2), mats.grey); g.add(pile);
      const zsT = []; for (let i = 0; i <= 40; i++) zsT.push(zi + (zt + 1 - zi) * i / 40);
      const tower = new THREE.Mesh(lathe(zsT, (z) => Dz(Math.max(z, zi + 0.01)) / 2), mats.white); g.add(tower);
      const rTP = Dz(zi - 0.01) / 2 + 0.15;
      const tp = new THREE.Mesh(new THREE.CylinderGeometry(rTP, rTP, 5, 40), mats.yellow); tp.position.y = zi - 2.4; g.add(tp);
      const zp = m.z_platform, rP = Math.max(rTP, Dz(zp) / 2) + 3.2;
      const plat = new THREE.Group(); plat.position.y = zp; g.add(plat);
      const deck = new THREE.Mesh(new THREE.CylinderGeometry(rP, rP, 0.5, 48), mats.grey); plat.add(deck);
      const rail = new THREE.Mesh(new THREE.TorusGeometry(rP - 0.1, 0.09, 6, 64), mats.yellow); rail.rotation.x = Math.PI / 2; rail.position.y = 1.2; plat.add(rail);
      const landing = []; [-1, 1].forEach((sx) => { const b = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, zp + 4, 10), mats.yellow); b.position.set(sx * 1.3, -(zp + 4) / 2 + 0.2, -(rTP + 1.6)); plat.add(b); landing.push(b); });
      const doorZ = m.z_door, rd = Dz(doorZ) / 2;
      const door = new THREE.Mesh(new THREE.BoxGeometry(1.2, 2.3, 0.4), mats.dark); door.position.set(0, doorZ, -rd); g.add(door);
      // nacelle, spinner and rotor
      const yaw = new THREE.Group(); yaw.position.y = m.z_hub; g.add(yaw);
      const wN = 1.45 * Dt, hN = 1.3 * Dt, lN = 3.1 * Dt;
      const ng = new THREE.ExtrudeGeometry(roundRect(wN, hN, 0.24 * hN), { depth: lN, bevelEnabled: true, bevelThickness: 0.5, bevelSize: 0.4, bevelSegments: 3, curveSegments: 6 });
      ng.translate(0, 0, -0.62 * lN);
      const nac = new THREE.Mesh(ng, mats.white); nac.position.y = 0.15 * hN; yaw.add(nac);
      const cool = new THREE.Mesh(new THREE.BoxGeometry(0.7 * wN, 0.22 * hN, 0.18 * lN), mats.grey); cool.position.set(0, 0.15 * hN + hN / 2 + 0.11 * hN, -0.5 * lN); yaw.add(cool);
      const tilt = new THREE.Group(); tilt.position.z = 0.38 * lN + 0.6; tilt.rotation.x = -6 * Math.PI / 180; yaw.add(tilt);
      const rotor = new THREE.Group(); rotor.position.z = 0.04 * R + 1.5; tilt.add(rotor);
      const Rs = 0.026 * R + 0.8, Ls = 1.5 * Rs, prof = [];
      for (let i = 0; i <= 16; i++) { const a = i / 16; prof.push(new THREE.Vector2(Math.max(0.02, Rs * Math.sin(a * Math.PI / 2) ** 0.8), Ls * (1 - a) - 0.45 * Ls)); }
      prof.push(new THREE.Vector2(Rs, -0.8 * Ls));
      const sg = new THREE.LatheGeometry(prof, 36); sg.rotateX(Math.PI / 2);
      const hub = new THREE.Mesh(sg, mats.white); rotor.add(hub);
      const blades = [], bg = bladeGeo(R, 0.85 * Rs);
      for (let b = 0; b < 3; b++) {
        const piv = new THREE.Group(); piv.rotation.z = b * 2 * Math.PI / 3; rotor.add(piv);
        const bl = new THREE.Mesh(bg, mats.blade); bl.rotation.x = 4 * Math.PI / 180; piv.add(bl); blades.push(bl);
      }
      return { g, yaw, rotor, nac, hub, blades, tower, tp, plat: deck, pile, door, extras: [cool, rail, ...landing] };
    }
    return { build, bladeGeo };
  })();

  // ---------- scene ----------
  const U = 10, TS = 3, VE = 2.5;                      // metres per scene unit; turbines and depth exaggerated for legibility
  const host = $("view"), renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1)); host.append(renderer.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 1, 8000), controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.maxPolarAngle = Math.PI / 2 - 0.05; controls.minDistance = 30; controls.maxDistance = 2600;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 0.8)); const sun = new THREE.DirectionalLight(0xffffff, 0.85); sun.position.set(-400, 600, 300); scene.add(sun);
  const kmX = (x) => x * 1000 / U - G.map.W * 500 / U, kmZ = (y) => y * 1000 / U - G.map.H * 500 / U;
  const mats = { white: new THREE.MeshStandardMaterial({ color: 0xe9edf0, roughness: 0.55 }), blade: new THREE.MeshStandardMaterial({ color: 0xf6f7f8, roughness: 0.45 }),
    grey: new THREE.MeshStandardMaterial({ color: 0x8b98a3, roughness: 0.7 }), yellow: new THREE.MeshStandardMaterial({ color: 0xf2c230, roughness: 0.6 }), dark: new THREE.MeshStandardMaterial({ color: 0x4a5560 }) };
  const ghost = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false });
  const ghostMats = { white: ghost, blade: ghost, grey: ghost, yellow: ghost, dark: ghost };
  // sea surface: a few travelling waves, fresnel, see-through so the seabed shows
  const seaU = { uT: { value: 0 }, uDeep: { value: new THREE.Color() }, uShal: { value: new THREE.Color() }, uSky: { value: new THREE.Color() }, uA: { value: 1 } };
  const sea = new THREE.Mesh(new THREE.PlaneGeometry(1, 1, 200, 200).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({ uniforms: seaU, transparent: true,
    vertexShader: `uniform float uT, uA; varying vec3 vP; varying vec3 vN;
      void main() { vec3 p = (modelMatrix * vec4(position, 1.0)).xyz; float h = 0.0, hx = 0.0, hz = 0.0;
        vec4 W[4]; W[0] = vec4(0.05, 0.03, 0.55, 0.9); W[1] = vec4(-0.02, 0.06, 0.35, 1.2); W[2] = vec4(0.09, -0.04, 0.2, 1.6); W[3] = vec4(0.13, 0.08, 0.12, 2.1);
        for (int i = 0; i < 4; i++) { float ph = W[i].x * p.x + W[i].y * p.z - W[i].w * uT; h += uA * W[i].z * sin(ph); hx += uA * W[i].z * W[i].x * cos(ph); hz += uA * W[i].z * W[i].y * cos(ph); }
        p.y += h; vP = p; vN = normalize(vec3(-hx, 1.0, -hz)); gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0); }`,
    fragmentShader: `uniform vec3 uDeep, uShal, uSky; varying vec3 vP; varying vec3 vN;
      void main() { vec3 v = normalize(cameraPosition - vP); float fr = pow(1.0 - max(dot(vN, v), 0.0), 3.0); vec3 c = mix(uDeep, uShal, 0.5 + 0.5 * vN.x * 3.0); c = mix(c, uSky, fr * 0.6);
        float d = length(cameraPosition - vP), fog = smoothstep(1800.0, 3400.0, d); gl_FragColor = vec4(mix(c, uSky, fog), mix(0.3 + 0.55 * fr, 1.0, fog)); }` }));
  sea.scale.set(6000, 1, 6000); sea.renderOrder = -1; scene.add(sea);
  // seabed: chart depths until the geophysical survey, then the surveyed bathymetry; soil and exclusions as tints
  const bedMat = new THREE.MeshLambertMaterial({ vertexColors: true }); let bed = null, bedKey = "", overlay = "depth";
  const RAMP = [[0, 0xeadcae], [0.25, 0xb9d1a6], [0.5, 0x6fb3a9], [0.75, 0x3b7fa6], [1, 0x1c3d6b]].map(([u, c]) => [u, new THREE.Color(c)]);
  const ramp = (u, c) => { let i = 0; while (i < RAMP.length - 2 && u > RAMP[i + 1][0]) i++; return c.copy(RAMP[i][1]).lerp(RAMP[i + 1][1], Math.min(1, Math.max(0, (u - RAMP[i][0]) / (RAMP[i + 1][0] - RAMP[i][0])))); };
  const SOILC = [new THREE.Color(0xe0c070), new THREE.Color(0xb98a5a), new THREE.Color(0x8d7bb0)];
  let dMean = 40;
  const depthArr = () => (G.known.geophys ? G.map.depth : G.map.chart);
  const dispDepth = (d) => Math.max(4, dMean + (d - dMean) * VE);
  function buildBed() {
    const key = G.lease.round + ":" + G.known.geophys + ":" + G.known.geotech + ":" + overlay; if (key === bedKey) return; bedKey = key;
    if (bed) { scene.remove(bed); bed.geometry.dispose(); }
    const m = G.map, arr = depthArr(), ext = 3, g = new THREE.PlaneGeometry(m.W * 1000 / U * ext, m.H * 1000 / U * ext, 180, 135).rotateX(-Math.PI / 2), P = g.attributes.position, col = new Float32Array(P.count * 3), c = new THREE.Color();
    let lo = 1e9, hi = -1e9; for (let i = 0; i < arr.length; i++) { lo = Math.min(lo, arr[i]); hi = Math.max(hi, arr[i]); } dMean = (lo + hi) / 2;
    for (let k = 0; k < P.count; k++) {
      const x = (P.getX(k) + m.W * 500 / U) * U / 1000, y = (P.getZ(k) + m.H * 500 / U) * U / 1000, inside = x >= 0 && y >= 0 && x <= m.W && y <= m.H;
      const cx = Math.min(m.W, Math.max(0, x)), cy = Math.min(m.H, Math.max(0, y)), d = S.sample(m, arr, cx, cy) + 0.6 * Math.hypot(x - cx, y - cy);
      P.setY(k, -dispDepth(d) * TS / U);
      ramp((d - lo) / Math.max(hi - lo, 4), c);
      if (inside && overlay === "soil" && G.known.geotech) c.lerp(SOILC[m.soil[S.cell(m, cx, cy)]], 0.55);
      if (inside && m.excl[S.cell(m, cx, cy)]) c.lerp(new THREE.Color(0xd1495b), 0.55);
      if (!inside) c.multiplyScalar(0.8);
      col.set([c.r, c.g, c.b], 3 * k);
    }
    g.setAttribute("color", new THREE.BufferAttribute(col, 3)); g.computeVertexNormals();
    bed = new THREE.Mesh(g, bedMat); bed.renderOrder = -2; scene.add(bed);
    turbObjs.forEach((o) => { o.key = ""; });      // piles must meet the new seabed
  }
  // lease boundary: yellow buoys and a line on the water
  const boundary = new THREE.Group(); scene.add(boundary);
  function buildBoundary() {
    boundary.clear(); const m = G.map, pts = [[0, 0], [m.W, 0], [m.W, m.H], [0, m.H], [0, 0]].map(([x, y]) => new THREE.Vector3(kmX(x), 0.6, kmZ(y)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0xf2c230, dashSize: 14, gapSize: 9 })); line.computeLineDistances(); boundary.add(line);
    pts.slice(0, 4).forEach((p) => { const b = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 2, 6, 10), mats.yellow); b.position.copy(p); b.position.y = 2; boundary.add(b); });
    const port = makeLabel("To port: 38 km west"); port.position.set(kmX(0.9), 10, kmZ(m.H / 2)); port.scale.set(55, 10, 1); boundary.add(port);
  }
  function makeLabel(text, w = 256) {
    const c = document.createElement("canvas"); c.width = w; c.height = 48; const g = c.getContext("2d"); g.font = "600 26px IBM Plex Mono, monospace"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillStyle = "rgba(255,255,255,0.88)"; g.fillRect(4, 4, w - 8, 40); g.fillStyle = "#0e1b27"; g.fillText(text, w / 2, 25);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false })); sp.renderOrder = 10; return sp;
  }
  // turbines
  const farm = new THREE.Group(); scene.add(farm); const turbObjs = new Map();
  const tmeta = () => { const t = S.CFG.turbine; return { z_interface: 15, z_hub: t.hub, rotor_D: t.rotorD, z_platform: 20, z_door: 21, Dz: (z) => (z <= 15 ? 9.5 : 9.5 - 3 * (z - 15) / (t.hub - 20)) }; };
  function syncTurbines() {
    const seen = new Set();
    G.turbines.forEach((t) => {
      seen.add(t.id); const dd = dispDepth(S.sample(G.map, depthArr(), t.x, t.y)), key = t.state + ":" + fmt(dd, 1) + ":" + (t.down ? 1 : 0);
      let o = turbObjs.get(t.id);
      if (o && o.key === key) return;
      if (o) farm.remove(o.g);
      const g = new THREE.Group(), T3 = TM.build(tmeta(), t.state === "planned" ? ghostMats : mats, -dd - 0.3); T3.g.scale.setScalar(1 / U); g.add(T3.g);
      if (t.state === "foundation") { T3.tower.visible = false; T3.yaw.visible = false; T3.plat.visible = true; }
      const ring = new THREE.Mesh(new THREE.TorusGeometry(2.4, 0.18, 6, 32), new THREE.MeshBasicMaterial({ color: 0xffffff })); ring.rotation.x = Math.PI / 2; ring.position.y = 0.5; g.add(ring);
      const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.7, 10, 8), new THREE.MeshBasicMaterial({ color: 0xff3b30 })); beacon.position.y = (S.CFG.turbine.hub + 9) / U; beacon.visible = !!t.down; g.add(beacon);
      const lab = makeLabel(t.name, 128); lab.position.y = (S.CFG.turbine.hub + S.CFG.turbine.rotorD / 2 + 14) / U; lab.scale.set(7, 2.6, 1); g.add(lab);
      g.position.set(kmX(t.x), 0, kmZ(t.y)); g.scale.setScalar(TS); g.traverse((q) => { q.userData.tid = t.id; });
      farm.add(g); o = { g, T3, ring, beacon, key, spin: Math.random() * 6 }; turbObjs.set(t.id, o);
    });
    turbObjs.forEach((o, id) => { if (!seen.has(id)) { farm.remove(o.g); turbObjs.delete(id); } });
  }
  // vessels: crew transfer vessels, the service vessel, the jack-up, and the installation vessels during construction
  const fleet = new THREE.Group(); scene.add(fleet);
  function boat(color, len) {
    const g = new THREE.Group(), hull = new THREE.Mesh(new THREE.BoxGeometry(len, len * 0.22, len * 0.36), new THREE.MeshStandardMaterial({ color })); hull.position.y = len * 0.08; g.add(hull);
    const cab = new THREE.Mesh(new THREE.BoxGeometry(len * 0.32, len * 0.24, len * 0.28), mats.white); cab.position.set(len * 0.12, len * 0.3, 0); g.add(cab); return g;
  }
  function jackupMesh() {
    const g = new THREE.Group(), deck = new THREE.Mesh(new THREE.BoxGeometry(14, 1.6, 9), new THREE.MeshStandardMaterial({ color: 0xd8dde2 })); deck.position.y = 5; g.add(deck);
    [[-6, -3.6], [6, -3.6], [-6, 3.6], [6, 3.6]].forEach(([x, z]) => { const l = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 34, 6), mats.yellow); l.position.set(x, 3, z); g.add(l); });
    const crane = new THREE.Mesh(new THREE.BoxGeometry(0.8, 18, 0.8), mats.yellow); crane.position.set(-4, 14, 0); crane.rotation.z = 0.35; g.add(crane); return g;
  }
  const ctvs = [], sovMesh = boat(0xe05a2b, 9), jack = jackupMesh(), fdnShip = jackupMesh(), wtgShip = jackupMesh(); sovMesh.visible = jack.visible = fdnShip.visible = wtgShip.visible = false;
  [sovMesh, jack, fdnShip, wtgShip].forEach((m) => fleet.add(m));
  const portPos = () => new THREE.Vector3(kmX(-0.3), 0, kmZ(G.map.H / 2));
  function syncFleet(dt) {
    while (ctvs.length < G.ctv) { const b = boat(0xf2c230, 4.5); b.position.copy(portPos()); fleet.add(b); ctvs.push({ m: b }); }
    while (ctvs.length > G.ctv) { fleet.remove(ctvs.pop().m); }
    const jobs = [...new Set(G.crewToday || [])];
    ctvs.forEach((c, i) => {
      const t = G.turbines.find((q) => q.id === jobs[i]), tgt = t ? new THREE.Vector3(kmX(t.x) + 5, 0, kmZ(t.y) + 4) : portPos().add(new THREE.Vector3(0, 0, i * 8 - 12));
      const d = tgt.clone().sub(c.m.position); d.y = 0; const L = d.length(), v = Math.min(L, dt * 260);
      if (L > 0.5) { c.m.position.add(d.multiplyScalar(v / L)); c.m.rotation.y = Math.atan2(-d.z, d.x); }
    });
    const firstJob = G.turbines.find((q) => (G.crewToday || []).includes(q.id));
    sovMesh.visible = G.sov; if (G.sov) { const t = firstJob || G.turbines[0]; if (t) sovMesh.position.lerp(new THREE.Vector3(kmX(t.x) - 9, 0, kmZ(t.y) - 7), Math.min(1, dt * 1.5)); }
    const J = G.jackup && G.jackup.onsite; jack.visible = !!J; if (J) { const t = G.turbines.find((q) => q.id === G.jackupAt) || G.turbines.find((q) => G.tasks.some((k) => k.vessel === "jackup" && k.tid === q.id)); if (t) jack.position.set(kmX(t.x) + 10, 0, kmZ(t.y)); }
    const b = G.build; fdnShip.visible = !!(b && b.fdnIdx < G.turbines.length); wtgShip.visible = !!(b && b.wtgIdx < b.fdnIdx);
    if (fdnShip.visible) { const t = G.turbines[b.fdnIdx]; fdnShip.position.set(kmX(t.x) + 10, 0, kmZ(t.y)); }
    if (wtgShip.visible) { const t = G.turbines[b.wtgIdx]; wtgShip.position.set(kmX(t.x) - 10, 0, kmZ(t.y)); }
  }
  function theme() {
    const sky = new THREE.Color(css("--sky")); scene.background = sky; seaU.uSky.value = sky; seaU.uDeep.value = new THREE.Color(css("--ocean-deep")); seaU.uShal.value = new THREE.Color(css("--ocean-shallow"));
  }
  function resize() { const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(host);
  function resetView() {            // frame the turbines if there are any, else the lease
    const ts = G.turbines, cx = ts.length ? ts.reduce((a, t) => a + t.x, 0) / ts.length : G.map.W / 2, cy = ts.length ? ts.reduce((a, t) => a + t.y, 0) / ts.length : G.map.H / 2;
    const span = ts.length ? Math.max(2, Math.max(...ts.map((t) => Math.hypot(t.x - cx, t.y - cy)))) : G.map.W / 2;
    controls.target.set(kmX(cx), 0, kmZ(cy)); camera.position.set(kmX(cx - span * 0.5), span * 70 + 120, kmZ(cy + span * 1.25 + 1)); controls.update();
  }

  // ---------- picking and placing ----------
  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2(), plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  function at(ev) { const r = renderer.domElement.getBoundingClientRect(); mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(mouse, camera); }
  function seaPoint(ev) { at(ev); const p = new THREE.Vector3(); if (!ray.ray.intersectPlane(plane, p)) return null; return [(p.x + G.map.W * 500 / U) * U / 1000, (p.z + G.map.H * 500 / U) * U / 1000]; }
  let down = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    at(e); const hit = ray.intersectObjects(farm.children, true)[0];
    if (hit && hit.object.userData.tid) { sel = hit.object.userData.tid; dirty = true; return; }
    if (placing) { const p = seaPoint(e); if (p) { const r = cmd("place", p[0], p[1]); if (r.ok) toast(r.msg, "good"); } return; }
    sel = null; dirty = true;
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    const p = seaPoint(e), m = G.map; if (!p || p[0] < 0 || p[1] < 0 || p[0] > m.W || p[1] > m.H) { $("readout").textContent = ""; return; }
    const d = S.sample(m, depthArr(), p[0], p[1]), fc = S.foundationCost(G, p[0], p[1], G.known);
    $("readout").textContent = `${fmt(p[0], 2)}, ${fmt(p[1], 2)} km · depth ${fmt(d, 1)} m${G.known.geophys ? "" : " (chart, ±3 m)"} · ${G.known.geotech ? S.SOILS[m.soil[S.cell(m, p[0], p[1])]].name : "soil unknown"}${m.excl[S.cell(m, p[0], p[1])] ? " · exclusion zone" : ""}${placing ? ` · foundation ≈ ${eur(fc.cost)} ±${fmt(100 * fc.unc, 0)} %` : ""}`;
  });

  // ---------- interface ----------
  let dirty = true, toastT = 0;
  function toast(msg, lvl = "warn") { G.log.push({ t: G.day, lvl, text: msg }); dirty = true; }
  const STEPS = [["lease", "Lease"], ["develop", "Develop"], ["finance", "Finance"], ["build", "Build"], ["operate", "Operate"]];
  function stepNow() { if (G.stage === "lease") return 0; if (G.stage === "develop") return G.cfd.state === "won" || G.cfd.state === "merchant" ? 2 : G.consent === "granted" && G.turbines.length ? 2 : 1; return G.stage === "build" ? 3 : 4; }
  function header() {
    const s = S.summary(G); $("h-date").textContent = s.date; $("h-cash").textContent = eur(s.cash); $("h-cash").classList.toggle("neg", s.cash < 0); $("h-debt").textContent = eur(s.debt);
    $("h-pow").textContent = s.operating ? `${fmt(s.powerNow, 0)} / ${fmt(s.capacity, 0)} MW` : "–"; $("h-av").textContent = s.availability !== null ? fmt(100 * s.availability, 1) + " %" : "–";
    const w = G.w, ctv = w.HsMax < S.CFG.access.ctv, sov = w.HsMax < S.CFG.access.sov;
    $("wx").innerHTML = `<span>Wind <b class="mono">${fmt(w.U, 1)}</b> m/s at hub</span><span>Waves H<sub>s</sub> <b class="mono">${fmt(w.Hs, 1)}</b> m</span><span>Access: <span class="pill ${ctv ? "ok" : "crit"}">CTV</span> <span class="pill ${sov ? "ok" : "crit"}">SOV</span></span>${G.cfd.state === "merchant" ? `<span>Price <b class="mono">€${fmt(G.price, 0)}</b>/MWh</span>` : ""}`;
    $("log").innerHTML = G.log.slice(-4).reverse().map((l) => `<div class="${l.lvl}"><span class="mono muted">${S.dateStr(l.t)}</span> ${l.text}</div>`).join("");
  }
  function stage() {
    const k = stepNow(); $("stepper").innerHTML = STEPS.map(([, l], i) => `<div class="${i < k ? "done" : i === k ? "now" : ""}">${l}</div>`).join("");
    const s = S.summary(G), el = $("stage"); let h = "";
    if (G.stage === "lease") {
      const m = G.map, arr = m.chart; let lo = 1e9, hi = -1e9; arr.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      h += `<div class="card"><h3>Seabed lease round ${G.lease.round}</h3><div class="kv"><span>Area</span><span class="v">${m.W} × ${m.H} km</span><span>Charted depth</span><span class="v">${fmt(lo, 0)}–${fmt(hi, 0)} m</span><span>Distance to port</span><span class="v">${S.CFG.map.portDist} km</span><span>Mean wind at hub (est.)</span><span class="v">≈ 10.3 m/s</span><span>Rival bidders</span><span class="v">${G.lease.rivals.length}</span></div>` +
        `<div class="slider"><span>Your sealed bid for the option</span><b class="mono" id="bidv">€14.0 M</b><input type="range" id="bid" min="2" max="30" step="0.5" value="14"></div><button class="btn go" type="button" id="dobid">Submit sealed bid</button>` +
        `<p class="cap">The highest bid wins the right to develop the area. Every euro here is a euro you cannot spend on turbines. Rivals typically bid €5–16 M.</p></div>`;
    }
    if (G.stage === "develop") {
      const job = (k) => G.jobs.find((j) => j.kind === k), st = (k) => (G.known[k] ? `<span class="pill ok">done</span>` : job(k) ? `<span class="pill info">${Math.ceil(job(k).done - G.day)} d</span>` : "");
      h += `<div class="card"><h3>Surveys and consent</h3>` +
        `<div class="row" style="justify-content:space-between"><span>Geophysical survey (bathymetry)</span>${st("geophys") || `<button class="btn sm" data-c="survey:geophys">${eur(S.CFG.cost.geophys)}</button>`}</div>` +
        `<div class="row" style="justify-content:space-between;margin-top:4px"><span>Geotechnical survey (soil)</span>${st("geotech") || `<button class="btn sm" data-c="survey:geotech">${eur(S.CFG.cost.geotech)}</button>`}</div>` +
        `<div class="row" style="justify-content:space-between;margin-top:4px"><span>Consent application</span>${G.consent === "granted" ? `<span class="pill ok">granted</span>` : G.consent === "pending" ? `<span class="pill info">${Math.ceil(job("consent").done - G.day)} d</span>` : `<button class="btn sm" data-c="consent">${eur(S.CFG.cost.consent)}</button>`}</div>` +
        `<p class="cap">Until the surveys are in, depths come from an old chart and foundation costs are uncertain.</p></div>`;
      const cx = s.capex, y = s.yield;
      h += `<div class="card"><h3>Layout <span class="muted" style="font: 0.8rem var(--font-body)">${G.turbines.length} × ${S.CFG.turbine.P} MW</span></h3><div class="row"><button class="btn ${placing ? "on" : ""}" id="place">${placing ? "Placing: click the sea" : "Place turbines"}</button>${G.turbines.length ? `<button class="btn" id="clear">Clear planned</button>` : ""}</div>` +
        (G.turbines.length ? `<div class="kv" style="margin-top:8px"><span>Capital cost (est.)</span><span class="v">${eur(cx.total)} ± ${eur(cx.unc)}</span><span>of which foundations</span><span class="v">${eur(cx.foundations)}</span><span>Energy, P50</span><span class="v">${fmt(y.energy / 1e3, 0)} GWh/yr</span><span>Capacity factor</span><span class="v">${fmt(100 * y.cf * (1 - y.wake), 1)} %</span><span>Wake loss</span><span class="v">${fmt(100 * y.wake, 1)} %</span></div>` : `<p class="cap">Click "Place turbines", then click the sea inside the yellow boundary. Keep out of the red exclusion zones and at least 3.5 rotor diameters apart.</p>`) + `</div>`;
      if (G.turbines.length) {
        const share = G.cfd.state === "merchant" ? S.CFG.debt.merchant : S.CFG.debt.cfd, eq = cx.total * (1 - share);
        h += `<div class="card"><h3>Route to market</h3>` + (G.cfd.state === "won" ? `<p>Contract for difference: <b>€${fmt(G.cfd.strike, 0)}/MWh</b> for 15 years.</p>` : G.cfd.state === "merchant" ? `<p>Merchant: selling at the market price (now €${fmt(G.price, 0)}/MWh).</p>` : G.cfd.state === "waiting" ? `<p class="muted">Next allocation round in ${Math.ceil(G.cfd.next - G.day)} days.</p><button class="btn" data-c="merchant">Go merchant instead</button>` :
          `<div class="slider"><span>Strike price bid</span><b class="mono" id="cfdv">€82/MWh</b><input type="range" id="cfd" min="55" max="110" step="1" value="82"></div><div class="row"><button class="btn go" id="docfd">Bid in the CfD round</button><button class="btn" data-c="merchant">Go merchant</button></div><p class="cap">Lower bids are likelier to win but earn less. A CfD lets lenders cover ${100 * S.CFG.debt.cfd}% of the cost; merchant only ${100 * S.CFG.debt.merchant}%.</p>`) +
          `<div class="kv" style="margin-top:6px"><span>Equity you need (est.)</span><span class="v" style="color:var(${eq > G.cash ? "--crit" : "--ok"})">${eur(eq)}</span><span>Cash now</span><span class="v">${eur(G.cash)}</span></div></div>`;
        const can = { consent: G.consent === "granted", market: G.cfd.state === "won" || G.cfd.state === "merchant", layout: G.turbines.length > 0 };
        h += `<div class="card"><h3>Construction</h3><ul class="check"><li class="${can.consent ? "y" : ""}">Consent granted</li><li class="${can.market ? "y" : ""}">Route to market and debt financing</li><li class="${can.layout ? "y" : ""}">Layout planned</li><li class="${G.known.geophys && G.known.geotech ? "y" : ""}">Surveys complete (recommended)</li></ul><button class="btn go" data-c="construct" ${can.consent && can.market && can.layout ? "" : "disabled"}>Start construction</button></div>`;
      }
    }
    if (G.stage === "build") {
      const b = G.build, n = G.turbines.length, bar = (v) => `<div class="bar"><i style="width:${Math.min(100, 100 * v)}%"></i></div>`;
      h += `<div class="card"><h3>Construction</h3><div class="kv"><span>Substation and export cable</span><span class="v">${b.exportDone ? "done" : fmt(100 * b.exportDays / S.CFG.durations.exportBuild, 0) + " %"}</span></div>${bar(b.exportDone ? 1 : b.exportDays / S.CFG.durations.exportBuild)}` +
        `<div class="kv" style="margin-top:6px"><span>Foundations</span><span class="v">${b.fdnIdx} / ${n}</span></div>${bar(b.fdnIdx / n)}<div class="kv" style="margin-top:6px"><span>Turbines</span><span class="v">${b.wtgIdx} / ${n}</span></div>${bar(b.wtgIdx / n)}` +
        `<div class="kv" style="margin-top:8px"><span>Spent so far</span><span class="v">${eur(G.capexSpent)}</span><span>Debt drawn</span><span class="v">${eur(G.debt)} of ${eur(G.facility)}</span></div><p class="cap">Installation stops when waves exceed ${S.CFG.access.installHs} m or wind ${S.CFG.access.installU} m/s; vessels cost their day rate either way. Winter builds are slow and expensive.</p></div>`;
      h += opsCards();
    }
    if (G.stage === "operate") h += opsCards();
    el.innerHTML = h; wire(el);
  }
  function opsCards() {
    const Y = G.year, ops = G.turbines.filter((t) => t.state === "operating"); if (!ops.length) return "";
    let h = `<div class="card"><h3>${Y.y} so far</h3><div class="kv"><span>Energy</span><span class="v">${fmt(Y.energy / 1e3, 1)} GWh</span><span>Revenue</span><span class="v">${eur(Y.revenue)}</span><span>Operating costs</span><span class="v">${eur(Y.opex)}</span><span>Debt service</span><span class="v">${eur(Y.debt)}</span><span>Availability</span><span class="v">${Y.possibleHours ? fmt(100 * Y.upHours / Y.possibleHours, 1) + " %" : "–"}</span><span>Failures · repaired early</span><span class="v">${Y.failures} · ${Y.prevent}</span><span>Lost to downtime</span><span class="v">${fmt(Y.lostMWh / 1e3, 1)} GWh</span></div></div>`;
    h += `<div class="card"><h3>Crews and vessels</h3><div class="row" style="justify-content:space-between"><span>Crew transfer vessels (H<sub>s</sub> &lt; ${S.CFG.access.ctv} m)</span><span class="row"><button class="btn sm" data-c="ctv:${G.ctv - 1}">−</button><b class="mono">${G.ctv}</b><button class="btn sm" data-c="ctv:${G.ctv + 1}">+</button></span></div>` +
      `<div class="row" style="justify-content:space-between;margin-top:4px"><span>Service operation vessel (H<sub>s</sub> &lt; ${S.CFG.access.sov} m)</span><button class="btn sm ${G.sov ? "on" : ""}" data-c="sov:${G.sov ? 0 : 1}">${G.sov ? "chartered" : "charter"}</button></div>` +
      `<div class="row" style="justify-content:space-between;margin-top:4px"><span>StormTwin monitoring</span><button class="btn sm ${G.monitoring ? "on" : ""}" data-c="monitoring:${G.monitoring ? 0 : 1}">${G.monitoring ? "on" : "switch on"}</button></div>` +
      `<p class="cap">CTV €${fmt(S.CFG.cost.ctvDay * 1000, 1)} k/day each · SOV €${fmt(S.CFG.cost.sovDay * 1000, 0)} k/day · monitoring €${fmt(S.CFG.cost.monitoring * 1000, 0)} k per turbine per year. Monitoring keeps component health estimates fresh and raises alerts; without it you only learn at services and inspections.</p>` +
      `<div class="row" style="margin-top:6px"><button class="btn" data-c="jackup" ${G.jackup || !G.tasks.some((q) => q.vessel === "jackup") ? "disabled" : ""}>${G.jackup ? (G.jackup.onsite ? "Jack-up on site" : "Jack-up due " + S.dateStr(G.jackup.arrive)) : "Book jack-up vessel"}</button><button class="btn" data-c="rockdump" ${G.rock || !G.tasks.some((q) => q.vessel === "rock") ? "disabled" : ""}>${G.rock ? "Rock vessel booked" : "Book rock placement"}</button></div></div>`;
    h += `<div class="card"><h3>Work orders <span class="muted" style="font: 0.8rem var(--font-body)">${G.tasks.length}</span></h3>` + (G.tasks.length ? `<ul class="tasks">${G.tasks.map((q) => { const t = G.turbines.find((x) => x.id === q.tid);
      return `<li><span>${t ? t.name : ""}: ${q.label}${t && t.down ? ` <span class="pill crit">down</span>` : ""}</span><span class="a"><button class="btn sm" data-c="prioritise:${q.id}" title="Do first">↑</button><button class="btn sm" data-c="cancel:${q.id}" title="Cancel">×</button></span><span class="s">${q.vessel === "jackup" ? "needs the jack-up" : q.vessel === "rock" ? "needs rock placement" : `crew ${fmt(q.done || 0, 0)}/${q.work} days`} · parts ${eur(q.parts)}</span></li>`; }).join("")}</ul>` : `<p class="cap">No open work. Click a turbine to inspect it or plan repairs.</p>`) + `</div>`;
    return h;
  }
  function wire(el) {
    el.querySelectorAll("[data-c]").forEach((b) => b.addEventListener("click", () => { const [n, a] = b.dataset.c.split(":"); const r = cmd(n, a === undefined ? undefined : isNaN(+a) ? a : +a); if (r.ok && r.msg) toast(r.msg, "good"); }));
    const bid = el.querySelector("#bid"); if (bid) { bid.addEventListener("input", () => { el.querySelector("#bidv").textContent = eur(+bid.value); }); el.querySelector("#dobid").addEventListener("click", () => { const r = cmd("bidLease", +bid.value); if (r.ok) { buildBoundary(); bedKey = ""; } else { bedKey = ""; buildBoundary(); } }); }
    const cfd = el.querySelector("#cfd"); if (cfd) { cfd.addEventListener("input", () => { el.querySelector("#cfdv").textContent = `€${cfd.value}/MWh`; }); el.querySelector("#docfd").addEventListener("click", () => cmd("bidCfd", +cfd.value)); }
    const pl = el.querySelector("#place"); if (pl) pl.addEventListener("click", () => { placing = !placing; dirty = true; });
    const cl = el.querySelector("#clear"); if (cl) cl.addEventListener("click", () => { G.turbines.filter((t) => t.state === "planned").forEach((t) => S.command(G, "remove", t.id)); dirty = true; });
  }
  function tpanel() {
    const p = $("tpanel"), t = G.turbines.find((q) => q.id === sel); if (!t) { p.hidden = true; return; }
    const st = t.down ? `<span class="pill crit">stopped</span>` : t.state === "operating" ? `<span class="pill ok">operating</span>` : `<span class="pill info">${t.state}</span>`;
    const d = S.sample(G.map, depthArr(), t.x, t.y), fc = S.foundationCost(G, t.x, t.y, G.known);
    let h = `<h3 style="display:flex;justify-content:space-between;align-items:center">${t.name} ${st}<button class="btn sm" id="tclose">×</button></h3><div class="kv" style="margin-top:6px"><span>Water depth</span><span class="v">${fmt(t.depth || d, 1)} m</span><span>Soil</span><span class="v">${t.soil || (G.known.geotech ? S.SOILS[G.map.soil[S.cell(G.map, t.x, t.y)]].name : "unknown")}</span><span>Foundation</span><span class="v">${t.fdnCost ? eur(t.fdnCost) : `≈ ${eur(fc.cost)}`}</span><span>Wake factor</span><span class="v">${fmt(100 * t.wake, 1)} %</span>${t.state === "operating" ? `<span>Energy to date</span><span class="v">${fmt(t.energy / 1e3, 1)} GWh</span>` : ""}</div>`;
    if (t.state === "planned") h += `<div class="row" style="margin-top:8px"><button class="btn" id="tremove">Remove</button></div>`;
    if (t.state === "operating") {
      h += `<div class="row" style="margin-top:8px;justify-content:space-between"><b style="font-family:var(--font-display)">Component health</b><button class="btn sm" data-t="inspect">Inspect (€12 k, 1 crew day)</button></div>` +
        `<p class="cap" style="margin-top:2px">Bar: the estimate (line) and its 90 % band. ${G.monitoring ? "StormTwin updates it weekly." : "Without monitoring it goes stale between services and inspections."}</p>`;
      S.COMPS.forEach((c) => { const e = t.est[c.k], sg = t.sig[c.k], lo = Math.max(0, e - 1.645 * sg), hi = Math.min(1, e + 1.645 * sg), col = lo < 0.3 ? "--crit" : lo < 0.5 ? "--warn" : "--ok", isDown = t.down && t.down.comp === c.k;
        h += `<div class="comp"><span>${c.name}${isDown ? ` <span class="pill crit">failed</span>` : ""}</span><span class="mono" style="color:var(${col})">${isDown ? "0" : fmt(100 * e, 0)} %</span>` +
          `<div class="hb"><div class="band" style="left:${100 * lo}%;width:${100 * (hi - lo)}%"></div><div class="est" style="left:calc(${100 * (isDown ? 0 : e)}% - 1px)"></div></div>` +
          `<span class="muted" style="font-size:0.7rem">seen ${Math.round(G.day - t.seen[c.k])} d ago</span><span></span>` +
          `<div class="acts">${c.rock ? `<button class="btn sm" data-t="replace:${c.k}">Rock placement (${eur(S.CFG.cost.rockPer)})</button>` : `<button class="btn sm" data-t="repair:${c.k}" ${t.health[c.k] < 0.2 && !isDown ? "" : ""}>Repair up-tower (${eur(c.repair)})</button><button class="btn sm" data-t="replace:${c.k}">Replace (${eur(c.replace)}${c.major ? " + jack-up" : ""})</button>`}</div></div>`; });
    }
    p.innerHTML = h; p.hidden = false;
    p.querySelector("#tclose").addEventListener("click", () => { sel = null; dirty = true; });
    const rm = p.querySelector("#tremove"); if (rm) rm.addEventListener("click", () => { cmd("remove", t.id); sel = null; });
    p.querySelectorAll("[data-t]").forEach((b) => b.addEventListener("click", () => { const [k, comp] = b.dataset.t.split(":"); const r = cmd("task", t.id, comp || null, k); if (r.ok) toast(r.msg, "good"); }));
  }
  function tools() {
    $("tools").innerHTML = `<button class="btn sm ${overlay === "depth" ? "on" : ""}" data-o="depth">Depth</button><button class="btn sm ${overlay === "soil" ? "on" : ""}" data-o="soil" ${G.known.geotech ? "" : "disabled title='Needs the geotechnical survey'"}>Soil</button><button class="btn sm" id="rv">Reset view</button>`;
    $("tools").querySelectorAll("[data-o]").forEach((b) => b.addEventListener("click", () => { overlay = b.dataset.o; dirty = true; }));
    $("rv").addEventListener("click", resetView);
  }
  function modal(html, onClose) { $("mbox").innerHTML = html; $("modal").hidden = false; speed = 0; speedUI(); const c = $("mbox").querySelector("[data-close]"); if (c) c.addEventListener("click", () => { $("modal").hidden = true; if (onClose) onClose(); }); }
  function yearReport(y) {
    const net = y.revenue - y.opex - y.debt;
    modal(`<h2>${y.y} in review</h2><div class="score"><div><span class="k">Energy</span><span class="v">${fmt(y.energy / 1e3, 0)} GWh</span></div><div><span class="k">Revenue</span><span class="v">${eur(y.revenue)}</span></div><div><span class="k">Net cash flow</span><span class="v" style="color:var(${net < 0 ? "--crit" : "--ok"})">${eur(net)}</span></div>` +
      `<div><span class="k">Availability</span><span class="v">${y.possibleHours ? fmt(100 * y.upHours / y.possibleHours, 1) : "–"} %</span></div><div><span class="k">Failures</span><span class="v">${y.failures}</span></div><div><span class="k">Repaired early</span><span class="v">${y.prevent}</span></div></div>` +
      `<p class="muted">Operating costs ${eur(y.opex)}, debt service ${eur(y.debt)}${y.capex ? `, construction ${eur(y.capex)}` : ""}. ${y.failures > y.prevent ? "More failures than early repairs: StormTwin monitoring and summer repair campaigns would cut the downtime." : "Good asset management: problems caught before they stop turbines."}</p><button class="btn go" data-close>Continue</button>`);
  }
  function finalReport() {
    const ys = G.history.filter((y) => y.possibleHours), av = ys.reduce((a, y) => a + y.upHours, 0) / Math.max(1, ys.reduce((a, y) => a + y.possibleHours, 0)), E = ys.reduce((a, y) => a + y.energy, 0), value = G.cash - G.debt;
    const grade = value > 0 && av > 0.97 ? "A" : value > -150 && av > 0.95 ? "B" : value > -300 ? "C" : "D";
    modal(`<h2>Five years of operation: grade ${grade}</h2><div class="score"><div><span class="k">Energy delivered</span><span class="v">${fmt(E / 1e6, 2)} TWh</span></div><div><span class="k">Availability</span><span class="v">${fmt(100 * av, 1)} %</span></div><div><span class="k">Cash less debt</span><span class="v">${eur(value)}</span></div></div>` +
      `<p>Your farm powered the equivalent of about ${fmt(E / ys.length / 3.5 / 1000, 0)} thousand homes a year. Keep playing to run it longer, or start a new game with what you learned.</p><div class="row"><button class="btn go" data-close>Keep playing</button><button class="btn" id="newg">New game</button></div>`);
    $("newg").addEventListener("click", () => location.reload());
  }
  function gameOver() { modal(`<h2>Game over</h2><p>${G.over.why}</p><p class="muted">Tip: keep enough equity for construction (see the estimate under Route to market), and watch winter vessel costs.</p><button class="btn go" id="again">Try again</button>`); $("again").addEventListener("click", () => location.reload()); }

  // ---------- clock ----------
  function speedUI() { document.querySelectorAll("[data-sp]").forEach((b) => b.classList.toggle("on", +b.dataset.sp === speed)); }
  document.querySelectorAll("[data-sp]").forEach((b) => b.addEventListener("click", () => { speed = +b.dataset.sp; speedUI(); }));
  document.addEventListener("keydown", (e) => { if (e.key === " " && e.target === document.body) { e.preventDefault(); speed = speed ? 0 : 1; speedUI(); } if (e.key === "Escape") { placing = false; sel = null; dirty = true; } });
  $("theme").addEventListener("click", () => { const r = document.documentElement, dk = r.dataset.theme ? r.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches; r.dataset.theme = dk ? "light" : "dark"; theme(); dirty = true; });
  let acc = 0, last = performance.now(), uiT = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (speed && $("modal").hidden) {
      acc += dt * speed; let n = 0; const before = G.history.length;
      while (acc >= 1 && n < 40) { S.step(G); acc -= 1; n++; }
      if (n) dirty = true;
      if (G.history.length > before) { const y = G.history[G.history.length - 1]; if (y.possibleHours) yearReport(y); }
      if (G.cod && !finalShown && G.day >= G.cod + 5 * 365.25) { finalShown = true; finalReport(); }
      if (G.over) gameOver();
    }
    if (G.stage !== "lease" && !G.lease.won) {}
    seaU.uT.value += dt; seaU.uA.value = 0.4 + 0.5 * G.w.Hs;
    turbObjs.forEach((o, id) => { const t = G.turbines.find((q) => q.id === id); if (!t || t.state !== "operating" || t.down) return;
      const om = Math.min(7.5, Math.max(0, (G.w.U - 3) / 7.6 * 7.5)) * (G.w.U > 25 ? 0 : 1); o.spin += dt * om * 2 * Math.PI / 60 * Math.max(1, Math.min(speed, 3)); o.T3.rotor.rotation.z = o.spin;
      o.T3.yaw.rotation.y = -(G.w.dir - 180) * Math.PI / 180 + Math.PI; });
    turbObjs.forEach((o) => { if (o.beacon.visible) o.beacon.material.color.setHSL(0, 1, 0.35 + 0.25 * Math.sin(now / 180)); o.ring.material.color.set(o.g.userData.sel ? 0x43b3d5 : 0xffffff); });
    syncFleet(dt);
    uiT += dt;
    if (dirty && uiT > 0.2) { uiT = 0; dirty = false; buildBed(); syncTurbines(); header(); stage(); tpanel(); tools(); turbObjs.forEach((o, id) => { o.g.userData.sel = id === sel; }); }
    controls.update(); renderer.render(scene, camera);
  }
  theme(); resize(); buildBoundary(); resetView(); speedUI();
  modal(`<h2>Offshore Wind Developer</h2><p>You have <b>€${S.CFG.startCash} M</b> of investors' money and a seabed lease round about to open.</p>` +
    `<p>Win the lease, survey the site, apply for consent, design the layout and secure a contract for difference. Then build the farm through the weather, and run it for five years: sell the power, send crews out when the sea allows, book jack-up vessels for the big repairs, and use the StormTwin digital twin to fix problems before they stop your turbines.</p>` +
    `<p class="muted">Controls: drag to orbit, scroll to zoom, right-drag to pan. Space pauses. Speeds run 1 day, 1 week or 1 month per second.</p><button class="btn go" data-close>Start</button>`, () => { speed = 1; speedUI(); });
  window.__ows = { select: (id) => { sel = id; dirty = true; }, view: resetView, game: () => G, cmd: (...a) => cmd(...a), step: (n) => { for (let i = 0; i < n; i++) S.step(G); dirty = true; }, refresh: () => { dirty = true; } };   // for automated play-tests
  requestAnimationFrame(frame);
})();
