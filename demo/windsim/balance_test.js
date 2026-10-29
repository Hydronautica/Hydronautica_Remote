// Headless balance check: plays whole games with a reactive and a twin-guided maintenance strategy.
// Run with: node demo/windsim/balance_test.js
const S = require('./sim.js');
function play(seed, strat, years = 9) {
  const s = S.newGame(seed); const cmd = (...a) => S.command(s, ...a);
  for (let k = 0; k < 6 && s.stage === 'lease'; k++) cmd('bidLease', 14);
  if (s.stage === 'lease') return { fail: 'lease' };
  cmd('survey', 'geophys'); cmd('survey', 'geotech'); cmd('consent');
  // grid of 12
  const m = s.map; let placed = 0;
  for (let gx = 1.2; gx < m.W - 0.5 && placed < 12; gx += 1.75) for (let gy = 1.0; gy < m.H - 0.5 && placed < 12; gy += 1.5) { if (cmd('place', gx, gy).ok) placed++; }
  const end = s.day + years * 365;
  while (s.day < end && !s.over) {
    if (s.consent === 'granted' && s.cfd.state === 'none') { const r = cmd('bidCfd', 82); if (!r.ok && s.cfd.state !== 'won') cmd('merchant'); }
    if (s.consent === 'granted' && !s.build && s.stage === 'develop') { cmd('construct'); cmd('ctv', 2); if (strat === 'twin') cmd('monitoring', true); }
    if (s.stage === 'operate' || s.turbines.some(t => t.state === 'operating')) {
      if (strat === 'twin') s.turbines.filter(t => t.state === 'operating').forEach(t => S.COMPS.forEach(c => { const lo = t.est[c.k] - 1.645 * t.sig[c.k]; if (lo < 0.35 && t.health[c.k] >= 0.2 && !c.rock && !s.tasks.some(q => q.tid === t.id && q.comp === c.k)) cmd('task', t.id, c.k, 'repair'); }));
      if (s.tasks.some(q => q.vessel === 'jackup') && !s.jackup) cmd('jackup');
      if (s.tasks.some(q => q.vessel === 'rock') && !s.rock) cmd('rockdump');
    }
    S.step(s);
  }
  return s;
}
for (const strat of ['reactive', 'twin']) for (const seed of [3, 7]) {
  const s = play(seed, strat); if (s.fail) { console.log(strat, seed, s.fail); continue; }
  console.log(`\n== ${strat} seed ${seed}: cod ${s.cod ? S.dateStr(s.cod) : '-'} cash ${s.cash.toFixed(0)} debt ${s.debt.toFixed(0)} capex ${s.capexSpent.toFixed(0)} cfd ${s.cfd.state} ${s.cfd.strike} over ${s.over ? s.over.why : ''}`);
  s.history.concat([s.year]).forEach(y => console.log(y.y, 'GWh', (y.energy / 1e3).toFixed(0), 'rev', y.revenue.toFixed(1), 'opex', y.opex.toFixed(1), 'capex', y.capex.toFixed(0), 'debt', y.debt.toFixed(1), 'fail', y.failures, 'prev', y.prevent, 'avail', y.possibleHours ? (100 * y.upHours / y.possibleHours).toFixed(1) : '-'));
}
