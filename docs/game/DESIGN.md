# Offshore Wind Developer (working title)

A management sim in the spirit of SimCity and Farming Simulator. You bid for a seabed lease, survey it, design the turbines and the layout, finance and build the farm, and then run it for 25–30 years: selling power, sending crews out to fix what breaks, and using the digital twin to catch damage before it becomes a failure.

The difference from other tycoon games is that the simulation underneath is real engineering. The StormTwin models already in this repository predict the sea, the wind, wakes, loads, fatigue damage and natural frequencies. The player sees those numbers through the same twin an operator would use, and makes the same calls an operator makes.

## Core loop

1. **Find a site.** A national leasing round opens. Lease areas come with bathymetry, soil, metocean data, distance to port and grid connection, and existing constraints (shipping lanes, cables, wrecks). This reuses the Map builder.
2. **Win the lease.** Sealed-bid or option-fee auction against AI developers. Higher bids mean less money for everything after.
3. **Survey and permit.** Pay for geophysical and geotechnical campaigns, which reveal the soil and lift the uncertainty on foundation cost. Environmental consent takes time and can come with conditions, such as piling noise windows or bird curtailment.
4. **Design.** Choose the turbine model and design the foundations with the Turbine builder. Lay out the farm with the Site builder: energy yield against wake loss and foundation cost, with checks for pile length and soil.
5. **Finance.** Bid into a contract-for-difference (CfD) auction or sign a power purchase agreement (PPA). Debt and equity terms depend on how bankable the design and the yield estimate are.
6. **Build.** Book installation vessels and ports, then install foundations, transition pieces, turbines, array cables and the export cable. Weather windows decide progress, and delays cost vessel day rates.
7. **Operate.** Sell power. Turbines degrade and fail, driven by the twin's fatigue and condition models. Crews go out by crew transfer vessel (CTV), service operation vessel (SOV) or helicopter, limited by sea state. Big repairs need a jack-up vessel. Preventive inspections and repairs informed by the twin cost money now and save outages later.
8. **Extend or decommission.** At end of life, decide whether to extend operation (the twin's remaining-life estimate decides whether that's safe), repower or decommission.

Seasons, storms and market prices keep pushing the player between growth and keeping the existing assets running.

## Systems

### Economy
| Item | Starting value (to tune) |
|---|---|
| Development costs (surveys, consent, design) | 3–5 % of capital cost |
| Capital cost | €2.5–4 M per MW (turbines ~35 %, foundations ~20 %, electrical ~20 %, installation ~15 %) |
| Operations and maintenance cost | €60–100 k per MW per year |
| Revenue | CfD strike price, or a merchant price that moves with the market |
| Lease | Option fee at award, then annual rent |
| Finance | Debt share, interest rate and debt-service cover ratio covenants; breaching a covenant locks dividends |

### Weather and access
- Wind and sea come from the StormTwin closed-form wind history and the metocean data of each map.
- **Access limits** (significant wave height H<sub>s</sub>): CTV about 1.5 m, SOV walk-to-work about 2.5–3 m, helicopter limited by wind and visibility, jack-up limits on both legs and lifting.
- Winter access windows make maintenance planning the central puzzle of the operations phase.

### Failures and maintenance (where the twin earns its place)
- Each component carries a damage or health state: blades (erosion, cracks), main bearing, gearbox, generator, converter, pitch system, bolts, welds, cables and scour.
- Failure rates rise with the twin's damage measures: fatigue life used, flange preload, scour depth and frequency drift.
- **Reactive maintenance:** wait for the failure, lose production until a crew and a weather window arrive.
- **Preventive maintenance:** run a twin assessment (costs analyst time and sensor upkeep), see predicted remaining life with uncertainty, and schedule a repair in the summer window.
- **Inspections** (rope access, drones, ROV, ultrasonic bolt checks) narrow the twin's uncertainty band. That's the game version of the virtual sensing already in StormTwin.
- Major component exchanges need a jack-up vessel: book months ahead, at a high day rate.

### Research and upgrades
- Better sensors, which give tighter twin estimates.
- Larger turbines.
- Floating foundations for deep water.
- Better wake steering, which recovers part of the wake loss.

### Events
Storms, cable faults, vessel breakdowns, grid curtailment, price spikes, supply-chain delays, regulator audits.

## How existing code maps to game systems
| StormTwin today | Game system |
|---|---|
| Map builder (bathymetry, soil zones, exclusions, metocean) | Lease areas and surveys |
| Turbine builder (structure, soil and p-y curves, ULS, fatigue, drivetrain) | Design phase |
| Site builder (layout, yield, wake loss, pile and soil checks) | Layout and yield |
| Wind climate, power, wakes | Revenue |
| Fatigue, frequency tracking, virtual sensing | Asset health and preventive decisions |
| Storm replay, what-if | Event and stress testing |

## Engine options

| | Web (three.js, as StormTwin) | Godot 4 | Unreal Engine 5 |
|---|---|---|---|
| Reuse of current code | Direct | Port the models to GDScript/C#, or call a library | Port the models to C++, or run a sidecar service |
| Visual ceiling | Good (already looks good) | Good | Best: water, lighting, vessels, weather |
| Time to a playable prototype | Days | Weeks | Months |
| Build and test in this cloud environment | Yes, end to end | Partly (headless) | No: the editor and its builds need a desktop GPU and an Epic account |
| Distribution | A link, any device | Desktop, web export | Desktop and console; Steam |

### Recommendation
1. **Prototype the game loop in the browser on this branch first.** It reuses every model as is, it can be played and shared as a link right away, and it shows whether the economy and maintenance loop is fun before money goes into art.
2. **Keep the simulation core separate from the rendering** (a pure module with a small API: `step(dt)`, `getState()`, `command(...)`). That core is what would port.
3. **Move to Unreal Engine 5 once the loop works,** if the goal is a commercial PC game with high-end visuals. Two routes:
   - **Port the core to C++** as an Unreal plugin.
   - **Run the core as a separate process** (Python or Node) that Unreal talks to locally.
   The Unreal project itself would have to be built on a desktop with the editor; this cloud session can write the C++ and the design, but cannot compile or run it.

## Milestones
1. **Vertical slice (web):** one lease area, one turbine model, build a 12-turbine farm, run 5 years at accelerated time with failures, crews and weather access, and a profit and loss statement.
2. **Full development phase:** auctions, surveys, consent, finance.
3. **Twin gameplay:** preventive assessments, inspections that narrow uncertainty, remaining-life decisions.
4. **Content:** more regions, turbine models, events, AI competitors.
5. **Engine decision:** stay on the web, or port the core to Unreal.

## Prototype
A playable browser vertical slice is in `demo/windsim/`:
- `sim.js`: the simulation core (no DOM; runs in Node and the browser). This is the part to port.
- `game.js`: three.js scene and interface.
- `balance_test.js`: plays whole games headlessly with reactive and twin-guided maintenance, to check the economy.

## Parked ideas (from play-testing the first slice)
- **Put the engineering back in the loop.** The slice skips the most fun part: designing and analysing the turbine and foundation. Bring the Turbine builder into the game, so the player sizes the pile and tower, runs the analyses and lives with the consequences (fatigue, frequency, cost).
- **Skill trees for analysis fidelity.** Analyses start simple and unlock deeper models: hand calculations, then the beam and p-y model, then nonlinear p-y and PISA, then 3D soil FE; spectral fatigue, then time-domain aero-elastic, then full design-load-case sets. Better models narrow the uncertainty bands, which lowers contingency, cuts overdesign and makes the twin's predictions sharper.
