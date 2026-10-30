# Wind & Steel (working title)

A day-by-day offshore wind farm game in the spirit of Poly Bridge. You run a farm: every day the wind blows, your turbines make energy and you get paid for it. With the money you lease new plots, design and build new turbines, unlock better turbines and technology, and keep everything running.

**The heart of the game is the turbine builder.** You design the tower and the monopile so the turbine survives the sea, doesn't resonate and lasts, and you try to do it with as little steel as you dare. It is a simplified version of the StormTwin Turbine builder: a handful of handles instead of a full form, but driven by the same physics.

## The core loop

1. **Every day:** the weather happens (wind, waves, sometimes a storm). Turbines produce energy and you are paid per MWh. Components wear, storms load the structures, things break.
2. **Design and build:** pick a plot (its depth, soil and wave climate), pick a turbine, then shape the structure and test it in a storm. Building costs money and a few days at sea.
3. **Research:** engineers produce research points. Spend them (and money) to unlock bigger turbines, stronger steel, active control, better software, better inspections and more vessels.
4. **Look after the farm:** inspect turbines to find damage before it becomes a failure, repair in calm weather, and get broken turbines back online.

## The builder (Poly Bridge-style)

A side view of the turbine: sky, sea and seabed, with the soil layers. You drag handles to set:

| Handle | Effect |
|---|---|
| Tower base and top diameters | Stiffness and strength of the tower; steel and cost |
| Tower wall thickness | Strength and buckling of the tower |
| Monopile diameter and wall | Strength at the mudline, stiffness, wave load (a fatter pile catches more wave) |
| Embedment (pile toe) | Holding capacity in the soil and foundation stiffness |
| Steel grade | Strength (S355 → S420 → S460 after research) |

Four checks update live, each shown as an estimate with a confidence band against its limit:

- **Strength:** storm stresses at the tower base and the mudline against steel strength (with buckling for thin walls).
- **Holding:** the soil's moment capacity against the storm overturning moment.
- **Resonance:** the first natural frequency must sit between the rotor's 1P and 3P bands (the soft-stiff window).
- **Fatigue life:** years before the welds crack under everyday wind and waves.

**Storm test:** run the design through the plot's design storm at the confidence you choose (P50 or P90). The turbine sways, stresses colour the steel and, if a check fails, it breaks in the way it would: the tower buckles, the pile pulls over, or it shakes itself apart in resonance. Testing is free; building is not.

### Uncertainty is the game

Every check is an estimate. The plot's true soil strength and storm severity, and the error of your analysis methods, are hidden. You see an estimate and a confidence band. You can design to the safe end of the band (more steel, less profit) or to the middle (cheaper, riskier). The truth shows up when a real storm hits.

Narrowing the bands is how you progress:

| What you unlock | Which band it narrows |
|---|---|
| Software: hand calcs → spreadsheet → beam FE → time-domain simulation → digital twin | Model error on every check |
| Geotechnical survey of a plot (paid, takes days) | Soil strength |
| Metocean engineer, wave buoy data | Storm loads and fatigue loads |
| Geotechnical and structural engineers | Soil and structure model error; they also produce research |

## Running the farm

- **Energy and money:** each day's wind gives each working turbine its energy through the turbine's power curve, less wake losses (active wake control reduces them). You're paid a fixed price per MWh.
- **Wear:** blades, drivetrain and electrics lose health every day, faster in rough weather; structures accumulate fatigue at the rate their design predicted (the true rate, not your estimate).
- **Failures:** a component at the end of its health fails and the turbine stops. In a storm, a structure that is truly too weak fails outright and the turbine is lost.
- **Inspections:** visual, drone, ROV and ultrasonic checks, then structural-health sensors, each detecting more problems with tighter health estimates.
- **Repairs:** need a vessel and a weather window (crew transfer vessel: waves under 1.5 m; later a service vessel: under 2.5 m). Planned repairs are cheaper and quicker than repairing after a failure, which also waits for spare parts.

## Research tree

| Branch | Unlocks |
|---|---|
| Turbines | 6 MW → 10 MW → 15 MW → 20 MW |
| Materials | S420 steel, S460 steel, better weld detailing and coatings (fatigue) |
| Control | Individual pitch control (lower fatigue loads), tower damper, wake steering (less wake loss) |
| Software | Spreadsheet, beam FE, time-domain simulation, digital twin |
| Inspection | Drones, ROV and ultrasonic testing, structural-health sensors |
| Vessels | Second crew vessel, service operation vessel |

## Architecture

- `demo/windsim/core.js`: the whole simulation, pure and deterministic from a seed (state in, state out, no DOM), so it can be tested headlessly (`balance_test.js`) and later ported (for example to an Unreal plugin as C++).
- `demo/windsim/ui.js` and `index.html`: the browser front end (2D canvas, flat Poly Bridge-like look).

The engineering formulas are deliberately simple versions of the StormTwin models: a cantilever with a soil spring for the frequency, a Morison inertia storm load and a rated-thrust gust load, a Broms-type lateral pile capacity, and S-N fatigue scaled from a reference design. They keep the trends right (fatter is stiffer and catches more wave, deeper water means longer piles and lower frequencies, thin walls buckle) without exposing the full engineering forms.
