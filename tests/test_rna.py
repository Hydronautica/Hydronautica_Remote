import numpy as np

from rna import SN, Turbine, assess, rainflow, shaft_supports, synthetic
from rna.components import _design_rate, design_climate


def test_rainflow_astm_example():
    # ASTM E1049-85 rainflow example
    rg, _, ct, _ = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    hist = {}
    for r, c in zip(rg, ct):
        hist[r] = hist.get(r, 0) + c
    assert hist == {3: 0.5, 4: 1.5, 6: 0.5, 8: 1.0, 9: 0.5}


def test_sn_bilinear():
    sn = SN(100.0, 1e6, 3, 5)
    assert np.isclose(sn.cycles(200.0), 1.25e5)
    assert np.isclose(sn.cycles(50.0), 3.2e7)


def test_hub_reduction_balances():
    tb = Turbine()
    r = synthetic("op", tb)
    # mean vertical force is the rotor weight, mean torque is the aerodynamic torque
    assert np.isclose(r["hub_F"][:, 1].mean(), -tb.rotor_mass * 9.81, rtol=0.01)
    assert np.isclose(r["hub_M"][:, 0].mean(), r["torque_aero"].mean(), rtol=0.01)
    # tower top carries the whole tower-head mass
    assert np.isclose(r["tt_F"][:, 1].mean(), -(tb.rotor_mass + tb.nacelle_mass) * 9.81, rtol=0.01)


def test_shaft_supports_static():
    tb = Turbine()
    r = synthetic("op", tb)
    mb, gb, fa = shaft_supports(r, tb)
    # bearing loads sum to the hub forces
    assert np.allclose(mb[:, 0] + gb[:, 0], r["hub_F"][:, 1])
    assert np.allclose(mb[:, 1] + gb[:, 1], r["hub_F"][:, 2])


def test_design_climate_sane():
    tb = Turbine()
    U, p, rpm, T, _ = design_climate(tb)
    assert 0.9 < p.sum() < 1.0
    assert _design_rate(6.6, tb) > 0


def test_assess_runs():
    out = assess(synthetic("storm"))
    keys = [c["key"] for c in out]
    assert "main_bearing" in keys and "bolts" in keys
    gears = next(c for c in out if c["key"] == "gears")
    assert gears["ageing"] < 1e-3          # a parked rotor does not age the gears
