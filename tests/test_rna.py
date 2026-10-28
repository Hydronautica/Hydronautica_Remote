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


def test_rbe3_resultant():
    from rna.fe import rbe3
    rng = np.random.default_rng(0)
    X = rng.random((40, 3))
    F, M, p = np.array([1.0, -2, 3]), np.array([0.5, 0.2, -1]), np.array([2.0, 0, 1])
    f = rbe3(X, p, F, M)
    assert np.allclose(f.sum(0), F)
    assert np.allclose(np.cross(X - p, f).sum(0), M)


def test_voxel_bar_axial_stiffness_and_reaction():
    from rna.fe import VoxelModel
    L, a = 1.0, 0.2
    m = VoxelModel(lambda p: np.ones(p.shape[:-1], bool), ((0, L), (0, a), (0, a)), 0.05, E=200e9, nu=0.0)
    m.fix("root", m.where(lambda X: X[:, 0] < 1e-9))
    tip = m.where(lambda X: X[:, 0] > L - 1e-9)
    U = m.solve(m.load(tip, [L, a / 2, a / 2], F=[1e6, 0, 0]))
    u_tip = U[3 * tip].mean()
    # equal forces on every tip node (corners included) add a little local flexibility at the loaded end
    assert np.isclose(u_tip, 1e6 * L / (200e9 * a * a), rtol=0.04)
    R = m.reaction("root", np.array([0, a / 2, a / 2]))
    assert np.allclose(R[:3, 0], [1e6, 0, 0], atol=1.0)


def test_chain_closes_if_built():
    import os
    import pytest
    from rna.chain import closure, load, run, turbine_for
    path = os.path.join(os.path.dirname(__file__), "..", "demo", "rna", "fe_chain.npz")
    if not os.path.exists(path):
        pytest.skip("run examples/rna_fe_chain.py first")
    fe = load(path)
    tb = turbine_for(fe)
    for kind in ("op", "storm"):
        rec = synthetic(kind, tb)
        c = closure(rec, run(rec, fe))
        assert c["hub"] < 0.02 and c["tower"] < 0.02


def test_virtual_sensing_observability_if_built():
    import os
    import pytest
    from rna.chain import load, run, turbine_for
    from rna.virtual import as_record, estimate, prior_cov, r2
    path = os.path.join(os.path.dirname(__file__), "..", "demo", "rna", "fe_chain.npz")
    if not os.path.exists(path):
        pytest.skip("run examples/rna_fe_chain.py first")
    fe = load(path)
    tb = turbine_for(fe)
    rec = synthetic("op", tb, duration=200.0)
    P = prior_cov(tb, "op", seeds=range(101, 105))
    tilt = run(rec, fe)["flange"][5]
    fit = []
    for which in (0, 1):
        _, x_hat, _, _ = estimate(rec, fe, P, which, tb)
        fit.append(r2(tilt, run(as_record(rec, x_hat, fe), fe)["flange"][5]))
    assert fit[0] < 0.3 < 0.9 < fit[1]       # SCADA alone cannot see hub bending; tower-top gauges can
