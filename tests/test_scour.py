import numpy as np
import pytest

from scour import (Pile, Sediment, assess, bed_orbital_velocity, combined_shear, live_bed_depth,
                   onset_current, simulate, synthetic, time_scale)
from scour.core import G, NU, depth_factor, intensity_factor


def test_threshold_matches_soulsby_whitehouse():
    sed = Sediment(d50=0.2e-3)
    ds = 0.2e-3 * (G * (2650 / 1025 - 1) / NU**2) ** (1 / 3)
    assert sed.d_star == pytest.approx(ds)
    assert sed.theta_cr == pytest.approx(0.30 / (1 + 1.2 * ds) + 0.055 * (1 - np.exp(-0.02 * ds)))
    # sands sit in the familiar 0.03-0.1 band, coarser grains need less Shields stress
    th = [Sediment(d * 1e-3).theta_cr for d in (0.1, 0.2, 0.4, 1.0)]
    assert all(0.025 < t < 0.1 for t in th) and np.all(np.diff(th) < 0)


def test_bed_velocity_linear_in_hs_and_decays_with_depth():
    u1, u2 = bed_orbital_velocity([1.0, 2.0], 8.0, 20.0)
    assert u2 == pytest.approx(2 * u1, rel=1e-9)
    deep, shallow = bed_orbital_velocity(2.0, 8.0, 80.0), bed_orbital_velocity(2.0, 8.0, 15.0)
    assert deep < 0.1 * shallow
    # shallow-water limit: U_m ~ (Hs/2) sqrt(g/h) for a long-period sea
    assert bed_orbital_velocity(1.0, 25.0, 5.0) == pytest.approx(0.5 * np.sqrt(G / 5.0), rel=0.2)


def test_combined_shear_reduces_to_components():
    tm, tmax = combined_shear(1.0, 0.0)
    assert tm == pytest.approx(1.0) and tmax == pytest.approx(1.0)
    tm, tmax = combined_shear(0.0, 2.0)
    assert tm == pytest.approx(0.0) and tmax == pytest.approx(2.0)
    tm, tmax = combined_shear(1.0, 1.0)
    assert tm == pytest.approx(1 + 1.2 * 0.5**3.2) and tmax == pytest.approx(tm + 1.0)


def test_live_bed_depth_limits():
    # current alone -> 1.3 D
    assert live_bed_depth(0.0, 1.0) == pytest.approx(1.3)
    # waves alone, Sumer & Fredsøe: 1.3 (1 - exp(-0.03 (KC - 6)))
    assert live_bed_depth(10.0, 0.0) == pytest.approx(1.3 * (1 - np.exp(-0.03 * 4)))
    assert live_bed_depth(5.0, 0.0) == 0.0
    # more current, deeper scour
    s = live_bed_depth(1.0, np.linspace(0, 1, 11))
    assert np.all(np.diff(s) >= -1e-12)


def test_clear_water_ramp_and_depth_factor():
    assert intensity_factor(0.25 * 0.05, 0.05) == pytest.approx(0.0)
    assert intensity_factor(0.05, 0.05) == pytest.approx(1.0)
    assert intensity_factor(0.5, 0.05) == pytest.approx(1.0)
    assert depth_factor(40.0, 10.0) == pytest.approx(1.0, abs=1e-3)
    assert depth_factor(10.0, 10.0) < 0.8


def test_onset_current_brackets_the_threshold():
    pile = Pile(10.0, 40.0, Sediment(0.25e-3))
    u_on, u_live = onset_current(pile), onset_current(pile, level="live")
    assert 0 < u_on < u_live
    assert not assess(pile, 0.95 * u_on, 0.0, 8.0)["onset"]
    assert assess(pile, 1.05 * u_on, 0.0, 8.0)["onset"]
    assert assess(pile, 1.05 * u_live, 0.0, 8.0)["live"]
    # waves lower the current needed
    assert onset_current(pile, 3.0, 8.0) < u_on


def test_constant_current_reaches_equilibrium_with_sumer_time_scale():
    pile = Pile(10.0, 40.0, Sediment(0.25e-3))
    st = assess(pile, 0.9, 0.0, 8.0)
    seq, T = float(st["s_eq"]), float(st["T"])
    assert seq == pytest.approx(1.3 * 10.0 * np.tanh(4.0), rel=1e-6)
    theta = float(st["theta"])
    tstar = (1 / 2000) * 4.0 * theta**-2.2
    assert T == pytest.approx(tstar * 100 / np.sqrt(G * (2650 / 1025 - 1) * 0.25e-3**3), rel=1e-6)
    n = int(np.ceil(T / 3600))
    S, _ = simulate(pile, 1.0, np.full(n, 0.9), np.zeros(n), np.full(n, 8.0))
    assert S[-1] == pytest.approx(seq * (1 - np.exp(-n * 3600 / T)), rel=1e-6)


def test_backfill_under_waves_but_not_in_clear_water():
    pile = Pile(10.0, 40.0, Sediment(0.25e-3))
    n = 24 * 60
    # current scour, then a wave-dominated spell: the bed is mobile and the equilibrium is shallower
    uc, hs, tp = np.r_[np.full(n, 0.9), np.full(n, 0.1)], np.r_[np.zeros(n), np.full(n, 4.0)], np.r_[np.full(n, 8.0), np.full(n, 9.0)]
    S, st = simulate(pile, 1.0, uc, hs, tp)
    assert st["live"][-1] and st["s_eq"][-1] < S[n - 1]
    assert S[-1] < S[n - 1]
    S2, _ = simulate(pile, 1.0, np.r_[np.full(n, 0.9), np.full(n, 0.05)], np.zeros(2 * n), np.full(2 * n, 8.0))
    assert S2[-1] == pytest.approx(S2[n - 1])      # no transport, no backfill


def test_time_scale_uses_wave_form_only_for_large_kc():
    pile = Pile(2.0, 20.0, Sediment(0.2e-3))
    small = time_scale(0.1, 1.0, 0.0, pile)
    assert small == pytest.approx(time_scale(0.1, 1.0, 0.0, pile, current_only=True))
    big = time_scale(0.1, 20.0, 0.0, pile)
    tw = 1e-6 * (20.0 / 0.1) ** 3 * 4 / np.sqrt(G * (2650 / 1025 - 1) * 0.2e-3**3)
    assert big == pytest.approx(tw, rel=1e-6)


def test_synthetic_record_and_cli(tmp_path):
    rec = synthetic("2024-01-01", 0.25, u_spring=0.6, surge_per_m=0.0, seed=3)
    assert len(rec["hs"]) == int(0.25 * 8766)
    assert rec["uc"].max() == pytest.approx(0.6, abs=0.02) and rec["tp"].min() >= 4.0
    from scour.__main__ import main
    out = tmp_path / "s.json"
    assert main(["--D", "8", "--h", "25", "--d50", "0.3", "--years", "0.3", "--no-plots", "--json", str(out)]) == 0
    assert out.exists()


def test_protection_holds_or_fails_with_rock_size():
    from scour.core import Armour, rock_size_for, simulate_protected
    pile = Pile(10.0, 42.0, Sediment(0.30e-3))
    rec = synthetic("2020-01-01", 2.0, hs_median=1.6, u_spring=0.65, seed=5)
    ok = simulate_protected(pile, 1.0, rec["uc"], rec["hs"], rec["tp"], Armour(d50=0.4))
    assert ok["integrity"].min() == 1.0 and ok["s_pile"].max() == 0.0
    assert 0 < ok["s_edge"].max() < 0.5 * simulate(pile, 1.0, rec["uc"], rec["hs"], rec["tp"])[0].max()
    weak = simulate_protected(pile, 1.0, rec["uc"], rec["hs"], rec["tp"], Armour(d50=0.03))
    assert weak["integrity"].min() < 1.0 and weak["s_pile"].max() > 0.0
    d = rock_size_for(rec["uc"], rec["hs"], rec["tp"], 42.0)
    assert simulate_protected(pile, 1.0, rec["uc"], rec["hs"], rec["tp"], Armour(d50=d))["utilisation"].max() <= 0.8 + 1e-6
    assert Armour(thickness=1.5).surcharge == pytest.approx(1.5 * 0.6 * (2650 - 1025) * G)


def test_backfill_and_surcharge_in_global_model():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "stormtwin"))
    from structure import Model, Turbine
    f = lambda **k: Model(Turbine(depth=42.0, **k)).frequencies(n=1)[0]
    f0 = f()
    assert f(scour=0.0, scour_max=0.0) == pytest.approx(f0)
    # a refilled hole is softer than undisturbed soil but stiffer than an open hole
    assert f(scour=8.0) < f(scour=0.0, scour_max=8.0) < f0
    assert f(scour=4.0, scour_max=8.0) < f(scour=4.0)
    assert f(surcharge=14e3) > f0
