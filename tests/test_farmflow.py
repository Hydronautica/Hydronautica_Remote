"""Physics checks of the StormTwin farm-flow simulator (demo/stormtwin/farmflow.js), run in Node."""

import json
import math
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(__file__)
NODE = shutil.which("node") or (os.path.exists("/opt/node22/bin/node") and "/opt/node22/bin/node")


@pytest.fixture(scope="module")
def res():
    if not NODE:
        pytest.skip("Node.js not available")
    out = subprocess.run([NODE, os.path.join(HERE, "farmflow_check.js")], capture_output=True, text=True, timeout=600, check=True)
    return json.loads(out.stdout)


def test_static_wake_matches_the_gaussian(res):
    s = res["static"]
    ct, D, x = s["ct"], 240.0, 7.0
    beta = (1 + math.sqrt(1 - ct)) / (2 * math.sqrt(1 - ct))
    sig = 0.003678 * x + 0.2 * math.sqrt(beta)                     # k* = 0.3837 * TI + 0.003678 with TI ~ 0
    q = 1 - ct / (8 * sig * sig)                                   # the Gaussian is capped at the momentum deficit in the near wake
    expect = min(1 - math.sqrt(1 - ct), 1 - math.sqrt(q)) if q > 0 else 1 - math.sqrt(1 - ct)
    assert s["centre"] == pytest.approx(expect, abs=0.02)
    assert abs(s["nAtMax"]) < 0.1 * D
    assert s["integral"] == pytest.approx(expect * math.sqrt(2 * math.pi) * sig * D, rel=0.05)


def test_meandering_spreads_but_keeps_the_deficit(res):
    m = res["meander"]
    assert m["centre"] < 0.95 * m["centreStatic"]                # time-averaged wake is shallower...
    assert 0.1 <= m["wander"] <= 0.5                               # wake centre wanders a fraction of a diameter at 7 D
    assert m["integral"] == pytest.approx(m["integralStatic"], rel=0.2)   # ...and wider, with about the same deficit


def test_row_losses_are_realistic(res):
    lo, hi = res["row0.06"], res["row0.1"]
    assert 0.45 <= lo["ratios"][1] <= 0.70
    assert min(lo["ratios"][1:]) >= 0.45
    assert hi["eff"] > lo["eff"]                                   # more turbulence, faster wake recovery
    assert abs(lo["frontLoss"]) < 0.01 and abs(hi["frontLoss"]) < 0.01   # the front turbine has no wake loss


def test_yaw_deflects_the_wake_to_the_left(res):
    assert 0.3 <= res["yaw"]["nAtMax"] <= 0.7                    # Jimenez: about half a diameter at 7 D for 25 degrees


def test_prefilled_runs_match_spin_up(res):
    assert res["prefill"]["prefill"] == pytest.approx(res["prefill"]["spin"], abs=0.025)


def test_speed(res):
    assert res["ms12x600"] < 15000
