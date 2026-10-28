"""Structural data for StormTwin's live whole-turbine filter, per farm and turbine: live_struct_<F>.json.

For each turbine:
  * a "true" soil stiffness (the design p-y stiffness times a site factor, seeded per turbine)
    with its surveyed scour; the plant that stands in for the real turbine uses 6 modes at it;
  * the filter's modes (4 per plane) on a grid of soil stiffness factors, so the page can use the
    design soil or the soil implied by the tracked natural frequency;
  * the first-mode frequency on that grid (to turn a tracked frequency into a soil factor) and the
    tracked frequency itself: the twin's operational-modal-analysis history, rescaled so it is
    the frequency of the true soil (the Frequencies tab uses the same scaling);
  * bending-moment rows at every element, static moment per unit top force and moment, and the
    hot spots' section data.
Run after build_modal.py; make_page.py embeds the result.
"""
import base64
import copy
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import farms  # noqa: E402
from structure import Model, Turbine, section  # noqa: E402

GRID = np.array([0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0])     # soil stiffness factors for the filter's model
N_EST, N_PLANT = 4, 6


def b64(a):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).decode()


def modal(tb, kappa, n_modes):
    m = Model(tb)
    m.k0 = m.k0 * kappa
    kt = np.zeros(m.ndof); kt[2 * m.soil] = m.k0                  # linear (initial) p-y stiffness
    K = m.K + np.diag(kt)
    w2, Phi = eigh(K, m.M)
    w = np.sqrt(w2[:n_modes]); Phi = Phi[:, :n_modes]
    top = 2 * (m.z.size - 1)
    Phi = Phi * np.sign(Phi[top])                                  # consistent signs: top moves +
    zc = 0.5 * (m.z[1:] + m.z[:-1])
    rows = []
    for zt in zc:
        e = int(np.clip(np.searchsorted(m.z, zt) - 1, 0, m.z.size - 2))
        L = m.z[e + 1] - m.z[e]; xi = (zt - m.z[e]) / L
        r = np.zeros(m.ndof)
        r[2 * e:2 * e + 4] = m.EI[e] * np.array([(-6 + 12 * xi) / L ** 2, (-4 + 6 * xi) / L, (6 - 12 * xi) / L ** 2, (-2 + 6 * xi) / L])
        rows.append(r)
    rows = np.array(rows)
    Lt = m.z[-1] - m.z[-2]
    F1 = np.zeros(m.ndof); F1[top] = 1.0
    M1 = np.zeros(m.ndof); M1[top] = 1.0 / Lt; M1[top - 2] = -1.0 / Lt
    stat = np.stack([rows @ np.linalg.solve(K, F1), rows @ np.linalg.solve(K, M1)], 1)      # N m per N, per N m
    return dict(m=m, w=w, Phi=Phi, phiL=Phi[0::2], rows=rows @ Phi, stat=stat, zc=zc, f1=w[0] / (2 * np.pi))


def main():
    for fid in ("A", "B", "C"):
        os.environ["STORMTWIN_FARM"] = fid
        cfg = farms.current()
        T = farms.TURBINES[cfg["turbine"]]
        with open(os.path.join(HERE, f"demo_data_{fid}.json")) as fh:
            data = json.load(fh)
        rng = np.random.default_rng(500 + ord(fid))
        out = dict(farm=fid, turbines=[], grid=GRID.tolist(), n_est=N_EST, n_plant=N_PLANT)
        hot = None
        for t in data["turbines"]:
            spec = t["spec"]
            tb = Turbine(**dict(T, scour=spec["scour"], mean_thrust=0.0), depth=spec["depth"])
            kappa = float(np.exp(rng.normal(0, 0.3)))                  # the site's soil against the design p-y curves
            plant = modal(copy.deepcopy(tb), kappa, N_PLANT)
            grid = [modal(copy.deepcopy(tb), k, N_EST) for k in GRID]
            design = grid[list(GRID).index(1.0)]
            # tracked frequency: the twin's OMA history rescaled from design soil to the true soil
            fa = t["modal"]["modes"]["tower_fa"]
            r = plant["f1"] / design["f1"]
            hist = np.array(fa["f"][-40:]) * r
            m = plant["m"]
            wet = np.flatnonzero((m.z > -tb.depth) & (m.z < 0))
            tp = m.node(tb.z_interface)
            out["turbines"].append(dict(
                name=spec["name"], depth=tb.depth, scour=spec["scour"], kappa_true=kappa, freq_scale=float(r),
                f1_design=float(design["f1"]), f1_true=float(plant["f1"]), f1_track=float(plant["f1"] * (1 + rng.normal(0, 0.002))),             # environment-normalised OMA estimate
                f1_track_sd=float(0.002 * plant["f1"]), track_hist=hist.round(5).tolist(),
                z=m.z.round(3).tolist(), zc=plant["zc"].round(3).tolist(), wet=wet.tolist(), trib=m.trib[wet].round(4).tolist(),
                tp=int(tp), D=float(tb.D_pile), m_rna=float(tb.m_rna),
                plant=dict(w=plant["w"].tolist(), phi=b64(plant["phiL"]), rows=b64(plant["rows"]), stat=b64(plant["stat"])),
                est=dict(f1=[float(g["f1"]) for g in grid], w=b64(np.array([g["w"] for g in grid])),
                         phi=b64(np.array([g["phiL"] for g in grid])), rows=b64(np.array([g["rows"] for g in grid])),
                         stat=b64(np.array([g["stat"] for g in grid])))))
            if hot is None:
                spots = {"tower_base": (tb.z_interface + 2, 1.15, "air", "Tower base weld"), "door": (tb.z_door, 2.4, "air", "Door frame"),
                         "flange": (tb.z_interface, 0, "bolt", "Tower-base flange bolts"), "mudline": (-tb.depth + 1.0, 1.25, "sea", "Monopile weld at the mudline"),
                         "pile": (-tb.depth - 6.0, 1.0, "sea", "Pile below the seabed (−6 m)")}
                hot = {}
                for k, (z, scf, curve, nm) in spots.items():
                    D, th = float(tb.diameter(z)), float(tb.thickness(z))
                    A, I, W = section(D, th)
                    hot[k] = dict(z_rel_mud=z + tb.depth if z < 0 else None, z=z, scf=scf, curve=curve, name=nm, W=W, A=A, t=th,
                                  N=float(m.weight_above(z)), n_bolts=tb.n_bolts, Dbc=tb.D_bolt_circle, As=tb.bolt_As,
                                  FV=0.7 * tb.bolt_fub * tb.bolt_As, d_bolt=tb.bolt_d)
            print(fid, spec["name"], f"kappa {kappa:.2f} f1 design {design['f1']:.4f} true {plant['f1']:.4f} tracked {out['turbines'][-1]['f1_track']:.4f}")
        out["hot"] = hot
        out["rated_U"], out["rotor_D"] = T["rated_U"], T["rotor_D"]
        with open(os.path.join(HERE, f"live_struct_{fid}.json"), "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        print("wrote", fid, os.path.getsize(os.path.join(HERE, f"live_struct_{fid}.json")) // 1024, "kB")


if __name__ == "__main__":
    main()
