"""Synthetic farm portfolio for the StormTwin demo.

Three illustrative farms owned by one developer. Names, layouts, storms and
conditions are invented; turbine types are loosely sized after public
reference designs. Select one with the environment variable STORMTWIN_FARM
(A, B or C) when running build_data.py and build_history.py.
"""

import os
from datetime import datetime

TURBINES = {
    "15MW": dict(D_pile=10.0, t_pile=0.095, embedment=42.0, z_interface=15.0, z_hub=150.0,
                 D_tower_top=6.5, t_tower_base=0.042, t_tower_top=0.022,
                 m_rna=1.017e6, J_rna=4.0e8, mean_thrust=0.8e6,
                 n_bolts=160, bolt_d=0.072, bolt_As=3.46e-3, D_bolt_circle=9.80,
                 z_door=21.0, z_platform=20.0,
                 rated_thrust=2.4e6, rated_U=10.6, rotor_D=240.0, label="15 MW class"),
    "10MW": dict(D_pile=8.0, t_pile=0.080, embedment=34.0, z_interface=14.0, z_hub=120.0,
                 D_tower_top=5.5, t_tower_base=0.036, t_tower_top=0.020,
                 m_rna=6.0e5, J_rna=1.8e8, mean_thrust=0.55e6,
                 n_bolts=128, bolt_d=0.064, bolt_As=2.68e-3, D_bolt_circle=7.82,
                 z_door=19.0, z_platform=18.5,
                 rated_thrust=1.6e6, rated_U=11.0, rotor_D=190.0, label="10 MW class"),
}


def grid(prefix, cols, rows, depth0, ddepth_c, ddepth_r, hs0, dhs_c, dhs_r, dx=1.8, dy=1.6, seed0=100):
    out = []
    for r in range(rows):
        for c in range(cols):
            i = cols * r + c
            out.append(dict(name=f"{prefix}{i + 1:02d}", x=c * dx, y=r * dy,
                            depth=round(depth0 + ddepth_c * c + ddepth_r * r, 1),
                            Hs=round(hs0 + dhs_c * (cols - 1 - c) + dhs_r * r, 2),
                            seed=seed0 + i, scour=0.0, preload=1.0))
    return out


def _farm_A():
    farm = grid("A", 4, 3, 41.0, 2.5, 1.2, 8.0, 0.4, 0.2, seed0=100)
    farm[4]["scour"] = 3.5
    farm[1]["preload"] = 0.60
    return dict(
        id="A", name="Northern Bank", region="North Sea type, 41–51 m",
        turbine="15MW", Tp=11.0, farm=farm,
        condition={"A05": "Scour 3.5 m (last survey)", "A02": "Flange bolt preload 60% (last campaign)"},
        storm_name="Winter storm", storm_peak=datetime(2026, 1, 14, 6),
        commissioned=datetime(2019, 3, 1), climate_median=1.55, seasonal=0.42, tp_coef=3.6,
        named_storms=[(datetime(2019, 12, 9, 3), 7.4, 20), (datetime(2020, 2, 15, 18), 8.6, 26),
                      (datetime(2020, 2, 23, 9), 7.1, 16), (datetime(2021, 1, 28, 12), 7.8, 22),
                      (datetime(2022, 2, 18, 15), 8.9, 24), (datetime(2023, 11, 2, 6), 7.6, 18),
                      (datetime(2024, 12, 7, 21), 8.2, 22), (datetime(2025, 10, 30, 0), 6.9, 16)],
        history_seed=7)


def _farm_B():
    farm = grid("B", 4, 2, 30.0, 2.2, 1.5, 7.0, 0.4, 0.3, dx=1.9, dy=1.9, seed0=300)
    farm[2]["scour"] = 2.0
    return dict(
        id="B", name="Atlantic Shelf", region="US East Coast type, 30–38 m",
        turbine="15MW", Tp=9.5, farm=farm,
        condition={"B03": "Scour 2.0 m (post-installation survey)"},
        storm_name="Nor'easter", storm_peak=datetime(2026, 2, 20, 18),
        commissioned=datetime(2023, 9, 1), climate_median=1.35, seasonal=0.38, tp_coef=3.7,
        named_storms=[(datetime(2023, 12, 18, 6), 6.2, 18), (datetime(2024, 1, 10, 0), 6.8, 20),
                      (datetime(2024, 9, 28, 12), 5.9, 16), (datetime(2025, 1, 21, 9), 7.1, 22),
                      (datetime(2025, 11, 14, 3), 6.5, 18)],
        history_seed=11)


def _farm_C():
    farm = grid("C", 3, 2, 22.0, 3.5, 1.5, 5.5, 0.4, 0.2, dx=1.4, dy=1.4, seed0=500)
    farm[3]["preload"] = 0.70
    return dict(
        id="C", name="Eastern Gulf", region="Baltic type, 22–30 m",
        turbine="10MW", Tp=8.0, farm=farm,
        condition={"C04": "Flange bolt preload 70% (last campaign)"},
        storm_name="Autumn gale", storm_peak=datetime(2025, 12, 5, 12),
        commissioned=datetime(2016, 6, 1), climate_median=1.05, seasonal=0.45, tp_coef=3.0,
        named_storms=[(datetime(2016, 12, 26, 6), 5.1, 18), (datetime(2017, 10, 29, 9), 4.8, 16),
                      (datetime(2019, 1, 1, 12), 5.6, 20), (datetime(2020, 2, 9, 18), 5.3, 18),
                      (datetime(2022, 1, 30, 3), 5.8, 22), (datetime(2023, 10, 20, 15), 5.0, 16),
                      (datetime(2024, 11, 23, 0), 5.4, 18)],
        history_seed=19)


FARMS = {"A": _farm_A, "B": _farm_B, "C": _farm_C}


def current():
    """The farm selected by STORMTWIN_FARM (default A)."""
    return FARMS[os.environ.get("STORMTWIN_FARM", "A")]()
