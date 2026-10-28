"""Turbine description, illustrative load records, and the reductions the components need.

Frame (fixed to the nacelle): x along the shaft, positive downwind; y up; z lateral.
Blade k sits at azimuth psi_k = psi + 2*pi*k/3 measured from straight up, with radial unit
vector e_r = (0, cos psi_k, sin psi_k) and in-plane tangent e_t = x × e_r = (0, -sin, cos).

All blade-root loads are the loads the blade puts on the hub, so they simply add up.
Accelerations follow the bedplate FE convention: A = (ax, 9.81 + ay, az) and every mass
feels -m A, so gravity is the vertical case with 9.81.
"""
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class Turbine:
    """Representative 14 MW class rotor-nacelle assembly (geared, 3-point mount)."""
    rotor_radius: float = 118.0
    r_root: float = 1.85               # blade-root flange radius from the hub centre, m
    blade_mass: float = 60e3
    blade_cg: float = 30.0             # from the root, m
    hub_mass: float = 136e3            # matches the hub FE model
    nacelle_mass: float = 400e3        # everything on the tower top except the rotor
    nacelle_cg: tuple = (1.0, 2.0, 0.0)
    hub: tuple = (-7.6, 2.55, 0.0)     # hub centre (matches the FE models)
    yaw_centre: tuple = (-0.2, 0.0, 0.0)
    x_main_bearing: float = -3.2
    x_gear_bearing: float = -1.2       # gearbox front carrier bearing, the second shaft support
    rated_rpm: float = 7.2
    rated_torque: float = 18.5e6       # N m (13.9 MW mechanical)
    r_flap: float = 72.0               # radius of the aerodynamic thrust centre, m
    r_edge: float = 65.0               # radius of the aerodynamic torque centre, m

    @property
    def rotor_mass(self):
        return 3 * self.blade_mass + self.hub_mass


def _band(rng, t, fc, bw, sd, n=24):
    """Narrow-band Gaussian-looking process: sum of random-phase cosines around fc (Hz)."""
    f = np.maximum(0.003, fc + bw * (2 * rng.random(n) - 1))
    ph = 2 * np.pi * rng.random(n)
    return sd * np.sqrt(2.0 / n) * np.cos(2 * np.pi * np.outer(t, f) + ph).sum(axis=1)


def synthetic(kind="op", tb=None, dt=0.25, duration=600.0, seed=None):
    """Ten-minute record built blade by blade.

    kind "op": power production just above rated (12 m/s), turbulence, shear, 1P pitch.
    kind "storm": parked, feathered and idling, gusty side loads, a wave slam at 330 s
    that shakes the nacelle.
    """
    tb = tb or Turbine()
    rng = np.random.default_rng(seed if seed is not None else (11 if kind == "op" else 23))
    t = np.arange(0, duration, dt)
    n = t.size
    f1 = 0.167                         # tower first mode, Hz
    if kind == "op":
        rpm = tb.rated_rpm + _band(rng, t, 0.03, 0.03, 0.12)
        u = _band(rng, t, 0.03, 0.03, 0.08)                     # rotor-average wind, relative
        a, b = _band(rng, t, 0.02, 0.02, 0.06), _band(rng, t, 0.02, 0.02, 0.06)   # wind shear modes
        torque = tb.rated_torque * (1 + _band(rng, t, 0.05, 0.05, 0.035))
        ax, az = _band(rng, t, f1, 0.01, 0.12), _band(rng, t, f1, 0.01, 0.06)
        ay = _band(rng, t, 0.5, 0.3, 0.02)
        yaw = np.clip((t - 200) / 40, 0, 1) * 8.0               # one 8° yaw manoeuvre
    elif kind == "storm":
        rpm = 0.25 + _band(rng, t, 0.02, 0.02, 0.05)
        u = _band(rng, t, 0.02, 0.02, 0.5)
        a = b = np.zeros(n)
        torque = _band(rng, t, 0.02, 0.02, 0.8e6)
        slam = np.where(t > 330, np.exp(-(t - 330) / 3.0) * np.sin(2 * np.pi * f1 * (t - 330)), 0)
        ax = _band(rng, t, f1, 0.01, 0.45) + 2.2 * slam
        az = _band(rng, t, f1, 0.01, 0.25)
        ay = _band(rng, t, 0.5, 0.3, 0.05)
        yaw = np.zeros(n)
        f_lat = _band(rng, t, 0.02, 0.02, 0.08e6)               # side gusts on the feathered blades
    else:
        raise ValueError(kind)

    psi = np.cumsum(rpm / 60 * 2 * np.pi) * dt
    A = np.stack([ax, G + ay, az], axis=1)                      # n × 3
    omega = rpm / 60 * 2 * np.pi
    xh = np.array([1.0, 0, 0])
    F_root = np.zeros((3, n, 3)); M_root = np.zeros((3, n, 3))
    blade6 = np.zeros((3, 6, n)); pitch = np.zeros((3, n))
    for k in range(3):
        pk = psi + 2 * np.pi * k / 3
        er = np.stack([np.zeros(n), np.cos(pk), np.sin(pk)], 1)
        et = np.stack([np.zeros(n), -np.sin(pk), np.cos(pk)], 1)
        own = _band(rng, t, 0.25, 0.2, 0.03 if kind == "op" else 0.15)   # what only this blade sees
        if kind == "op":
            rel = 1 + u + 0.05 * np.cos(pk) + a * np.cos(pk) + b * np.sin(pk) + own
            f_flap = 0.70e6 * rel
            q = torque / 3 * (1 + 0.6 * (rel - 1 - u))
            f_in = (q / tb.r_edge)[:, None] * et
            pitch[k] = 4.0 + 20 * u - 0.4 * np.cos(pk)          # collective + 1P individual pitch
        else:
            f_flap = 0.23e6 * (1 + u + own)
            f_in = np.outer(f_lat, [0, 0, 1.0]) + (torque / 3 / tb.r_edge)[:, None] * et
            pitch[k] = 88.0
        Ff = f_flap[:, None] * xh
        r_f, r_e = tb.r_flap - tb.r_root, tb.r_edge - tb.r_root
        F_inert = -tb.blade_mass * A
        F_cf = (tb.blade_mass * omega ** 2 * (tb.r_root + tb.blade_cg))[:, None] * er
        F = Ff + f_in + F_inert + F_cf
        m_pitch = -0.3e6 + _band(rng, t, 0.2, 0.15, 0.15e6)
        M = (np.cross(r_f * er, Ff) + np.cross(r_e * er, f_in) + np.cross(tb.blade_cg * er, F_inert)
             + m_pitch[:, None] * er)
        F_root[k], M_root[k] = F, M
        blade6[k] = [np.sum(F * xh, 1), np.sum(F * et, 1), np.sum(F * er, 1),
                     -np.sum(M * et, 1), np.sum(M * xh, 1), np.sum(M * er, 1)]

    rec = dict(kind=kind, t=t, dt=dt, rpm=rpm, psi=psi, pitch=pitch, yaw=yaw, A=A,
               blade=blade6, F_root=F_root, M_root=M_root, torque_aero=torque)
    rec["hub_F"], rec["hub_M"] = hub_loads(rec, tb)
    rec["tt_F"], rec["tt_M"] = tower_top(rec, tb)
    return rec


def hub_loads(rec, tb):
    """Blade-root loads -> six loads at the hub centre in the fixed frame (plus the hub's own weight)."""
    F = rec["F_root"].sum(0) - tb.hub_mass * rec["A"]
    M = np.zeros_like(F)
    for k in range(3):
        pk = rec["psi"] + 2 * np.pi * k / 3
        er = np.stack([np.zeros_like(pk), np.cos(pk), np.sin(pk)], 1)
        M += rec["M_root"][k] + np.cross(tb.r_root * er, rec["F_root"][k])
    return F, M


def tower_top(rec, tb):
    """Loads the nacelle puts on the tower at the yaw-bearing centre."""
    p_hub = np.array(tb.hub) - tb.yaw_centre
    p_cg = np.array(tb.nacelle_cg) - tb.yaw_centre
    W = -tb.nacelle_mass * rec["A"]
    F = rec["hub_F"] + W
    M = rec["hub_M"] + np.cross(p_hub, rec["hub_F"]) + np.cross(p_cg, W)
    return F, M


def shaft_supports(rec, tb):
    """Statics of the main shaft on two supports (main bearing, gearbox carrier).

    Returns radial forces on the main bearing and the gearbox support (n × 2: y, z) and the
    axial force, all as loads on the shaft from the hub side reversed, i.e. bearing loads.
    """
    F, M = rec["hub_F"], rec["hub_M"]
    a = tb.x_main_bearing - tb.hub[0]               # hub overhang
    s = tb.x_gear_bearing - tb.x_main_bearing       # span
    # moment balance about the main bearing for the gearbox reaction, then force balance
    Rg_y = -(M[:, 2] - a * F[:, 1]) / s
    Rg_z = (M[:, 1] + a * F[:, 2]) / s
    Rb_y = -F[:, 1] - Rg_y
    Rb_z = -F[:, 2] - Rg_z
    # the bearings carry the opposite of the reactions they apply to the shaft
    return np.stack([-Rb_y, -Rb_z], 1), np.stack([-Rg_y, -Rg_z], 1), F[:, 0]


def bedplate_inputs(rec):
    """The nine bedplate unit-case multipliers: hub loads in MN, MN m and accelerations in m/s²."""
    return np.vstack([rec["hub_F"].T / 1e6, rec["hub_M"].T / 1e6, rec["A"].T])
