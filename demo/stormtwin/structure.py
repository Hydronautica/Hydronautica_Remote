"""Demo-grade global and hot-spot models for the StormTwin showcase.

These are compact stand-ins for the production chain (beam tower/RNA/blade
model, hybrid shell monopile, nonlinear p-y foundation, local FE submodels of
hot spots). They are physically reasonable but simplified, and every number
they produce in the demo is labelled as synthetic.

Turbine: a 15 MW-class monopile turbine loosely sized after public reference
designs (10 m monopile, hub at +150 m, RNA ~1000 t). Not a real asset.
"""

from dataclasses import dataclass, field

import numpy as np

RHO_W = 1025.0
G = 9.81
E_STEEL = 210e9
RHO_STEEL = 7850.0


# --- geometry ------------------------------------------------------------------

@dataclass
class Turbine:
    depth: float = 45.0              # m
    embedment: float = 42.0          # m below mudline
    scour: float = 0.0               # m of soil removed around the pile (current scour hole)
    scour_max: float = 0.0           # deepest scour reached so far; soil between scour and scour_max is loose backfill
    backfill_factor: float = 0.25    # p-y stiffness and capacity of backfilled sand relative to the original soil
    surcharge: float = 0.0           # Pa on the seabed around the pile (rock scour protection)
    D_pile: float = 10.0
    t_pile: float = 0.095
    z_interface: float = 15.0        # tower base flange (m above MSL)
    z_hub: float = 150.0
    D_tower_top: float = 6.5
    t_tower_base: float = 0.042
    t_tower_top: float = 0.022
    m_rna: float = 1.017e6           # kg
    J_rna: float = 4.0e8             # kg m^2 about the horizontal axis
    mean_thrust: float = 0.8e6       # N, parked rotor + nacelle drag at storm wind
    damping_ratio: float = 0.015
    # flange
    n_bolts: int = 160
    bolt_d: float = 0.072            # M72
    bolt_As: float = 3.46e-3         # m^2
    bolt_fub: float = 1000e6
    preload_frac: float = 1.0        # fraction of design preload remaining
    D_bolt_circle: float = 9.80
    # hot-spot elevations
    z_door: float = 21.0
    z_platform: float = 20.0         # underside of the access platform
    fy: float = 355e6
    rated_thrust: float = 2.4e6      # N at rated wind
    rated_U: float = 10.6            # m/s
    rotor_D: float = 240.0
    label: str = "15 MW class"
    name: str = "T00"

    def diameter(self, z):
        z = np.asarray(z, float)
        frac = np.clip((z - self.z_interface) / (self.z_hub - 5 - self.z_interface), 0, 1)
        return np.where(z <= self.z_interface, self.D_pile,
                        self.D_pile + frac * (self.D_tower_top - self.D_pile))

    def thickness(self, z):
        z = np.asarray(z, float)
        frac = np.clip((z - self.z_interface) / (self.z_hub - 5 - self.z_interface), 0, 1)
        return np.where(z <= self.z_interface, self.t_pile,
                        self.t_tower_base + frac * (self.t_tower_top - self.t_tower_base))


def section(D, t):
    """Area, second moment and elastic modulus of a thin circular tube."""
    A = np.pi * D * t
    I = np.pi * D**3 * t / 8
    return A, I, I / (D / 2)


# --- p-y springs (API sand, cyclic) --------------------------------------------

def api_sand(z_below, D, phi=35.0, gamma=10e3):
    """Ultimate resistance p_u (N/m) and initial modulus k z (N/m^2) at depth z_below."""
    C1, C2, C3 = 2.97, 3.42, 53.7      # phi' = 35 deg
    k = 22e6                           # N/m^3, medium dense sand, submerged
    z = np.maximum(z_below, 0.05)
    pu = np.minimum((C1 * z + C2 * D) * gamma * z, C3 * D * gamma * z)
    return 0.9 * pu, k * z             # A = 0.9 (cyclic)


# --- finite-element model --------------------------------------------------------

@dataclass
class Model:
    tb: Turbine
    z: np.ndarray = field(init=False)

    def __post_init__(self):
        tb = self.tb
        mud = -tb.depth
        toe = mud - tb.embedment
        z = np.concatenate([
            np.linspace(toe, mud, 22, endpoint=False),
            np.linspace(mud, -12, 17, endpoint=False),
            np.linspace(-12, 20, 33, endpoint=False),
            np.linspace(20, tb.z_hub, 27),
        ])
        self.z = z
        n = z.size
        self.ndof = 2 * n
        K = np.zeros((2 * n, 2 * n))
        M = np.zeros((2 * n, 2 * n))
        self.EI = []
        for e in range(n - 1):
            L = z[e + 1] - z[e]
            zm = 0.5 * (z[e] + z[e + 1])
            D, t = tb.diameter(zm), tb.thickness(zm)
            A, I, _ = section(D, t)
            EI = E_STEEL * I
            self.EI.append(EI)
            ke = EI / L**3 * np.array([[12, 6 * L, -12, 6 * L],
                                       [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                                       [-12, -6 * L, 12, -6 * L],
                                       [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
            m_line = RHO_STEEL * A * (1.1 if zm > tb.z_interface else 1.0)
            if mud < zm < 0:            # internal + added water mass in the water column
                m_line += RHO_W * np.pi * (D - 2 * t) ** 2 / 4 + RHO_W * np.pi * D**2 / 4
            me = m_line * L / 420 * np.array([[156, 22 * L, 54, -13 * L],
                                              [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                                              [54, 13 * L, 156, -22 * L],
                                              [-13 * L, -3 * L**2, -22 * L, 4 * L**2]])
            idx = np.arange(2 * e, 2 * e + 4)
            K[np.ix_(idx, idx)] += ke
            M[np.ix_(idx, idx)] += me
        top = 2 * (n - 1)
        M[top, top] += tb.m_rna
        M[top + 1, top + 1] += tb.J_rna
        self.K, self.M = K, M
        # soil springs at nodes below the (scoured) mudline
        trib = np.zeros(n)
        trib[1:-1] = 0.5 * (z[2:] - z[:-2])
        trib[0] = 0.5 * (z[1] - z[0])
        trib[-1] = 0.5 * (z[-1] - z[-2])
        self.trib = trib
        z_soil_top = mud - tb.scour
        below = z_soil_top - z
        self.soil = np.flatnonzero(below > 0)
        # a surcharge acts as extra overburden: equivalent depth of submerged sand
        pu, kz = api_sand(below[self.soil] + tb.surcharge / 10e3, tb.D_pile)
        # refilled sand above the deepest scour level is loose
        loose = np.where(below[self.soil] < max(tb.scour_max - tb.scour, 0.0), tb.backfill_factor, 1.0)
        self.p_u = pu * trib[self.soil] * loose
        self.k0 = kz * trib[self.soil] * loose
        self.wet = np.flatnonzero((z > mud) & (z < tb.z_interface + 10))
        # weight above each node (for axial stress at hot spots)
        self._mass_above = None
        self.C = None

    # nonlinear spring force and tangent stiffness at lateral displacements y
    def springs(self, y):
        arg = self.k0 * y / self.p_u
        return self.p_u * np.tanh(arg), self.k0 / np.cosh(arg) ** 2

    def restoring(self, u):
        r = self.K @ u
        kt = np.zeros(self.ndof)
        f, k = self.springs(u[2 * self.soil])
        r[2 * self.soil] += f
        kt[2 * self.soil] = k
        return r, kt

    def frequencies(self, u=None, n=3):
        u = np.zeros(self.ndof) if u is None else u
        _, kt = self.restoring(u)
        Kt = self.K + np.diag(kt)
        w2 = np.linalg.eigvals(np.linalg.solve(self.M, Kt))
        w = np.sort(np.sqrt(np.abs(w2.real)))
        return w[:n] / (2 * np.pi)

    def setup_damping(self):
        f = self.frequencies(n=2)
        w1, w2 = 2 * np.pi * f
        zeta = self.tb.damping_ratio
        a = 2 * zeta * w1 * w2 / (w1 + w2)
        b = 2 * zeta / (w1 + w2)
        self.C = a * self.M + b * (self.K + np.diag(self.restoring(np.zeros(self.ndof))[1]))
        return f

    def static(self, F, u0=None, tol=1e-6):
        u = np.zeros(self.ndof) if u0 is None else u0.copy()
        for _ in range(50):
            r, kt = self.restoring(u)
            du = np.linalg.solve(self.K + np.diag(kt), F - r)
            u += du
            if np.max(np.abs(du)) < tol * max(1e-3, np.max(np.abs(u))):
                break
        return u

    def mean_load(self):
        F = np.zeros(self.ndof)
        F[2 * (self.z.size - 1)] = self.tb.mean_thrust
        return F

    def run(self, t, F_dyn, u0=None, v0=None):
        """Newmark (average acceleration) with Newton iterations.

        F_dyn: (len(t), n_nodes) lateral nodal forces in addition to the mean load.
        Returns displacements (len(t), ndof) and velocities at the end.
        """
        if self.C is None:
            self.setup_damping()
        dt = t[1] - t[0]
        F0 = self.mean_load()
        u = self.static(F0) if u0 is None else u0.copy()
        v = np.zeros(self.ndof) if v0 is None else v0.copy()
        r, _ = self.restoring(u)
        F = F0.copy()
        F[0::2] += F_dyn[0]
        a = np.linalg.solve(self.M, F - self.C @ v - r)
        U = np.empty((t.size, self.ndof))
        U[0] = u
        b0, b1 = 4 / dt**2, 2 / dt
        for i in range(1, t.size):
            F = F0.copy()
            F[0::2] += F_dyn[i]
            u_new = u + dt * v + 0.25 * dt**2 * a      # predictor
            for _ in range(20):
                a_new = b0 * (u_new - u) - 4 / dt * v - a
                v_new = v + 0.5 * dt * (a + a_new)
                r, kt = self.restoring(u_new)
                res = F - self.M @ a_new - self.C @ v_new - r
                Keff = b0 * self.M + b1 * self.C + self.K + np.diag(kt)
                du = np.linalg.solve(Keff, res)
                u_new += du
                if np.max(np.abs(du)) < 1e-9 + 1e-7 * np.max(np.abs(u_new)):
                    break
            a_new = b0 * (u_new - u) - 4 / dt * v - a
            v = v + 0.5 * dt * (a + a_new)
            u, a = u_new, a_new
            U[i] = u
        return U, v

    def moment(self, U, z_target):
        """Bending moment history (N m) at elevation z_target from element end forces."""
        e = int(np.clip(np.searchsorted(self.z, z_target) - 1, 0, self.z.size - 2))
        L = self.z[e + 1] - self.z[e]
        EI = self.EI[e]
        xi = (z_target - self.z[e]) / L
        ue = U[:, 2 * e:2 * e + 4]
        # curvature from Hermite shape functions: M = EI * w''
        B = np.array([(-6 + 12 * xi) / L**2, (-4 + 6 * xi) / L,
                      (6 - 12 * xi) / L**2, (-2 + 6 * xi) / L])
        return EI * ue @ B

    def node(self, z_target):
        return int(np.argmin(np.abs(self.z - z_target)))

    def weight_above(self, z_target):
        tb = self.tb
        m = tb.m_rna
        for e in range(self.z.size - 1):
            zm = 0.5 * (self.z[e] + self.z[e + 1])
            if zm > z_target:
                A, _, _ = section(tb.diameter(zm), tb.thickness(zm))
                m += RHO_STEEL * A * 1.1 * (self.z[e + 1] - self.z[e])
        return m * G


def thrust(U, tb=None):
    """Rotor thrust (N): cut-in 3 m/s, peak at rated wind, cut-out 25 m/s, then
    idling (feathered drag). Scaled to the turbine's rated thrust."""
    Tr = tb.rated_thrust if tb else 2.4e6
    Ur = tb.rated_U if tb else 10.6
    idle = tb.mean_thrust if tb else 0.8e6          # storm idling thrust at ~35 m/s
    if U < 3:
        return 2e4 * U * U / 9 * Tr / 2.4e6
    if U <= Ur:
        return Tr * (U / Ur) ** 2
    if U <= 25:
        return Tr * (Ur / U) ** 1.1
    return idle * (U / 35) ** 2


def aero_damping(U):
    """Fore-aft aerodynamic damping ratio added by the rotor."""
    return 0.035 if 3 <= U <= 25 else 0.003


# --- hydrodynamic loads ------------------------------------------------------------

def morison_nodal(model, eta, u, ax, Cd=1.0, Cm=2.0):
    """Nodal Morison forces (N) for kinematics u, ax of shape (n_t, n_wet)."""
    tb = model.tb
    zw = model.z[model.wet]
    D = tb.D_pile
    f = RHO_W * Cm * np.pi * D**2 / 4 * ax + 0.5 * RHO_W * Cd * D * np.abs(u) * u
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.where(zw[None, :] <= eta[:, None], f, 0.0)
    F = np.zeros((eta.size, model.z.size))
    F[:, model.wet] = f * model.trib[model.wet][None, :]
    return F


def slam_nodal(model, t, t_impact, crest, c, curl=0.46):
    """Wienke & Oumeraci (2005) slamming line load, distributed over the top
    curl * crest of the wave front, as nodal forces (N)."""
    R = model.tb.D_pile / 2
    Ts = 13 * R / (32 * c)
    tau = t - t_impact
    shape = np.where((tau >= 0) & (tau <= Ts), 1 - tau / Ts, 0.0)
    f_line = RHO_W * R * c**2 * 2 * np.pi * shape        # N/m, triangular decay
    z = model.z
    top, bottom = crest, crest - curl * crest
    band = (z >= bottom) & (z <= top)
    F = np.zeros((t.size, z.size))
    F[:, band] = f_line[:, None] * model.trib[band][None, :]
    return F


# --- bolted flange (Schmidt-Neuper-type trilinear) -----------------------------------

def bolt_force(tb, Z):
    """Bolt force (N) for segment tension Z (N) per bolt.

    Trilinear Schmidt-Neuper-type model: closed joint (small load share p),
    progressive gap opening, open joint with prying (lever lam). Stand-in for
    the local FE flange segment with contact.
    """
    Fv = 0.7 * tb.bolt_fub * tb.bolt_As * tb.preload_frac
    p, lam = 0.12, 2.1
    Z1 = 0.40 * Fv      # gap starts to open (sized to stay closed in normal operation)
    Z2 = 0.80 * Fv      # gap fully open
    F1 = Fv + p * Z1
    F2 = max(lam * Z2, F1 * 1.05)
    Z = np.asarray(Z, float)
    return np.where(Z <= Z1, Fv + p * Z,
                    np.where(Z <= Z2, F1 + (F2 - F1) * (Z - Z1) / (Z2 - Z1),
                             F2 + lam * (Z - Z2))), dict(Fv=Fv, Z1=Z1, Z2=Z2, F1=F1, F2=F2)


def segment_force(tb, M, N):
    """Tension per bolt segment from bending moment M and compressive weight N."""
    return 4 * M / (tb.n_bolts * tb.D_bolt_circle) - N / tb.n_bolts


# --- fatigue ------------------------------------------------------------------------

def rainflow(x):
    """Rainflow cycle counting (ASTM E1049). Returns (ranges, counts)."""
    x = np.asarray(x, float)
    d = np.diff(x)
    keep = np.concatenate([[True], d[1:] * d[:-1] < 0, [True]])
    tp = x[np.concatenate([[True], keep[1:-1], [True]])] if x.size > 2 else x
    # turning points
    s = [tp[0]]
    for v in tp[1:]:
        if len(s) >= 2 and (s[-1] - s[-2]) * (v - s[-1]) > 0:
            s[-1] = v
        elif v != s[-1]:
            s.append(v)
    ranges, counts = [], []
    stack = []
    for v in s:
        stack.append(v)
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            if len(stack) == 3:
                ranges.append(Y)
                counts.append(0.5)
                stack.pop(0)
            else:
                ranges.append(Y)
                counts.append(1.0)
                del stack[-3:-1]
    for i in range(len(stack) - 1):
        ranges.append(abs(stack[i + 1] - stack[i]))
        counts.append(0.5)
    return np.array(ranges), np.array(counts)


SN_CURVES = {
    # DNV-RP-C203: log a1, m1, log a2, m2, N at knee, thickness exponent k
    "D_air": (12.164, 3.0, 15.606, 5.0, 1e7, 0.20),
    "D_seawater_cp": (11.764, 3.0, 15.606, 5.0, 1e6, 0.20),
}


def miner_damage(ranges_pa, counts, curve, t_ref=None, t=0.025):
    """Miner damage for stress ranges (Pa) on a DNV-RP-C203 S-N curve."""
    if curve == "bolt":
        # EN 1993-1-9 detail 50 for bolts in tension, size factor (30/d)^0.25
        return None
    la1, m1, la2, m2, Nk, k = SN_CURVES[curve]
    s = ranges_pa / 1e6 * max(1.0, t / 0.025) ** k
    N1 = 10 ** (la1 - m1 * np.log10(np.maximum(s, 1e-9)))
    N2 = 10 ** (la2 - m2 * np.log10(np.maximum(s, 1e-9)))
    N = np.where(N1 <= Nk, N1, N2)
    return float(np.sum(counts / N))


def bolt_damage(ranges_pa, counts, d=0.072):
    """EN 1993-1-9 detail category 50 (bolts in tension), m = 3 to 5e6, m = 5 to 1e8."""
    ks = (0.030 / d) ** 0.25 if d > 0.030 else 1.0
    sc = 50e6 * ks
    sD = sc * (2 / 5) ** (1 / 3)          # constant amplitude limit at 5e6
    sL = sD * (5 / 100) ** (1 / 5)        # cut-off at 1e8
    s = ranges_pa
    N = np.where(s >= sD, 2e6 * (sc / np.maximum(s, 1)) ** 3,
                 np.where(s >= sL, 5e6 * (sD / np.maximum(s, 1)) ** 5, np.inf))
    return float(np.sum(counts / N))
