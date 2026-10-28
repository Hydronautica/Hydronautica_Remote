"""Small voxel (8-node hexahedron) linear FE for unit-load analysis of castings and forgings.

A part is described by a solid(p) -> bool function on a box. The model is built on a
regular grid, so every element shares one stiffness matrix and assembly is cheap.
Supports are either fixed nodes or springs to ground; both record their reactions so a
part's support loads can be handed to the next part in the load chain.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
FACES = [((-1, 0, 0), [0, 4, 7, 3]), ((1, 0, 0), [1, 2, 6, 5]), ((0, -1, 0), [0, 1, 5, 4]),
         ((0, 1, 0), [3, 7, 6, 2]), ((0, 0, -1), [0, 3, 2, 1]), ((0, 0, 1), [4, 5, 6, 7])]


def elasticity(E, nu):
    c = E / ((1 + nu) * (1 - 2 * nu))
    g = (1 - 2 * nu) / 2
    return c * np.array([[1 - nu, nu, nu, 0, 0, 0], [nu, 1 - nu, nu, 0, 0, 0], [nu, nu, 1 - nu, 0, 0, 0],
                         [0, 0, 0, g, 0, 0], [0, 0, 0, 0, g, 0], [0, 0, 0, 0, 0, g]])


def hex_matrices(h, D):
    """Element stiffness (2×2×2 Gauss) and the stress-recovery matrix at the centre for a cube of side h."""
    s, t, w = (2 * CORNERS - 1).T

    def B_at(xi, eta, zeta):
        dN = np.array([s * (1 + t * eta) * (1 + w * zeta), t * (1 + s * xi) * (1 + w * zeta),
                       w * (1 + s * xi) * (1 + t * eta)]) / 8 * 2 / h
        B = np.zeros((6, 24))
        for a in range(8):
            dx, dy, dz = dN[:, a]
            B[:, 3 * a:3 * a + 3] = [[dx, 0, 0], [0, dy, 0], [0, 0, dz], [dy, dx, 0], [0, dz, dy], [dz, 0, dx]]
        return B
    g = 1 / np.sqrt(3)
    K = sum(B_at(a, b, c).T @ D @ B_at(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)) * (h / 2) ** 3
    return K, D @ B_at(0, 0, 0)


def rbe3(X, point, F, M):
    """Nodal forces on points X (n×3) with resultant F and moment M about `point` (least-squares spread)."""
    c = X.mean(0)
    r = X - c
    Mc = np.asarray(M, float) + np.cross(np.asarray(point, float) - c, F)
    J = np.sum(r * r) * np.eye(3) - r.T @ r
    w = np.linalg.solve(J, Mc)
    return np.asarray(F, float) / len(X) + np.cross(w, r)


class VoxelModel:
    def __init__(self, solid, bounds, h, E=169e9, nu=0.275, rho=7100.0, extra_mass=None):
        (x0, x1), (y0, y1), (z0, z1) = bounds
        self.h, self.origin = h, np.array([x0, y0, z0])
        nx, ny, nz = (round((b - a) / h) for a, b in bounds)
        self.shape = (nx, ny, nz)
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        cen = np.stack([x0 + (ii + 0.5) * h, y0 + (jj + 0.5) * h, z0 + (kk + 0.5) * h], -1)
        self.filled = solid(cen)
        self.el = np.argwhere(self.filled)
        nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        conn = np.stack([nid(*(self.el + c).T) for c in CORNERS], 1)
        used, conn = np.unique(conn, return_inverse=True)
        self.conn = conn.reshape(-1, 8)
        self.grid = np.stack([used // ((ny + 1) * (nz + 1)), (used // (nz + 1)) % (ny + 1), used % (nz + 1)], 1)
        self.X = self.origin + self.grid * h
        self.nn = len(used)
        self.ndof = 3 * self.nn
        D = elasticity(E, nu)
        self.Ke, self.DB0 = hex_matrices(h, D)
        dens = rho * np.ones(len(self.el))
        if extra_mass is not None:
            dens = dens * extra_mass(self.origin + (self.el + 0.5) * h)
        self.mass_el = dens * h ** 3
        self.nodal_mass = np.zeros(self.nn)
        np.add.at(self.nodal_mass, self.conn.ravel(), np.repeat(self.mass_el / 8, 8))
        self.dofs = (3 * self.conn[:, :, None] + np.arange(3)).reshape(-1, 24)
        self.springs, self.fixed = {}, {}
        self._extra = []

    @property
    def mass(self):
        return self.mass_el.sum()

    @property
    def cg(self):
        return (self.nodal_mass[:, None] * self.X).sum(0) / self.nodal_mass.sum()

    def where(self, fn):
        return np.flatnonzero(fn(self.X))

    def add_springs(self, name, nodes, k_total, proj):
        """Springs from nodes to ground; proj is one 3×3 direction matrix or a list per node."""
        P = np.broadcast_to(proj, (len(nodes), 3, 3)) if np.ndim(proj) == 2 else np.asarray(proj)
        kb = k_total / len(nodes) * P
        self.springs[name] = (np.asarray(nodes), kb)
        d = 3 * np.asarray(nodes)[:, None] + np.arange(3)
        self._extra.append((np.repeat(d, 3, 1).ravel(), np.tile(d, 3).ravel(), kb.reshape(len(nodes), 9).ravel()))

    def add_mean_spring(self, name, nodes, k, direction):
        """Spring on the mean displacement of a node set along one direction (no tilt stiffness)."""
        nodes = np.asarray(nodes)
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        idx = (3 * nodes[:, None] + np.arange(3)).ravel()
        w = np.tile(d, len(nodes)) / len(nodes)
        self._extra.append((np.repeat(idx, len(idx)), np.tile(idx, len(idx)), k * np.outer(w, w).ravel()))
        self.means = getattr(self, "means", {})
        self.means[name] = (nodes, k, d)

    def add_ties(self, name, a, b, k_total, proj):
        """Springs between paired nodes a[i]-b[i] (bearings, gear mesh) inside one model."""
        P = np.broadcast_to(proj, (len(a), 3, 3)) if np.ndim(proj) == 2 else np.asarray(proj)
        kb = k_total / len(a) * P
        da = 3 * np.asarray(a)[:, None] + np.arange(3)
        db = 3 * np.asarray(b)[:, None] + np.arange(3)
        for d1, d2, sgn in ((da, da, 1), (db, db, 1), (da, db, -1), (db, da, -1)):
            self._extra.append((np.repeat(d1, 3, 1).ravel(), np.tile(d2, 3).ravel(), (sgn * kb).reshape(len(a), 9).ravel()))
        self.ties = getattr(self, "ties", {})
        self.ties[name] = (np.asarray(a), np.asarray(b), kb)

    def fix(self, name, nodes):
        self.fixed[name] = np.asarray(nodes)

    def load(self, nodes, point, F=(0, 0, 0), M=(0, 0, 0)):
        f = np.zeros(self.ndof)
        f[(3 * np.asarray(nodes)[:, None] + np.arange(3)).ravel()] = rbe3(self.X[nodes], point, F, M).ravel()
        return f

    def accel(self, a):
        """Body force for the support frame accelerating at a (d'Alembert: every mass feels -m a)."""
        return -(self.nodal_mass[:, None] * np.asarray(a, float)).ravel()

    def solve(self, F):
        F = np.atleast_2d(F.T).T
        n = len(self.el)
        rows = np.repeat(self.dofs, 24, axis=1).ravel()
        cols = np.tile(self.dofs, (1, 24)).ravel()
        vals = np.tile(self.Ke.ravel(), n)
        R = [rows] + [e[0] for e in self._extra]
        C = [cols] + [e[1] for e in self._extra]
        V = [vals] + [e[2] for e in self._extra]
        self.K = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(self.ndof, self.ndof)).tocsc()
        fixed = np.concatenate([n for n in self.fixed.values()]) if self.fixed else np.zeros(0, int)
        fdof = (3 * fixed[:, None] + np.arange(3)).ravel()
        free = np.setdiff1d(np.arange(self.ndof), fdof)
        lu = splu(self.K[free][:, free], permc_spec="MMD_AT_PLUS_A")
        U = np.zeros_like(F, dtype=float)
        U[free] = lu.solve(F[free])
        self.F, self.U = F, U
        return U

    def reaction(self, name, point):
        """Resultant load (Fx, Fy, Fz, Mx, My, Mz about point) the part puts on a support, per case."""
        if name in self.springs:
            nodes, kb = self.springs[name]
            u = self.U[(3 * nodes[:, None] + np.arange(3))]                # n × 3 × cases
            f = np.einsum("nij,njc->nic", kb, u)
            if name in getattr(self, "means", {}):
                _, k, d = self.means[name]
                um = np.einsum("j,njc->c", d, u) / len(nodes)
                f = f + (k * um / len(nodes))[None, None, :] * d[None, :, None]
        else:
            nodes = self.fixed[name]
            d = (3 * nodes[:, None] + np.arange(3))
            f = (self.F - self.K @ self.U)[d]
        r = self.X[nodes] - point
        M = np.cross(r[:, :, None], f, axis=1).sum(0)
        return np.vstack([f.sum(0), M])

    def element_stress(self):
        return np.einsum("sk,ekc->ecs", self.DB0, self.U[self.dofs])        # el × cases × 6

    def surface(self):
        """Outer faces as triangles, vertex positions, per-vertex stress (nv × cases × 6) and displacement."""
        nx, ny, nz = self.shape
        occ = np.zeros((nx + 2, ny + 2, nz + 2), bool)
        occ[1:-1, 1:-1, 1:-1] = self.filled
        quads, qel = [], []
        for (di, dj, dk), cidx in FACES:
            sel = np.flatnonzero(~occ[self.el[:, 0] + 1 + di, self.el[:, 1] + 1 + dj, self.el[:, 2] + 1 + dk])
            quads.append(self.conn[sel][:, cidx]); qel.append(sel)
        quads, qel = np.vstack(quads), np.concatenate(qel)
        sv, quads = np.unique(quads, return_inverse=True)
        quads = quads.reshape(-1, 4)
        sig = self.element_stress()
        acc = np.zeros((len(sv),) + sig.shape[1:]); cnt = np.zeros(len(sv))
        for q in range(4):
            np.add.at(acc, quads[:, q], sig[qel]); np.add.at(cnt, quads[:, q], 1)
        tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        disp = self.U[(3 * sv[:, None] + np.arange(3))]
        return dict(nodes=sv, grid=self.grid[sv], tris=tris, stress=acc / cnt[:, None, None], disp=disp)


def von_mises(s):
    a, b, c, xy, yz, zx = np.moveaxis(s, -1, 0)
    return np.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) + 3 * (xy ** 2 + yz ** 2 + zx ** 2))
