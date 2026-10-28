"""Cycle counting and S-N damage shared by all components."""
from dataclasses import dataclass

import numpy as np

YEAR = 8766 * 3600.0          # s
DESIGN_LIFE = 20 * YEAR       # s, the reference every ageing rate is measured against


def turning_points(x):
    """Indices of the peaks and valleys of x (ends included, flat runs collapsed)."""
    x = np.asarray(x, float)
    keep = np.r_[True, np.diff(x) != 0]
    idx = np.flatnonzero(keep)
    if idx.size < 3:
        return idx
    d = np.sign(np.diff(x[idx]))
    turn = np.flatnonzero(d[1:] != d[:-1]) + 1
    return np.r_[idx[0], idx[turn], idx[-1]]


def rainflow(x):
    """ASTM E1049 rainflow count.

    Returns (ranges, means, counts, ends): counts are 1 for full cycles and 0.5 for the
    residual half cycles; ends is the sample index at which each cycle closes, so damage
    can be placed in time.
    """
    x = np.asarray(x, float)
    tp = turning_points(x)
    rng, mean, cnt, end = [], [], [], []
    stack = []
    for i in tp:
        stack.append(i)
        while len(stack) >= 3:
            a, b, c = x[stack[-3]], x[stack[-2]], x[stack[-1]]
            X, Y = abs(c - b), abs(b - a)
            if X < Y:
                break
            if len(stack) == 3:       # Y contains the starting point: half cycle
                rng.append(Y); mean.append(0.5 * (a + b)); cnt.append(0.5); end.append(stack[-2])
                stack.pop(0)
            else:
                rng.append(Y); mean.append(0.5 * (a + b)); cnt.append(1.0); end.append(i)
                del stack[-3:-1]
    for j in range(len(stack) - 1):
        a, b = x[stack[j]], x[stack[j + 1]]
        rng.append(abs(b - a)); mean.append(0.5 * (a + b)); cnt.append(0.5); end.append(stack[j + 1])
    return np.array(rng), np.array(mean), np.array(cnt), np.array(end, int)


@dataclass
class SN:
    """Bilinear S-N curve: N = n_knee (s_knee / s)^m, slope m1 above the knee and m2 below.

    s is whatever the curve is written in (a stress range for welds and bolts, an amplitude
    for shafts); the caller passes the matching quantity.
    """
    s_knee: float
    n_knee: float
    m1: float
    m2: float = None

    def cycles(self, s):
        s = np.maximum(np.asarray(s, float), 1e-30)
        m = np.where(s >= self.s_knee, self.m1, self.m2 if self.m2 else self.m1)
        return self.n_knee * (self.s_knee / s) ** m

    def damage(self, s, counts=1.0):
        return np.sum(np.asarray(counts, float) / self.cycles(s))


def damage_equivalent(ranges, counts, m, n_ref):
    """Constant range that does the same damage in n_ref cycles with a single-slope curve."""
    return (np.sum(np.asarray(counts) * np.asarray(ranges) ** m) / n_ref) ** (1.0 / m)


def ageing(damage, duration):
    """Design-life seconds used per second: 1 means using life exactly at the design rate."""
    return damage * DESIGN_LIFE / duration


def place(n, ends, dmg):
    """Damage per sample from per-cycle damage closing at the given sample indices."""
    out = np.zeros(n)
    np.add.at(out, ends, dmg)
    return out
