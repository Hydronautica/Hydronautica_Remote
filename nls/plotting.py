"""Matplotlib helpers for NLS results."""

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_evolution(result, ax=None, cmap="viridis", title=None):
    """Space-time map of |psi|."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(result.x, result.t, result.amplitude, cmap=cmap,
                       shading="auto", rasterized=True)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="|ψ|")
    return ax


def plot_surface_3d(result, ax=None, stride=2, cmap="viridis", title=None):
    """3D surface of |psi|(x, t)."""
    plt = _plt()
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
    X, T = np.meshgrid(result.x[::stride], result.t[::stride])
    ax.plot_surface(X, T, result.amplitude[::stride, ::stride], cmap=cmap,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("|ψ|")
    if title:
        ax.set_title(title)
    return ax


def plot_peak(result, ax=None, exact=None, label="numerical"):
    """Max |psi| vs t, optionally against an exact solution ``exact(x, t)``."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(result.t, result.max_amplitude(), label=label)
    if exact is not None:
        ref = [np.abs(exact(result.x, t)).max() for t in result.t]
        ax.plot(result.t, ref, "--", label="exact")
        ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("max |ψ|")
    return ax


def plot_snapshot(result, t, ax=None, exact=None):
    """|psi|(x) at the stored time closest to t."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    i = int(np.argmin(np.abs(result.t - t)))
    ax.plot(result.x, result.amplitude[i], label=f"numerical, t={result.t[i]:.2f}")
    if exact is not None:
        ax.plot(result.x, np.abs(exact(result.x, result.t[i])), "--", label="exact")
    ax.set_xlabel("x")
    ax.set_ylabel("|ψ|")
    ax.legend()
    return ax
