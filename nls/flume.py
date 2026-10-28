"""Design a Peregrine breather experiment in a wave flume.

In a flume the wave train evolves in space, so the NLS is written with x as the
evolution variable (Chabchoub et al. 2011):

    i A_x - (k0 / w0^2) A_tt' - k0^3 |A|^2 A = 0,     t' = t - x / cg,

with eta = Re[A exp(i(k0 x - w0 t))]. With A = a0 conj(psi), eps = k0 a0,

    T = eps^2 k0 (x - x_f),    X = eps w0 (t - t_f - (x - x_f) / cg) / sqrt(2)

this is i psi_T + psi_XX / 2 + |psi|^2 psi = 0, whose Peregrine solution peaks
at 3 a0 at the focus x_f, time t_f. The paddle (x = 0) is driven so that the
water surface there follows the breather at T0 = -eps^2 k0 x_f: a uniform wave
train with a weak, slowly varying modulation that the flume's own nonlinearity
amplifies into one extreme wave at x_f.

The carrier is chosen in deep water (k h > ~3) where the deep-water NLS holds;
k0 uses the finite-depth dispersion relation.
"""

import argparse
import csv

import numpy as np

from .solutions import peregrine

G = 9.81


def wavenumber(omega, h):
    k = omega**2 / G
    for _ in range(50):
        k = omega**2 / (G * np.tanh(k * h))
    return k


def design(T0, eps, h, x_focus, g=G):
    """Key numbers for a carrier period T0 (s), steepness eps = k0 a0, depth h (m)
    and focus distance x_focus (m) from the paddle."""
    w0 = 2 * np.pi / T0
    k0 = wavenumber(w0, h)
    a0 = eps / k0
    cg = 0.5 * w0 / k0 * (1 + 2 * k0 * h / np.sinh(2 * k0 * h))
    Tn0 = -eps**2 * k0 * x_focus
    amp0 = abs(peregrine(np.array([0.0]), Tn0)[0])          # modulation at the paddle
    return dict(T0=T0, eps=eps, h=h, x_focus=x_focus, w0=w0, k0=k0, kh=k0 * h, wavelength=2 * np.pi / k0,
                a0=a0, crest_peak=3 * a0, cg=cg, T_paddle=Tn0, paddle_modulation=amp0,
                peak_steepness=3 * eps, travel_time=x_focus / cg,
                # time half-width of the breather (|X| < 2) at the focus
                half_width=2 * np.sqrt(2) / (eps * w0))


def surface(x, t, d, t_focus, phase=0.0):
    """Linear surface elevation eta(x, t) of the designed breather (deep-water NLS)."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    Tn = d["eps"] ** 2 * d["k0"] * (x - d["x_focus"])
    Xn = d["eps"] * d["w0"] * (t - t_focus - (x - d["x_focus"]) / d["cg"]) / np.sqrt(2)
    A = d["a0"] * np.conj(peregrine(Xn, Tn))
    return np.real(A * np.exp(1j * (d["k0"] * x - d["w0"] * t + phase)))


def crest_phase(d, t_focus, trough=False):
    """Carrier phase that puts a crest (or trough) exactly at the envelope peak. At
    typical steepness the breather envelope is only a few carrier periods wide, so
    an arbitrary phase can miss the 3 a0 peak by 10% or more."""
    # at the focus A = -3 a0, so eta = -3 a0 cos(k0 x_f - w0 t_f + phase)
    return np.pi - (d["k0"] * d["x_focus"] - d["w0"] * t_focus) + (np.pi if trough else 0.0)


def paddle_signal(d, duration=None, fs=50.0, ramp_waves=5, phase=None, trough=False):
    """Surface elevation to generate at the paddle, eta(0, t).

    The breather passes the paddle at t_focus - x_focus / cg; the record starts
    early enough for a smooth ramp of `ramp_waves` carrier periods and ends after
    the breather has passed the focus. By default the carrier phase is set so a
    crest (trough=True: a trough) coincides with the envelope peak.
    Returns t, eta, t_focus, phase.
    """
    lead = 6 * d["half_width"] + ramp_waves * d["T0"]
    t_focus = lead + d["travel_time"]
    if phase is None:
        phase = crest_phase(d, t_focus, trough)
    if duration is None:
        duration = t_focus + 4 * d["half_width"]
    t = np.arange(0.0, duration, 1.0 / fs)
    eta = surface(0.0, t, d, t_focus, phase)
    tr = ramp_waves * d["T0"]
    ramp = np.clip(t / tr, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    end = np.clip((duration - t) / tr, 0, 1)
    end = end * end * (3 - 2 * end)
    return t, eta * ramp * end, t_focus, phase


def flap_transfer(k, h, hinge_depth=None):
    """Wave height / paddle stroke at the water surface for a flap wavemaker hinged at
    the bottom (Biesel); pass hinge_depth for a flap hinged above the bed. Written
    with tanh and sech so it stays finite in deep water, where it tends to 2(kh-1)/kh."""
    d = h if hinge_depth is None else hinge_depth
    kh = np.maximum(np.asarray(k, float) * h, 1e-9)
    kd = np.asarray(k, float) * (h - d)
    th = np.tanh(kh)
    sech2 = 1 / np.cosh(np.minimum(kh, 350)) ** 2
    ratio = np.exp(kd - kh) * (1 + np.exp(-2 * kd)) / (1 + np.exp(-2 * kh))      # cosh(kd) / cosh(kh)
    return 4 * th / kh * (kh * th - 1 + ratio) / (2 * th + 2 * kh * sech2)


def piston_transfer(k, h):
    """Wave height / stroke for a piston wavemaker (Biesel); tends to 2 in deep water."""
    kh = np.maximum(np.asarray(k, float) * h, 1e-9)
    th = np.tanh(kh)
    sech2 = 1 / np.cosh(np.minimum(kh, 350)) ** 2
    return 2 * (2 * th * th) / (2 * th + 2 * kh * sech2)


def paddle_stroke(t, eta, h, kind="flap"):
    """Convert a surface elevation signal at the paddle into paddle stroke (m at the
    still-water level), applying the linear transfer function per frequency."""
    n = len(eta)
    E = np.fft.rfft(eta)
    f = np.fft.rfftfreq(n, t[1] - t[0])
    w = 2 * np.pi * f
    k = np.where(w > 0, wavenumber(np.maximum(w, 1e-6), h), 0.0)
    tf = flap_transfer(k, h) if kind == "flap" else piston_transfer(k, h)
    S = np.where(tf > 1e-6, E / np.where(tf > 1e-6, tf, 1.0), 0.0)
    S[f > 4 / (t[1] - t[0]) / 10] = 0.0                     # drop content far above the carrier
    return np.fft.irfft(S, n)


def table(h, x_focus, periods=(0.6, 0.7, 0.8, 0.9, 1.0), steepness=(0.08, 0.10, 0.12)):
    rows = []
    for T0 in periods:
        for eps in steepness:
            d = design(T0, eps, h, x_focus)
            rows.append(d)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Peregrine breather paddle signal for a wave flume")
    ap.add_argument("--h", type=float, default=1.0, help="water depth (m)")
    ap.add_argument("--T0", type=float, default=0.7, help="carrier period (s)")
    ap.add_argument("--eps", type=float, default=0.10, help="background steepness k0 a0")
    ap.add_argument("--focus", type=float, default=15.0, help="distance from paddle to focus (m)")
    ap.add_argument("--fs", type=float, default=50.0, help="sample rate of the paddle file (Hz)")
    ap.add_argument("--kind", choices=["flap", "piston"], default="flap")
    ap.add_argument("--gauges", default="0,5,10,13,15,17,20", help="gauge positions (m) for predicted records")
    ap.add_argument("--out", default="breather")
    ap.add_argument("--table", action="store_true", help="print a design table for this depth and focus distance")
    a = ap.parse_args(argv)
    if a.table:
        print(f"depth {a.h} m, focus {a.focus} m from the paddle")
        print(" T0 s  eps   lambda m  kh    a0 mm  peak crest mm  peak ak  paddle modulation  T at paddle")
        for d in table(a.h, a.focus):
            print(f" {d['T0']:.1f}  {d['eps']:.2f}  {d['wavelength']:6.2f}  {d['kh']:5.1f}  {1e3 * d['a0']:5.1f}  {1e3 * d['crest_peak']:8.1f}"
                  f"      {d['peak_steepness']:.2f}    {d['paddle_modulation']:.2f} x a0         {d['T_paddle']:.1f}")
        return 0
    d = design(a.T0, a.eps, a.h, a.focus)
    xs = [float(v) for v in a.gauges.split(",")]
    lead = 6 * d["half_width"] + 5 * d["T0"]
    duration = lead + max(max(xs), a.focus) / d["cg"] + 4 * d["half_width"]   # until the breather has passed the last gauge
    t, eta, tf, ph = paddle_signal(d, duration=duration, fs=a.fs)
    stroke = paddle_stroke(t, eta, a.h, a.kind)
    print(f"carrier T0 {a.T0} s, wavelength {d['wavelength']:.2f} m, kh {d['kh']:.1f}, a0 {1e3 * d['a0']:.1f} mm, "
          f"expected peak crest about {1e3 * d['crest_peak']:.0f} mm (ak {d['peak_steepness']:.2f}) at x = {a.focus} m, t = {tf:.1f} s")
    print(f"group velocity {d['cg']:.2f} m/s; normalised time at the paddle {d['T_paddle']:.2f}, modulation there {d['paddle_modulation']:.2f} a0; "
          f"record {t[-1]:.0f} s at {a.fs:g} Hz; peak {a.kind} stroke {1e3 * np.abs(stroke).max():.0f} mm")
    with open(a.out + "_paddle.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "eta_m", f"{a.kind}_stroke_m"])
        for row in zip(t, eta, stroke):
            w.writerow([f"{row[0]:.4f}", f"{row[1]:.6f}", f"{row[2]:.6f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(xs), 1, figsize=(10, 1.4 * len(xs) + 1), sharex=True)
    for ax, x in zip(np.atleast_1d(axs), xs):
        ax.plot(t, 1e3 * surface(x, t, d, tf, ph), lw=0.7, color="#0a6a8a")
        ax.set_ylabel(f"{x:g} m", rotation=0, ha="right", va="center")
        ax.axhline(3e3 * d["a0"], color="#bb3328", ls=":", lw=0.8)
    np.atleast_1d(axs)[0].set_title(f"Predicted surface (mm) at the gauges, linear deep-water NLS: T0 {a.T0} s, ak {a.eps}, focus {a.focus} m", loc="left", fontsize=10)
    np.atleast_1d(axs)[-1].set_xlabel("time since paddle start (s)")
    fig.savefig(a.out + "_gauges.png", dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}_paddle.csv and {a.out}_gauges.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
