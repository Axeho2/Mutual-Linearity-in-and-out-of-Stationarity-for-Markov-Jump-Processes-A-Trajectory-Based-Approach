#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theorem-compatible Gillespie simulation for an N-site simple exclusion process (SEP)
coupled to left/right particle reservoirs.

This file is configured for N = 5 sites, giving a larger system than the two-site
example while keeping a physically transparent negative-slope setup.

Negative-slope version:
- ONLY ONE microscopic jump channel is perturbed:
      00...0 -> 10...0  via the left reservoir
- Q1 is the net particle current into the right reservoir
- Q2 is the dwelling time in the completely empty state 00...0

As the left-boundary injection rate lambda increases, the right-reservoir current
typically increases while the time spent in the empty configuration decreases,
leading to a negative slope in the parametric plot of Q1-hat versus Q2-hat.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


# ============================================================
# User-facing model parameters
# ============================================================

N_SITES = 8  # number of sites in the SEP (excluding reservoirs)

SINGLE_COLUMN_WIDTH_IN = 1.95
FIG_HEIGHT_IN = 1.55

OMEGAS = np.array([0.4, 0.5, 0.6], dtype=np.float64)
LAMBDAS = np.linspace(0.1, 5.0, 9)

# Boundary and hopping rates
ALPHA_OUT_L = 0.60   # left extraction
BETA_IN_R = 0.25     # right injection
BETA_OUT_R = 1.35    # right extraction
HOP_RIGHT = 1.00
HOP_LEFT = 0.45

OMEGA_STYLES = {
    0.4: {"marker": "o", "linestyle": "-",  "linewidth": 0.95, "markersize": 2.7, "color": "blue"},
    0.5: {"marker": "s", "linestyle": "--", "linewidth": 0.95, "markersize": 2.6, "color": "orange"},
    0.6: {"marker": "^", "linestyle": "-.", "linewidth": 0.95, "markersize": 2.7, "color": "green"},
}


# ============================================================
# Plot style
# ============================================================

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def format_axes(ax):
    ax.tick_params(which="major", length=3.2)
    ax.tick_params(which="minor", length=1.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def get_script_dir():
    return Path(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Generic multichannel Gillespie generator
# ============================================================

@njit(cache=True)
def draw_initial_state_from_prob(p0):
    u = np.random.random()
    c = 0.0
    for i in range(p0.shape[0]):
        c += p0[i]
        if u <= c:
            return i
    return p0.shape[0] - 1


@njit(cache=True)
def traj_generator_channels(n_states, p0, t_max, max_jumps,
                            ch_from, ch_to, ch_rate):
    n_channels = ch_from.shape[0]
    jump_times = np.empty(max_jumps, dtype=np.float64)
    states = np.empty(max_jumps + 1, dtype=np.int64)
    channel_ids = np.empty(max_jumps, dtype=np.int64)

    s = draw_initial_state_from_prob(p0)
    states[0] = s

    t = 0.0
    n_jumps = 0

    while t < t_max and n_jumps < max_jumps:
        total_rate = 0.0
        for a in range(n_channels):
            if ch_from[a] == s:
                total_rate += ch_rate[a]

        if total_rate <= 0.0:
            break

        u = np.random.random()
        if u < 1e-15:
            u = 1e-15
        dt = -np.log(u) / total_rate
        t_next = t + dt

        if t_next > t_max:
            break

        u2 = np.random.random() * total_rate
        accum = 0.0
        chosen = -1

        for a in range(n_channels):
            if ch_from[a] == s:
                accum += ch_rate[a]
                if u2 <= accum:
                    chosen = a
                    break

        if chosen < 0:
            break

        jump_times[n_jumps] = t_next
        channel_ids[n_jumps] = chosen
        s = ch_to[chosen]
        states[n_jumps + 1] = s

        n_jumps += 1
        t = t_next

    return n_jumps, jump_times, states, channel_ids


@njit(cache=True)
def qhat_from_trajectory_general(n_jumps, jump_times, states, channel_ids, t_max,
                                 omegas, a_state, c_channel):
    """
    General state-counting observable:
        Q(tau) = ∫ a_{x_t} dt + Σ_k c_{channel_k}

    and
        Qhat(omega) = (1/omega) [ ∫ e^{-omega t} a_{x_t} dt + Σ_k c_k e^{-omega t_k} ]
    """
    n_omega = omegas.shape[0]
    accum = np.zeros(n_omega, dtype=np.float64)

    s = states[0]
    t_prev = 0.0

    for k in range(n_jumps):
        t_jump = jump_times[k]

        a_s = a_state[s]
        if a_s != 0.0:
            for m in range(n_omega):
                w = omegas[m]
                accum[m] += a_s * (np.exp(-w * t_prev) - np.exp(-w * t_jump)) / w

        coeff = c_channel[channel_ids[k]]
        if coeff != 0.0:
            for m in range(n_omega):
                accum[m] += coeff * np.exp(-omegas[m] * t_jump)

        s = states[k + 1]
        t_prev = t_jump

    a_s = a_state[s]
    if a_s != 0.0:
        for m in range(n_omega):
            w = omegas[m]
            accum[m] += a_s * (np.exp(-w * t_prev) - np.exp(-w * t_max)) / w

    for m in range(n_omega):
        accum[m] /= omegas[m]

    return accum


@njit(cache=True)
def simulate_observable_pair(n_states, p0, t_max, max_jumps, n_traj, omegas,
                             ch_from, ch_to, ch_rate,
                             a1, c1, a2, c2):
    n_omega = omegas.shape[0]
    q1_samples = np.empty((n_traj, n_omega), dtype=np.float64)
    q2_samples = np.empty((n_traj, n_omega), dtype=np.float64)

    for n in range(n_traj):
        n_jumps, jump_times, states, channel_ids = traj_generator_channels(
            n_states, p0, t_max, max_jumps, ch_from, ch_to, ch_rate
        )
        q1_samples[n, :] = qhat_from_trajectory_general(
            n_jumps, jump_times, states, channel_ids, t_max, omegas, a1, c1
        )
        q2_samples[n, :] = qhat_from_trajectory_general(
            n_jumps, jump_times, states, channel_ids, t_max, omegas, a2, c2
        )

    return q1_samples, q2_samples


# ============================================================
# N-site SEP model (configured for N=5)
# ============================================================

def state_to_bits(state, n_sites):
    return tuple((state >> i) & 1 for i in range(n_sites))


def bits_to_state(bits):
    s = 0
    for i, b in enumerate(bits):
        if b:
            s |= (1 << i)
    return s


def sepN_channels(lambda_rate, n_sites=N_SITES):
    beta = 0.1  # 1/(k_B T)

    # site energies
    eps = np.array([(i+1)/n_sites for i in range(n_sites)], dtype=np.float64)

    # barriers between sites
    B = 1.5 * np.ones(n_sites - 1)

    # chemical potentials
    mu_L = 2.0
    mu_R = 0.0

    channels = []

    for state in range(1 << n_sites):
        bits = list(state_to_bits(state, n_sites))

        # ---------- LEFT RESERVOIR ----------
        if bits[0] == 0:
            new_bits = bits.copy()
            new_bits[0] = 1
            new_state = bits_to_state(new_bits)

            # injection from left reservoir
            rate = np.exp(-beta * (1.5 - mu_L))

            # perturb ONLY empty → injection
            if state == 0:
                rate = lambda_rate

            channels.append((state, new_state, rate, "L_in"))

        else:
            new_bits = bits.copy()
            new_bits[0] = 0
            new_state = bits_to_state(new_bits)

            # extraction to left reservoir
            rate = np.exp(-beta * (1.5 - eps[0]))

            channels.append((state, new_state, rate, "L_out"))


        # ---------- RIGHT RESERVOIR ----------
        if bits[-1] == 0:
            new_bits = bits.copy()
            new_bits[-1] = 1
            new_state = bits_to_state(new_bits)

            # injection from right reservoir
            rate = np.exp(-beta * (1.5 - mu_R))

            channels.append((state, new_state, rate, "R_in"))

        else:
            new_bits = bits.copy()
            new_bits[-1] = 0
            new_state = bits_to_state(new_bits)

            # extraction to right reservoir
            rate = np.exp(-beta * (1.5 - eps[-1]))

            channels.append((state, new_state, rate, "R_out"))

        # ---------- BULK HOPPING ----------
        for k in range(n_sites - 1):
            if bits[k] == 1 and bits[k + 1] == 0:
                new_bits = bits.copy()
                new_bits[k] = 0
                new_bits[k + 1] = 1
                new_state = bits_to_state(new_bits)

                dE = eps[k + 1] - eps[k]
                rate = np.exp(-beta * (B[k] + max(dE, 0.0)))

                channels.append((state, new_state, rate, f"hop_{k}_right"))

            if bits[k] == 0 and bits[k + 1] == 1:
                new_bits = bits.copy()
                new_bits[k] = 1
                new_bits[k + 1] = 0
                new_state = bits_to_state(new_bits)

                dE = eps[k] - eps[k + 1]
                rate = np.exp(-beta * (B[k] + max(dE, 0.0)))

                channels.append((state, new_state, rate, f"hop_{k}_left"))

    ch_from = np.array([c[0] for c in channels], dtype=np.int64)
    ch_to = np.array([c[1] for c in channels], dtype=np.int64)
    ch_rate = np.array([c[2] for c in channels], dtype=np.float64)
    ch_name = [c[3] for c in channels]

    return ch_from, ch_to, ch_rate, ch_name


def make_right_reservoir_net_current_coeffs(ch_name):
    """
    Net particle current INTO the right reservoir:
        +1 for system -> right reservoir
        -1 for right reservoir -> system
    """
    c = np.zeros(len(ch_name), dtype=np.float64)
    for i, name in enumerate(ch_name):
        if name == "R_out":
            c[i] = +1.0
        elif name == "R_in":
            c[i] = -1.0
    return c


# ============================================================
# Data collection / plotting
# ============================================================

def summarize_samples(samples):
    mean = np.mean(samples, axis=0)
    if samples.shape[0] > 1:
        se = np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])
    else:
        se = np.zeros_like(mean)
    return mean, se


def panel_limits(data, pad_frac=0.14):
    x_low = (data["q2_mean"] - data["q2_se"]).ravel()
    x_high = (data["q2_mean"] + data["q2_se"]).ravel()
    y_low = (data["q1_mean"] - data["q1_se"]).ravel()
    y_high = (data["q1_mean"] + data["q1_se"]).ravel()

    xmin = np.min(x_low)
    xmax = np.max(x_high)
    ymin = np.min(y_low)
    ymax = np.max(y_high)

    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0

    return (xmin - pad_frac * dx, xmax + pad_frac * dx), (ymin - pad_frac * dy, ymax + pad_frac * dy)


def linear_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


def collect_model_data(n_traj=30000, t_max=250.0, max_jumps=300000):
    n_states = 1 << N_SITES
    p0 = np.zeros(n_states, dtype=np.float64)
    p0[0] = 1.0  # start from the empty configuration 00...0

    # Q1 = net current into the right reservoir
    a1 = np.zeros(n_states, dtype=np.float64)

    # Q2 = dwelling time in the completely empty configuration 00...0
    a2 = np.zeros(n_states, dtype=np.float64)
    a2[0] = 1.0

    q1_mean = np.empty((LAMBDAS.shape[0], OMEGAS.shape[0]), dtype=np.float64)
    q2_mean = np.empty((LAMBDAS.shape[0], OMEGAS.shape[0]), dtype=np.float64)
    q1_se = np.empty_like(q1_mean)
    q2_se = np.empty_like(q1_mean)

    for i, lam in enumerate(LAMBDAS):
        ch_from, ch_to, ch_rate, ch_name = sepN_channels(lam, N_SITES)
        c1 = make_right_reservoir_net_current_coeffs(ch_name)
        c2 = np.zeros(len(ch_name), dtype=np.float64)

        s1, s2 = simulate_observable_pair(
            n_states=n_states,
            p0=p0,
            t_max=t_max,
            max_jumps=max_jumps,
            n_traj=n_traj,
            omegas=OMEGAS,
            ch_from=ch_from,
            ch_to=ch_to,
            ch_rate=ch_rate,
            a1=a1,
            c1=c1,
            a2=a2,
            c2=c2,
        )
        m1, e1 = summarize_samples(s1)
        m2, e2 = summarize_samples(s2)

        q1_mean[i, :] = m1
        q2_mean[i, :] = m2
        q1_se[i, :] = e1
        q2_se[i, :] = e2

    return {
        "name": f"{N_SITES}-site SEP",
        "q1_mean": q1_mean,
        "q2_mean": q2_mean,
        "q1_se": q1_se,
        "q2_se": q2_se,
    }


def plot_panel(ax, data):
    xlim, ylim = panel_limits(data)

    for j, omega in enumerate(OMEGAS):
        style = OMEGA_STYLES[float(omega)]
        color = style["color"]
        x = data["q2_mean"][:, j]
        y = data["q1_mean"][:, j]

        ax.plot(
            x, y,
            linestyle="None",
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.7,
            label=fr"$\omega={omega}$",
            zorder=3,
        )

        m, b = linear_fit(x, y)
        xfit = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(
            xfit, m * xfit + b,
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            color=color,
            zorder=2,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\hat{Q}_3(\omega)$")
    ax.set_ylabel(r"$\hat{Q}_1(\omega)$")
    ax.grid(True, linestyle="--", linewidth=0.2)
    format_axes(ax)
    ax.legend(loc="upper right", handlelength=0.5, borderpad=0.1, labelspacing=0.05, handletextpad=0.5)


def main():
    set_style()

    n_traj = 50000
    t_max = 500.0
    max_jumps = 500000

    print("Numba available:", NUMBA_AVAILABLE)
    print(f"N_sites={N_SITES}")
    print(f"Simulating with n_traj={n_traj}, t_max={t_max}, max_jumps={max_jumps}")
    print("Perturbed single channel: 00...0 -> 10...0 via left reservoir")
    print("Q1: net particle current into right reservoir")
    print("Q2: dwelling time in the empty state 00...0")

    data = collect_model_data(n_traj=n_traj, t_max=t_max, max_jumps=max_jumps)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    plot_panel(ax, data)

    outdir = get_script_dir() / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    pdf_path = outdir / f"sep{N_SITES}_negative.pdf"
    svg_path = outdir / f"sep{N_SITES}_negative.svg"

    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
