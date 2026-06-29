#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gillespie simulation for frequency-domain mutual linearity in an open N-site SEP.

This script extends the previous SEP simulations by adding Q4, a local
entropy-production observable associated with the first bulk bond 1 <-> 2.
The perturbed channel remains the single left-boundary injection

    00...0 -> 10...0.

The local entropy-production observable Q4 is constructed only from the
microscopic hopping channels across the first bulk bond. Therefore it does not
contain the perturbed channel and is theorem-compatible.

The script produces four separate single-column figures:
    Q4-hat versus Q2-hat for N=3,
    Q4-hat versus Q3-hat for N=3,
    Q4-hat versus Q2-hat for N=8,
    Q4-hat versus Q3-hat for N=8.
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
# User-facing model and simulation parameters
# ============================================================

N_SITES_LIST = (3, 8)

OMEGAS = np.array([0.4, 0.5, 0.6], dtype=np.float64)
LAMBDAS = np.linspace(0.1, 5.0, 9)

BETA = 0.1
MU_L = 2.0
MU_R = 0.0
BARRIER = 1.5

# These values reproduce the manuscript-scale run. For a quick test, reduce
# N_TRAJ and T_MAX in main().
N_TRAJ = 50000
T_MAX = 500.0
MAX_JUMPS = 500000

SINGLE_COLUMN_WIDTH_IN = 1.95
FIG_HEIGHT_IN = 1.55

OMEGA_STYLES = {
    0.4: {"marker": "o", "linestyle": "-",  "linewidth": 0.9, "markersize": 2.6, "color": "blue"},
    0.5: {"marker": "s", "linestyle": "--", "linewidth": 0.9, "markersize": 2.5, "color": "orange"},
    0.6: {"marker": "^", "linestyle": "-.", "linewidth": 0.9, "markersize": 2.6, "color": "green"},
}


# ============================================================
# Plot style
# ============================================================

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.0,
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
    General additive observable:
        Q(tau) = int_0^tau a_{x_t} dt + sum_k c_{channel_k}.

    This function returns
        Qhat(omega) = int_0^infty exp(-omega tau) Q(tau) d tau,
    estimated from a finite trajectory up to t_max. For t_max large enough,
    the missing tail is exponentially suppressed.
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
def simulate_three_observables(n_states, p0, t_max, max_jumps, n_traj, omegas,
                               ch_from, ch_to, ch_rate,
                               a2, c2, a3, c3, a4, c4):
    n_omega = omegas.shape[0]
    q2_samples = np.empty((n_traj, n_omega), dtype=np.float64)
    q3_samples = np.empty((n_traj, n_omega), dtype=np.float64)
    q4_samples = np.empty((n_traj, n_omega), dtype=np.float64)

    for n in range(n_traj):
        n_jumps, jump_times, states, channel_ids = traj_generator_channels(
            n_states, p0, t_max, max_jumps, ch_from, ch_to, ch_rate
        )
        q2_samples[n, :] = qhat_from_trajectory_general(
            n_jumps, jump_times, states, channel_ids, t_max, omegas, a2, c2
        )
        q3_samples[n, :] = qhat_from_trajectory_general(
            n_jumps, jump_times, states, channel_ids, t_max, omegas, a3, c3
        )
        q4_samples[n, :] = qhat_from_trajectory_general(
            n_jumps, jump_times, states, channel_ids, t_max, omegas, a4, c4
        )

    return q2_samples, q3_samples, q4_samples


# ============================================================
# N-site SEP model
# ============================================================

def state_to_bits(state, n_sites):
    return tuple((state >> i) & 1 for i in range(n_sites))


def bits_to_state(bits):
    s = 0
    for i, b in enumerate(bits):
        if b:
            s |= (1 << i)
    return s


def sepN_channels(lambda_rate, n_sites):
    eps = np.array([(i + 1) / n_sites for i in range(n_sites)], dtype=np.float64)
    B = BARRIER * np.ones(n_sites - 1)

    channels = []

    for state in range(1 << n_sites):
        bits = list(state_to_bits(state, n_sites))

        # ---------- LEFT RESERVOIR ----------
        if bits[0] == 0:
            new_bits = bits.copy()
            new_bits[0] = 1
            new_state = bits_to_state(new_bits)

            # injection from the left reservoir
            rate = np.exp(-BETA * (BARRIER - MU_L))

            # perturb ONLY the single empty-state injection channel
            if state == 0:
                rate = lambda_rate

            channels.append((state, new_state, rate, "L_in"))

        else:
            new_bits = bits.copy()
            new_bits[0] = 0
            new_state = bits_to_state(new_bits)

            # extraction to the left reservoir
            rate = np.exp(-BETA * (BARRIER - eps[0]))
            channels.append((state, new_state, rate, "L_out"))

        # ---------- RIGHT RESERVOIR ----------
        if bits[-1] == 0:
            new_bits = bits.copy()
            new_bits[-1] = 1
            new_state = bits_to_state(new_bits)

            # injection from the right reservoir
            rate = np.exp(-BETA * (BARRIER - MU_R))
            channels.append((state, new_state, rate, "R_in"))

        else:
            new_bits = bits.copy()
            new_bits[-1] = 0
            new_state = bits_to_state(new_bits)

            # extraction to the right reservoir
            rate = np.exp(-BETA * (BARRIER - eps[-1]))
            channels.append((state, new_state, rate, "R_out"))

        # ---------- BULK HOPPING ----------
        for k in range(n_sites - 1):
            if bits[k] == 1 and bits[k + 1] == 0:
                new_bits = bits.copy()
                new_bits[k] = 0
                new_bits[k + 1] = 1
                new_state = bits_to_state(new_bits)

                dE = eps[k + 1] - eps[k]
                rate = np.exp(-BETA * (B[k] + max(dE, 0.0)))
                channels.append((state, new_state, rate, f"hop_{k}_right"))

            if bits[k] == 0 and bits[k + 1] == 1:
                new_bits = bits.copy()
                new_bits[k] = 1
                new_bits[k + 1] = 0
                new_state = bits_to_state(new_bits)

                dE = eps[k] - eps[k + 1]
                rate = np.exp(-BETA * (B[k] + max(dE, 0.0)))
                channels.append((state, new_state, rate, f"hop_{k}_left"))

    ch_from = np.array([c[0] for c in channels], dtype=np.int64)
    ch_to = np.array([c[1] for c in channels], dtype=np.int64)
    ch_rate = np.array([c[2] for c in channels], dtype=np.float64)
    ch_name = [c[3] for c in channels]

    return ch_from, ch_to, ch_rate, ch_name


def make_local_entropy_production_coeffs(ch_from, ch_to, ch_rate, ch_name,
                                          bond_index=0):
    """
    Jump coefficients for the medium entropy production on one bulk bond.

    For each selected microscopic transition x -> x' across the chosen bond,
    the coefficient is
        log k_{x',x} / k_{x,x'}.

    The selected bond_index=0 corresponds to the bond between sites 1 and 2.
    This observable excludes the perturbed channel 00...0 -> 10...0.
    """
    rates_by_edge = {}
    for f, t, r in zip(ch_from, ch_to, ch_rate):
        rates_by_edge[(int(f), int(t))] = float(r)

    c = np.zeros(len(ch_name), dtype=np.float64)
    prefix = f"hop_{bond_index}_"

    for a, name in enumerate(ch_name):
        if name.startswith(prefix):
            f = int(ch_from[a])
            t = int(ch_to[a])
            reverse_rate = rates_by_edge[(t, f)]
            c[a] = np.log(float(ch_rate[a]) / reverse_rate)

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


def panel_limits(x_mean, x_se, y_mean, y_se, pad_frac=0.14):
    x_low = (x_mean - x_se).ravel()
    x_high = (x_mean + x_se).ravel()
    y_low = (y_mean - y_se).ravel()
    y_high = (y_mean + y_se).ravel()

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


def collect_model_data(n_sites, n_traj=N_TRAJ, t_max=T_MAX, max_jumps=MAX_JUMPS):
    n_states = 1 << n_sites
    p0 = np.zeros(n_states, dtype=np.float64)
    p0[0] = 1.0  # start from the empty configuration 00...0

    # Q2 = dwelling time in the fully occupied state 11...1
    a2 = np.zeros(n_states, dtype=np.float64)
    a2[(1 << n_sites) - 1] = 1.0

    # Q3 = dwelling time in the completely empty state 00...0
    a3 = np.zeros(n_states, dtype=np.float64)
    a3[0] = 1.0

    # Q4 = local entropy production on the first bulk bond 1 <-> 2
    a4 = np.zeros(n_states, dtype=np.float64)

    c2_ref = None
    c3_ref = None

    q2_mean = np.empty((LAMBDAS.shape[0], OMEGAS.shape[0]), dtype=np.float64)
    q3_mean = np.empty_like(q2_mean)
    q4_mean = np.empty_like(q2_mean)
    q2_se = np.empty_like(q2_mean)
    q3_se = np.empty_like(q2_mean)
    q4_se = np.empty_like(q2_mean)

    for i, lam in enumerate(LAMBDAS):
        ch_from, ch_to, ch_rate, ch_name = sepN_channels(lam, n_sites)

        c2 = np.zeros(len(ch_name), dtype=np.float64)
        c3 = np.zeros(len(ch_name), dtype=np.float64)
        c4 = make_local_entropy_production_coeffs(
            ch_from, ch_to, ch_rate, ch_name, bond_index=0
        )

        if c2_ref is None:
            c2_ref = c2
            c3_ref = c3

        s2, s3, s4 = simulate_three_observables(
            n_states=n_states,
            p0=p0,
            t_max=t_max,
            max_jumps=max_jumps,
            n_traj=n_traj,
            omegas=OMEGAS,
            ch_from=ch_from,
            ch_to=ch_to,
            ch_rate=ch_rate,
            a2=a2,
            c2=c2,
            a3=a3,
            c3=c3,
            a4=a4,
            c4=c4,
        )

        m2, e2 = summarize_samples(s2)
        m3, e3 = summarize_samples(s3)
        m4, e4 = summarize_samples(s4)

        q2_mean[i, :] = m2
        q3_mean[i, :] = m3
        q4_mean[i, :] = m4
        q2_se[i, :] = e2
        q3_se[i, :] = e3
        q4_se[i, :] = e4

    return {
        "name": f"{n_sites}-site SEP",
        "q2_mean": q2_mean,
        "q3_mean": q3_mean,
        "q4_mean": q4_mean,
        "q2_se": q2_se,
        "q3_se": q3_se,
        "q4_se": q4_se,
    }


def plot_relation_panel(ax, data, x_key, x_label):
    x_mean = data[f"{x_key}_mean"]
    x_se = data[f"{x_key}_se"]
    y_mean = data["q4_mean"]
    y_se = data["q4_se"]

    xlim, ylim = panel_limits(x_mean, x_se, y_mean, y_se)

    for j, omega in enumerate(OMEGAS):
        style = OMEGA_STYLES[float(omega)]
        color = style["color"]
        x = x_mean[:, j]
        y = y_mean[:, j]

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
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\hat{Q}_4(\omega)$")
    ax.grid(True, linestyle="--", linewidth=0.2)
    format_axes(ax)


def save_single_relation_figure(data, n_sites, x_key, x_label, filename_stem):
    fig, ax = plt.subplots(
        1, 1,
        figsize=(SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    plot_relation_panel(ax, data, x_key, x_label)
    ax.legend(
        loc="best", handlelength=0.7, borderpad=0.1,
        labelspacing=0.08, handletextpad=0.5
    )

    outdir = get_script_dir() / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    pdf_path = outdir / f"{filename_stem}.pdf"
    svg_path = outdir / f"{filename_stem}.svg"

    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")


def main():
    set_style()

    print("Numba available:", NUMBA_AVAILABLE)
    print("Perturbed single channel: 00...0 -> 10...0 via left reservoir")
    print("Q2: dwelling time in the fully occupied state 11...1")
    print("Q3: dwelling time in the empty state 00...0")
    print("Q4: local entropy production on the first bulk bond 1 <-> 2")
    print(f"n_traj={N_TRAJ}, t_max={T_MAX}, max_jumps={MAX_JUMPS}")

    for n_sites in N_SITES_LIST:
        print(f"Collecting data for N_sites={n_sites}")
        data = collect_model_data(
            n_sites=n_sites,
            n_traj=N_TRAJ,
            t_max=T_MAX,
            max_jumps=MAX_JUMPS,
        )

        save_single_relation_figure(
            data=data,
            n_sites=n_sites,
            x_key="q2",
            x_label=r"$\hat{Q}_2(\omega)$",
            filename_stem=f"SEP_local_entropy_N{n_sites}_Q4_vs_Q2",
        )
        save_single_relation_figure(
            data=data,
            n_sites=n_sites,
            x_key="q3",
            x_label=r"$\hat{Q}_3(\omega)$",
            filename_stem=f"SEP_local_entropy_N{n_sites}_Q4_vs_Q3",
        )


if __name__ == "__main__":
    main()
