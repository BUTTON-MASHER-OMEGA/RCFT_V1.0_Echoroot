
############################################################
#  SIM11.X — Basin-Local Invariant-Set Probe
#
#  Purpose:
#    Determine what kind of local stability object exists in
#    Z_x = (C_x, log V_eff,x) space.
#
#  Competing hypotheses:
#
#    H_point:
#        Z_x(t) -> Z_x*
#
#    H_cycle:
#        Z_x(t) approaches a recurrent / periodic orbit
#
#    H_distribution:
#        Z_x(t) samples a stationary local distribution
#
#    H_drift:
#        no stationary local object on the tested timescale
#
#  IMPORTANT:
#    - No perturbation
#    - No epsilon
#    - No eta sweep
#    - No R_global term
#    - No Lyapunov sign test yet
#
#  We only measure:
#        C_x(t)
#        W_x(t) = log(V_eff,x(t))
#
#  Then diagnose:
#        centroid stability
#        covariance stability
#        temporal drift
#        autocorrelation
#        recurrence
############################################################

import numpy as np
from collections import defaultdict
from dataclasses import dataclass


# ============================================================
# 0. GRAPH + BASINS
# ============================================================

neighbors = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 4],
    3: [1, 5],
    4: [2, 6],
    5: [3, 7],
    6: [4, 7],
    7: [5, 6],
}

proj_structured = {
    0: 0, 1: 0,
    2: 1, 3: 1,
    4: 2, 5: 2,
    6: 3, 7: 3,
}

DEFAULT_BASINS = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
}


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class Config:
    lam: float = 1.0
    delta: float = 0.95
    beta: float = 1.38
    alpha: float = 1.0

    T_tol: float = 1.0
    memory_sign: int = -1
    sat_mode: str = "tanh"

    seed: int = 1

    # Long enough to separate burn-in from late-time behavior
    burnin_steps: int = 3000
    observe_steps: int = 5000

    # Diagnostic cadence
    diagnostic_every: int = 10

    # Rolling statistics
    stats_window: int = 100

    basins: dict = None

    def __post_init__(self):
        if self.basins is None:
            self.basins = DEFAULT_BASINS


# ============================================================
# 2. MEMORY / FRICTION / TRANSITION
# ============================================================

def saturate_memory(M, mode="tanh"):
    if mode == "tanh":
        return np.tanh(M)
    return M


def friction(u_i, u_j, mem, cfg):
    delta_proj = (
        0.0
        if proj_structured[u_i] == proj_structured[u_j]
        else 1.0
    )

    M_raw = mem.get((u_i, u_j), 0.0)
    M_sat = saturate_memory(M_raw, cfg.sat_mode)

    return (
        cfg.alpha
        + cfg.beta * delta_proj
        + cfg.memory_sign * cfg.lam * M_sat
    )


def transition_probs(u, mem, cfg):
    nbrs = neighbors[u]

    costs = np.array(
        [friction(u, v, mem, cfg) for v in nbrs],
        dtype=float
    )

    weights = np.exp(-costs / cfg.T_tol)

    return nbrs, weights / weights.sum()


def update_memory(mem, u, v, cfg):
    for k in list(mem.keys()):
        mem[k] *= cfg.delta

        if mem[k] < 1e-12:
            del mem[k]

    mem[(u, v)] += 1.0


# ============================================================
# 3. LOCAL COHERENCE
# ============================================================

def compute_local_entropy_and_persistence(
    hidden,
    obs,
    basins,
    window=None
):
    if window is None:
        start = 0
    else:
        start = max(0, len(hidden) - window - 1)

    basin_trans = {
        x: defaultdict(int)
        for x in basins
    }

    basin_from = {
        x: defaultdict(int)
        for x in basins
    }

    counts_total = {
        x: 0
        for x in basins
    }

    counts_same = {
        x: 0
        for x in basins
    }

    for t in range(start, len(hidden) - 1):
        u = hidden[t]
        v = hidden[t + 1]
        b = obs[t]

        counts_total[b] += 1

        if obs[t + 1] == b:
            counts_same[b] += 1

        for x, states in basins.items():
            if u in states:
                basin_trans[x][(u, v)] += 1
                basin_from[x][u] += 1
                break

    H = {}
    P = {}

    for x in basins:
        H_sum = 0.0
        n_src = 0

        for u in basin_from[x]:

            probs = []

            for (uu, vv), c in basin_trans[x].items():
                if uu == u:
                    probs.append(
                        c / basin_from[x][u]
                    )

            if len(probs) == 0:
                continue

            p = np.array(probs, dtype=float)

            H_sum += -np.sum(
                p * np.log(p + 1e-12)
            )

            n_src += 1

        H[x] = (
            H_sum / n_src
            if n_src > 0
            else 0.0
        )

        P[x] = (
            counts_same[x] / counts_total[x]
            if counts_total[x] > 0
            else 0.0
        )

    return H, P


def compute_Cx(H, P, alpha):
    return {
        x: -H[x] + alpha * P[x]
        for x in H
    }


# ============================================================
# 4. EFFECTIVE VALENCE
# ============================================================

def compute_valence(mem, cfg):
    Veff = {
        x: 0.0
        for x in cfg.basins
    }

    counts = {
        x: 0
        for x in cfg.basins
    }

    for x, states in cfg.basins.items():

        for u in states:

            nbrs, probs = transition_probs(
                u,
                mem,
                cfg
            )

            p = np.array(
                probs,
                dtype=float
            )

            H_loc = -np.sum(
                p * np.log(p + 1e-12)
            )

            Veff[x] += np.exp(H_loc)

            counts[x] += 1

    for x in cfg.basins:
        if counts[x] > 0:
            Veff[x] /= counts[x]

    return Veff


# ============================================================
# 5. DIAGNOSTIC STATE
#
#    Z_x = (C_x, W_x)
#    W_x = log(V_eff,x)
# ============================================================

def compute_Z(hidden, obs, mem, cfg):
    H, P = compute_local_entropy_and_persistence(
        hidden,
        obs,
        cfg.basins,
        window=cfg.stats_window
    )

    Cx = compute_Cx(
        H,
        P,
        cfg.alpha
    )

    Veff = compute_valence(
        mem,
        cfg
    )

    Z = {}

    for x in cfg.basins:

        V = max(
            Veff[x],
            1e-12
        )

        Z[x] = np.array(
            [
                Cx[x],
                np.log(V)
            ],
            dtype=float
        )

    return Z


# ============================================================
# 6. SINGLE LONG TRAJECTORY
# ============================================================

def run_long_trajectory(cfg):
    rng = np.random.default_rng(
        cfg.seed
    )

    hidden = [0]
    obs = [proj_structured[0]]

    mem = defaultdict(float)

    # -------------------------
    # Burn-in
    # -------------------------

    for _ in range(cfg.burnin_steps):

        u = hidden[-1]

        nbrs, probs = transition_probs(
            u,
            mem,
            cfg
        )

        v = rng.choice(
            nbrs,
            p=probs
        )

        hidden.append(v)
        obs.append(
            proj_structured[v]
        )

        update_memory(
            mem,
            u,
            v,
            cfg
        )

    # -------------------------
    # Observation phase
    # -------------------------

    Z_logs = {
        x: []
        for x in cfg.basins
    }

    times = []

    for t in range(cfg.observe_steps):

        u = hidden[-1]

        nbrs, probs = transition_probs(
            u,
            mem,
            cfg
        )

        v = rng.choice(
            nbrs,
            p=probs
        )

        hidden.append(v)
        obs.append(
            proj_structured[v]
        )

        update_memory(
            mem,
            u,
            v,
            cfg
        )

        if t % cfg.diagnostic_every == 0:

            Z = compute_Z(
                hidden,
                obs,
                mem,
                cfg
            )

            times.append(t)

            for x in cfg.basins:
                Z_logs[x].append(
                    Z[x]
                )

    for x in Z_logs:
        Z_logs[x] = np.array(
            Z_logs[x]
        )

    return {
        "times": np.array(times),
        "Z": Z_logs
    }


# ============================================================
# 7. BASIC TIME-SERIES HELPERS
# ============================================================

def autocorrelation(series, max_lag=50):
    x = np.asarray(
        series,
        dtype=float
    )

    if len(x) < 3:
        return np.zeros(1)

    x = x - np.mean(x)

    denom = np.dot(x, x)

    if denom <= 1e-15:
        return np.ones(
            max_lag + 1
        )

    max_lag = min(
        max_lag,
        len(x) - 1
    )

    ac = []

    for lag in range(max_lag + 1):

        if lag == 0:
            ac.append(1.0)

        else:
            num = np.dot(
                x[:-lag],
                x[lag:]
            )

            ac.append(
                num / denom
            )

    return np.array(ac)


def linear_slope(y):
    y = np.asarray(
        y,
        dtype=float
    )

    if len(y) < 2:
        return 0.0

    t = np.arange(
        len(y),
        dtype=float
    )

    slope = np.polyfit(
        t,
        y,
        1
    )[0]

    return slope


# ============================================================
# 8. RECURRENCE SCORE
#
#    Measures how often trajectory returns close to earlier
#    states after a minimum temporal separation.
# ============================================================

def recurrence_fraction(
    Z,
    radius=0.05,
    min_separation=10
):
    Z = np.asarray(
        Z,
        dtype=float
    )

    n = len(Z)

    if n < 2:
        return 0.0

    hits = 0
    total = 0

    for i in range(n):

        for j in range(
            i + min_separation,
            n
        ):

            d = np.linalg.norm(
                Z[i] - Z[j]
            )

            total += 1

            if d <= radius:
                hits += 1

    if total == 0:
        return 0.0

    return hits / total


# ============================================================
# 9. STATIONARITY / INVARIANT-SET DIAGNOSTICS
# ============================================================

def analyze_basin_trajectory(
    Z,
    recurrence_radius=0.05
):
    Z = np.asarray(
        Z,
        dtype=float
    )

    C = Z[:, 0]
    W = Z[:, 1]

    n = len(Z)

    if n < 10:
        return None

    # --------------------------------------------------------
    # Split trajectory into early / late halves
    # --------------------------------------------------------

    mid = n // 2

    early = Z[:mid]
    late = Z[mid:]

    mu_early = np.mean(
        early,
        axis=0
    )

    mu_late = np.mean(
        late,
        axis=0
    )

    centroid_shift = np.linalg.norm(
        mu_late - mu_early
    )

    # --------------------------------------------------------
    # Covariance
    # --------------------------------------------------------

    cov_early = np.cov(
        early.T
    )

    cov_late = np.cov(
        late.T
    )

    covariance_shift = np.linalg.norm(
        cov_late - cov_early
    )

    variance_late = np.trace(
        cov_late
    )

    # --------------------------------------------------------
    # Component slopes
    # --------------------------------------------------------

    slope_C = linear_slope(C)

    slope_W = linear_slope(W)

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------

    ac_C = autocorrelation(
        C,
        max_lag=50
    )

    ac_W = autocorrelation(
        W,
        max_lag=50
    )

    # Ignore lag zero
    peak_ac_C = (
        np.max(ac_C[1:])
        if len(ac_C) > 1
        else 0.0
    )

    peak_ac_W = (
        np.max(ac_W[1:])
        if len(ac_W) > 1
        else 0.0
    )

    peak_lag_C = (
        int(
            np.argmax(ac_C[1:]) + 1
        )
        if len(ac_C) > 1
        else 0
    )

    peak_lag_W = (
        int(
            np.argmax(ac_W[1:]) + 1
        )
        if len(ac_W) > 1
        else 0
    )

    # --------------------------------------------------------
    # Recurrence
    # --------------------------------------------------------

    recurrence = recurrence_fraction(
        Z,
        radius=recurrence_radius,
        min_separation=10
    )

    return {
        "mean_C": np.mean(C),
        "mean_W": np.mean(W),

        "std_C": np.std(C),
        "std_W": np.std(W),

        "centroid_shift": centroid_shift,
        "covariance_shift": covariance_shift,
        "late_variance_trace": variance_late,

        "slope_C": slope_C,
        "slope_W": slope_W,

        "peak_ac_C": peak_ac_C,
        "peak_lag_C": peak_lag_C,

        "peak_ac_W": peak_ac_W,
        "peak_lag_W": peak_lag_W,

        "recurrence_fraction": recurrence,
    }


# ============================================================
# 10. HEURISTIC CLASSIFICATION
#
#     IMPORTANT:
#     This is only a diagnostic label.
#     It is NOT a proof.
# ============================================================

def classify_local_structure(stats):
    if stats is None:
        return "insufficient_data"

    drift_mag = np.sqrt(
        stats["slope_C"]**2
        +
        stats["slope_W"]**2
    )

    variance = stats[
        "late_variance_trace"
    ]

    centroid_shift = stats[
        "centroid_shift"
    ]

    recurrence = stats[
        "recurrence_fraction"
    ]

    ac_peak = max(
        stats["peak_ac_C"],
        stats["peak_ac_W"]
    )

    # --------------------------------------------------------
    # Fixed-point-like
    # --------------------------------------------------------

    if (
        variance < 1e-4
        and centroid_shift < 0.01
        and drift_mag < 1e-4
    ):
        return "fixed_point_like"

    # --------------------------------------------------------
    # Recurrent / cyclic
    # --------------------------------------------------------

    if (
        ac_peak > 0.5
        and recurrence > 0.05
        and centroid_shift < 0.1
    ):
        return "recurrent_or_cyclic"

    # --------------------------------------------------------
    # Stationary stochastic cloud
    # --------------------------------------------------------

    if (
        centroid_shift < 0.05
        and drift_mag < 1e-3
        and variance >= 1e-4
    ):
        return "stationary_distribution_like"

    # --------------------------------------------------------
    # Otherwise slow/nonstationary drift
    # --------------------------------------------------------

    return "drifting_or_unresolved"


# ============================================================
# 11. ENSEMBLE SIM11.1
# ============================================================

def run_SIM11_1(
    seeds=50,
    beta=1.38,
    recurrence_radius=0.05
):
    print(
        "=" * 72
    )

    print(
        "\nSIM11.1 — Basin-Local Invariant-Set Probe\n"
    )

    print(
        "=" * 72
    )

    print(
        f"\nbeta={beta}"
    )

    print(
        f"seeds={seeds}"
    )

    print(
        "\nNo perturbation. "
        "No epsilon. "
        "No global Lyapunov assumption.\n"
    )

    ensemble = {
        x: []
        for x in DEFAULT_BASINS
    }

    classifications = {
        x: defaultdict(int)
        for x in DEFAULT_BASINS
    }

    for seed in range(
        1,
        seeds + 1
    ):

        cfg = Config(
            beta=beta,
            seed=seed
        )

        out = run_long_trajectory(
            cfg
        )

        for x in cfg.basins:

            stats = analyze_basin_trajectory(
                out["Z"][x],
                recurrence_radius=recurrence_radius
            )

            if stats is None:
                continue

            label = classify_local_structure(
                stats
            )

            stats["classification"] = label

            ensemble[x].append(
                stats
            )

            classifications[x][
                label
            ] += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    for x in DEFAULT_BASINS:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"BASIN {x}"
        )

        print(
            "=" * 60
        )

        rows = ensemble[x]

        if len(rows) == 0:

            print(
                "No valid runs."
            )

            continue

        def avg(key):
            return np.mean(
                [
                    r[key]
                    for r in rows
                ]
            )

        print(
            f"n                              : {len(rows)}"
        )

        print(
            f"mean C                         : {avg('mean_C'):.6f}"
        )

        print(
            f"mean log(V_eff)                : {avg('mean_W'):.6f}"
        )

        print(
            f"mean std(C)                    : {avg('std_C'):.6f}"
        )

        print(
            f"mean std(log V_eff)            : {avg('std_W'):.6f}"
        )

        print(
            f"mean centroid shift            : {avg('centroid_shift'):.6f}"
        )

        print(
            f"mean covariance shift          : {avg('covariance_shift'):.6f}"
        )

        print(
            f"mean late variance trace       : {avg('late_variance_trace'):.6f}"
        )

        print(
            f"mean slope C                   : {avg('slope_C'):.8f}"
        )

        print(
            f"mean slope log(V_eff)          : {avg('slope_W'):.8f}"
        )

        print(
            f"mean peak autocorr C           : {avg('peak_ac_C'):.6f}"
        )

        print(
            f"mean peak autocorr log(V_eff)  : {avg('peak_ac_W'):.6f}"
        )

        print(
            f"mean recurrence fraction       : {avg('recurrence_fraction'):.6f}"
        )

        print(
            "\nClassification counts:"
        )

        total = len(rows)

        for label, count in sorted(
            classifications[x].items()
        ):

            frac = (
                count / total
            )

            print(
                f"  {label:28s}: "
                f"{count:3d} "
                f"({frac:.3f})"
            )

    return {
        "ensemble": ensemble,
        "classifications": classifications,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_SIM11_1(
        seeds=50,
        beta=1.38,
        recurrence_radius=0.05
    )





#########################





RESULTS:



========================================================================


SIM11.1 — Basin-Local Invariant-Set Probe

========================================================================


beta=1.38
seeds=50


No perturbation. No epsilon. No global Lyapunov assumption.


============================================================
BASIN 0

============================================================

n                              : 50

mean C                         : 0.475781

mean log(V_eff)                : 0.408024

mean std(C)                    : 0.219312

mean std(log V_eff)            : 0.055805

mean centroid shift            : 0.055982

mean covariance shift          : 0.014593

mean late variance trace       : 0.052869

mean slope C                   : 0.00004648

mean slope log(V_eff)          : -0.00000592

mean peak autocorr C           : 0.832055

mean peak autocorr log(V_eff)  : 0.689117

mean recurrence fraction       : 0.066126


Classification counts:
  drifting_or_unresolved      :   9 (0.180)

  recurrent_or_cyclic         :  41 (0.820)


============================================================
BASIN 1

============================================================

n                              : 50

mean C                         : -0.433005

mean log(V_eff)                : 0.671397

mean std(C)                    : 0.206650

mean std(log V_eff)            : 0.020325

mean centroid shift            : 0.043730

mean covariance shift          : 0.008296

mean late variance trace       : 0.043400

mean slope C                   : 0.00005514

mean slope log(V_eff)          : 0.00000062

mean peak autocorr C           : 0.778764

mean peak autocorr log(V_eff)  : 0.562903

mean recurrence fraction       : 0.184608



Classification counts:
  drifting_or_unresolved      :   4 (0.080)

  recurrent_or_cyclic         :  46 (0.920)


============================================================
BASIN 2

============================================================

n                              : 50

mean C                         : -0.431396

mean log(V_eff)                : 0.671263

mean std(C)                    : 0.208815

mean std(log V_eff)            : 0.020493

mean centroid shift            : 0.041458

mean covariance shift          : 0.009098

mean late variance trace       : 0.043725

mean slope C                   : -0.00001559

mean slope log(V_eff)          : 0.00000124

mean peak autocorr C           : 0.775243

mean peak autocorr log(V_eff)  : 0.575555

mean recurrence fraction       : 0.182190


Classification counts:
  drifting_or_unresolved      :   3 (0.060)

  recurrent_or_cyclic         :  47 (0.940)


============================================================
BASIN 3

============================================================

n                              : 50

mean C                         : 0.475861

mean log(V_eff)                : 0.408106

mean std(C)                    : 0.221246

mean std(log V_eff)            : 0.055791

mean centroid shift            : 0.059845

mean covariance shift          : 0.015731

mean late variance trace       : 0.049825

mean slope C                   : 0.00002796

mean slope log(V_eff)          : -0.00000582

mean peak autocorr C           : 0.829766

mean peak autocorr log(V_eff)  : 0.693808

mean recurrence fraction       : 0.065344


Classification counts:
  drifting_or_unresolved      :   8 (0.160)

  recurrent_or_cyclic         :  41 (0.820)

  stationary_distribution_like:   1 (0.020)
