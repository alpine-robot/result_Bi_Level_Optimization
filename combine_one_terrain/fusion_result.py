"""
fusion_result.py
----------------
Plots the "Median Best (Overall)" across multiple simulation groups.

Each GROUP is a directory containing M run sub-folders (run_1, run_2, …).
One line + IQR band is drawn per group, with a distinct color.

Usage
-----
# Default groups (edit GROUPS below, or pass via env):
python3 fusion_result.py

# Via environment variable (JSON list of {"name":…,"path":…} dicts):
FUSION_GROUPS='[{"name":"Gaussian","path":"exploitation_gaussian"},
                {"name":"Rocky","path":"exploitation_rocky"}]' \
python3 fusion_result.py
"""

import json
import os
import glob
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ──────────────────────────────────────────────
# Re-use SingleResult from combine_plot
# ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from combine_plot import SingleResult

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
_env = os.environ.get("FUSION_GROUPS")
if _env:
    GROUPS = json.loads(_env)
else:
    GROUPS = [
        # exploitation
        # {"name": "Gaussian", "path": "exploitation_gaussian"},
        # {"name": "Rocky",    "path": "exploitation_rocky"},
        # {"name": "Hemisphere",    "path": "exploitation_hemisphere"},
        
        
        # exploration
        {"name": "Gaussian", "path": "up_gaussian"},
        {"name": "Rocky",    "path": "up_rocky"},
        {"name": "Hemisphere",    "path": "up_hemisphere"},
    ]

OUTPUT_DIR = "pippo"
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
          "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def discover_runs(group_path: str) -> list[str]:
    """Return sorted list of run sub-folders inside group_path."""
    pattern = os.path.join(group_path, "run_*")
    runs = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()),
    )
    return [r for r in runs if os.path.isdir(r)]


def load_group(group: dict) -> list[SingleResult]:
    """Load all valid SingleResult objects for a group."""
    runs = discover_runs(group["path"])
    results = []
    for r in runs:
        try:
            results.append(SingleResult(r))
        except Exception as e:
            print(f"  [WARN] skipping {r}: {e}")
    print(f"  Group '{group['name']}': loaded {len(results)}/{len(runs)} runs.")
    return results


def global_best_matrix(results: list[SingleResult]) -> np.ndarray:
    """
    Returns a (n_runs × max_iters) array where each cell is the best
    fitness found *up to and including* that iteration (running maximum).
    """
    series_list = []
    for r in results:
        current_min = float("inf")
        series = []
        for it_elites in r.all_elites:
            valid = [e["fitness"] for e in it_elites if e["fitness"] < 1e4]
            it_best = min(valid) if valid else (series[-1] if series else float("inf"))
            current_min = min(current_min, it_best)
            series.append(current_min)
        series_list.append(series)

    max_len = max(len(s) for s in series_list)
    mat = np.full((len(series_list), max_len), np.nan)
    for i, s in enumerate(series_list):
        mat[i, : len(s)] = s
        if len(s) < max_len:          # pad with last value
            mat[i, len(s) :] = s[-1]
    return mat


def save_plot(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in (".png", ".pdf"):
        path = os.path.join(OUTPUT_DIR, filename + ext)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {OUTPUT_DIR}/{filename}")


# ──────────────────────────────────────────────
# Jump histogram across groups
# ──────────────────────────────────────────────
def plot_jump_histogram_fusion(groups: list[dict]):
    """
    Grouped bar chart of jump-count distribution for each group.
    One color per group (same palette as plot_median_best_overall).
    """
    # Collect all n_jumps for every group
    group_jumps = []
    group_names = []
    for group in groups:
        print(f"\nLoading group '{group['name']}' …")
        results = load_group(group)
        if not results:
            print(f"  [WARN] no valid runs for '{group['name']}', skipping.")
            group_jumps.append([])
        else:
            jumps = [e["n_jumps"] for r in results for it in r.all_elites for e in it]
            group_jumps.append(jumps)
        group_names.append(group["name"])

    if not any(group_jumps):
        return

    j_min = min(min(j) for j in group_jumps if j)
    j_max = max(max(j) for j in group_jumps if j)
    vals = np.arange(j_min, j_max + 1)
    n_groups = len(groups)
    bw = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(10, 6))

    for k, (jumps, name) in enumerate(zip(group_jumps, group_names)):
        counts = [jumps.count(v) for v in vals]
        color = COLORS[k % len(COLORS)]
        ax.bar(
            vals + (k - n_groups / 2 + 0.5) * bw,
            counts,
            width=bw,
            color=color,
            edgecolor="black",
            alpha=0.8,
            label=name, 
        )

    ax.set_xticks(vals)
    
    # Assi più grandi
    ax.set_xlabel("Number of Jumps", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=30)
    
    # Indici degli assi (tick) più grandi
    ax.tick_params(axis="both", labelsize=25)
    
    # Legenda in alto a destra e più grande
    ax.legend(loc="upper right", fontsize=25)
    
    ax.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()

    save_plot(fig, "fusion_jump_histogram")
    plt.show()


# ──────────────────────────────────────────────
# Main plot
# ──────────────────────────────────────────────
def plot_median_best_overall(groups: list[dict]):
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, group in enumerate(groups):
        print(f"\nLoading group '{group['name']}' …")
        results = load_group(group)
        if not results:
            print(f"  [WARN] no valid runs for '{group['name']}', skipping.")
            continue

        mat = global_best_matrix(results)          # (n_runs × max_iters)
        x = np.arange(1, mat.shape[1] + 1)

        med  = np.nanmedian(mat, axis=0)
        q1   = np.nanpercentile(mat, 25, axis=0)
        q3   = np.nanpercentile(mat, 75, axis=0)

        color = COLORS[idx % len(COLORS)]
        n_runs = mat.shape[0]

        ax.plot(x, med, color=color, lw=2.5, label=group["name"])
        ax.fill_between(x, q1, q3, color=color, alpha=0.20)

    # Assi più grandi
    ax.set_xlabel("Iteration", fontsize=30)
    ax.set_ylabel("Fitness", fontsize=30)
    
    # Indici degli assi (tick) più grandi
    ax.tick_params(axis="both", labelsize=25)
    
    # Legenda in alto a destra (upper right) e più grande
    ax.legend(loc="upper right", fontsize=25, frameon=True)
    
    ax.grid(ls=":", alpha=0.5)
    plt.tight_layout()

    save_plot(fig, "fusion_median_best_overall")
    plt.show()
    

# ──────────────────────────────────────────────
# Convergence rate across groups
# ──────────────────────────────────────────────
def plot_convergence_fusion(groups: list[dict]):
    """
    For each group plots the median convergence rate (%) per iteration,
    with an IQR band, computed across all runs in that group.
    A secondary panel shows the stacked (converged / failed) bar chart
    where each group is represented by its median counts.
    """
    # Collect per-run convergence-rate series for each group
    group_rate_matrices = []   # list of (n_runs × max_iters) arrays
    group_names = []
    group_results_list = []

    for group in groups:
        print(f"\nLoading group '{group['name']}' …")
        results = load_group(group)
        group_results_list.append(results)
        group_names.append(group["name"])

        if not results:
            print(f"  [WARN] no valid runs for '{group['name']}', skipping.")
            group_rate_matrices.append(None)
            continue

        # For each run, build a convergence-rate series indexed by iteration number
        all_iters_set = set()
        for r in results:
            for d in r.all_comb_data:
                all_iters_set.add(d["iteration"])

        if not all_iters_set:
            print(f"  [WARN] no combination data for '{group['name']}', skipping.")
            group_rate_matrices.append(None)
            continue

        all_iters = list(range(1, 11))  # fixed 1–10
        run_rates = []
        for r in results:
            rates = []
            for it in all_iters:
                d = next((x for x in r.all_comb_data if x["iteration"] == it), None)
                if d:
                    steps = d.get("steps", [])
                    cv = sum(1 for s in steps if s.get("converged"))
                    total = len(steps)
                    rates.append(100.0 * cv / total if total > 0 else np.nan)
                else:
                    rates.append(np.nan)  # run didn't reach this iteration
            run_rates.append(rates)

        mat = np.array(run_rates)          # (n_runs × 10)
        group_rate_matrices.append((all_iters, mat))

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=False)

    # ── Top panel: stacked bar (median counts) ────────────────────────────
    n_groups = len(groups)
    # Fixed iteration axis 1–10
    all_iters_union = list(range(1, 11))
    if not all_iters_union:
        print("[WARN] No convergence data found, skipping convergence plot.")
        plt.close(fig)
        return

    bw = 0.8 / n_groups
    x_pos = np.arange(len(all_iters_union))

    for idx, (group, entry) in enumerate(zip(groups, group_rate_matrices)):
        if entry is None:
            continue
        iters, mat = entry
        color = COLORS[idx % len(COLORS)]

        # For each iteration in the union, find the median converged / failed counts
        med_cv_list, med_fl_list = [], []
        for it in all_iters_union:
            if it in iters:
                it_idx = iters.index(it)
                # Recover raw counts from all runs
                cv_counts, fl_counts = [], []
                results = group_results_list[idx]
                for r in results:
                    d = next((x for x in r.all_comb_data if x["iteration"] == it), None)
                    if d:
                        steps = d.get("steps", [])
                        cv = sum(1 for s in steps if s.get("converged"))
                        fl = len(steps) - cv
                        cv_counts.append(cv)
                        fl_counts.append(fl)
                    # skip runs that didn't reach this iteration
                if cv_counts:
                    med_cv_list.append(float(np.median(cv_counts)))
                    med_fl_list.append(float(np.median(fl_counts)))
                else:
                    med_cv_list.append(np.nan)
                    med_fl_list.append(np.nan)
            else:
                med_cv_list.append(0.0)
                med_fl_list.append(0.0)

        med_cv = np.array(med_cv_list)
        med_fl = np.array(med_fl_list)
        xs = x_pos + (idx - n_groups / 2 + 0.5) * bw

        ax1.bar(xs, med_cv, bw, color=color, edgecolor="black",
                label=group["name"], alpha=0.85)
        ax1.bar(xs, med_fl, bw, bottom=med_cv, color=color,
                edgecolor="black", alpha=0.30, hatch="///")

        # Annotate median rate on top
        for x, cv, fl in zip(xs, med_cv, med_fl):
            total = cv + fl
            if total > 0:
                ax1.text(x, total * 1.02, f"{cv / total * 100:.0f}%",
                         ha="center", va="bottom", fontsize=11,
                         color=color, weight="bold")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(all_iters_union)
    ax1.set_ylabel("Population Count (median)", fontsize=18)
    ax1.tick_params(axis="both", labelsize=16)
    # ax1.set_title("Convergence per Iteration (median across runs)", fontsize=14)
    ax1.legend(fontsize=15)
    ax1.grid(ls=":", alpha=0.5)

    # ── Bottom panel: convergence-rate line + IQR ─────────────────────────
    for idx, (group, entry) in enumerate(zip(groups, group_rate_matrices)):
        if entry is None:
            continue
        iters, mat = entry
        color = COLORS[idx % len(COLORS)]
        x = np.array(iters)

        med_rate = np.nanmedian(mat, axis=0)
        q1_rate  = np.nanpercentile(mat, 25, axis=0)
        q3_rate  = np.nanpercentile(mat, 75, axis=0)

        ax2.plot(x, med_rate, color=color, lw=2.5,
                 marker="o", markersize=4, label=group["name"])
        ax2.fill_between(x, q1_rate, q3_rate, color=color, alpha=0.20)

    ax2.set_xlabel("Iteration", fontsize=18)
    ax2.set_ylabel("Convergence Rate [%]", fontsize=18)
    # ax2.set_title("Convergence Rate per Iteration ", fontsize=14)
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="both", labelsize=16)
    ax2.legend(fontsize=15)
    ax2.grid(ls=":", alpha=0.5)

    plt.tight_layout()
    save_plot(fig, "fusion_convergence")
    plt.show()


if __name__ == "__main__":
    plot_median_best_overall(GROUPS)   # --> Metodo per la Figura 6
    plot_jump_histogram_fusion(GROUPS) # --> Metodo per la Figura 6
    # plot_convergence_fusion(GROUPS)