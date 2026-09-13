import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Configuration
plot_str = os.environ.get("FOLDER_PLOT")
FOLDERS = json.loads(plot_str) if plot_str else [
    "up_hemisphere/run_1",
    "up_hemisphere/run_2",
    "up_hemisphere/run_3",
    "up_hemisphere/run_4",
    "up_hemisphere/run_5",
    "up_hemisphere/run_6",
    "up_hemisphere/run_7",
    "up_hemisphere/run_8",
    "up_hemisphere/run_9",
    "up_hemisphere/run_10"
    ]
COMBINED_OUTPUT = "results/combined_plots"

def _save_plot(fig, filename):
    """Helper to save plots in both PNG and PDF formats."""
    os.makedirs(COMBINED_OUTPUT, exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(f"{COMBINED_OUTPUT}/{filename}{ext}", bbox_inches="tight", dpi=150)
    print(f"Saved: {COMBINED_OUTPUT}/{filename}")

class SingleResult:
    """Loads and parses data for a single result folder."""
    def __init__(self, folder: str):
        self.folder = folder
        self.label = os.path.basename(folder.rstrip("/"))
        
        with open(f"{folder}/simulation_params.json") as f:
            dp = json.load(f)
        self.p0, self.pf = dp["START"], dp["GOAL"]
        
        with open(f"{folder}/actual_point_terrain.json") as f:
            self.points_t_data = json.load(f)["points"]

        self.all_elites, self.all_comb_data = [], []
        self.best_fit_ever, self.best_traj_ever, self.best_achieved_target_ever = None, [], None

        # Load Iteration Reports (using regex to extract just the numbers)
        iter_files = sorted(glob.glob(f"{folder}/iteration_reports/iteration_*.json"), 
                            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for fname in iter_files:
            with open(fname) as f:
                it = json.load(f)
            self.all_elites.append([{'fitness': -e['fitness'], 'n_jumps': e['n_jumps']} for e in it['elites']])
            self.best_fit_ever = -it['best_fitness_ever']
            self.best_traj_ever = it['best_trajectory_ever']
            self.best_achieved_target_ever = it.get('best_achieved_target_ever')
        
        # Load Combination History (using regex to extract just the numbers)
        comb_files = sorted(glob.glob(f"{folder}/iteration_reports/all_comb_in_iter_*.json"),
                            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for fname in comb_files:
            with open(fname) as f:
                self.all_comb_data.append(json.load(f))

        self.p0 = self._project(self.p0)
        self.pf = self._project(self.pf)

    def _project(self, point):
        """Projects start/goal points onto the terrain pointcloud."""
        cands = [p["position"] for p in self.points_t_data]
        for tol in (0.1, 0.2):
            valid = [p for p in cands if abs(p[1]-point[1]) < tol and abs(p[2]-point[2]) < tol]
            if valid:
                return min(valid, key=lambda p: abs(p[0] - point[0]))
        return point


class CombinePlot:
    def __init__(self, folders):
        print(f"Loading {len(folders)} folders...")
        self.results = [SingleResult(f) for f in folders if os.path.exists(f)]
        self.cmap = cm.get_cmap("tab10" if len(self.results) <= 10 else "tab20")
        print(f"Loaded {len(self.results)} valid results.\n")

    def plot_jump_histogram(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        all_jumps = [[e['n_jumps'] for it in r.all_elites for e in it] for r in self.results]
        if not any(all_jumps): return

        j_min, j_max = min(min(j) for j in all_jumps if j), max(max(j) for j in all_jumps if j)
        vals = np.arange(j_min, j_max + 1)
        bw = 0.8 / len(self.results)

        for k, (jumps, r) in enumerate(zip(all_jumps, self.results)):
            counts = [jumps.count(v) for v in vals]
            ax.bar(vals + (k - len(self.results)/2 + 0.5) * bw, counts, width=bw, 
                   color=self.cmap(k), edgecolor="black", alpha=0.8, label=r.label)

        ax.set(title="Jump Count Distribution - Combined", xlabel="Number of Jumps", ylabel="Frequency", xticks=vals)
        ax.legend(); ax.grid(axis="y", ls="--", alpha=0.5)
        _save_plot(fig, "combined_jump_histogram")

    def plot_fitness_by_iteration(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        for k, r in enumerate(self.results):
            iters, fits = [], []
            for i, it_elites in enumerate(r.all_elites):
                iters.extend([i+1] * len(it_elites))
                fits.extend([e['fitness'] for e in it_elites])
            
            iters, fits = np.array(iters), np.array(fits)
            mask = fits < 1e4
            iters, fits = iters[mask], fits[mask]
            
            best_mask = np.isclose(fits, r.best_fit_ever, atol=1e-8)
            c = self.cmap(k)
            ax.scatter(iters[~best_mask], fits[~best_mask], color=c, s=30, alpha=0.6, edgecolors='black')
            if any(best_mask):
                ax.scatter(iters[best_mask], fits[best_mask], color=c, s=60, marker='*', edgecolors='red', zorder=4)
            
            ax.axhline(r.best_fit_ever, color=c, ls='--', alpha=0.7)
            ax.plot([], [], color=c, label=f"{r.label} (best={r.best_fit_ever:.4f})")
        
        ax.set(title="Elite Fitness by Iteration", xlabel="Iteration", ylabel="Fitness")
        ax.legend(loc='lower right'); ax.grid(ls=':', alpha=0.5)
        _save_plot(fig, "combined_fitness_by_iteration")
        plt.show()
    
    def plot_best_fitness_line(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        for k, r in enumerate(self.results):
            best_vals = [min([e['fitness'] for e in it if e['fitness'] < 1e5], default=None) for it in r.all_elites]
            iters, vals = zip(*[(i+1, v) for i, v in enumerate(best_vals) if v is not None])
            ax.plot(iters, vals, marker='o', color=self.cmap(k), label=f"{r.label} (best={r.best_fit_ever:.4f})")
            
        ax.set(title="Best Elite Fitness per Iteration", xlabel="Iteration", ylabel="Best Fitness")
        ax.legend(loc='lower right'); ax.grid(ls=':', alpha=0.5)
        _save_plot(fig, "combined_best_fitness_line")

    def plot_mesh_pc_traj(self):
        if not self.results: return
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Terrain point cloud (same style as single_plot)
        ref = self.results[0]
        pts = np.array([p["position"] for p in ref.points_t_data])
        costs = np.array([p["cost"] for p in ref.points_t_data])
        
        pts_plot, costs_plot = pts, costs
        ax.scatter(*pts_plot.T, c=costs_plot, cmap="RdYlGn_r", s=10, alpha=0.2, zorder=1)

        # Accumulate all coords for 1:1:1 scaling
        all_x, all_y, all_z = [pts[:, 0]], [pts[:, 1]], [pts[:, 2]]

        # Find the result with the best (highest) fitness
        valid_results = [r for r in self.results if r.best_traj_ever and r.best_fit_ever is not None]
        best_label = min(valid_results, key=lambda r: r.best_fit_ever).label if valid_results else None

        lines_data = {}
        for k, r in enumerate(self.results):
            if not r.best_traj_ever: continue
            c = self.cmap(k)
            is_best = (r.label == best_label)
            draw_color = '#0080FF' if is_best else c
            traj_lw   = 4.5  if is_best else 1.5
            traj_alpha = 1.0 if is_best else 0.25
            scatter_alpha = 1.0 if is_best else 0.25
            zorder_traj = 12 if is_best else 9
            arts = []
            landing_points = []

            for i, seg in enumerate(r.best_traj_ever):
                s = np.array(seg)
                xs, ys, zs = s if s.shape[0] == 3 and s.shape[1] != 3 else s.T
                arts.extend(ax.plot(xs, ys, zs, color=draw_color, lw=traj_lw, alpha=traj_alpha, zorder=zorder_traj))
                all_x.append(xs); all_y.append(ys); all_z.append(zs)
                landing_points.append([xs[0], ys[0], zs[0]])
                if i == len(r.best_traj_ever) - 1:
                    landing_points.append([xs[-1], ys[-1], zs[-1]])

            lp = np.array(landing_points)
            arts.append(ax.scatter(lp[:, 0], lp[:, 1], lp[:, 2],
                                   c='yellow', s=70 if is_best else 30,
                                   edgecolors='black', alpha=scatter_alpha, zorder=zorder_traj + 1))

            arts.append(ax.scatter(*r.p0, c='lime', s=200, marker="^",
                                   edgecolors="k", linewidth=2, alpha=scatter_alpha, zorder=15))
            arts.append(ax.scatter(*r.pf, c="red", s=200, marker="X",
                                   edgecolors="k", linewidth=2, alpha=scatter_alpha, zorder=15))

            if r.best_achieved_target_ever:
                ach = np.array(r.best_achieved_target_ever).flatten()
                arts.append(ax.scatter(*ach, c="orange", s=250, marker="*",
                                       edgecolors="k", alpha=scatter_alpha, zorder=16))
                arts.extend(ax.plot(*zip(r.pf, ach), "--", color=draw_color, lw=2, alpha=0.7 * scatter_alpha, zorder=14))
                dist = np.linalg.norm(np.array(r.pf) - ach)
                print(f"[INFO] {r.label} — goal→achieved: {dist:.4f} m")

            label_str = f"★ {r.label} (fit={r.best_fit_ever:.4f})" if is_best else f"{r.label} (fit={r.best_fit_ever:.4f})"
            lines_data[r.label] = (arts, Patch(facecolor=draw_color, label=label_str,
                                               linewidth=2 if is_best else 1,
                                               edgecolor='gold' if is_best else 'none'))

        # Interactive legend
        # leg = ax.legend(handles=[d[1] for d in lines_data.values()],
        #                 loc="upper left", bbox_to_anchor=(1.05, 1), title="Click to hide/show")
        # interact_map = {id(p): lines_data[lbl][0] for p, lbl in zip(leg.get_patches(), lines_data.keys())}
        # interact_map.update({id(t): lines_data[lbl][0] for t, lbl in zip(leg.get_texts(), lines_data.keys())})

        # def on_pick(e):
        #     if id(e.artist) in interact_map:
        #         arts = interact_map[id(e.artist)]
        #         vis = not arts[0].get_visible()
        #         for a in arts: a.set_visible(vis)
        #         e.artist.set_alpha(1.0 if vis else 0.3)
        #         fig.canvas.draw_idle()

        # for a in leg.get_patches() + leg.get_texts(): a.set_picker(True)
        # fig.canvas.mpl_connect("pick_event", on_pick)

        # 1:1:1 aspect adjustment (terrain + trajectories)
        flat_x = np.concatenate(all_x)
        flat_y = np.concatenate(all_y)
        flat_z = np.concatenate(all_z)
        max_range = np.array([flat_x.ptp(), flat_y.ptp(), flat_z.ptp()]).max() / 2.0
        mid_x = (flat_x.max() + flat_x.min()) * 0.5
        mid_y = (flat_y.max() + flat_y.min()) * 0.5
        mid_z = (flat_z.max() + flat_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect((1, 1, 1))

        # Axis labels / ticks (match single_plot style)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.zaxis.set_major_locator(plt.MaxNLocator(2))
        ax.set_ylabel('Y [m]', fontsize=20)
        ax.set_zlabel('Z [m]', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='z', labelsize=16)

        ax.view_init(elev=20, azim=-20)
        plt.tight_layout()
        _save_plot(fig, "combined_mesh_pc_traj")
        plt.show()

    def plot_mesh_pc_traj_all_point(self):
        if not self.results: return
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Terrain point cloud (same style as single_plot)
        ref = self.results[0]
        pts = np.array([p["position"] for p in ref.points_t_data])
        costs = np.array([p["cost"] for p in ref.points_t_data])
        ax.scatter(*pts.T, c=costs, cmap="RdYlGn_r", s=10, alpha=0.2, zorder=1)

        # Accumulate all coords for 1:1:1 scaling
        all_x, all_y, all_z = [pts[:, 0]], [pts[:, 1]], [pts[:, 2]]

        # Find the result with the best (lowest) fitness
        valid_results = [r for r in self.results if r.best_traj_ever and r.best_fit_ever is not None]
        best_label = min(valid_results, key=lambda r: r.best_fit_ever).label if valid_results else None

        lines_data = {}
        for k, r in enumerate(self.results):
            if not r.best_traj_ever: continue
            c = self.cmap(k)
            is_best = (r.label == best_label)
            draw_color = '#0080FF' if is_best else c
            traj_lw   = 4.5  if is_best else 1.5
            traj_alpha = 1.0 if is_best else 0.25
            scatter_alpha = 1.0 if is_best else 0.25
            zorder_traj = 12 if is_best else 9
            arts = []
            landing_points = []

            for i, seg in enumerate(r.best_traj_ever):
                s = np.array(seg)
                xs, ys, zs = s if s.shape[0] == 3 and s.shape[1] != 3 else s.T
                arts.extend(ax.plot(xs, ys, zs, color=draw_color, lw=traj_lw, alpha=traj_alpha, zorder=zorder_traj))
                all_x.append(xs); all_y.append(ys); all_z.append(zs)
                landing_points.append([xs[0], ys[0], zs[0]])
                if i == len(r.best_traj_ever) - 1:
                    landing_points.append([xs[-1], ys[-1], zs[-1]])

            lp = np.array(landing_points)
            arts.append(ax.scatter(lp[:, 0], lp[:, 1], lp[:, 2],
                                   c='yellow', s=70 if is_best else 30,
                                   edgecolors='black', alpha=scatter_alpha, zorder=zorder_traj + 1))

            arts.append(ax.scatter(*r.p0, c='lime', s=200, marker="^",
                                   edgecolors="k", linewidth=2, alpha=scatter_alpha, zorder=15))
            arts.append(ax.scatter(*r.pf, c="red", s=200, marker="X",
                                   edgecolors="k", linewidth=2, alpha=scatter_alpha, zorder=15))

            if r.best_achieved_target_ever:
                ach = np.array(r.best_achieved_target_ever).flatten()
                arts.append(ax.scatter(*ach, c="orange", s=250, marker="*",
                                       edgecolors="k", alpha=scatter_alpha, zorder=16))
                arts.extend(ax.plot(*zip(r.pf, ach), "--", color=draw_color, lw=2, alpha=0.7 * scatter_alpha, zorder=14))
                dist = np.linalg.norm(np.array(r.pf) - ach)
                print(f"[INFO] {r.label} — goal→achieved: {dist:.4f} m")

            label_str = f"★ {r.label} (fit={r.best_fit_ever:.4f})" if is_best else f"{r.label} (fit={r.best_fit_ever:.4f})"
            lines_data[r.label] = (arts, Patch(facecolor=draw_color, label=label_str,
                                               linewidth=2 if is_best else 1,
                                               edgecolor='gold' if is_best else 'none'))

        # Interactive legend
        # leg = ax.legend(handles=[d[1] for d in lines_data.values()],
        #                 loc="upper left", bbox_to_anchor=(1.05, 1), title="Click to hide/show")
        # interact_map = {id(p): lines_data[lbl][0] for p, lbl in zip(leg.get_patches(), lines_data.keys())}
        # interact_map.update({id(t): lines_data[lbl][0] for t, lbl in zip(leg.get_texts(), lines_data.keys())})

        # def on_pick(e):
        #     if id(e.artist) in interact_map:
        #         arts = interact_map[id(e.artist)]
        #         vis = not arts[0].get_visible()
        #         for a in arts: a.set_visible(vis)
        #         e.artist.set_alpha(1.0 if vis else 0.3)
        #         fig.canvas.draw_idle()

        # for a in leg.get_patches() + leg.get_texts(): a.set_picker(True)
        # fig.canvas.mpl_connect("pick_event", on_pick)

        # 1:1:1 aspect adjustment (terrain + trajectories)
        flat_x = np.concatenate(all_x)
        flat_y = np.concatenate(all_y)
        flat_z = np.concatenate(all_z)
        max_range = np.array([flat_x.ptp(), flat_y.ptp(), flat_z.ptp()]).max() / 2.0
        mid_x = (flat_x.max() + flat_x.min()) * 0.5
        mid_y = (flat_y.max() + flat_y.min()) * 0.5
        mid_z = (flat_z.max() + flat_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect((1, 1, 1))

        # Axis labels / ticks (match single_plot style)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.zaxis.set_major_locator(plt.MaxNLocator(2))
        ax.set_ylabel('Y [m]', fontsize=20)
        ax.set_zlabel('Z [m]', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='z', labelsize=16)

        ax.view_init(elev=20, azim=-20)
        plt.tight_layout()
        _save_plot(fig, "combined_mesh_pc_traj")


    def plot_convergence(self):
        valid = [r for r in self.results if r.all_comb_data]
        if not valid: return
        
        all_iters = sorted({it for r in valid for d in r.all_comb_data for it in [d['iteration']]})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        bw = 0.8 / len(valid)

        for k, r in enumerate(valid):
            c = self.cmap(k)
            convs, fails, rates = [], [], []
            for it in all_iters:
                d = next((x for x in r.all_comb_data if x['iteration'] == it), None)
                cv = sum(1 for s in d.get('steps', []) if s.get('converged')) if d else 0
                fl = len(d.get('steps', [])) - cv if d else 0
                convs.append(cv); fails.append(fl)
                rates.append(100 * cv / max(cv + fl, 1) if (cv + fl) > 0 else 0)
            
            xs = np.arange(len(all_iters)) + (k - len(valid)/2 + 0.5) * bw
            ax1.bar(xs, convs, bw, color=c, edgecolor="black", label=r.label)
            ax1.bar(xs, fails, bw, bottom=convs, color=c, edgecolor="black", alpha=0.3, hatch="///")
            ax2.plot(all_iters, rates, marker='o', color=c, label=r.label)
            
            for x, cv, fl in zip(xs, convs, fails):
                if cv + fl > 0:
                    ax1.text(x, cv + fl * 1.02, f"{cv/(cv+fl)*100:.0f}%", ha="center", va="bottom", fontsize=7, color=c, weight="bold")

        ax1.set(title="Convergence per Iteration", xticks=np.arange(len(all_iters)), xticklabels=all_iters, ylabel="Population Count")
        ax2.set(title="Convergence Rate (%)", xlabel="Iteration", ylabel="Rate (%)", ylim=(0, 110))
        ax1.legend(); ax2.legend(); ax1.grid(ls=":", alpha=0.5); ax2.grid(ls=":", alpha=0.5)
        _save_plot(fig, "combined_convergence_histogram")
        plt.show()

    def plot_fitness_stats_over_time(self):
        """
        Plots the median of the best elite per iteration (Orange) and 
        the median of the absolute best fitness found so far (Blue).
        Individual runs are plotted dimly using their specific colors.
        """
        if not self.results: return

        # 1. Extract raw data lists
        all_iter_bests = []   # Best fitness found IN that iteration
        all_global_bests = [] # Best fitness found UP TO that iteration

        max_len = 0

        for r in self.results:
            iter_bests = []
            global_bests = []
            current_global_min = float('inf')

            for iteration_elites in r.all_elites:
                valid_fits = [e['fitness'] for e in iteration_elites if e['fitness'] < 1e4]
                
                if not valid_fits:
                    iter_best = iter_bests[-1] if iter_bests else 1000
                else:
                    iter_best = min(valid_fits)
                
                if iter_best < current_global_min:
                    current_global_min = iter_best
                
                iter_bests.append(iter_best)
                global_bests.append(current_global_min)

            if len(iter_bests) > max_len:
                max_len = len(iter_bests)
            
            all_iter_bests.append(iter_bests)
            all_global_bests.append(global_bests)

        # 2. Pad data to make them rectangular arrays
        def pad_data(data_list, length):
            arr = np.zeros((len(data_list), length))
            for i, row in enumerate(data_list):
                arr[i, :len(row)] = row
                if len(row) < length:
                    arr[i, len(row):] = row[-1]
            return arr

        mat_iter = pad_data(all_iter_bests, max_len)
        mat_global = pad_data(all_global_bests, max_len)
        x_axis = np.arange(1, max_len + 1)

        # 3. Calculate statistics (Median and Quartiles)
        med_iter = np.median(mat_iter, axis=0)
        q1_iter = np.percentile(mat_iter, 25, axis=0)
        q3_iter = np.percentile(mat_iter, 75, axis=0)

        med_global = np.median(mat_global, axis=0)
        q1_global = np.percentile(mat_global, 25, axis=0)
        q3_global = np.percentile(mat_global, 75, axis=0)

        # 4. Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Individual traces with dots
        for k, row_iter in enumerate(mat_iter):
            c = self.cmap(k)
            ax.plot(x_axis, row_iter, color=c, alpha=0.6, lw=1,
                    marker='o', markersize=3, markeredgewidth=0)

        # Iteration Best (Orange)
        ax.plot(x_axis, med_iter, color='orange', lw=3, label='Median Best (Iter)')
        ax.fill_between(x_axis, q1_iter, q3_iter, color='orange', alpha=0.25)

        # Global Best (Blue)
        ax.plot(x_axis, med_global, color='tab:blue', lw=2.5, ls='--', label='Median Best (Overall)')
        ax.fill_between(x_axis, q1_global, q3_global, color='tab:blue', alpha=0.15)

        ax.set(title="Fitness Trends: Median Performance & Stability",
            xlabel="Iteration", ylabel="Fitness")
        ax.legend(loc='lower right')
        ax.grid(ls=':', alpha=0.5)

        _save_plot(fig, "combined_fitness_median_stats")
        plt.show()
    
    def plot_fitness_by_iteration_2(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        all_iter_bests = []
        all_global_bests = []
        all_elite_fits = []  # all elite fitnesses per iteration, per run
        max_len = 0

        for k, r in enumerate(self.results):
            iters, fits = [], []
            for i, it_elites in enumerate(r.all_elites):
                iters.extend([i+1] * len(it_elites))
                fits.extend([e['fitness'] for e in it_elites])

            iters, fits = np.array(iters), np.array(fits)
            mask = fits < 1e4
            iters, fits = iters[mask], fits[mask]

            best_mask = np.isclose(fits, r.best_fit_ever, atol=1e-8)
            c = self.cmap(k)

            # Scatter: normal elites
            ax.scatter(iters[~best_mask], fits[~best_mask],
                    color=c, s=30, alpha=0.6, edgecolors='black')
            # Scatter: best ever points
            # if any(best_mask):
            #     ax.scatter(iters[best_mask], fits[best_mask],
            #             color=c, s=60, marker='*', edgecolors='red', zorder=4)

            # Build per-iteration series
            n_iters = int(iters.max()) if len(iters) > 0 else 0
            iter_bests = []
            global_bests = []
            elite_fits_per_iter = []   # list of arrays, one per iteration
            current_global_min = float('inf')

            for it in range(1, n_iters + 1):
                it_fits = fits[iters == it]
                it_best = float(it_fits.min()) if len(it_fits) > 0 else (iter_bests[-1] if iter_bests else np.nan)
                if it_best < current_global_min:
                    current_global_min = it_best
                iter_bests.append(it_best)
                global_bests.append(current_global_min)
                elite_fits_per_iter.append(it_fits)

            max_len = max(max_len, n_iters)
            all_iter_bests.append(iter_bests)
            all_global_bests.append(global_bests)
            all_elite_fits.append(elite_fits_per_iter)

        # Pad to same length
        def pad(data_list, length):
            arr = np.full((len(data_list), length), np.nan)
            for i, row in enumerate(data_list):
                arr[i, :len(row)] = row
                if len(row) < length:
                    arr[i, len(row):] = row[-1]
            return arr

        x_axis = np.arange(1, max_len + 1)

        # --- Green: median + IQR of ALL elite fitnesses across runs per iteration ---
        # For each iteration, pool all elite fitnesses from all runs
        med_elite = np.full(max_len, np.nan)
        p25_elite = np.full(max_len, np.nan)
        p75_elite = np.full(max_len, np.nan)

        for it_idx in range(max_len):
            pooled = []
            for run_elites in all_elite_fits:
                if it_idx < len(run_elites):
                    # use last available iteration's elites if past end
                    effective_idx = min(it_idx, len(run_elites) - 1)
                    if len(run_elites[effective_idx]) > 0:
                        pooled.extend(run_elites[effective_idx])
                if pooled:
                    med_elite[it_idx] = np.median(pooled)
                    p25_elite[it_idx] = np.percentile(pooled, 25)
                    p75_elite[it_idx] = np.percentile(pooled, 75)

        ax.plot(x_axis, med_elite, color='green', lw=2.5, zorder=5, label='Median Elite Fitness')
        ax.fill_between(x_axis, p25_elite, p75_elite, color='green', alpha=0.2, zorder=4,
                        label='25th–75th Percentile (Elites)')

        # --- Red: median of best-ever (global best) across runs per iteration ---
        mat_global = pad(all_global_bests, max_len)
        med_global = np.nanmedian(mat_global, axis=0)

        # ax.plot(x_axis, med_global, color='red', lw=2.5, zorder=5,
        #         marker='*', markersize=6, label='Median Best (Overall)')

        ax.set_title("Elite Fitness by Iteration", fontsize=16)
        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Fitness", fontsize=14)
        ax.tick_params(labelsize=13)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(ls=':', alpha=0.5)
        _save_plot(fig, "combined_fitness_by_iteration")
        plt.show()

    def plot_combined_dashboard(self):
        """Single figure: Elite Fitness | Jump Histogram | 3D Mesh + Trajectories."""
        if not self.results:
            return

        TICK_FS   = 13
        LABEL_FS  = 14
        LEGEND_FS = 12

        fig = plt.figure(figsize=(24, 13))
        gs  = fig.add_gridspec(2, 2,
                               height_ratios=[1, 1],
                               width_ratios=[1, 1.4],
                               hspace=0.45, wspace=0.1)
        ax_fitness = fig.add_subplot(gs[0, 0])
        ax_jump    = fig.add_subplot(gs[1, 0])
        ax_3d      = fig.add_subplot(gs[:, 1], projection='3d')

        # ── 1. Elite Fitness by Iteration ─────────────────────────────────────
        all_iter_bests   = []
        all_global_bests = []
        all_elite_fits   = []
        max_len = 0

        for k, r in enumerate(self.results):
            iters, fits = [], []
            for i, it_elites in enumerate(r.all_elites):
                iters.extend([i + 1] * len(it_elites))
                fits.extend([e['fitness'] for e in it_elites])
            iters, fits = np.array(iters), np.array(fits)
            mask = fits < 1e4
            iters, fits = iters[mask], fits[mask]

            best_mask = np.isclose(fits, r.best_fit_ever, atol=1e-8)
            c = self.cmap(k)
            ax_fitness.scatter(iters[~best_mask], fits[~best_mask],
                               color=c, s=30, alpha=0.6, edgecolors='black')
            if any(best_mask):
                ax_fitness.scatter(iters[best_mask], fits[best_mask],
                                   color=c, s=60, marker='*', edgecolors='red', zorder=4)

            n_iters = int(iters.max()) if len(iters) > 0 else 0
            iter_bests, global_bests, elite_fits_per_iter = [], [], []
            current_global_min = float('inf')
            for it in range(1, n_iters + 1):
                it_fits = fits[iters == it]
                it_best = float(it_fits.min()) if len(it_fits) > 0 else (iter_bests[-1] if iter_bests else np.nan)
                if it_best < current_global_min:
                    current_global_min = it_best
                iter_bests.append(it_best)
                global_bests.append(current_global_min)
                elite_fits_per_iter.append(it_fits)
            max_len = max(max_len, n_iters)
            all_iter_bests.append(iter_bests)
            all_global_bests.append(global_bests)
            all_elite_fits.append(elite_fits_per_iter)

        def _pad(data_list, length):
            arr = np.full((len(data_list), length), np.nan)
            for i, row in enumerate(data_list):
                arr[i, :len(row)] = row
                if len(row) < length:
                    arr[i, len(row):] = row[-1]
            return arr

        x_axis    = np.arange(1, max_len + 1)
        med_elite = np.full(max_len, np.nan)
        p25_elite = np.full(max_len, np.nan)
        p75_elite = np.full(max_len, np.nan)
        for it_idx in range(max_len):
            pooled = []
            for run_elites in all_elite_fits:
                if it_idx < len(run_elites):
                    eff = min(it_idx, len(run_elites) - 1)
                    if len(run_elites[eff]) > 0:
                        pooled.extend(run_elites[eff])
            if pooled:
                med_elite[it_idx] = np.median(pooled)
                p25_elite[it_idx] = np.percentile(pooled, 25)
                p75_elite[it_idx] = np.percentile(pooled, 75)

        ax_fitness.plot(x_axis, med_elite, color='green', lw=2.5, zorder=5,
                        label='Median Elite Fitness')
        ax_fitness.fill_between(x_axis, p25_elite, p75_elite,
                                color='green', alpha=0.2, zorder=4,
                                label='25th–75th Percentile (Elites)')
        mat_global = _pad(all_global_bests, max_len)
        med_global = np.nanmedian(mat_global, axis=0)
        ax_fitness.plot(x_axis, med_global, color='red', lw=2.5, zorder=5,
                        marker='*', markersize=6, label='Median Best (Overall)')

        ax_fitness.set_title("Elite Fitness by Iteration", fontsize=LABEL_FS + 2)
        ax_fitness.set_xlabel("Iteration",                 fontsize=LABEL_FS)
        ax_fitness.set_ylabel("Fitness",                   fontsize=LABEL_FS)
        ax_fitness.tick_params(labelsize=TICK_FS)
        ax_fitness.legend(loc='upper right', fontsize=LEGEND_FS)
        ax_fitness.grid(ls=':', alpha=0.5)

        # ── 2. Jump Count Distribution ─────────────────────────────────────────
        all_jumps = [[e['n_jumps'] for it in r.all_elites for e in it] for r in self.results]
        if any(all_jumps):
            j_min = min(min(j) for j in all_jumps if j)
            j_max = max(max(j) for j in all_jumps if j)
            vals  = np.arange(j_min, j_max + 1)
            bw    = 0.8 / len(self.results)
            for k, (jumps, r) in enumerate(zip(all_jumps, self.results)):
                counts = [jumps.count(v) for v in vals]
                ax_jump.bar(vals + (k - len(self.results) / 2 + 0.5) * bw,
                            counts, width=bw,
                            color=self.cmap(k), edgecolor="black", alpha=0.8, label=r.label)
            ax_jump.set_xticks(vals)
            ax_jump.set_title("Jump Count Distribution",  fontsize=LABEL_FS + 2)
            ax_jump.set_xlabel("Number of Jumps",         fontsize=LABEL_FS)
            ax_jump.set_ylabel("Frequency",               fontsize=LABEL_FS)
            ax_jump.tick_params(labelsize=TICK_FS)
            ax_jump.legend(fontsize=LEGEND_FS)
            ax_jump.grid(axis="y", ls="--", alpha=0.5)

        # ── 3. 3D Mesh + Trajectories ──────────────────────────────────────────
        ref = self.results[0]
        pts   = np.array([p["position"] for p in ref.points_t_data])
        costs = np.array([p["cost"]     for p in ref.points_t_data])
        max_pts = 5000
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts_plot, costs_plot = pts[idx], costs[idx]
        else:
            pts_plot, costs_plot = pts, costs
        ax_3d.scatter(*pts_plot.T, c=costs_plot, cmap="RdYlGn_r", s=10, alpha=0.2, zorder=1)

        all_x, all_y, all_z = [pts[:, 0]], [pts[:, 1]], [pts[:, 2]]
        valid_results = [r for r in self.results if r.best_traj_ever and r.best_fit_ever is not None]
        best_label    = min(valid_results, key=lambda r: r.best_fit_ever).label if valid_results else None

        lines_data = {}
        for k, r in enumerate(self.results):
            if not r.best_traj_ever:
                continue
            c = self.cmap(k)
            is_best       = (r.label == best_label)
            draw_color    = '#0080FF' if is_best else c
            traj_lw       = 4.5  if is_best else 1.5
            traj_alpha    = 1.0  if is_best else 0.25
            scatter_alpha = 1.0  if is_best else 0.25
            zorder_traj   = 12   if is_best else 9
            arts, landing_points = [], []

            for i, seg in enumerate(r.best_traj_ever):
                s = np.array(seg)
                xs, ys, zs = s if s.shape[0] == 3 and s.shape[1] != 3 else s.T
                arts.extend(ax_3d.plot(xs, ys, zs,
                                       color=draw_color, lw=traj_lw,
                                       alpha=traj_alpha, zorder=zorder_traj))
                all_x.append(xs); all_y.append(ys); all_z.append(zs)
                landing_points.append([xs[0], ys[0], zs[0]])
                if i == len(r.best_traj_ever) - 1:
                    landing_points.append([xs[-1], ys[-1], zs[-1]])

            lp = np.array(landing_points)
            arts.append(ax_3d.scatter(lp[:, 0], lp[:, 1], lp[:, 2],
                                      c='yellow', s=70 if is_best else 30,
                                      edgecolors='black', alpha=scatter_alpha,
                                      zorder=zorder_traj + 1))
            arts.append(ax_3d.scatter(*r.p0, c='lime',  s=200, marker="^",
                                      edgecolors="k", linewidth=2,
                                      alpha=scatter_alpha, zorder=15))
            arts.append(ax_3d.scatter(*r.pf, c="red",   s=200, marker="X",
                                      edgecolors="k", linewidth=2,
                                      alpha=scatter_alpha, zorder=15))
            if r.best_achieved_target_ever:
                ach = np.array(r.best_achieved_target_ever).flatten()
                arts.append(ax_3d.scatter(*ach, c="orange", s=250, marker="*",
                                          edgecolors="k", alpha=scatter_alpha, zorder=16))
                arts.extend(ax_3d.plot(*zip(r.pf, ach), "--",
                                       color=draw_color, lw=2,
                                       alpha=0.7 * scatter_alpha, zorder=14))
                print(f"[INFO] {r.label} — goal→achieved: "
                      f"{np.linalg.norm(np.array(r.pf) - ach):.4f} m")

            label_str = (f"★ {r.label} (fit={r.best_fit_ever:.4f})"
                         if is_best else f"{r.label} (fit={r.best_fit_ever:.4f})")
            lines_data[r.label] = (arts,
                                   Patch(facecolor=draw_color, label=label_str,
                                         linewidth=2 if is_best else 1,
                                         edgecolor='gold' if is_best else 'none'))

        leg3d = ax_3d.legend(handles=[d[1] for d in lines_data.values()],
                             loc="upper left", bbox_to_anchor=(1.05, 1),
                             title="Click to hide/show", fontsize=LEGEND_FS)
        interact_map = {id(p): lines_data[lbl][0]
                        for p, lbl in zip(leg3d.get_patches(), lines_data.keys())}
        interact_map.update({id(t): lines_data[lbl][0]
                             for t, lbl in zip(leg3d.get_texts(), lines_data.keys())})

        def on_pick(e):
            if id(e.artist) in interact_map:
                _arts = interact_map[id(e.artist)]
                vis = not _arts[0].get_visible()
                for a in _arts:
                    a.set_visible(vis)
                e.artist.set_alpha(1.0 if vis else 0.3)
                fig.canvas.draw_idle()

        for a in leg3d.get_patches() + leg3d.get_texts():
            a.set_picker(True)
        fig.canvas.mpl_connect("pick_event", on_pick)

        # 1:1:1 aspect
        flat_x = np.concatenate(all_x)
        flat_y = np.concatenate(all_y)
        flat_z = np.concatenate(all_z)
        max_range = np.array([flat_x.ptp(), flat_y.ptp(), flat_z.ptp()]).max() / 2.0
        mid_x = (flat_x.max() + flat_x.min()) * 0.5
        mid_y = (flat_y.max() + flat_y.min()) * 0.5
        mid_z = (flat_z.max() + flat_z.min()) * 0.5
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
        ax_3d.set_box_aspect((1, 1, 1))

        ax_3d.set_xlabel('')
        ax_3d.set_xticklabels([])
        ax_3d.xaxis.set_major_locator(plt.NullLocator())
        ax_3d.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax_3d.zaxis.set_major_locator(plt.MaxNLocator(2))
        ax_3d.set_ylabel('Y [m]', fontsize=LABEL_FS)
        ax_3d.set_zlabel('Z [m]', fontsize=LABEL_FS)
        ax_3d.tick_params(axis='y', labelsize=TICK_FS)
        ax_3d.tick_params(axis='z', labelsize=TICK_FS)
        ax_3d.view_init(elev=20, azim=-20)

        fig.suptitle("Combined Results Dashboard", fontsize=LABEL_FS + 4, y=1.01)
        _save_plot(fig, "combined_dashboard")
        plt.show()

if __name__ == "__main__":
    cp = CombinePlot(FOLDERS)
    if cp.results:
        # cp.plot_fitness_stats_over_time()
        # cp.plot_jump_histogram()
        # cp.plot_fitness_by_iteration_2()
        # cp.plot_best_fitness_line()
        cp.plot_mesh_pc_traj()
        # cp.plot_convergence()
        # cp.plot_combined_dashboard()
        print("All combined plots completed successfully!")