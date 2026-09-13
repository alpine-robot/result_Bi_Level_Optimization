import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

# Helper condivisi per i plot dinamici (deve stare nella stessa cartella)
try:
    import combine_one_terrain.dynamic_plots as dyn
except ImportError:
    dyn = None
    print("[WARN] dynamic_plots.py non trovato: i metodi animate_* non saranno disponibili.")

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

    def plot_mesh_pc_traj(self, elev=25, azim=-60):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.tri as mtri
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Patch

        if not self.results:
            return

        # ============================================================
        # FUNZIONI INTERNE DI SUPPORTO
        # ============================================================

        def plot_costmap_true_3d_mesh(
            ax,
            px,
            py,
            pz,
            costs,
            cmap='RdYlGn_r',
            alpha=1.0,
            max_edge_factor=5.0,
            edge_alpha=0.0
        ):
            """
            Disegna la cost map come vera mesh 3D triangolata.
            Usa direttamente i punti originali x, y, z.
            """

            px = np.asarray(px).flatten()
            py = np.asarray(py).flatten()
            pz = np.asarray(pz).flatten()
            costs = np.asarray(costs).flatten()

            valid = (
                np.isfinite(px)
                & np.isfinite(py)
                & np.isfinite(pz)
                & np.isfinite(costs)
            )

            px = px[valid]
            py = py[valid]
            pz = pz[valid]
            costs = costs[valid]

            xyz = np.column_stack((px, py, pz))

            if xyz.shape[0] < 3:
                print("[WARNING] Not enough valid terrain points to build a 3D mesh.")
                return None

            # Triangolazione sulle due coordinate che variano di più.
            # La geometria finale però resta 3D perché usiamo xyz[triangles].
            ranges = np.ptp(xyz, axis=0)
            uv_axes = np.argsort(ranges)[-2:]

            u = xyz[:, uv_axes[0]]
            v = xyz[:, uv_axes[1]]

            triang = mtri.Triangulation(u, v)
            triangles = triang.triangles

            if triangles.size == 0:
                print("[WARNING] Triangulation failed: no triangles generated.")
                return None

            # Filtro triangoli troppo lunghi
            uv = np.column_stack((u, v))
            tri_uv = uv[triangles]

            e01 = np.linalg.norm(tri_uv[:, 0, :] - tri_uv[:, 1, :], axis=1)
            e12 = np.linalg.norm(tri_uv[:, 1, :] - tri_uv[:, 2, :], axis=1)
            e20 = np.linalg.norm(tri_uv[:, 2, :] - tri_uv[:, 0, :], axis=1)

            edge_lengths = np.column_stack((e01, e12, e20))
            median_edge = np.median(edge_lengths)

            if median_edge > 0:
                max_allowed_edge = max_edge_factor * median_edge
                valid_triangles = np.all(edge_lengths < max_allowed_edge, axis=1)
                triangles = triangles[valid_triangles]

            if triangles.size == 0:
                print("[WARNING] All triangles removed by edge filtering.")
                return None

            verts = xyz[triangles]
            tri_costs = costs[triangles].mean(axis=1)

            # Normalizzazione robusta dei colori
            vmin, vmax = np.nanpercentile(costs, [2, 98])

            if np.isclose(vmin, vmax):
                vmin = np.nanmin(costs)
                vmax = np.nanmax(costs)

            if np.isclose(vmin, vmax):
                vmin = vmax - 1.0

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)

            facecolors = cmap_obj(norm(tri_costs))
            facecolors[:, 3] = alpha

            mesh = Poly3DCollection(
                verts,
                facecolors=facecolors,
                edgecolors=(0, 0, 0, edge_alpha),
                linewidths=0.0,
                zorder=0
            )

            ax.add_collection3d(mesh)

            return mesh

        def unpack_segment(segment):
            """
            Gestisce traiettorie in formato (3, N) oppure (N, 3).
            """

            s = np.asarray(segment)

            if s.ndim != 2:
                raise ValueError("Trajectory segment must be a 2D array.")

            if s.shape[0] == 3 and s.shape[1] != 3:
                xs = s[0, :]
                ys = s[1, :]
                zs = s[2, :]
            else:
                xs = s[:, 0]
                ys = s[:, 1]
                zs = s[:, 2]

            return xs, ys, zs

        def plot_line_with_halo(
            ax,
            x,
            y,
            z,
            color='blue',
            linewidth=3.4,
            alpha=1.0,
            halo=True,
            halo_alpha=0.95,
            halo_width=8.5,
            zorder=100,
            label=None
        ):
            """
            Disegna una traiettoria con alone bianco.
            Per le traiettorie non-best l'alone è molto più trasparente.
            """

            if halo:
                ax.plot(
                    x,
                    y,
                    z,
                    color='white',
                    linewidth=halo_width,
                    alpha=halo_alpha,
                    zorder=zorder - 1
                )

            line, = ax.plot(
                x,
                y,
                z,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                label=label
            )

            return line

        def scatter_marker_with_halo(
            ax,
            point,
            marker,
            color,
            size,
            alpha=1.0,
            label=None,
            zorder=200,
            halo=True
        ):
            """
            Disegna un marker con alone bianco.
            Per traiettorie non-best è più trasparente.
            """

            point = np.asarray(point).flatten()

            if point.size < 3:
                return []

            x, y, z = point[0], point[1], point[2]

            arts = []

            if halo:
                arts.append(
                    ax.scatter(
                        [x],
                        [y],
                        [z],
                        c='white',
                        s=size * 2.25,
                        marker=marker,
                        edgecolors='black',
                        linewidth=0.8,
                        alpha=alpha,
                        depthshade=False,
                        zorder=zorder - 1
                    )
                )

            arts.append(
                ax.scatter(
                    [x],
                    [y],
                    [z],
                    c=color,
                    s=size,
                    marker=marker,
                    edgecolors='black',
                    linewidth=2.0,
                    alpha=alpha,
                    depthshade=False,
                    label=label,
                    zorder=zorder
                )
            )

            return arts

        # ============================================================
        # FIGURA E ASSI 3D
        # ============================================================

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Sfondo bianco come nel plot singolo
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Nasconde completamente assi, box, tick e labels
        ax.axis('off')

        # Controllo manuale dello zorder
        ax.computed_zorder = False

        # Proiezione ortografica
        ax.set_proj_type('ortho')

        # ============================================================
        # 1. TERRENO DAL PRIMO RISULTATO
        # ============================================================

        ref = self.results[0]

        pts = np.array([p["position"] for p in ref.points_t_data])
        costs = np.array([p["cost"] for p in ref.points_t_data])

        px = pts[:, 0]
        py = pts[:, 1]
        pz = pts[:, 2]

        plot_costmap_true_3d_mesh(
            ax,
            px,
            py,
            pz,
            costs,
            cmap='RdYlGn_r',
            alpha=1.0,
            max_edge_factor=5.0,
            edge_alpha=0.0
        )

        # ============================================================
        # 2. COORDINATE PER LIMITI
        # ============================================================

        all_x = [px]
        all_y = [py]
        all_z = [pz]

        # ============================================================
        # 3. TROVA LA TRAIETTORIA MIGLIORE
        # ============================================================

        valid_results = [
            r for r in self.results
            if r.best_traj_ever is not None
            and len(r.best_traj_ever) > 0
            and r.best_fit_ever is not None
        ]

        best_label = None

        if valid_results:
            best_result = min(valid_results, key=lambda r: r.best_fit_ever)
            best_label = best_result.label

        # ============================================================
        # 4. DISEGNA TUTTE LE TRAIETTORIE
        # ============================================================

        lines_data = {}

        for k, r in enumerate(self.results):

            if r.best_traj_ever is None or len(r.best_traj_ever) == 0:
                continue

            if r.best_fit_ever is None:
                continue

            is_best = r.label == best_label

            # Colori e trasparenze
            base_color = self.cmap(k)
            draw_color = '#0080FF' if is_best else base_color

            traj_lw = 3.4 if is_best else 1.5
            traj_alpha = 1.0 if is_best else 0.22

            marker_alpha = 1.0 if is_best else 0.20

            halo_alpha = 0.95 if is_best else 0.18
            halo_width = 8.5 if is_best else 4.5

            zorder_traj = 100 if is_best else 60
            zorder_marker = 220 if is_best else 120

            arts = []
            landing_points = []

            for i, seg in enumerate(r.best_traj_ever):

                xs, ys, zs = unpack_segment(seg)

                line = plot_line_with_halo(
                    ax,
                    xs,
                    ys,
                    zs,
                    color=draw_color,
                    linewidth=traj_lw,
                    alpha=traj_alpha,
                    halo=True,
                    halo_alpha=halo_alpha,
                    halo_width=halo_width,
                    zorder=zorder_traj,
                    label=None
                )

                arts.append(line)

                all_x.append(xs)
                all_y.append(ys)
                all_z.append(zs)

                landing_points.append([xs[0], ys[0], zs[0]])

                if i == len(r.best_traj_ever) - 1:
                    landing_points.append([xs[-1], ys[-1], zs[-1]])

            # ========================================================
            # CONTACT POINTS
            # ========================================================

            if len(landing_points) > 0:
                lp = np.asarray(landing_points)

                # Alone bianco contact points
                arts.append(
                    ax.scatter(
                        lp[:, 0],
                        lp[:, 1],
                        lp[:, 2],
                        c='white',
                        s=210 if is_best else 80,
                        marker='o',
                        edgecolors='black',
                        linewidth=0.8,
                        alpha=marker_alpha,
                        depthshade=False,
                        zorder=zorder_marker
                    )
                )

                # Contact points veri
                arts.append(
                    ax.scatter(
                        lp[:, 0],
                        lp[:, 1],
                        lp[:, 2],
                        c='yellow',
                        s=95 if is_best else 35,
                        marker='o',
                        edgecolors='black',
                        linewidth=1.8 if is_best else 0.8,
                        alpha=marker_alpha,
                        depthshade=False,
                        zorder=zorder_marker + 1
                    )
                )

            # ========================================================
            # START E DESIRED GOAL
            # ========================================================

            arts.extend(
                scatter_marker_with_halo(
                    ax,
                    r.p0,
                    marker='^',
                    color='lime',
                    size=250 if is_best else 120,
                    alpha=marker_alpha,
                    zorder=zorder_marker + 20,
                    halo=True
                )
            )

            arts.extend(
                scatter_marker_with_halo(
                    ax,
                    r.pf,
                    marker='X',
                    color='red',
                    size=260 if is_best else 120,
                    alpha=marker_alpha,
                    zorder=zorder_marker + 30,
                    halo=True
                )
            )

            # ========================================================
            # REACHED GOAL E DISCREPANZA
            # ========================================================

            if r.best_achieved_target_ever is not None:
                ach = np.asarray(r.best_achieved_target_ever).flatten()

                if ach.size >= 3:
                    arts.extend(
                        scatter_marker_with_halo(
                            ax,
                            ach,
                            marker='*',
                            color='orange',
                            size=360 if is_best else 150,
                            alpha=marker_alpha,
                            zorder=zorder_marker + 40,
                            halo=True
                        )
                    )

                    # Discrepanza goal con alone bianco
                    arts.extend(
                        ax.plot(
                            [r.pf[0], ach[0]],
                            [r.pf[1], ach[1]],
                            [r.pf[2], ach[2]],
                            color='white',
                            linestyle='--',
                            linewidth=6.5 if is_best else 3.0,
                            alpha=0.9 if is_best else 0.15,
                            zorder=zorder_marker + 10
                        )
                    )

                    arts.extend(
                        ax.plot(
                            [r.pf[0], ach[0]],
                            [r.pf[1], ach[1]],
                            [r.pf[2], ach[2]],
                            color='red' if is_best else draw_color,
                            linestyle='--',
                            linewidth=2.7 if is_best else 1.2,
                            alpha=1.0 if is_best else 0.22,
                            zorder=zorder_marker + 11
                        )
                    )

                    dist = np.linalg.norm(np.asarray(r.pf).flatten()[:3] - ach[:3])
                    print(f"[INFO] {r.label} — goal→achieved: {dist:.4f} m")

            # ========================================================
            # DATI PER LEGENDA INTERATTIVA
            # ========================================================

            if is_best:
                label_str = f"★ {r.label} (fit={r.best_fit_ever:.4f})"
            else:
                label_str = f"{r.label} (fit={r.best_fit_ever:.4f})"

            patch = Patch(
                facecolor=draw_color,
                label=label_str,
                linewidth=2 if is_best else 1,
                edgecolor='gold' if is_best else 'none',
                alpha=1.0 if is_best else 0.35
            )

            lines_data[r.label] = (arts, patch)

        # # ============================================================
        # # 5. LEGENDA INTERATTIVA OPZIONALE
        # # ============================================================
        # # Se vuoi il plot identico a quello singolo senza legenda,
        # # commenta tutto questo blocco.

        # if len(lines_data) > 0:
        #     leg = ax.legend(
        #         handles=[d[1] for d in lines_data.values()],
        #         loc="upper left",
        #         bbox_to_anchor=(1.02, 1.0),
        #         title="Click to hide/show",
        #         frameon=False
        #     )

        #     interact_map = {
        #         id(p): lines_data[lbl][0]
        #         for p, lbl in zip(leg.get_patches(), lines_data.keys())
        #     }

        #     interact_map.update({
        #         id(t): lines_data[lbl][0]
        #         for t, lbl in zip(leg.get_texts(), lines_data.keys())
        #     })

        #     def on_pick(e):
        #         if id(e.artist) in interact_map:
        #             arts = interact_map[id(e.artist)]

        #             if len(arts) == 0:
        #                 return

        #             vis = not arts[0].get_visible()

        #             for a in arts:
        #                 a.set_visible(vis)

        #             e.artist.set_alpha(1.0 if vis else 0.3)
        #             fig.canvas.draw_idle()

        #     for a in leg.get_patches() + leg.get_texts():
        #         a.set_picker(True)

        #     fig.canvas.mpl_connect("pick_event", on_pick)

        # ============================================================
        # 6. LIMITI COME NEL PLOT SINGOLO
        # ============================================================

        flat_x = np.concatenate(all_x)
        flat_y = np.concatenate(all_y)
        flat_z = np.concatenate(all_z)

        x_min = np.nanmin(flat_x)
        x_max = np.nanmax(flat_x)

        y_min_data = np.nanmin(flat_y)
        y_max_data = np.nanmax(flat_y)

        z_min = np.nanmin(flat_z)
        z_max = np.nanmax(flat_z)

        max_range = np.array([
            x_max - x_min,
            y_max_data - y_min_data,
            z_max - z_min
        ]).max() / 2.0

        mid_x = (x_max + x_min) * 0.5
        mid_z = (z_max + z_min) * 0.5

        y_min = 0
        y_max = 15

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_box_aspect((1, 1, 1))

        # ============================================================
        # 7. VISTA E SALVATAGGIO
        # ============================================================

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()

        _save_plot(fig, "combined_mesh_pc_traj")

        plt.show()


    # ════════════════════════════════════════════════════════════════════════
    #  VERSIONE DINAMICA di plot_mesh_pc_traj
    #  Le traiettorie vengono disegnate una dopo l'altra: quella in corso è
    #  in evidenza, quelle già finite restano a schermo ma diventano opache.
    # ════════════════════════════════════════════════════════════════════════
    def animate_mesh_pc_traj(self,
                             elev=20,
                             azim=-20,
                             n_frames=260,
                             fps=25,
                             hold_frames=45,
                             order="best_last",
                             faded_alpha=0.20,
                             ylim=(0.0, 15.0),
                             show_label=True,
                             gif_name="combined_mesh_pc_traj_animated",
                             dpi=100,
                             show=True):
        """
        Animazione 3-D delle traiettorie combinate.

        Ogni run viene tracciato progressivamente (il percorso si costruisce
        segmento dopo segmento). Appena un run è completo la sua traiettoria
        NON sparisce: passa allo stile "sfondo" (linea sottile, alpha basso),
        mentre il run successivo viene disegnato in primo piano. Alla fine
        restano tutte visibili con il migliore in evidenza.

        Parametri
        ---------
        order        "best_last"  → il run migliore viene disegnato per ultimo
                     "fitness"    → dal peggiore al migliore
                     "folder"     → nell'ordine delle cartelle
        faded_alpha  alpha delle traiettorie già completate
        ylim         limiti Y fissi come nel plot statico (None = automatici)
        n_frames     numero di frame totali distribuiti su TUTTI i run
        """
        if dyn is None:
            raise RuntimeError("Serve dynamic_plots.py nella stessa cartella.")
        if not self.results:
            return None

        # ── run validi + ordine di disegno ───────────────────────────────────
        valid = [(k, r) for k, r in enumerate(self.results)
                 if r.best_traj_ever and r.best_fit_ever is not None]
        if not valid:
            print("[WARN] Nessuna traiettoria valida da animare.")
            return None

        best_label = min(valid, key=lambda kr: kr[1].best_fit_ever)[1].label

        if order == "fitness":
            draw_order = sorted(valid, key=lambda kr: -kr[1].best_fit_ever)
        elif order == "best_last":
            draw_order = sorted(valid, key=lambda kr: kr[1].label == best_label)
        else:
            draw_order = list(valid)

        # ── figura / assi come nel plot statico ──────────────────────────────
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.axis('off')
        ax.computed_zorder = False
        ax.set_proj_type('ortho')

        ref = self.results[0]
        pts = np.array([p["position"] for p in ref.points_t_data])
        costs = np.array([p["cost"] for p in ref.points_t_data])
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        dyn.costmap_true_3d_mesh(ax, px, py, pz, costs, cmap='RdYlGn_r',
                                 alpha=1.0, max_edge_factor=5.0, edge_alpha=0.0)

        all_x, all_y, all_z = [px], [py], [pz]

        # ── timeline unica: tutti i segmenti di tutti i run in fila ──────────
        segs, offset = [], 0
        runs = []
        for pos, (k, r) in enumerate(draw_order):
            r_segs, offset = dyn.collect_path(r.best_traj_ever, tag=pos, offset=offset)
            if not r_segs:
                continue
            for s in r_segs:
                all_x.append(s["x"]); all_y.append(s["y"]); all_z.append(s["z"])
            segs.extend(r_segs)
            runs.append(dict(pos=pos, res=r, cmap_idx=k,
                             is_best=(r.label == best_label),
                             segs=r_segs,
                             lands=dyn.landing_points_of(r.best_traj_ever)))
        total_pts = offset
        if total_pts == 0:
            print("[WARN] Traiettorie vuote.")
            plt.close(fig)
            return None

        # ── artisti per ogni run ─────────────────────────────────────────────
        def _marker_pair(color, marker, zorder):
            halo, = ax.plot([], [], [], linestyle="none", marker=marker,
                            color='white', markeredgecolor='black',
                            markeredgewidth=0.8, zorder=zorder - 1)
            main, = ax.plot([], [], [], linestyle="none", marker=marker,
                            color=color, markeredgecolor='black',
                            markeredgewidth=1.6, zorder=zorder)
            return [halo, main]

        for run in runs:
            r = run["res"]
            run["color"] = '#0080FF' if run["is_best"] else self.cmap(run["cmap_idx"])
            c = run["color"]

            run["halos"], run["lines"] = [], []
            for _ in run["segs"]:
                h, = ax.plot([], [], [], color='white', lw=8.5, alpha=0.95, zorder=99)
                m, = ax.plot([], [], [], color=c, lw=3.4, alpha=1.0, zorder=100)
                run["halos"].append(h)
                run["lines"].append(m)

            run["contact"] = _marker_pair('yellow', 'o', 190)
            run["p0"] = _marker_pair('lime', '^', 230)
            run["pf"] = _marker_pair('red', 'X', 240)
            run["ach"] = _marker_pair('orange', '*', 250)

            dw, = ax.plot([], [], [], color='white', ls='--', lw=6.5,
                          alpha=0.9, zorder=210)
            dc, = ax.plot([], [], [], color='red' if run["is_best"] else c,
                          ls='--', lw=2.7, alpha=1.0, zorder=211)
            run["disc"] = [dw, dc]

            ach = r.best_achieved_target_ever
            run["ach_pt"] = (np.asarray(ach).flatten()[:3]
                             if ach is not None and np.asarray(ach).size >= 3 else None)

        head, = ax.plot([], [], [], linestyle="none", marker='o', markersize=11,
                        color='white', markeredgecolor='black',
                        markeredgewidth=1.6, zorder=300)

        txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=15,
                        fontweight='bold', va='top', zorder=400,
                        bbox=dict(boxstyle="round,pad=0.35", fc='white',
                                  ec='0.7', alpha=0.85))
        txt.set_visible(show_label)

        # ── stile attivo / opaco ─────────────────────────────────────────────
        def _style(run, active):
            lw = 3.4 if active else 1.5
            a_line = 1.0 if active else faded_alpha
            a_halo = 0.95 if active else 0.18
            hw = 8.5 if active else 4.5
            a_mk = 1.0 if active else faded_alpha
            big = 1.0 if active else 0.62

            for h, m in zip(run["halos"], run["lines"]):
                h.set_linewidth(hw); h.set_alpha(a_halo)
                m.set_linewidth(lw); m.set_alpha(a_line)

            for key, ms_halo, ms_main in (("contact", 15, 10),
                                          ("p0", 17, 12),
                                          ("pf", 17, 12),
                                          ("ach", 21, 15)):
                halo, main = run[key]
                halo.set_markersize(ms_halo * big); halo.set_alpha(a_mk)
                main.set_markersize(ms_main * big); main.set_alpha(a_mk)

            run["disc"][0].set_linewidth(6.5 if active else 3.0)
            run["disc"][0].set_alpha(0.9 if active else 0.15)
            run["disc"][1].set_linewidth(2.7 if active else 1.2)
            run["disc"][1].set_alpha(1.0 if active else faded_alpha)

        def _hide(run):
            for a in (run["halos"] + run["lines"] + run["contact"] + run["p0"]
                      + run["pf"] + run["ach"] + run["disc"]):
                a.set_visible(False)

        def _show(run):
            for a in (run["halos"] + run["lines"] + run["contact"] + run["p0"]
                      + run["pf"] + run["ach"] + run["disc"]):
                a.set_visible(True)

        # ── update ───────────────────────────────────────────────────────────
        def _update(k):
            head_xyz, active_run = None, None

            for run in runs:
                started, done = False, 0
                for s, h, m in zip(run["segs"], run["halos"], run["lines"]):
                    n = dyn.visible_count(s, k)
                    dyn.set_line3d(h, s["x"][:n], s["y"][:n], s["z"][:n])
                    dyn.set_line3d(m, s["x"][:n], s["y"][:n], s["z"][:n])
                    if n > 0:
                        started = True
                        head_xyz = (s["x"][n - 1], s["y"][n - 1], s["z"][n - 1])
                    if n == s["n"]:
                        done += 1
                finished = done == len(run["segs"])

                if not started:
                    _hide(run)
                    continue

                _show(run)
                active = not finished
                if active:
                    active_run = run
                _style(run, active)

                r = run["res"]
                lp = run["lands"]
                n_lp = min(done + 1, len(lp))
                if n_lp > 0:
                    for a in run["contact"]:
                        dyn.set_line3d(a, lp[:n_lp, 0], lp[:n_lp, 1], lp[:n_lp, 2])
                else:
                    for a in run["contact"]:
                        dyn.set_line3d(a, [], [], [])

                p0 = np.asarray(r.p0).flatten()
                for a in run["p0"]:
                    dyn.set_line3d(a, [p0[0]], [p0[1]], [p0[2]])

                pf = np.asarray(r.pf).flatten()
                if finished:
                    for a in run["pf"]:
                        dyn.set_line3d(a, [pf[0]], [pf[1]], [pf[2]])
                    ach = run["ach_pt"]
                    if ach is not None:
                        for a in run["ach"]:
                            dyn.set_line3d(a, [ach[0]], [ach[1]], [ach[2]])
                        for a in run["disc"]:
                            dyn.set_line3d(a, [pf[0], ach[0]], [pf[1], ach[1]],
                                           [pf[2], ach[2]])
                    else:
                        for a in run["ach"] + run["disc"]:
                            dyn.set_line3d(a, [], [], [])
                else:
                    for a in run["pf"] + run["ach"] + run["disc"]:
                        dyn.set_line3d(a, [], [], [])

            # a fine animazione l'ultimo run disegnato (il migliore, con
            # order="best_last") resta in evidenza invece di sbiadire
            if active_run is None and runs:
                _style(runs[-1], True)

            if head_xyz is not None and k < total_pts and active_run is not None:
                dyn.set_line3d(head, [head_xyz[0]], [head_xyz[1]], [head_xyz[2]])
            else:
                dyn.set_line3d(head, [], [], [])

            if show_label:
                ref_run = active_run if active_run is not None else runs[-1]
                r = ref_run["res"]
                star = "★ " if ref_run["is_best"] else ""
                txt.set_text(f"{star}{r.label}   fit = {r.best_fit_ever:.4f}")
                txt.set_color(ref_run["color"] if not ref_run["is_best"] else '#0080FF')

            return []

        # ── limiti (come nel plot statico) ───────────────────────────────────
        flat_x = np.concatenate(all_x)
        flat_y = np.concatenate(all_y)
        flat_z = np.concatenate(all_z)
        x_min, x_max = np.nanmin(flat_x), np.nanmax(flat_x)
        y_min_d, y_max_d = np.nanmin(flat_y), np.nanmax(flat_y)
        z_min, z_max = np.nanmin(flat_z), np.nanmax(flat_z)
        max_range = np.array([x_max - x_min, y_max_d - y_min_d, z_max - z_min]).max() / 2.0
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max_d + y_min_d) * 0.5
        mid_z = (z_max + z_min) * 0.5

        y_lo, y_hi = ylim if ylim is not None else (mid_y - max_range, mid_y + max_range)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()

        frames = dyn.frame_indices(total_pts, n_frames, hold_frames)
        anim = FuncAnimation(fig, _update, frames=frames,
                             interval=1000.0 / fps, blit=False, repeat=True)
        self._anim_mesh = anim          # evita la garbage collection

        dyn.save_animation(anim, COMBINED_OUTPUT, gif_name, fps=fps, dpi=dpi)
        _update(total_pts)
        dyn.set_line3d(head, [], [], [])
        dyn.save_still(fig, COMBINED_OUTPUT, gif_name)
        if show:
            plt.show()
        return anim

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
        cp.plot_fitness_stats_over_time()
        cp.plot_jump_histogram()
        cp.plot_fitness_by_iteration_2()
        cp.plot_best_fitness_line()
        cp.plot_mesh_pc_traj(elev=20, azim=-20) #-->  Metodo per la Figura 5
        cp.animate_mesh_pc_traj(elev=20, azim=-20)  # --> versione dinamica (GIF)
        cp.plot_convergence()
        cp.plot_combined_dashboard()
        print("All combined plots completed successfully!")