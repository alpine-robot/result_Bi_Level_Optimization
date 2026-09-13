import json
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    # Fallback se PyQt5 non è installato nell'ambiente
    matplotlib.use('TkAgg')

from typing import Any, List, Optional
from attr import dataclass
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy.interpolate import griddata
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation

# Helper condivisi per i plot dinamici (deve stare nella stessa cartella)
try:
    import combine_one_terrain.dynamic_plots as dyn
except ImportError:
    dyn = None
    print("[WARN] dynamic_plots.py non trovato: i metodi animate_* non saranno disponibili.")


plot_str = os.environ.get("FOLDER_PLOT")
if plot_str:
    FOLDER_MAIN = np.array(json.loads(plot_str))
else:
    FOLDER_MAIN = "up_rocky/run_1"

FILE_TERRAIN_POINTS = f"{FOLDER_MAIN}/actual_point_terrain.json"
FILE_TERRAIN_PATCHES = f"{FOLDER_MAIN}/actual_patch_terrain.json"
ITERATIONS_FOLDER = f"{FOLDER_MAIN}/iteration_reports"
FILE_SAVE_PARAMS = f"{FOLDER_MAIN}/simulation_params.json"


@dataclass
class InnerParams:
    Fleg_max: float
    Fr_max: float
    Fr_min: float
    mass: float
    anchor_distance: float
    fitness_weights: List[float]
    filter_weights: List[float]
    inner_opt_params: Any


@dataclass
class CemParams:
    seed: int
    n_threads: int
    cem_iters: int
    pop_size: int
    n_elites: int
    decrease_pop_factor: float
    fraction_elites_reused: float
    dim_discrete: int
    n_values: int
    init_probs: List[float]
    min_prob: float
    dim_continuous: int
    max_value_continuous: List[float]
    min_value_continuous: List[float]
    init_mu_continuous: List[float]
    init_std_continuous: List[float]
    min_std_continuous: List[float]
    alpha: float


@dataclass
class eliteData:
    fitness: float
    n_jumps: int
    consumed_energy: float
    landing_cost: float
    points: List[List[float]]
    traj: List[List[List[float]]]
    patch_ids: List[int]
    achieved_target: Optional[List[float]]


class PlotResultCemMjumps:

    def __init__(self):
        # LOAD PARAMS
        self.data_params = self.load_simulation_params()

        self.p0 = self.data_params['START']
        self.pf = self.data_params['GOAL']
        self.n_jumps = self.data_params['MAX_JUMP']
        self.n_threads = self.data_params['THREADS']
        self.inner_params = InnerParams(**self.data_params['inner_opt_params_order'])
        self.cem_params = CemParams(**self.data_params['cem_params'])

        # LOAD POINTS
        self.data_terrain_points = self.load_terrain_points()
        self.num_points = self.data_terrain_points['metadata']['num_points']
        self.mesh_bounds = self.data_terrain_points['mesh_bounds']
        self.points_t_data = self.data_terrain_points['points']

        # LOAD PATCHES
        self.data_terrain_patches = self.load_terrain_patches()
        self.num_patches = self.data_terrain_patches['metadata']['num_patches']
        self.patch_width = self.data_terrain_patches['metadata']['patch_width']
        self.patch_height = self.data_terrain_patches['metadata']['patch_height']
        self.patches = self.data_terrain_patches['patches']

        # LOAD ITERATION HISTORY
        self.best_fit_ever = []
        self.best_energy_ever = []
        self.best_land_cost_ever = []
        self.best_traj_ever = []
        self.best_achieved_target_ever = None
        self.best_fit_each_iter = []
        self.all_elites = []

        if os.path.exists(ITERATIONS_FOLDER):
            self.iteration_files = sorted(
                [f for f in os.listdir(ITERATIONS_FOLDER) if f.startswith('iteration_') and f.endswith('.json')],
                key=lambda x: int(x.split('_')[1].split('.')[0])
            )
        else:
            self.iteration_files = []
            print(f"[WARNING] Cartella {ITERATIONS_FOLDER} non trovata.")

        self.correct_start_goal_positions()
        self.load_iteration_history()

    def load_iteration_history(self):
        for file_name in self.iteration_files:
            with open(os.path.join(ITERATIONS_FOLDER, file_name), 'r') as file:
                iteration_data = json.load(file)
                self.best_fit_each_iter.append(iteration_data['best_fitness_this_iter'])

                iteration_elites = []
                for e in iteration_data['elites']:
                    elite_instance = eliteData(
                        fitness=e['fitness'],
                        n_jumps=e['n_jumps'],
                        consumed_energy=e['consumed_energy'],
                        landing_cost=e['landing_cost'],
                        points=e['points'],
                        traj=e['traj'],
                        patch_ids=e['patch_ids'],
                        achieved_target=e.get('achieved_target', None)
                    )
                    iteration_elites.append(elite_instance)

                self.all_elites.append(iteration_elites)
                self.best_fit_ever = iteration_data['best_fitness_ever']
                self.best_energy_ever = iteration_data['best_consumed_energy_ever']
                self.best_land_cost_ever = iteration_data['best_landing_cost_ever']
                self.best_traj_ever = iteration_data['best_trajectory_ever']
                self.best_achieved_target_ever = iteration_data.get('best_achieved_target_ever', None)

    def load_simulation_params(self):
        with open(FILE_SAVE_PARAMS, 'r') as file:
            return json.load(file)

    def load_terrain_points(self):
        with open(FILE_TERRAIN_POINTS, 'r') as file:
            return json.load(file)

    def load_terrain_patches(self):
        with open(FILE_TERRAIN_PATCHES, 'r') as file:
            return json.load(file)

    def project_point_to_surface(self, point):
        target_y, target_z = point[1], point[2]
        tolerance_y = 0.1
        tolerance_z = 0.1

        candidates = []
        for p in self.points_t_data:
            pos = p['position']
            if (abs(pos[1] - target_y) < tolerance_y and
                    abs(pos[2] - target_z) < tolerance_z):
                candidates.append(pos)

        if not candidates:
            tolerance_y *= 2
            tolerance_z *= 2
            for p in self.points_t_data:
                pos = p['position']
                if (abs(pos[1] - target_y) < tolerance_y and
                        abs(pos[2] - target_z) < tolerance_z):
                    candidates.append(pos)

        if not candidates:
            return point

        candidates_np = np.array(candidates)
        distances = np.abs(candidates_np[:, 0] - point[0])
        closest_idx = np.argmin(distances)
        projected_point = candidates_np[closest_idx]

        return projected_point.tolist()

    def correct_start_goal_positions(self):
        self.p0 = self.project_point_to_surface(self.p0)
        self.pf = self.project_point_to_surface(self.pf)

    # =================
    # PLOTTER METHODS
    # =================
    @staticmethod
    def plot_cost_map_as_heatmap_plane(ax, px, py, pz, costs,
                                        cmap='RdYlGn_r',
                                        alpha=0.75,
                                        n_grid=250,
                                        plane_shift=0.00):
        """
        Disegna la cost map come una heatmap 2D su un piano 3D,
        invece che come scatter di punti.

        plane_shift sposta leggermente il piano lungo la sua normale.
        Se la mappa copre il percorso, prova plane_shift = 0.03 oppure -0.03.
        """
        xyz = np.vstack([px, py, pz])

        # Trova quale asse varia meno: quello è la normale del piano
        ranges = np.ptp(xyz, axis=1)
        normal_axis = int(np.argmin(ranges))
        uv_axes = [i for i in range(3) if i != normal_axis]

        u = xyz[uv_axes[0]]
        v = xyz[uv_axes[1]]

        u_lin = np.linspace(u.min(), u.max(), n_grid)
        v_lin = np.linspace(v.min(), v.max(), n_grid)
        U, V = np.meshgrid(u_lin, v_lin)

        # Interpola i costi sulla griglia 2D
        C = griddata((u, v), costs, (U, V), method='linear')

        # Scala colori robusta: evita che pochi outlier rovinino la colormap
        vmin, vmax = np.nanpercentile(costs, [2, 98])
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        try:
            cmap_obj = matplotlib.colormaps[cmap]
        except AttributeError:
            # fallback per versioni più vecchie di matplotlib
            cmap_obj = cm.get_cmap(cmap)

        facecolors = cmap_obj(norm(C))

        # Nasconde le zone fuori interpolazione
        mask = np.isnan(C)
        facecolors[mask, -1] = 0.0
        facecolors[~mask, -1] = alpha

        # Ricostruisci X, Y, Z della superficie
        grids = [None, None, None]

        plane_value = np.median(xyz[normal_axis])
        plane_value += plane_shift

        grids[normal_axis] = np.full_like(U, plane_value)
        grids[uv_axes[0]] = U
        grids[uv_axes[1]] = V

        X, Y, Z = grids

        surf = ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
            zorder=0
        )

        return surf

    def plot_surface_mesh_traj(self, elev=25, azim=-40):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.tri as mtri
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
            alpha=0.48,
            max_edge_factor=5.0,
            edge_alpha=0.0
        ):
            px = np.asarray(px).flatten()
            py = np.asarray(py).flatten()
            pz = np.asarray(pz).flatten()
            costs = np.asarray(costs).flatten()

            xyz = np.column_stack((px, py, pz))

            valid = (
                np.isfinite(px)
                & np.isfinite(py)
                & np.isfinite(pz)
                & np.isfinite(costs)
            )

            xyz = xyz[valid]
            costs = costs[valid]

            if xyz.shape[0] < 3:
                print("[WARNING] Not enough valid terrain points to build a 3D mesh.")
                return None

            ranges = np.ptp(xyz, axis=0)
            uv_axes = np.argsort(ranges)[-2:]

            u = xyz[:, uv_axes[0]]
            v = xyz[:, uv_axes[1]]

            triang = mtri.Triangulation(u, v)
            triangles = triang.triangles

            if triangles.size == 0:
                print("[WARNING] Triangulation failed: no triangles generated.")
                return None

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

        def plot_line_with_halo(ax, x, y, z, label=None):
            ax.plot(x, y, z, color='white', linewidth=8.5, alpha=0.95, zorder=90)
            ax.plot(x, y, z, color='blue', linewidth=3.4, alpha=1.0, zorder=100, label=label)

        def scatter_marker_with_halo(ax, point, marker, color, size, label=None, zorder=200):
            point = np.asarray(point).flatten()
            if point.size < 3:
                return

            x, y, z = point[0], point[1], point[2]

            ax.scatter(
                [x], [y], [z],
                c='white', s=size * 2.25, marker=marker,
                edgecolors='black', linewidth=0.8, depthshade=False, zorder=zorder - 1
            )

            ax.scatter(
                [x], [y], [z],
                c=color, s=size, marker=marker,
                edgecolors='black', linewidth=2.0, depthshade=False, label=label, zorder=zorder
            )

        # ============================================================
        # FIGURA E ASSI 3D
        # ============================================================

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sfondo completamente bianco
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # NASCONDE COMPLETAMENTE GLI ASSI (Mantiene solo il contenuto)
        ax.axis('off')

        ax.computed_zorder = False
        ax.set_proj_type('ortho')

        # ============================================================
        # 1. ESTRAZIONE DATI TERRENO
        # ============================================================

        px = np.array([p['position'][0] for p in self.points_t_data])
        py = np.array([p['position'][1] for p in self.points_t_data])
        pz = np.array([p['position'][2] for p in self.points_t_data])

        costs = np.array([p['cost'] for p in self.points_t_data])

        # ============================================================
        # 2. MAPPA COME VERA MESH 3D
        # ============================================================

        plot_costmap_true_3d_mesh(
            ax, px, py, pz, costs,
            cmap='RdYlGn_r', alpha=1., max_edge_factor=5.0, edge_alpha=0.0
        )

        # ============================================================
        # 3. TRAIETTORIA E CONTACT POINTS
        # ============================================================

        all_x = [px]
        all_y = [py]
        all_z = [pz]

        landing_points = []
        blue_label_added = False

        if self.best_traj_ever is not None and len(self.best_traj_ever) > 0:
            for i, segment in enumerate(self.best_traj_ever):
                segment_np = np.asarray(segment)

                if (segment_np.ndim == 2 and segment_np.shape[0] == 3 and segment_np.shape[1] != 3):
                    x_s = segment_np[0, :]
                    y_s = segment_np[1, :]
                    z_s = segment_np[2, :]
                else:
                    x_s = segment_np[:, 0]
                    y_s = segment_np[:, 1]
                    z_s = segment_np[:, 2]

                label = "Trajectory" if not blue_label_added else None
                blue_label_added = True

                plot_line_with_halo(ax, x_s, y_s, z_s, label=label)

                all_x.append(x_s)
                all_y.append(y_s)
                all_z.append(z_s)

                landing_points.append([x_s[0], y_s[0], z_s[0]])

                if i == len(self.best_traj_ever) - 1:
                    landing_points.append([x_s[-1], y_s[-1], z_s[-1]])

            if len(landing_points) > 0:
                lp = np.asarray(landing_points)

                ax.scatter(
                    lp[:, 0], lp[:, 1], lp[:, 2],
                    c='white', s=210, marker='o',
                    edgecolors='black', linewidth=0.8, depthshade=False, zorder=180
                )

                ax.scatter(
                    lp[:, 0], lp[:, 1], lp[:, 2],
                    c='yellow', s=95, marker='o',
                    edgecolors='black', linewidth=1.8, depthshade=False, zorder=190, label='Contact Points'
                )

        # ============================================================
        # 4. START, DESIRED GOAL, REACHED GOAL
        # ============================================================

        scatter_marker_with_halo(
            ax, self.p0, marker='^', color='lime', size=250, label='START', zorder=230
        )

        scatter_marker_with_halo(
            ax, self.pf, marker='X', color='red', size=260, label='DESIRED GOAL', zorder=240
        )

        if self.best_achieved_target_ever is not None:
            achieved = np.asarray(self.best_achieved_target_ever).flatten()

            if achieved.size >= 3:
                scatter_marker_with_halo(
                    ax, achieved, marker='*', color='orange', size=360, label='REACHED GOAL', zorder=250
                )

                ax.plot(
                    [self.pf[0], achieved[0]], [self.pf[1], achieved[1]], [self.pf[2], achieved[2]],
                    color='white', linestyle='--', linewidth=6.5, alpha=0.9, zorder=210
                )

                ax.plot(
                    [self.pf[0], achieved[0]], [self.pf[1], achieved[1]], [self.pf[2], achieved[2]],
                    color='red', linestyle='--', linewidth=2.7, alpha=1.0, zorder=220, label='Goal Discrepancy'
                )

                distance = np.linalg.norm(np.asarray(self.pf).flatten()[:3] - achieved[:3])
                print(f"[INFO] Distance desired goal → achieved target: {distance:.4f} m")

        # ============================================================
        # 5. SCALATURA 1:1:1
        # ============================================================
        # Nonostante gli assi non siano visibili, i limiti determinano il field of view.

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
        mid_y = (y_max_data + y_min_data) * 0.5
        mid_z = (z_max + z_min) * 0.5

        y_min = 0
        y_max = 6

        # Manteniamo i limiti come nel codice originale per fissare l'inquadratura
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Mantiene il rapporto di forma del bounding box
        ax.set_box_aspect((1, 1, 1))

        # ============================================================
        # 6. VISTA E RENDER
        # ============================================================
        # Tutto il codice relativo alle etichette degli assi è stato rimosso
        # perché ax.axis('off') le ha già disabilitate alla radice.

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        plt.show()

    # ════════════════════════════════════════════════════════════════════════
    #  VERSIONI DINAMICHE (GIF)
    # ════════════════════════════════════════════════════════════════════════
    def _anim_scene(self, elev, azim, figsize=(12, 9)):
        """Crea figura/assi 3-D + mesh del terreno, come nel plot statico."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.axis('off')
        ax.computed_zorder = False
        ax.set_proj_type('ortho')

        px = np.array([p['position'][0] for p in self.points_t_data])
        py = np.array([p['position'][1] for p in self.points_t_data])
        pz = np.array([p['position'][2] for p in self.points_t_data])
        costs = np.array([p['cost'] for p in self.points_t_data])

        dyn.costmap_true_3d_mesh(ax, px, py, pz, costs, cmap='RdYlGn_r',
                                 alpha=1.0, max_edge_factor=5.0, edge_alpha=0.0)

        ax.view_init(elev=elev, azim=azim)
        return fig, ax, (px, py, pz)

    @staticmethod
    def _anim_limits(ax, all_x, all_y, all_z, ylim=(0.0, 6.0)):
        """Stessa inquadratura del plot statico (box 1:1:1)."""
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

    def _best_elite_of_iter(self, idx):
        """Elite migliore dell'iterazione idx (robusta al segno della fitness)."""
        elites = [e for e in self.all_elites[idx] if getattr(e, "traj", None)]
        if not elites:
            return None
        if idx < len(self.best_fit_each_iter) and self.best_fit_each_iter[idx] is not None:
            b = float(self.best_fit_each_iter[idx])
            return min(elites, key=lambda e: min(abs(e.fitness - b), abs(e.fitness + b)))
        return elites[0]

    def animate_surface_mesh_traj(self,
                                  elev=20,
                                  azim=-20,
                                  n_frames=200,
                                  fps=25,
                                  hold_frames=35,
                                  ylim=(0.0, 6.0),
                                  out_dir=None,
                                  gif_name="single_surface_mesh_traj_animated",
                                  dpi=100,
                                  show_head=True,
                                  show=True):
        """
        Versione dinamica di plot_surface_mesh_traj(): la traiettoria migliore
        viene disegnata man mano, salto dopo salto, e quello che è già stato
        tracciato resta sempre a schermo fino al punto finale. Salva una GIF.
        """
        if dyn is None:
            raise RuntimeError("Serve dynamic_plots.py nella stessa cartella.")
        if not self.best_traj_ever:
            print("[WARNING] Nessuna traiettoria da animare.")
            return None

        out_dir = out_dir or str(FOLDER_MAIN)
        fig, ax, (px, py, pz) = self._anim_scene(elev, azim)

        segs, total_pts = dyn.collect_path(self.best_traj_ever)
        lands = dyn.landing_points_of(self.best_traj_ever)

        all_x = [px] + [s["x"] for s in segs]
        all_y = [py] + [s["y"] for s in segs]
        all_z = [pz] + [s["z"] for s in segs]
        self._anim_limits(ax, all_x, all_y, all_z, ylim)

        halos, lines = [], []
        for _ in segs:
            h, = ax.plot([], [], [], color='white', lw=8.5, alpha=0.95, zorder=90)
            m, = ax.plot([], [], [], color='blue', lw=3.4, alpha=1.0, zorder=100)
            halos.append(h); lines.append(m)

        def _pair(color, marker, zorder, ms_halo, ms_main):
            halo, = ax.plot([], [], [], linestyle='none', marker=marker, color='white',
                            markersize=ms_halo, markeredgecolor='black',
                            markeredgewidth=0.8, zorder=zorder - 1)
            main, = ax.plot([], [], [], linestyle='none', marker=marker, color=color,
                            markersize=ms_main, markeredgecolor='black',
                            markeredgewidth=1.8, zorder=zorder)
            return [halo, main]

        contact = _pair('yellow', 'o', 190, 15, 10)
        p0_a = _pair('lime', '^', 230, 18, 13)
        pf_a = _pair('red', 'X', 240, 18, 13)
        ach_a = _pair('orange', '*', 250, 23, 17)

        dw, = ax.plot([], [], [], color='white', ls='--', lw=6.5, alpha=0.9, zorder=210)
        dc, = ax.plot([], [], [], color='red', ls='--', lw=2.7, alpha=1.0, zorder=220)

        head, = ax.plot([], [], [], linestyle='none', marker='o', markersize=11,
                        color='white', markeredgecolor='black',
                        markeredgewidth=1.6, zorder=300)

        achieved = None
        if self.best_achieved_target_ever is not None:
            a = np.asarray(self.best_achieved_target_ever).flatten()
            if a.size >= 3:
                achieved = a[:3]

        p0 = np.asarray(self.p0).flatten()
        pf = np.asarray(self.pf).flatten()

        def _update(k):
            done = 0
            for s, h, m in zip(segs, halos, lines):
                n = dyn.visible_count(s, k)
                dyn.set_line3d(h, s["x"][:n], s["y"][:n], s["z"][:n])
                dyn.set_line3d(m, s["x"][:n], s["y"][:n], s["z"][:n])
                if n == s["n"]:
                    done += 1
                if n > 0:
                    last = (s["x"][n - 1], s["y"][n - 1], s["z"][n - 1])

            n_lp = min(done + 1, len(lands))
            for a in contact:
                if n_lp > 0:
                    dyn.set_line3d(a, lands[:n_lp, 0], lands[:n_lp, 1], lands[:n_lp, 2])
                else:
                    dyn.set_line3d(a, [], [], [])

            for a in p0_a:
                dyn.set_line3d(a, [p0[0]], [p0[1]], [p0[2]])

            finished = done == len(segs)
            for a in pf_a:
                if finished:
                    dyn.set_line3d(a, [pf[0]], [pf[1]], [pf[2]])
                else:
                    dyn.set_line3d(a, [], [], [])

            if finished and achieved is not None:
                for a in ach_a:
                    dyn.set_line3d(a, [achieved[0]], [achieved[1]], [achieved[2]])
                for a in (dw, dc):
                    dyn.set_line3d(a, [pf[0], achieved[0]], [pf[1], achieved[1]],
                                   [pf[2], achieved[2]])
            else:
                for a in ach_a + [dw, dc]:
                    dyn.set_line3d(a, [], [], [])

            if show_head and k < total_pts:
                dyn.set_line3d(head, [last[0]], [last[1]], [last[2]])
            else:
                dyn.set_line3d(head, [], [], [])
            return []

        plt.tight_layout()
        frames = dyn.frame_indices(total_pts, n_frames, hold_frames)
        anim = FuncAnimation(fig, _update, frames=frames,
                             interval=1000.0 / fps, blit=False, repeat=True)
        self._anim_traj = anim

        dyn.save_animation(anim, out_dir, gif_name, fps=fps, dpi=dpi)
        _update(total_pts)
        dyn.set_line3d(head, [], [], [])
        dyn.save_still(fig, out_dir, gif_name)
        if show:
            plt.show()
        return anim

    def animate_best_traj_evolution(self,
                                    elev=20,
                                    azim=-20,
                                    n_frames=320,
                                    fps=25,
                                    hold_frames=45,
                                    faded_alpha=0.18,
                                    cmap_name="plasma",
                                    ylim=(0.0, 6.0),
                                    out_dir=None,
                                    gif_name="single_best_traj_evolution",
                                    dpi=100,
                                    show=True):
        """
        Mostra come la soluzione migliora lungo le iterazioni CEM: per ogni
        iterazione viene disegnata progressivamente la traiettoria dell'elite
        migliore; quando è completa resta a schermo ma diventa opaca, e parte
        quella dell'iterazione successiva (in primo piano e di colore diverso).
        """
        if dyn is None:
            raise RuntimeError("Serve dynamic_plots.py nella stessa cartella.")
        if not self.all_elites:
            print("[WARNING] Nessuna iterazione da animare.")
            return None

        out_dir = out_dir or str(FOLDER_MAIN)
        fig, ax, (px, py, pz) = self._anim_scene(elev, azim)

        cmap = plt.get_cmap(cmap_name)
        n_it = len(self.all_elites)

        iters, segs_all, offset = [], [], 0
        for i in range(n_it):
            e = self._best_elite_of_iter(i)
            if e is None:
                continue
            i_segs, offset = dyn.collect_path(e.traj, tag=i, offset=offset)
            if not i_segs:
                continue
            color = cmap(0.12 + 0.8 * (i / max(1, n_it - 1)))
            iters.append(dict(idx=i, elite=e, segs=i_segs, color=color,
                              lands=dyn.landing_points_of(e.traj)))
            segs_all.extend(i_segs)
        total_pts = offset

        if total_pts == 0:
            print("[WARNING] Le elite non contengono traiettorie.")
            plt.close(fig)
            return None

        all_x = [px] + [s["x"] for s in segs_all]
        all_y = [py] + [s["y"] for s in segs_all]
        all_z = [pz] + [s["z"] for s in segs_all]
        self._anim_limits(ax, all_x, all_y, all_z, ylim)

        for it in iters:
            it["halos"], it["lines"] = [], []
            for _ in it["segs"]:
                h, = ax.plot([], [], [], color='white', lw=8.5, alpha=0.95, zorder=90)
                m, = ax.plot([], [], [], color=it["color"], lw=3.4, alpha=1.0, zorder=100)
                it["halos"].append(h); it["lines"].append(m)
            it["pts"], = ax.plot([], [], [], linestyle='none', marker='o',
                                 markersize=10, color='yellow',
                                 markeredgecolor='black', markeredgewidth=1.5,
                                 zorder=190)

        p0 = np.asarray(self.p0).flatten()
        pf = np.asarray(self.pf).flatten()
        ax.plot([p0[0]], [p0[1]], [p0[2]], linestyle='none', marker='^',
                markersize=13, color='lime', markeredgecolor='black',
                markeredgewidth=1.8, zorder=230)
        ax.plot([pf[0]], [pf[1]], [pf[2]], linestyle='none', marker='X',
                markersize=13, color='red', markeredgecolor='black',
                markeredgewidth=1.8, zorder=240)

        head, = ax.plot([], [], [], linestyle='none', marker='o', markersize=11,
                        color='white', markeredgecolor='black',
                        markeredgewidth=1.6, zorder=300)

        txt = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, fontsize=15,
                        fontweight='bold', va='top', zorder=400,
                        bbox=dict(boxstyle="round,pad=0.35", fc='white',
                                  ec='0.7', alpha=0.85))

        def _update(k):
            head_xyz, active = None, None
            for it in iters:
                started, done = False, 0
                for s, h, m in zip(it["segs"], it["halos"], it["lines"]):
                    n = dyn.visible_count(s, k)
                    dyn.set_line3d(h, s["x"][:n], s["y"][:n], s["z"][:n])
                    dyn.set_line3d(m, s["x"][:n], s["y"][:n], s["z"][:n])
                    if n > 0:
                        started = True
                        head_xyz = (s["x"][n - 1], s["y"][n - 1], s["z"][n - 1])
                    if n == s["n"]:
                        done += 1
                finished = started and done == len(it["segs"])
                if not finished and started:
                    active = it

                vis = started
                is_act = started and not finished
                for h, m in zip(it["halos"], it["lines"]):
                    h.set_visible(vis); m.set_visible(vis)
                    h.set_linewidth(8.5 if is_act else 4.0)
                    h.set_alpha(0.95 if is_act else 0.12)
                    m.set_linewidth(3.4 if is_act else 1.6)
                    m.set_alpha(1.0 if is_act else faded_alpha)

                lp = it["lands"]
                n_lp = min(done + 1, len(lp)) if started else 0
                it["pts"].set_visible(vis)
                it["pts"].set_markersize(10 if is_act else 6)
                it["pts"].set_alpha(1.0 if is_act else faded_alpha)
                if n_lp > 0:
                    dyn.set_line3d(it["pts"], lp[:n_lp, 0], lp[:n_lp, 1], lp[:n_lp, 2])
                else:
                    dyn.set_line3d(it["pts"], [], [], [])

            if active is None and iters:      # fine: l'ultima resta in evidenza
                last = iters[-1]
                for h, m in zip(last["halos"], last["lines"]):
                    h.set_linewidth(8.5); h.set_alpha(0.95)
                    m.set_linewidth(3.4); m.set_alpha(1.0)
                last["pts"].set_markersize(10); last["pts"].set_alpha(1.0)
                active = last

            if head_xyz is not None and k < total_pts:
                dyn.set_line3d(head, [head_xyz[0]], [head_xyz[1]], [head_xyz[2]])
            else:
                dyn.set_line3d(head, [], [], [])

            if active is not None:
                fit = getattr(active["elite"], "fitness", None)
                fit_str = f"   fit = {fit:.4f}" if fit is not None else ""
                txt.set_text(f"Iteration {active['idx'] + 1}/{n_it}{fit_str}")
                txt.set_color(active["color"])
            return []

        plt.tight_layout()
        frames = dyn.frame_indices(total_pts, n_frames, hold_frames)
        anim = FuncAnimation(fig, _update, frames=frames,
                             interval=1000.0 / fps, blit=False, repeat=True)
        self._anim_evo = anim

        dyn.save_animation(anim, out_dir, gif_name, fps=fps, dpi=dpi)
        _update(total_pts)
        dyn.set_line3d(head, [], [], [])
        dyn.save_still(fig, out_dir, gif_name)
        if show:
            plt.show()
        return anim

    def plot_fitness_by_iteration(self, ax=None):
        if not self.all_elites:
            print("WARNING: No data available for plotting fitness.")
            return

        created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))
            created = True
        else:
            fig = ax.get_figure()

        x_normal, y_normal = [], []
        x_best, y_best = [], []

        tolerance = 1e-8
        fitness_threshold = -1e4

        all_fitness_values_for_norm = []

        for i, iteration_list in enumerate(self.all_elites):
            current_iter = i + 1
            for elite in iteration_list:
                if elite.fitness < fitness_threshold:
                    continue

                all_fitness_values_for_norm.append(elite.fitness)

                if np.isclose(elite.fitness, self.best_fit_ever, atol=tolerance):
                    x_best.append(current_iter)
                    y_best.append(elite.fitness)
                else:
                    x_normal.append(current_iter)
                    y_normal.append(elite.fitness)

        if not all_fitness_values_for_norm:
            print("WARNING: No fitness values found below threshold.")
            return

        vmin = min(all_fitness_values_for_norm)
        vmax = max(all_fitness_values_for_norm)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors_cmap = ['green', 'orange']
        cmap_name = 'green_to_orange_gradient'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_cmap, N=256)
        ax.scatter(x_normal, y_normal, c=y_normal, cmap=custom_cmap, norm=norm,
                   s=35, edgecolors='black', linewidths=0.3, alpha=0.8, zorder=3,
                   label='Elite Solutions (Gradient)')
        if x_best:
            ax.scatter(x_best, y_best, c='red',
                       s=50, edgecolors='black', linewidths=0.5, alpha=1.0, zorder=4,
                       label='Best Ever Reached')
        ax.set_title('Elite Fitness Distribution', fontsize=20, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=20)
        ax.set_ylabel('Fitness Value', fontsize=20)
        ax.axhline(y=self.best_fit_ever, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=2)

        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
        max_iter = len(self.all_elites)
        if max_iter <= 25:
            ax.set_xticks(range(1, max_iter + 1))
        else:
            ax.xaxis.get_major_locator().set_params(integer=True)

        fig.tight_layout()

        if created:
            save_path = f'{FOLDER_MAIN}/plot_fitness_colormap_redbest.png'
            fig.savefig(save_path)
            save_path_pdf = f'{FOLDER_MAIN}/plot_fitness_colormap_redbest.pdf'
            fig.savefig(save_path_pdf, bbox_inches='tight', pad_inches=0.23)
            print(f"Fitness plot saved to: {save_path}")
            plt.show()


def main():
    plotter = PlotResultCemMjumps()
    print("[INFO] Plotter initialized successfully")
    plotter.plot_surface_mesh_traj(elev=20, azim=-20)
    # plotter.plot_fitness_by_iteration()

    # ── versioni dinamiche (salvano una GIF in FOLDER_MAIN) ──────────────────
    plotter.animate_surface_mesh_traj(elev=20, azim=-20)
    plotter.animate_best_traj_evolution(elev=20, azim=-20)
    print("[INFO] All plots completed!")


if __name__ == "__main__":
    main()