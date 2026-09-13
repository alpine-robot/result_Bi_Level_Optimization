"""
dynamic_plots.py
================
Utility condivise per i plot "dinamici" (animati) usati da:

    combine_plot.py      → CombinePlot.animate_mesh_pc_traj()
    single_plot.py       → PlotResultCemMjumps.animate_surface_mesh_traj()
                           PlotResultCemMjumps.animate_best_traj_evolution()
    fusion_result*.py    → animate_median_best_overall()
                           animate_jump_histogram_fusion()

Tutte le animazioni vengono salvate come GIF (richiede Pillow: pip install pillow).
Il file deve stare nella STESSA cartella degli script che lo importano.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ═════════════════════════════════════════════════════════════════════════════
#  GIF / frame
# ═════════════════════════════════════════════════════════════════════════════
def save_animation(anim, out_dir, name, fps=25, dpi=100):
    """Salva l'animazione come GIF in out_dir/name.gif."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.gif")
    print(f"[GIF ] Rendering → {path}   (può richiedere un po')")
    try:
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        print(f"[SAVE] {path}")
    except Exception as e:
        print(f"[WARN] GIF non salvata ({e}).  Serve Pillow:  pip install pillow")
    return path


def save_still(fig, out_dir, name):
    """Salva il fotogramma finale (PNG + PDF)."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in (".png", ".pdf"):
        p = os.path.join(out_dir, name + ext)
        fig.savefig(p, bbox_inches="tight", dpi=150)
    print(f"[SAVE] {os.path.join(out_dir, name)}.png/.pdf")


def frame_indices(total, n_frames=None, hold=25):
    """Lista di indici 1..total campionati su n_frames, con pausa finale."""
    total = int(total)
    if total <= 0:
        return [0]
    if not n_frames or n_frames >= total:
        frames = list(range(1, total + 1))
    else:
        frames = sorted(set(np.linspace(1, total, int(n_frames)).astype(int).tolist()))
    return frames + [total] * int(max(0, hold))


# ═════════════════════════════════════════════════════════════════════════════
#  Traiettorie
# ═════════════════════════════════════════════════════════════════════════════
def unpack_segment(segment):
    """Gestisce segmenti in formato (3, N) oppure (N, 3) → (xs, ys, zs)."""
    s = np.asarray(segment, dtype=float)
    if s.ndim != 2:
        raise ValueError("Il segmento di traiettoria non è un array 2-D.")
    if s.shape[0] == 3 and s.shape[1] != 3:
        return s[0, :], s[1, :], s[2, :]
    return s[:, 0], s[:, 1], s[:, 2]


def collect_path(traj, tag=None, offset=0):
    """
    Mette in fila i segmenti di UNA traiettoria assegnando a ciascuno un
    offset cumulativo, così l'animazione può avanzare su un unico contatore.

    Ritorna (segs, next_offset) dove ogni seg è un dict con
    tag, seg, n, offset, x, y, z.
    """
    segs = []
    for i, seg in enumerate(traj or []):
        try:
            x, y, z = unpack_segment(seg)
        except Exception:
            continue
        if x.size == 0:
            continue
        segs.append(dict(tag=tag, seg=i, n=int(x.size), offset=int(offset),
                         x=x, y=y, z=z))
        offset += int(x.size)
    return segs, offset


def visible_count(seg, k):
    """Quanti campioni del segmento sono già stati disegnati al frame k."""
    return int(np.clip(k - seg["offset"], 0, seg["n"]))


def set_line3d(line, x, y, z):
    """set_data 3-D compatibile con tutte le versioni di matplotlib."""
    line.set_data(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    line.set_3d_properties(np.asarray(z, dtype=float))


def landing_points_of(traj):
    """Punti di contatto: inizio di ogni segmento + fine dell'ultimo."""
    pts = []
    segs, _ = collect_path(traj)
    for i, s in enumerate(segs):
        pts.append([s["x"][0], s["y"][0], s["z"][0]])
        if i == len(segs) - 1:
            pts.append([s["x"][-1], s["y"][-1], s["z"][-1]])
    return np.asarray(pts) if pts else np.empty((0, 3))


# ═════════════════════════════════════════════════════════════════════════════
#  Cost map 3-D (stessa resa di combine_plot / single_plot)
# ═════════════════════════════════════════════════════════════════════════════
def costmap_true_3d_mesh(ax, px, py, pz, costs, cmap="RdYlGn_r", alpha=1.0,
                         max_edge_factor=5.0, edge_alpha=0.0):
    """Disegna la cost map come vera mesh 3-D triangolata."""
    px = np.asarray(px).flatten()
    py = np.asarray(py).flatten()
    pz = np.asarray(pz).flatten()
    costs = np.asarray(costs).flatten()

    valid = (np.isfinite(px) & np.isfinite(py)
             & np.isfinite(pz) & np.isfinite(costs))
    xyz = np.column_stack((px, py, pz))[valid]
    costs = costs[valid]

    if xyz.shape[0] < 3:
        print("[WARNING] Not enough valid terrain points to build a 3D mesh.")
        return None

    ranges = np.ptp(xyz, axis=0)
    uv_axes = np.argsort(ranges)[-2:]
    u, v = xyz[:, uv_axes[0]], xyz[:, uv_axes[1]]

    triangles = mtri.Triangulation(u, v).triangles
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
        triangles = triangles[np.all(edge_lengths < max_edge_factor * median_edge, axis=1)]
    if triangles.size == 0:
        print("[WARNING] All triangles removed by edge filtering.")
        return None

    verts = xyz[triangles]
    tri_costs = costs[triangles].mean(axis=1)

    vmin, vmax = np.nanpercentile(costs, [2, 98])
    if np.isclose(vmin, vmax):
        vmin, vmax = np.nanmin(costs), np.nanmax(costs)
    if np.isclose(vmin, vmax):
        vmin = vmax - 1.0

    facecolors = plt.get_cmap(cmap)(mcolors.Normalize(vmin=vmin, vmax=vmax)(tri_costs))
    facecolors[:, 3] = alpha

    mesh = Poly3DCollection(verts, facecolors=facecolors,
                            edgecolors=(0, 0, 0, edge_alpha),
                            linewidths=0.0, zorder=0)
    ax.add_collection3d(mesh)
    return mesh


# ═════════════════════════════════════════════════════════════════════════════
#  Curve che crescono (usato dai fusion_result*)
# ═════════════════════════════════════════════════════════════════════════════
def animate_growing_curves(series,
                           out_dir,
                           name,
                           xlabel="Iteration",
                           ylabel="Fitness",
                           figsize=(12, 6),
                           n_frames=200,
                           fps=25,
                           hold_frames=35,
                           label_fs=30,
                           tick_fs=25,
                           legend_fs=25,
                           legend_loc="upper right",
                           sequential=False,
                           faded_alpha=0.25,
                           smooth=20,
                           marker_head=True,
                           dpi=100,
                           save_png=True,
                           show=True):
    """
    Anima una o più curve che si "disegnano" da sinistra a destra, con la
    banda IQR che cresce insieme alla linea.

    series : lista di dict con chiavi
        name        etichetta in legenda
        x, y        array della curva (stessa lunghezza)
        lo, hi      (opzionali) estremi della banda
        color       colore
        ls          stile linea ('-', '--', …)
        lw          spessore (default 2.5)
        fill_alpha  alpha della banda (default 0.20)

    sequential : se True le curve vengono disegnate una dopo l'altra e quelle
                 già completate diventano opache (alpha = faded_alpha).
                 Se False crescono tutte insieme.
    """
    series = [s for s in series if s.get("x") is not None and len(s["x"]) > 1]
    if not series:
        print("[WARN] Nessuna serie da animare.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # ── infittimento per un tratto fluido ────────────────────────────────────
    dense = []
    for s in series:
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        n_out = max(2, (len(x) - 1) * int(max(1, smooth)) + 1)
        xd = np.linspace(x[0], x[-1], n_out)
        d = dict(s)
        d["xd"] = xd
        d["yd"] = np.interp(xd, x, y)
        for key in ("lo", "hi"):
            arr = s.get(key)
            d[key + "d"] = (np.interp(xd, x, np.asarray(arr, dtype=float))
                            if arr is not None else None)
        dense.append(d)

    # ── limiti fissi ─────────────────────────────────────────────────────────
    xs = np.concatenate([d["xd"] for d in dense])
    ys = [d["yd"] for d in dense]
    for d in dense:
        for key in ("lod", "hid"):
            if d[key] is not None:
                ys.append(d[key])
    ys = np.concatenate(ys)
    y_lo, y_hi = np.nanmin(ys), np.nanmax(ys)
    pad = 0.06 * max(y_hi - y_lo, 1e-9)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(y_lo - pad, y_hi + pad)

    lines, heads = [], []
    bands = [None] * len(dense)
    for d in dense:
        ln, = ax.plot([], [], color=d.get("color"), ls=d.get("ls", "-"),
                      lw=d.get("lw", 2.5), label=d.get("name"))
        lines.append(ln)
        hd, = ax.plot([], [], linestyle="none", marker="o", markersize=9,
                      color=d.get("color"), markeredgecolor="white",
                      markeredgewidth=1.5, zorder=10)
        hd.set_visible(marker_head)
        heads.append(hd)

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(loc=legend_loc, fontsize=legend_fs, frameon=True)
    ax.grid(ls=":", alpha=0.5)
    fig.tight_layout()

    n_ser = len(dense)
    total = int(n_frames)

    def _update(k):
        p = k / float(total)
        for i, (d, ln, hd) in enumerate(zip(dense, lines, heads)):
            if sequential:
                lo_p, hi_p = i / n_ser, (i + 1) / n_ser
                local = 0.0 if p <= lo_p else min(1.0, (p - lo_p) / (hi_p - lo_p))
                alpha = faded_alpha if p > hi_p else 1.0
            else:
                local, alpha = p, 1.0

            m = int(round(local * (len(d["xd"]) - 1))) + 1 if local > 0 else 0
            ln.set_data(d["xd"][:m], d["yd"][:m])
            ln.set_alpha(alpha)

            if marker_head and 0 < m < len(d["xd"]):
                hd.set_data([d["xd"][m - 1]], [d["yd"][m - 1]])
                hd.set_alpha(alpha)
            else:
                hd.set_data([], [])

            if bands[i] is not None:
                bands[i].remove()
                bands[i] = None
            if m > 1 and d["lod"] is not None and d["hid"] is not None:
                bands[i] = ax.fill_between(
                    d["xd"][:m], d["lod"][:m], d["hid"][:m],
                    color=d.get("color"),
                    alpha=d.get("fill_alpha", 0.20) * (alpha if sequential else 1.0),
                )
        return lines + heads

    frames = list(range(0, total + 1)) + [total] * int(max(0, hold_frames))
    anim = FuncAnimation(fig, _update, frames=frames,
                         interval=1000.0 / fps, blit=False, repeat=True)

    save_animation(anim, out_dir, name, fps=fps, dpi=dpi)
    if save_png:
        _update(total)
        for hd in heads:
            hd.set_data([], [])
        save_still(fig, out_dir, name)
    if show:
        plt.show()
    return anim


# ═════════════════════════════════════════════════════════════════════════════
#  Barre raggruppate che crescono (istogramma dei salti)
# ═════════════════════════════════════════════════════════════════════════════
def animate_grouped_bars(categories,
                         groups,
                         out_dir,
                         name,
                         xlabel="Number of Jumps",
                         ylabel="Frequency",
                         figsize=(10, 6),
                         n_frames=160,
                         fps=25,
                         hold_frames=35,
                         label_fs=(20, 30),
                         tick_fs=25,
                         legend_fs=25,
                         legend_loc="upper right",
                         sequential=True,
                         dpi=100,
                         save_png=True,
                         show=True):
    """
    Anima un bar chart raggruppato: le barre salgono da zero.
    sequential=True → un gruppo per volta; False → tutti insieme.

    categories : valori dell'asse x (es. numero di salti)
    groups     : lista di dict(name, values, color, hatch=None)
    """
    groups = [g for g in groups if g.get("values") is not None and len(g["values"])]
    if not groups:
        print("[WARN] Nessun gruppo da animare.")
        return None

    categories = np.asarray(categories, dtype=float)
    n_g = len(groups)
    bw = 0.8 / n_g

    fig, ax = plt.subplots(figsize=figsize)
    bar_sets = []
    for k, g in enumerate(groups):
        xs = categories + (k - n_g / 2 + 0.5) * bw
        bars = ax.bar(xs, np.zeros(len(categories)), width=bw,
                      color=g.get("color"), edgecolor="black", alpha=0.8,
                      hatch=g.get("hatch"), label=g.get("name"))
        bar_sets.append(bars)

    y_max = max(float(np.nanmax(g["values"])) for g in groups)
    ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1.0)
    ax.set_xticks(categories)

    xl_fs, yl_fs = (label_fs if isinstance(label_fs, (tuple, list))
                    else (label_fs, label_fs))
    ax.set_xlabel(xlabel, fontsize=xl_fs)
    ax.set_ylabel(ylabel, fontsize=yl_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(loc=legend_loc, fontsize=legend_fs)
    ax.grid(axis="y", ls="--", alpha=0.5)
    fig.tight_layout()

    total = int(n_frames)

    def _update(k):
        p = k / float(total)
        arts = []
        for i, (g, bars) in enumerate(zip(groups, bar_sets)):
            if sequential:
                lo_p, hi_p = i / n_g, (i + 1) / n_g
                local = 0.0 if p <= lo_p else min(1.0, (p - lo_p) / (hi_p - lo_p))
            else:
                local = p
            for bar, v in zip(bars, g["values"]):
                bar.set_height(float(v) * local)
                arts.append(bar)
        return arts

    frames = list(range(0, total + 1)) + [total] * int(max(0, hold_frames))
    anim = FuncAnimation(fig, _update, frames=frames,
                         interval=1000.0 / fps, blit=False, repeat=True)

    save_animation(anim, out_dir, name, fps=fps, dpi=dpi)
    if save_png:
        _update(total)
        save_still(fig, out_dir, name)
    if show:
        plt.show()
    return anim