import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ==========================================
# CONFIGURAZIONE GENERALE
# ==========================================
NAME_FOLDER = "test_patches"

# Cartelle base dei risultati "a patch" (W/H multipli), per terreno
TERRAINS = {
    "rock": "result_rock_patch",
    "gaussian": "result_gaussian_patch",
    "hemisphere": "result_hemisphere_patch"
}

PATCH_CONFIGS = [
    {"W": 10, "H": 5},
    {"W": 10, "H": 20},
    {"W": 20, "H": 20},
]

# Dati "centroide a 05": cartella result_05, con sottocartelle dirette
# tipo test_05_<run>_<terrain_suffix> (stessa logica di fusion_result_totals.py)
CENTER_PATCH_05_DIR = "result_05"
CENTER_PATCH_05_GLOBS = {
    "rock": "test_05_*_rock",
    "gaussian": "test_05_*_gaussian_bumps",
    "hemisphere": "test_05_*_hemisphere",
}

# Dati "legacy" (centroide storico): cartelle up_<terrain>, con sottocartelle
# run_1, run_2, ... (pattern "run_*")
CENTER_PATCH_LEGACY_DIRS = {
    "rock": "up_rocky",
    "gaussian": "up_gaussian",
    "hemisphere": "up_hemisphere",
}

COMBINED_OUTPUT = "result/combined_terrain_patch_plots"


def _save_plot(fig, filename):
    """Helper per salvare i plot in PNG e PDF."""
    os.makedirs(COMBINED_OUTPUT, exist_ok=True)
    for ext in ['.png', '.pdf']:
        fig.savefig(f"{COMBINED_OUTPUT}/{filename}{ext}", bbox_inches="tight", dpi=150)
    print(f"[SAVE] Salvato: {COMBINED_OUTPUT}/{filename}")


class SingleResult:
    """Carica e parsa i dati per una singola cartella di risultato."""
    def __init__(self, folder: str):
        self.folder = folder
        self.all_elites = []

        iter_files = sorted(
            glob.glob(f"{folder}/iteration_reports/iteration_*.json"),
            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
        )
        for fname in iter_files:
            with open(fname) as f:
                it = json.load(f)
            self.all_elites.append([{'fitness': e['fitness'], 'n_jumps': e['n_jumps']} for e in it['elites']])


def _discover_runs(group_path: str) -> list[str]:
    """Ritorna le sottocartelle run_* dentro group_path (per i dati legacy up_<terrain>)."""
    pattern = os.path.join(group_path, "run_*")
    runs = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r'\d+', os.path.basename(p)).group())
    )
    return [r for r in runs if os.path.isdir(r)]


def _raw_series(results: list) -> list:
    """
    Da una lista di SingleResult calcola, per ogni run, la serie del best-so-far
    (senza padding). Ritorna una lista di liste (una per run).
    """
    series_list = []
    for run in results:
        current_min = float("inf")
        series = []
        for iteration_elites in run.all_elites:
            valid_fits = [abs(e['fitness']) for e in iteration_elites if abs(e['fitness']) < 1e4]
            if valid_fits:
                it_best = min(valid_fits)
            else:
                it_best = series[-1] if series else float("inf")
            current_min = min(current_min, it_best)
            series.append(current_min)
        series_list.append(series)
    return series_list


def _median_iqr_from_padded(series_list: list, global_max_len: int) -> tuple:
    """
    Prende le serie grezze di un gruppo e le estende (padding) fino a
    global_max_len ripetendo l'ultimo valore acquisito, poi calcola
    x, mediana, q1, q3 su quella lunghezza comune.
    """
    if not series_list or global_max_len == 0:
        return None

    mat = np.full((len(series_list), global_max_len), np.nan)
    for i, s in enumerate(series_list):
        if not s:
            continue
        mat[i, :len(s)] = s
        if len(s) < global_max_len:
            mat[i, len(s):] = s[-1]

    x = np.arange(1, global_max_len + 1)
    med = np.nanmedian(mat, axis=0)
    q1 = np.nanpercentile(mat, 25, axis=0)
    q3 = np.nanpercentile(mat, 75, axis=0)
    return x, med, q1, q3


class CompareAllTerrains:
    def __init__(self):
        # Struttura: self.grouped_results[terrain][patch_label] = [run1, run2, ...]
        self.grouped_results = {t: {} for t in TERRAINS}
        # Struttura: self.center_patch_results[terrain] = [run1, run2, ...]
        self.center_patch_results = {t: [] for t in CENTER_PATCH_05_GLOBS}
        # Struttura: self.legacy_results[terrain] = [run1, run2, ...]
        self.legacy_results = {t: [] for t in CENTER_PATCH_LEGACY_DIRS}

        print("Inizio il caricamento dei risultati per tutti i terreni...\n")

        # --- Risultati "a patch" (W/H) ---
        for terrain, base_dir in TERRAINS.items():
            print(f"--- TERRENO: {terrain.upper()} ---")
            for config in PATCH_CONFIGS:
                W, H = config["W"], config["H"]
                label = f"W{W}_H{H}"
                self.grouped_results[terrain][label] = []

                pattern = f"{base_dir}/{NAME_FOLDER}_W{W}_H{H}_run*_{terrain}"
                folders = glob.glob(pattern)

                for folder in folders:
                    if os.path.isdir(folder):
                        try:
                            res = SingleResult(folder)
                            if res.all_elites:
                                self.grouped_results[terrain][label].append(res)
                        except Exception as e:
                            print(f"Errore nel caricamento di {folder}: {e}")

                print(f"[{label}]: Trovate {len(self.grouped_results[terrain][label])} run valide.")
            print("")

        # --- Risultati "centroide a 05" ---
        print("--- CENTROIDE 05 ---")
        for terrain, run_glob in CENTER_PATCH_05_GLOBS.items():
            pattern = os.path.join(CENTER_PATCH_05_DIR, run_glob)
            folders = sorted(glob.glob(pattern))
            for run_folder in folders:
                if not os.path.isdir(run_folder):
                    continue
                try:
                    res = SingleResult(run_folder)
                    if res.all_elites:
                        self.center_patch_results[terrain].append(res)
                except Exception as e:
                    print(f"Errore nel caricamento di {run_folder}: {e}")
            print(f"[{terrain}_center_patch]: Trovate {len(self.center_patch_results[terrain])} run valide.")
        print("")

        # --- Risultati "legacy" (up_<terrain>) ---
        print("--- LEGACY (up_xxx) ---")
        for terrain, path in CENTER_PATCH_LEGACY_DIRS.items():
            runs = _discover_runs(path)
            for run_folder in runs:
                try:
                    res = SingleResult(run_folder)
                    if res.all_elites:
                        self.legacy_results[terrain].append(res)
                except Exception as e:
                    print(f"Errore nel caricamento di {run_folder}: {e}")
            print(f"[{terrain}_100_patches]: Trovate {len(self.legacy_results[terrain])} run valide.")
        print("")

    def plot_median_best_overall(self):
        """
        Mostra l'andamento del Best Overall.
        Colori = Terreni (stessi per patch e center patch).
        Linestyle = tipologia patch (W/H) oppure center patch (05).
        Tutte le curve vengono estese fino alla lunghezza massima globale
        (numero di iterazioni), ripetendo l'ultimo valore acquisito.
        """
        # Stessi colori per terreno, usati sia per le patch che per il center patch (05)
        terrain_colors = {
            "rock": "tab:orange",
            "gaussian": "tab:blue",
            "hemisphere": "tab:green"
        }

        # Linestyle per le dimensioni patch (W/H), indicizzate per numero totale di patch (W*H)
        patch_linestyles_by_index = ["-", "--", ":", "-."]

        # Linestyle dedicato ai dati center patch (05), distinto da tutti gli altri
        center_patch_linestyle = (0, (3, 1, 1, 1, 1, 1))

        # Linestyle dedicato ai dati legacy (up_xxx)
        legacy_linestyle = (0, (1, 1))

        # ------------------------------------------------------------------
        # 1) Raccogliamo tutte le serie grezze (non ancora paddate) di ogni
        #    linea, per poter calcolare la lunghezza massima globale.
        # ------------------------------------------------------------------
        lines_data = []  # lista di dict: {label, color, linestyle, series_list}

        for terrain in TERRAINS:
            color = terrain_colors[terrain]
            for i, config in enumerate(PATCH_CONFIGS):
                W, H = config["W"], config["H"]
                config_label = f"W{W}_H{H}"
                runs = self.grouped_results[terrain].get(config_label, [])
                if not runs:
                    continue

                n_patches = W * H
                label = f"{terrain}_{n_patches}_patches"
                linestyle = patch_linestyles_by_index[i % len(patch_linestyles_by_index)]

                series_list = _raw_series(runs)
                lines_data.append({
                    "label": label,
                    "color": color,
                    "linestyle": linestyle,
                    "series_list": series_list,
                })

        for terrain in CENTER_PATCH_05_GLOBS:
            runs = self.center_patch_results[terrain]
            if not runs:
                continue

            color = terrain_colors[terrain]
            label = f"{terrain}_center_patch"
            series_list = _raw_series(runs)
            lines_data.append({
                "label": label,
                "color": color,
                "linestyle": center_patch_linestyle,
                "series_list": series_list,
                "is_center_patch": True,
            })

        for terrain in CENTER_PATCH_LEGACY_DIRS:
            runs = self.legacy_results[terrain]
            if not runs:
                continue

            color = terrain_colors[terrain]
            label = f"{terrain}_100_patches"
            series_list = _raw_series(runs)
            lines_data.append({
                "label": label,
                "color": color,
                "linestyle": legacy_linestyle,
                "series_list": series_list,
            })

        if not lines_data:
            print("Nessun dato disponibile per il plot.")
            return

        # Lunghezza massima globale (numero massimo di iterazioni tra tutte le curve)
        global_max_len = max(
            len(s) for line in lines_data for s in line["series_list"] if s
        )

        # ------------------------------------------------------------------
        # 2) Plot: ogni curva viene paddata fino a global_max_len ripetendo
        #    l'ultimo valore acquisito.
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 8))

        for line in lines_data:
            result = _median_iqr_from_padded(line["series_list"], global_max_len)
            if result is None:
                continue
            x, med, q1, q3 = result

            is_center_patch = line.get("is_center_patch", False)
            lw = 3
            # Contorno nero per distinguere i dati "centroide 05"
            path_effects = (
                [pe.Stroke(linewidth=lw + 2.5, foreground="black"), pe.Normal()]
                if is_center_patch else None
            )

            ax.plot(x, med, color=line["color"], linestyle=line["linestyle"],
                     lw=lw, label=line["label"], path_effects=path_effects,
                     zorder=3 if is_center_patch else 2)
            ax.fill_between(x, q1, q3, color=line["color"], alpha=0.10)

        ax.set_xlabel("Iteration", fontsize=26)
        ax.set_ylabel("Fitness", fontsize=26)
        ax.tick_params(axis="both", labelsize=20)

        # Legenda più marcata (font e linee grandi), fuori dal grafico
        legend = ax.legend(
            bbox_to_anchor=(1.02, 1), loc='upper left',
            fontsize=18, frameon=True, borderaxespad=0.,
            handlelength=3, handletextpad=0.6, labelspacing=0.6
        )
        for legend_line in legend.get_lines():
            legend_line.set_linewidth(4)

        ax.grid(ls=":", alpha=0.5)

        plt.tight_layout()

        _save_plot(fig, "combined_median_best_overall")
        plt.show()


    def plot_jump_histogram(self):
        """
        Distribuzione del numero di salti (Number of Jumps), stile
        plot_jump_histogram_fusion di fusion_result.py: barre raggruppate,
        conteggi assoluti (Frequency) sull'asse Y.
        Stesso colore per terreno (patch e center patch); le patch W/H si
        distinguono per sfumatura di trasparenza, il center patch (05) per
        una texture a righe (hatch).
        """
        terrain_colors = {
            "rock": "tab:orange",
            "gaussian": "tab:blue",
            "hemisphere": "tab:green"
        }

        # ------------------------------------------------------------------
        # 1) Raccogliamo i n_jumps di ogni gruppo (patch W/H + center patch)
        # ------------------------------------------------------------------
        bars_data = []  # lista di dict: {label, color, alpha, hatch, jumps}

        for terrain in TERRAINS:
            color = terrain_colors[terrain]
            n_configs = len(PATCH_CONFIGS)
            for i, config in enumerate(PATCH_CONFIGS):
                W, H = config["W"], config["H"]
                config_label = f"W{W}_H{H}"
                runs = self.grouped_results[terrain].get(config_label, [])
                if not runs:
                    continue

                jumps = [e['n_jumps'] for run in runs for it in run.all_elites for e in it]
                if not jumps:
                    continue

                n_patches = W * H
                label = f"{terrain}_{n_patches}_patches"
                # Sfumatura di trasparenza in base alla dimensione della patch (i=0 più scura)
                alpha = 0.9 - (i * (0.5 / max(1, n_configs - 1)))

                bars_data.append({
                    "label": label,
                    "color": color,
                    "alpha": alpha,
                    "hatch": None,
                    "jumps": jumps,
                })

        for terrain in CENTER_PATCH_05_GLOBS:
            runs = self.center_patch_results[terrain]
            if not runs:
                continue

            jumps = [e['n_jumps'] for run in runs for it in run.all_elites for e in it]
            if not jumps:
                continue

            label = f"{terrain}_center_patch"
            bars_data.append({
                "label": label,
                "color": terrain_colors[terrain],
                "alpha": 1.0,
                "hatch": "///",
                "jumps": jumps,
            })

        for terrain in CENTER_PATCH_LEGACY_DIRS:
            runs = self.legacy_results[terrain]
            if not runs:
                continue

            jumps = [e['n_jumps'] for run in runs for it in run.all_elites for e in it]
            if not jumps:
                continue

            label = f"{terrain}_legacy"
            bars_data.append({
                "label": label,
                "color": terrain_colors[terrain],
                "alpha": 1.0,
                "hatch": "xx",
                "jumps": jumps,
            })

        if not bars_data:
            print("Nessun dato per i salti.")
            return

        # ------------------------------------------------------------------
        # 2) Plot a barre raggruppate
        # ------------------------------------------------------------------
        j_min = min(min(item["jumps"]) for item in bars_data)
        j_max = max(max(item["jumps"]) for item in bars_data)
        vals = np.arange(j_min, j_max + 1)

        n_bars = len(bars_data)
        bw = 0.8 / n_bars

        fig, ax = plt.subplots(figsize=(14, 8))

        for k, item in enumerate(bars_data):
            counts = [item["jumps"].count(v) for v in vals]
            x_pos = vals + (k - n_bars / 2 + 0.5) * bw
            ax.bar(
                x_pos, counts, width=bw,
                color=item["color"], edgecolor="black",
                alpha=item["alpha"], hatch=item["hatch"],
                label=item["label"]
            )

        ax.set_xticks(vals)
        ax.set_xlabel("Number of Jumps", fontsize=26)
        ax.set_ylabel("Frequency", fontsize=26)
        ax.tick_params(axis="both", labelsize=20)

        # Legenda più marcata, fuori dal grafico (troppe voci per starci dentro)
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc='upper left',
            fontsize=18, frameon=True, borderaxespad=0.,
            handlelength=3, handletextpad=0.6, labelspacing=0.6
        )

        ax.grid(axis="y", ls="--", alpha=0.5)
        plt.tight_layout()

        _save_plot(fig, "combined_jump_histogram")
        plt.show()


if __name__ == "__main__":
    comparator = CompareAllTerrains()
    comparator.plot_median_best_overall()
    comparator.plot_jump_histogram()
    print("\nGrafici generati con successo!")