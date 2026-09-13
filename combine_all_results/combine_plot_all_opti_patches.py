import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# CONFIGURAZIONE GENERALE
# ==========================================
NAME_FOLDER = "test_patches"

# Definizione dei terreni e delle relative cartelle base
TERRAINS = {
    "rock": "result_rock_patch",
    "gaussian": "result_gaussian_patch",
    "hemisphere": "result_hemisphere_patch"
}

PATCH_CONFIGS = [
    {"W": 10, "H": 5},
    {"W": 10, "H": 10},
    {"W": 10, "H": 20},
    {"W": 20, "H": 20},
]

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
        
        # Carica solo i file delle iterazioni (bastano per questi due plot)
        iter_files = sorted(glob.glob(f"{folder}/iteration_reports/iteration_*.json"), 
                            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for fname in iter_files:
            with open(fname) as f:
                it = json.load(f)
            self.all_elites.append([{'fitness': e['fitness'], 'n_jumps': e['n_jumps']} for e in it['elites']])


class CompareAllTerrains:
    def __init__(self):
        # Struttura: self.grouped_results[terrain][patch_label] = [run1, run2, ...]
        self.grouped_results = {t: {} for t in TERRAINS}
        
        print("Inizio il caricamento dei risultati per tutti i terreni...\n")
        
        for terrain, base_dir in TERRAINS.items():
            print(f"--- TERRENO: {terrain.upper()} ---")
            for config in PATCH_CONFIGS:
                W, H = config["W"], config["H"]
                label = f"W{W}_H{H}"
                self.grouped_results[terrain][label] = []
                
                # Cerca tutte le run per questa configurazione e terreno
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

    def plot_jump_histogram(self):
        """Confronta la distribuzione dei salti. Colori base per terreno, sfumature per patch."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Mappe colori per le sfumature (shades) in base al terreno
        cmaps = {
            "rock": plt.get_cmap("Blues"),
            "gaussian": plt.get_cmap("Greens"),
            "hemisphere": plt.get_cmap("Reds")
        }
        
        all_jumps_info = [] # (label_completa, lista_jumps, colore)
        
        # Raccogliamo tutti i dati e assegniamo il colore sfumato
        for terrain in TERRAINS:
            cmap = cmaps[terrain]
            configs = list(self.grouped_results[terrain].keys())
            
            for i, config_label in enumerate(configs):
                runs = self.grouped_results[terrain][config_label]
                jumps = [e['n_jumps'] for run in runs for it in run.all_elites for e in it]
                
                if jumps:
                    # Calcoliamo la sfumatura. i=0 -> più scuro, i=2 -> più chiaro
                    # usiamo range tra 0.4 e 0.9 per evitare che sia troppo bianco o troppo nero
                    shade_intensity = 0.9 - (i * (0.5 / max(1, len(configs) - 1))) 
                    color = cmap(shade_intensity)
                    
                    full_label = f"{terrain.capitalize()} {config_label}"
                    all_jumps_info.append((full_label, jumps, color))

        if not all_jumps_info:
            print("Nessun dato per i salti.")
            return

        # Trova il range globale dei salti
        j_min = min(min(info[1]) for info in all_jumps_info)
        j_max = max(max(info[1]) for info in all_jumps_info)
        vals = np.arange(j_min, j_max + 1)
        
        total_bars = len(all_jumps_info)
        bw = 0.8 / total_bars # Larghezza barre

        for k, (label, jumps, color) in enumerate(all_jumps_info):
            counts = [jumps.count(v) for v in vals]
            total = sum(counts)
            percs = [c / total * 100 if total > 0 else 0 for c in counts]
            
            x_pos = vals + (k - total_bars/2 + 0.5) * bw
            ax.bar(x_pos, percs, width=bw, color=color, edgecolor="black", alpha=0.85, label=label)

        ax.set_title("Jump Count Distribution (Shades by Patch Size, Base Color by Terrain)", fontsize=16)
        ax.set_xlabel("Number of Jumps", fontsize=14)
        ax.set_ylabel("Frequency (%)", fontsize=14)
        ax.set_xticks(vals)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(axis="y", ls="--", alpha=0.5)
        
        plt.tight_layout()
        _save_plot(fig, "combined_jump_histogram")
        plt.show()


    def plot_median_best_overall(self):
        """
        Mostra l'andamento del Best Overall. 
        Colori = Terreni, Linestyles = Patch sizes.
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # 3 Colori differenti per i terreni
        terrain_colors = {
            "rock": "tab:blue",
            "gaussian": "tab:green",
            "hemisphere": "tab:red"
        }
        
        # 4 Tipologie di linee differenti per le dimensioni patch
        patch_linestyles = {
            "W10_H5": "-",      # Continua
            "W10_H10": "--",    # Tratteggiata
            "W10_H20": ":",     # Punteggiata
            "W20_H20": "-."     # Tratto-punto
        }

        for terrain in TERRAINS:
            color = terrain_colors[terrain]
            
            for config_label, runs in self.grouped_results[terrain].items():
                if not runs:
                    continue
                
                linestyle = patch_linestyles.get(config_label, "-")
                series_list = []
                
                for run in runs:
                    current_min = float("inf")
                    series = []
                    
                    for iteration_elites in run.all_elites:
                        # Trasformiamo la fitness in un costo positivo e scartiamo i fallimenti enormi
                        valid_fits = [abs(e['fitness']) for e in iteration_elites if abs(e['fitness']) < 1e4]
                        
                        if valid_fits:
                            it_best = min(valid_fits)
                        else:
                            it_best = series[-1] if series else float("inf")
                            
                        current_min = min(current_min, it_best)
                        series.append(current_min)
                        
                    series_list.append(series)

                max_len = max((len(s) for s in series_list), default=0)
                if max_len == 0: continue
                
                mat = np.full((len(series_list), max_len), np.nan)
                
                for i, s in enumerate(series_list):
                    mat[i, :len(s)] = s
                    if len(s) < max_len:
                        mat[i, len(s):] = s[-1]

                # Mediana e IQR
                x = np.arange(1, max_len + 1)
                med = np.nanmedian(mat, axis=0)
                q1 = np.nanpercentile(mat, 25, axis=0)
                q3 = np.nanpercentile(mat, 75, axis=0)

                label = f"{terrain.capitalize()} {config_label}"
                
                ax.plot(x, med, color=color, linestyle=linestyle, lw=2.5, label=label)
                # La fill_between è stata tenuta leggera per non fare troppa confusione visiva, 
                # ma commentala se preferisci vedere solo le linee
                ax.fill_between(x, q1, q3, color=color, alpha=0.10)

        ax.set_title("Global Best Fitness Trend Across Terrains and Patch Sizes", fontsize=18)
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel("Global Best Fitness", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        
        # Mettiamo la legenda fuori per non coprire i dati
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True)
        ax.grid(ls=":", alpha=0.5)
        
        plt.tight_layout()

        _save_plot(fig, "combined_median_best_overall")
        plt.show()

if __name__ == "__main__":
    comparator = CompareAllTerrains()
    comparator.plot_jump_histogram()
    comparator.plot_median_best_overall()
    print("\nGrafici generati con successo!")