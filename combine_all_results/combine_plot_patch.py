import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


BASE_DIR = "result_rock_patch"
NAME_FOLDER = "test_patches"
TERRAIN_TYPE = "rock"
PATCH_CONFIGS = [
    {"W": 10, "H": 5},
    {"W": 10, "H": 10},
    {"W": 10, "H": 20},
    
]

COMBINED_OUTPUT = "result/patch_comparison_plots"

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
        
        # Estrai info dai file iterazione
        self.all_elites = []
        self.all_comb_data = []
        self.best_fit_ever = None
        
        iter_files = sorted(glob.glob(f"{folder}/iteration_reports/iteration_*.json"), 
                            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for fname in iter_files:
            with open(fname) as f:
                it = json.load(f)
            # Salviamo le fitness positive per comodità di plot (min is better o max is better in base a BilevelOpt)
            # NOTA: Nel tuo script originale usavi -e['fitness']. Adatto in base alla tua logica:
            self.all_elites.append([{'fitness': e['fitness'], 'n_jumps': e['n_jumps']} for e in it['elites']])
            self.best_fit_ever = it['best_fitness_ever']
        
        comb_files = sorted(glob.glob(f"{folder}/iteration_reports/all_comb_in_iter_*.json"),
                            key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for fname in comb_files:
            with open(fname) as f:
                self.all_comb_data.append(json.load(f))


class ComparePatchSizes:
    def __init__(self):
        self.grouped_results = {}
        self.colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        print(f"Cercando risultati nella cartella '{BASE_DIR}'...\n")
        
        for config in PATCH_CONFIGS:
            W, H = config["W"], config["H"]
            label = f"W{W}_H{H}"
            self.grouped_results[label] = []
            
            # Cerca tutte le run per questa configurazione
            pattern = f"{BASE_DIR}/{NAME_FOLDER}_W{W}_H{H}_run*_{TERRAIN_TYPE}"
            folders = glob.glob(pattern)
            
            for folder in folders:
                if os.path.isdir(folder):
                    try:
                        res = SingleResult(folder)
                        if res.all_elites:  # Aggiungi solo se ha dati validi
                            self.grouped_results[label].append(res)
                    except Exception as e:
                        print(f"Errore nel caricamento di {folder}: {e}")
            
            print(f"[{label}]: Trovate {len(self.grouped_results[label])} run valide.")

    def plot_jump_histogram(self):
        """Confronta la distribuzione dei salti tra le diverse configurazioni di patch."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(self.grouped_results.keys())
        all_jumps_by_config = []
        
        for label in labels:
            jumps = [e['n_jumps'] for run in self.grouped_results[label] for it in run.all_elites for e in it]
            all_jumps_by_config.append(jumps)
            
        if not any(all_jumps_by_config):
            return

        # Trova il range globale dei salti
        j_min = min(min(j) for j in all_jumps_by_config if j)
        j_max = max(max(j) for j in all_jumps_by_config if j)
        vals = np.arange(j_min, j_max + 1)
        
        bw = 0.8 / len(labels) # Larghezza barre

        for k, (label, jumps) in enumerate(zip(labels, all_jumps_by_config)):
            if not jumps: continue
            counts = [jumps.count(v) for v in vals]
            # Normalizzazione percentuale per rendere confrontabili run con un numero diverso di iterazioni totali
            total = sum(counts)
            percs = [c / total * 100 if total > 0 else 0 for c in counts]
            
            x_pos = vals + (k - len(labels)/2 + 0.5) * bw
            color = self.colors[k % len(self.colors)]
            ax.bar(x_pos, percs, width=bw, color=color, edgecolor="black", alpha=0.8, label=label)

        ax.set_title(f"Jump Count Distribution by Patch Size ({TERRAIN_TYPE})")
        ax.set_xlabel("Number of Jumps")
        ax.set_ylabel("Frequency (%)")
        ax.set_xticks(vals)
        ax.legend()
        ax.grid(axis="y", ls="--", alpha=0.5)
        
        _save_plot(fig, "patch_comparison_jump_histogram")
        plt.show()

    def plot_fitness_stats_over_time(self):
        """Mostra la mediana della fitness della migliore elite per iterazione, aggregata per patch size."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for k, (label, runs) in enumerate(self.grouped_results.items()):
            if not runs: continue
            
            # Estrai i best fitness per iterazione per ogni run
            run_iter_bests = []
            max_len = 0
            
            for run in runs:
                iter_bests = []
                for iteration_elites in run.all_elites:
                    valid_fits = [e['fitness'] for e in iteration_elites if e['fitness'] < 1e5]
                    if valid_fits:
                        # NOTA: Uso max() perché nel tuo script Main usi: if log_result['fitness'] > best_fitness 
                        # Adatta a min() se il tuo obiettivo è minimizzare.
                        iter_bests.append(max(valid_fits)) 
                    else:
                        iter_bests.append(iter_bests[-1] if iter_bests else 0)
                
                run_iter_bests.append(iter_bests)
                max_len = max(max_len, len(iter_bests))
                
            if max_len == 0: continue
            
            # Padding per allineare le run che si sono fermate prima (early stop)
            padded_bests = np.zeros((len(run_iter_bests), max_len))
            for i, row in enumerate(run_iter_bests):
                padded_bests[i, :len(row)] = row
                if len(row) < max_len:
                    padded_bests[i, len(row):] = row[-1] # Propaga l'ultimo valore
            
            x_axis = np.arange(1, max_len + 1)
            med_iter = np.median(padded_bests, axis=0)
            q1_iter = np.percentile(padded_bests, 25, axis=0)
            q3_iter = np.percentile(padded_bests, 75, axis=0)
            
            c = self.colors[k % len(self.colors)]
            ax.plot(x_axis, med_iter, color=c, lw=3, label=f'{label} (Median)')
            ax.fill_between(x_axis, q1_iter, q3_iter, color=c, alpha=0.2)

        ax.set_title(f"Best Fitness Trend per Iteration by Patch Size")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness Score")
        ax.legend()
        ax.grid(ls=':', alpha=0.5)
        
        _save_plot(fig, "patch_comparison_fitness_trend")
        plt.show()

    def plot_final_best_fitness_boxplot(self):
        """Compara i risultati finali globali tra le configurazioni usando un boxplot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = []
        labels = []
        
        for label, runs in self.grouped_results.items():
            bests = [r.best_fit_ever for r in runs if r.best_fit_ever is not None]
            if bests:
                data.append(bests)
                labels.append(f"{label}\n(n={len(bests)})")
                
        if not data: return

        bplot = ax.boxplot(data, 
                    patch_artist=True, 
                    labels=labels, 
                    zorder=3,
                    flierprops=dict(marker='', markersize=0))  # rimuove completamente gli outlier

        # Colora i box
        for patch, color in zip(bplot['boxes'], [self.colors[i % len(self.colors)] for i in range(len(bplot['boxes']))]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        # Aggiungi i singoli punti per vedere l'esatta distribuzione delle run
        # for i, d in enumerate(data):
        #     y = d
        #     x = np.random.normal(i + 1, 0.04, size=len(y))
        #     ax.scatter(x, y, alpha=0.9, color='black', zorder=4, s=20)

        ax.set_title("Global Best Fitness Distribution Across Runs")
        ax.set_ylabel("Best Fitness Reached")
        ax.grid(axis='y', ls='--', alpha=0.6, zorder=0)
        
        _save_plot(fig, "patch_comparison_best_fitness_boxplot")
        plt.show()

    def plot_convergence_rate(self):
        """Confronta la percentuale media di traiettorie convergenti per iterazione."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for k, (label, runs) in enumerate(self.grouped_results.items()):
            valid_runs = [r for r in runs if r.all_comb_data]
            if not valid_runs: continue
            
            max_iters = max(max((d['iteration'] for d in r.all_comb_data), default=0) for r in valid_runs)
            rates_matrix = np.full((len(valid_runs), max_iters), np.nan)
            
            for i, r in enumerate(valid_runs):
                for d in r.all_comb_data:
                    it_idx = d['iteration'] - 1
                    cv = sum(1 for s in d.get('steps', []) if s.get('converged'))
                    total = len(d.get('steps', []))
                    if total > 0:
                        rates_matrix[i, it_idx] = (cv / total) * 100

            # Media ignorando i NaN (per le run finite con early stop)
            mean_rates = np.nanmean(rates_matrix, axis=0)
            x_axis = np.arange(1, max_iters + 1)
            
            color = self.colors[k % len(self.colors)]
            ax.plot(x_axis, mean_rates, color=color, marker='o', markersize=4, lw=2, label=label)

        ax.set_title("Average Convergence Rate by Patch Size")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Convergence Rate (%)")
        ax.set_ylim(-5, 105)
        ax.legend()
        ax.grid(ls=":", alpha=0.5)
        
        _save_plot(fig, "patch_comparison_convergence_rate")
        plt.show()

    def plot_median_best_overall(self):
        """
        Mostra l'andamento del "Miglior Risultato Globale" (Global Best) raggiunto 
        fino a ciascuna iterazione, calcolando la mediana e l'area interquartile (IQR).
        Scarta in automatico le penalità di non-convergenza.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for k, (label, runs) in enumerate(self.grouped_results.items()):
            if not runs:
                continue

            series_list = []
            for run in runs:
                current_min = float("inf")
                series = []
                
                for iteration_elites in run.all_elites:
                    # Trasformiamo la fitness in un costo positivo (assoluto)
                    # e scartiamo le penalità enormi (es. > 10000) dovute alla non-convergenza
                    valid_fits = [abs(e['fitness']) for e in iteration_elites if abs(e['fitness']) < 1e4]
                    
                    if valid_fits:
                        it_best = min(valid_fits) # Cerchiamo il costo più basso
                    else:
                        it_best = series[-1] if series else float("inf")
                        
                    current_min = min(current_min, it_best)
                    series.append(current_min)
                    
                series_list.append(series)

            # Padding (riempimento) per allineare array di lunghezze diverse
            max_len = max((len(s) for s in series_list), default=0)
            if max_len == 0: continue
            
            mat = np.full((len(series_list), max_len), np.nan)
            
            for i, s in enumerate(series_list):
                mat[i, :len(s)] = s
                if len(s) < max_len:
                    mat[i, len(s):] = s[-1] # Propaga l'ultimo valore valido

            # Calcolo Statistico: Mediana e IQR
            x = np.arange(1, max_len + 1)
            med = np.nanmedian(mat, axis=0)
            q1 = np.nanpercentile(mat, 25, axis=0)
            q3 = np.nanpercentile(mat, 75, axis=0)

            color = self.colors[k % len(self.colors)]
            

            ax.plot(x, med, color=color, lw=2.5, label=label)
            ax.fill_between(x, q1, q3, color=color, alpha=0.20)

        # Stile visivo identico a fusion_result.py
        ax.set_xlabel("Iteration", fontsize=30)
        ax.set_ylabel("Global Best Fitness", fontsize=30)
        ax.tick_params(axis="both", labelsize=25)
        ax.legend(loc="upper right", fontsize=25, frameon=True)
        ax.grid(ls=":", alpha=0.5)
        
        plt.tight_layout()

        _save_plot(fig, "patch_comparison_median_best_overall")
        plt.show()

if __name__ == "__main__":
    comparator = ComparePatchSizes()
    comparator.plot_jump_histogram()
    comparator.plot_fitness_stats_over_time()
    comparator.plot_final_best_fitness_boxplot()
    comparator.plot_convergence_rate()
    comparator.plot_median_best_overall()
    print("\nTutti i grafici di confronto sono stati generati con successo!")