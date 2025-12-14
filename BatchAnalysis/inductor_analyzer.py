import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import datetime
import sys
from typing import List, Dict

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from main.debug_utils import write_debug_log, create_error_entry

def load_spirals_data(spirals_file: Path) -> tuple[str, List[Dict]]:
    """Reads, validates, and resolves paths from a spirals.json file."""
    if not spirals_file.is_file():
        raise FileNotFoundError(f"Spirals JSON file not found: {spirals_file}")

    with open(spirals_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    project_name = data.get("project_name", spirals_file.stem)
    spirals = data.get("spirals", [])
    if not isinstance(spirals, list):
        raise ValueError("JSON 'spirals' field is not a list.")

    for spiral in spirals:
        path_str = spiral.get("path")
        if not path_str:
            continue
        p = Path(path_str)
        if not p.is_absolute():
            p = spirals_file.parent / p
        spiral["path"] = p.resolve()
    
    return project_name, spirals

def save_spirals_data(spirals_file: Path, project_name: str, spirals: List[Dict]):
    """Saves the updated list of spirals back to the JSON file."""
    try:
        with open(spirals_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
    except Exception:
        output_data = {}

    output_data["project_name"] = project_name
    output_data["spirals"] = spirals

    for spiral in output_data["spirals"]:
        if isinstance(spiral.get("path"), Path):
            spiral["path"] = str(spiral["path"])

    with open(spirals_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)


def sum_k_from_name(name: str) -> float:
    """Return the sum of all K-values encoded in the folder name."""
    k_matches = re.findall(r"K(\d+(?:\.\d+)?)", name)
    return float(np.sum([float(k) for k in k_matches])) if k_matches else 0.0

def get_freq_index(frequencies, target_freq=None):
    """Finds the index of the target frequency, or the highest frequency if not specified."""
    if target_freq is None: return np.argmax(frequencies)
    try:
        target_freq = float(target_freq)
        return np.argmin(np.abs(frequencies - target_freq))
    except (ValueError, TypeError):
        return np.argmax(frequencies)

def analyze_inductor_json(json_path: Path, spiral: Dict, target_freq: float | None) -> dict | None:
    """Analyzes a single inductor matrix JSON file and returns a KPI dictionary."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    analysis_type = data.get('analysis_type', 'unknown')
    port_name = data.get('port_names', ['unknown'])[0]
    
    port_type = 'unknown'
    if 'series' in analysis_type: port_type = 'series'
    elif 'parallel' in analysis_type: port_type = 'parallel'

    frequencies = np.array(data['frequencies_Hz'])
    L_port = np.array(data['matrices']['L_port'])
    R_port = np.array(data['matrices']['R_port'])
    C_port = np.array(data['matrices']['C_port'])

    freq_idx = get_freq_index(frequencies, target_freq)
    f_selected = frequencies[freq_idx]
    min_freq_idx = np.argmin(frequencies)
    omega = 2 * np.pi * f_selected

    L = L_port[freq_idx] if L_port.ndim == 3 else L_port
    R = R_port[freq_idx] if R_port.ndim == 3 else R_port
    C = C_port
    R_dc = R_port[min_freq_idx] if R_port.ndim == 3 else R_port

    l_eff, r_ac, c_self, r_dc_val = L[0, 0], R[0, 0], C[0, 0], R_dc[0, 0]

    q_factor = (omega * l_eff) / r_ac if r_ac > 0 else 0
    ac_dc_ratio = r_ac / r_dc_val if r_dc_val > 0 else 0
    srf_mhz = (1 / (2 * np.pi * np.sqrt(l_eff * c_self))) / 1e6 if l_eff > 0 and c_self > 0 else 0

    return {
        'folder': spiral.get("name", json_path.stem), 'port_name': port_name, 'port_type': port_type,
        'frequency_Hz': f_selected, 'effective_inductance_uH': l_eff * 1e6,
        'quality_factor_Q': q_factor, 'ac_dc_resistance_ratio': ac_dc_ratio,
        'self_capacitance_pF': c_self * 1e12, 'estimated_srf_MHz': srf_mhz,
        'total_k_sum': sum_k_from_name(spiral.get("name", ""))
    }

def main():
    parser = argparse.ArgumentParser(description="Run KPI analysis on inductor designs from a spirals.json file.")
    parser.add_argument("spirals_json_file", type=Path, help="Path to the spirals.json file.")
    parser.add_argument("--frequency", help="Optional: Specific frequency in Hz for analysis.", default=None)
    parser.add_argument("--label-mode", choices=["hover", "static", "none"], default="none", help="How to display design names on the plot.")
    parser.add_argument("--show-plot", action="store_true", help="Open an interactive window for the plot.")
    args = parser.parse_args()

    if not args.spirals_json_file.is_file():
        print(f"Error: Spirals JSON file not found at {args.spirals_json_file}"); sys.exit(1)

    try:
        project_name, spirals = load_spirals_data(args.spirals_json_file)
    except Exception as e:
        print(f"Error reading {args.spirals_json_file}: {e}"); sys.exit(1)

    output_dir = args.spirals_json_file.parent / "FinalInductorAnalysis"
    series_dir = output_dir / "Series"; parallel_dir = output_dir / "Parallel"
    series_dir.mkdir(parents=True, exist_ok=True); parallel_dir.mkdir(parents=True, exist_ok=True)
    
    results, errors = [], []
    processed_count, skipped_count = 0, 0

    for spiral in spirals:
        status = spiral.get("status", {})
        folder_path = spiral.get("path")

        if not status.get("results_processed") or not folder_path or not folder_path.is_dir():
            skipped_count += 1
            continue
        
        matrices_dir = folder_path / 'Analysis' / 'matrices'
        if not matrices_dir.exists():
            continue

        try:
            inductor_jsons_found = 0
            for json_file in matrices_dir.glob('*_inductor_matrices.json'):
                kpis = analyze_inductor_json(json_file, spiral, args.frequency)
                if kpis:
                    results.append(kpis)
                inductor_jsons_found += 1
            
            if inductor_jsons_found > 0:
                processed_count += 1
            
            # Mark as analyzed even if no inductor files were found, as the step ran
            status["analysis_complete"] = True
        except Exception as e:
            print(f"Error processing {spiral.get('name', 'N/A')}: {e}")
            errors.append(create_error_entry(spiral.get('name'), folder_path, "Inductor KPI analysis failed.", e))
            status["analysis_complete"] = False
        spiral["status"] = status

    print(f"\nProcessed {processed_count} spirals, skipped {skipped_count}.")
    print(f"Found and processed {len(results)} total inductor analysis files.")

    if errors:
        write_debug_log(__file__, errors, output_dir)
        print(f"Completed with {len(errors)} error(s). See debug log in {output_dir}")

    if not results:
        print("No valid inductor data found to plot. Exiting."); return

    df = pd.DataFrame(results)
    for port_type in ['series', 'parallel']:
        df_type = df[df['port_type'] == port_type]
        if df_type.empty:
            print(f"\nNo data found for '{port_type}' inductors."); continue

        target_dir = series_dir if port_type == 'series' else parallel_dir
        csv_path = target_dir / f'{port_type}_comparison.csv'
        df_type.to_csv(csv_path, index=False)
        print(f"\nSuccessfully saved '{port_type}' analysis to {csv_path}")

        plt.figure(figsize=(12, 8))
        ax = sns.scatterplot(
            data=df_type, 
            x='effective_inductance_uH', 
            y='quality_factor_Q', 
            hue='estimated_srf_MHz', 
            palette='viridis', 
            size='total_k_sum', 
            sizes=(50, 250)
        )
        plt.title(f"{port_type.capitalize()} Inductor Performance @ {args.frequency or 'Max'} Hz")
        plt.xlabel('Effective Inductance (uH)'), plt.ylabel('Quality Factor (Q)'), plt.grid(True)

        # Create a colorbar for the hue
        norm = plt.Normalize(df_type['estimated_srf_MHz'].min(), df_type['estimated_srf_MHz'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Estimated SRF (MHz)')

        # Get the original legend
        legend = ax.get_legend()
        if legend:
            # Get handles and labels
            handles = legend.legend_handles
            labels = [text.get_text() for text in legend.texts]
            
            # Separate size and hue handles/labels
            size_handles = []
            size_labels = []
            
            # We assume the 'total_k_sum' title is present for the size legend
            try:
                size_title_index = labels.index('total_k_sum')
                # The handles for the sizes are after the title
                size_handles = handles[size_title_index + 1:]
                size_labels = labels[size_title_index + 1:]
            except ValueError:
                # Fallback if title is not found
                pass

            # Remove the original legend
            legend.remove()

            # Create a new legend for the size
            if size_handles:
                ax.legend(
                    handles=size_handles,
                    labels=size_labels,
                    title='Total K Sum',
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left'
                )

        def _add_static_labels():
            for _, row in df_type.iterrows():
                plt.text(row['effective_inductance_uH'] * 1.01, row['quality_factor_Q'], row['folder'], fontsize=9)

        if args.label_mode == 'static': _add_static_labels()
        elif args.label_mode == 'hover':
            try:
                import mplcursors
                cursor = mplcursors.cursor(ax.collections[0], hover=True)
                cursor.connect("add", lambda sel: sel.annotation.set_text(df_type.iloc[sel.index]['folder']))
            except ImportError:
                print("mplcursors not installed; falling back to static labels."); _add_static_labels()
        
        plt.tight_layout()
        plot_path = target_dir / f'{port_type}_performance_plot.png'
        plt.savefig(plot_path)
        print(f"Successfully saved '{port_type}' performance plot to {plot_path}")

    if args.show_plot: plt.show()

    try:
        save_spirals_data(args.spirals_json_file, project_name, spirals)
        print(f"\nSuccessfully updated status in {args.spirals_json_file.name}")
    except Exception as e:
        print(f"Error saving status back to {args.spirals_json_file.name}: {e}")

if __name__ == "__main__":
    main()