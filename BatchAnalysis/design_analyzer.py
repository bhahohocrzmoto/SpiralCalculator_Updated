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

def analyze_design_folder(spiral: Dict, target_freq: float | None) -> dict | None:
    """Analyzes a single transformer_matrices.json file and returns a KPI dictionary."""
    folder_path = spiral.get("path")
    if not folder_path or not folder_path.is_dir():
        raise FileNotFoundError(f"Directory not found for spiral {spiral.get('name')}")

    json_path = folder_path / 'Analysis' / 'matrices' / 'transformer_matrices.json'
    if not json_path.exists():
        return None # This is not an error, just not a transformer design

    with open(json_path, 'r') as f:
        data = json.load(f)

    port_names = data.get('port_names', [])
    p_ports = [i for i, name in enumerate(port_names) if name.startswith('p')]
    s_ports = [i for i, name in enumerate(port_names) if name.startswith('s')]
    
    if not p_ports or not s_ports:
        return None
    
    p_idx, s_idx = p_ports[0], s_ports[0]

    frequencies = np.array(data['frequencies_Hz'])
    L_port_all = np.array(data['matrices']['L_port'])
    R_port_all = np.array(data['matrices']['R_port'])
    C_port_all = np.array(data['matrices']['C_port'])

    freq_idx = get_freq_index(frequencies, target_freq)
    f_selected = frequencies[freq_idx]
    min_freq_idx = np.argmin(frequencies)
    omega = 2 * np.pi * f_selected

    L, R, C = L_port_all[freq_idx], R_port_all[freq_idx], C_port_all
    R_dc = R_port_all[min_freq_idx]

    k = L[p_idx, s_idx] / np.sqrt(L[p_idx, p_idx] * L[s_idx, s_idx]) if L[p_idx, p_idx] > 0 and L[s_idx, s_idx] > 0 else 0
    Q = (omega * L[p_idx, p_idx]) / R[p_idx, p_idx] if R[p_idx, p_idx] != 0 else 0
    ac_dc_ratio = R[p_idx, p_idx] / R_dc[p_idx, p_idx] if R_dc[p_idx, p_idx] != 0 else 0
    
    primary_inductances = [L[i, i] for i in p_ports]
    symmetry_score = np.std(primary_inductances) if len(primary_inductances) > 1 else 0.0

    srf_mhz = (1 / (2 * np.pi * np.sqrt(L[p_idx, p_idx] * C[p_idx, p_idx]))) if L[p_idx, p_idx] > 0 and C[p_idx, p_idx] > 0 else 0
    srf_mhz /= 1e6

    return {
        'folder': folder_path.name,
        'frequency_Hz': f_selected,
        'coupling_coefficient_k': k,
        'quality_factor_Q': Q,
        'ac_dc_resistance_ratio': ac_dc_ratio,
        'symmetry_score': symmetry_score,
        'primary_self_capacitance_pF': C[p_idx, p_idx] * 1e12,
        'inter_winding_capacitance_pF': C[p_idx, s_idx] * 1e12,
        'estimated_srf_MHz': srf_mhz,
        'total_k_sum': sum_k_from_name(folder_path.name)
    }


def main():
    parser = argparse.ArgumentParser(description="Run KPI analysis on transformer designs from a spirals.json file.")
    parser.add_argument("spirals_json_file", type=Path, help="Path to the spirals.json file.")
    parser.add_argument("--frequency", help="Optional: Specific frequency in Hz for analysis.", default=None)
    parser.add_argument(
        "--label-mode", choices=["hover", "static", "none"], default="none",
        help="How to display design names on the plot."
    )
    parser.add_argument("--show-plot", action="store_true", help="Open an interactive window for the plot.")
    args = parser.parse_args()

    if not args.spirals_json_file.is_file():
        print(f"Error: Spirals JSON file not found at {args.spirals_json_file}"); sys.exit(1)

    try:
        project_name, spirals = load_spirals_data(args.spirals_json_file)
    except Exception as e:
        print(f"Error reading {args.spirals_json_file}: {e}"); sys.exit(1)

    output_dir = args.spirals_json_file.parent / "FinalTransformerAnalysis"
    output_dir.mkdir(exist_ok=True)
    
    results, errors = [], []
    processed_count, skipped_count = 0, 0

    for spiral in spirals:
        status = spiral.get("status", {})
        if not status.get("results_processed"):
            skipped_count += 1
            continue
        
        try:
            kpis = analyze_design_folder(spiral, args.frequency)
            if kpis:
                results.append(kpis)
            # We don't mark failure here, because not being a transformer isn't an error
            status["analysis_complete"] = True # Mark as analyzed
            processed_count +=1
        except Exception as e:
            print(f"Error processing {spiral.get('name', 'N/A')}: {e}")
            errors.append(create_error_entry(spiral.get('name'), spiral.get('path'), "KPI analysis failed.", e))
            status["analysis_complete"] = False
        spiral["status"] = status

    print(f"\nProcessed {processed_count} spirals, skipped {skipped_count}.")
    print(f"Found {len(results)} transformer-compatible designs.")

    if errors:
        write_debug_log(__file__, errors, output_dir)
        print(f"Completed with {len(errors)} error(s). See debug log in {output_dir}")

    if not results:
        print("No valid transformer data found to plot. Exiting."); return

    df = pd.DataFrame(results)
    csv_path = output_dir / 'design_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved analysis to {csv_path}")

    # Plotting
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=df, 
        x='coupling_coefficient_k', 
        y='quality_factor_Q', 
        hue='estimated_srf_MHz', 
        palette='viridis', 
        size='total_k_sum', 
        sizes=(50, 250)
    )
    plt.title(f"Pareto Plot: Design Comparison @ {args.frequency or 'Max'} Hz")
    plt.xlabel('Coupling Coefficient (k)')
    plt.ylabel('Quality Factor (Q)')
    plt.grid(True)

    # Create a colorbar for the hue
    norm = plt.Normalize(df['estimated_srf_MHz'].min(), df['estimated_srf_MHz'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Estimated SRF (MHz)')

    # Get the original legend
    legend = ax.get_legend()
    if legend:
        # Get handles and labels
        handles = legend.legendHandles
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
        for _, row in df.iterrows():
            plt.text(row['coupling_coefficient_k'] * 1.001, row['quality_factor_Q'], row['folder'], fontsize=9)

    if args.label_mode == 'static':
        _add_static_labels()
    elif args.label_mode == 'hover':
        try:
            import mplcursors
            cursor = mplcursors.cursor(ax.collections[0], hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(df.iloc[sel.index]['folder']))
        except ImportError:
            print("mplcursors not installed; falling back to static labels.")
            _add_static_labels()

    plt.tight_layout()
    plot_path = output_dir / 'design_pareto_plot.png'
    plt.savefig(plot_path)

    if args.show_plot:
        plt.show()

    # Save updated status back to JSON file
    try:
        save_spirals_data(args.spirals_json_file, project_name, spirals)
        print(f"Successfully updated status in {args.spirals_json_file.name}")
    except Exception as e:
        print(f"Error saving status back to {args.spirals_json_file.name}: {e}")

if __name__ == "__main__":
    main()