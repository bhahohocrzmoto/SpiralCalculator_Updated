#!/usr/bin/env python3
"""
Script for generating plots and CSV summaries from FastSolver outputs,
with multi-port reduction (phases / windings) on top of per-trace matrices.
"""

from __future__ import annotations

import argparse
import json
import datetime
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Add repo root to path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from main.debug_utils import write_debug_log, create_error_entry


matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY_FREQS = [10e3, 50e3, 100e3, 200e3, 500e3, 1e6]
REF_FREQ = 100e3
DEBUG_LOG_NAME = "PlotGeneration_Debug.json" # Kept for compatibility with SpiralsMain.py's popup

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


def ensure_analysis_dirs(spiral_path: Path) -> Dict[str, Path]:
    analysis = spiral_path / "Analysis"
    matrices_dir = analysis / "matrices"
    ports_dir = analysis / "ports"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    ports_dir.mkdir(parents=True, exist_ok=True)
    return {
        "analysis": analysis, "matrices": matrices_dir, "ports": ports_dir,
        "ports_config": analysis / "ports_config.json",
        "debug_log": spiral_path.parent / DEBUG_LOG_NAME,
    }

def load_capacitance_matrix(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [[float(x) for x in ln.replace(",", " ").split()] for ln in f if ln.strip()]
    if not lines: raise ValueError("Capacitance matrix is empty")
    return np.array(lines, dtype=float)

def select_first_match(data: dict, candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in data: return key
        for k in data:
            if k.lower() == key.lower(): return k
    return None

def load_impedance_and_freq(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    text = mat_path.read_text(encoding="utf-8", errors="ignore")
    if "Impedance matrix for frequency" in text:
        lines, freqs, mats, i = text.splitlines(), [], [], 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Impedance matrix for frequency"):
                m = re.search(r"=\s*([^\s]+)\s+(\d+)\s*x\s*(\d+)", line)
                if not m: raise ValueError(f"Could not parse header: {line!r}")
                freq_val, nrows, ncols = float(m.group(1)), int(m.group(2)), int(m.group(3))
                i += 1
                rows = [[complex(lines[i+r].split()[2*c] + lines[i+r].split()[2*c+1]) for c in range(ncols)] for r in range(nrows)]
                i += nrows
                mats.append(np.array(rows, dtype=complex))
                freqs.append(freq_val)
            else: i += 1
        if mats: return np.array(freqs, dtype=float), np.stack(mats, axis=0)

    data = loadmat(mat_path)
    freq_key, z_key = select_first_match(data, ["freq", "f"]), select_first_match(data, ["Zc", "Z"])
    if not freq_key or not z_key: raise ValueError("Could not find freq/impedance keys")
    freq, Z = np.squeeze(np.array(data[freq_key], dtype=float)), np.array(data[z_key])
    if Z.ndim == 2: Z = Z[np.newaxis, :, :]
    if Z.shape[0] != freq.shape[0]:
        if Z.shape[-1] == freq.shape[0]: Z = np.moveaxis(Z, -1, 0)
        else: raise ValueError(f"Freq/impedance shape mismatch")
    return freq, Z

def compute_current_pattern(port_def: Dict[str, object], n: int) -> np.ndarray:
    port_type, signs = str(port_def.get("type", "")).lower(), np.array(port_def.get("signs", []), dtype=float).reshape(-1)
    if signs.size != n: raise ValueError(f"Expected {n} signs, got {signs.size}")
    if port_type == "parallel":
        active_count = np.count_nonzero(signs)
        if active_count == 0: raise ValueError("Parallel port has no active conductors")
        return signs / active_count
    return signs

def build_grouping_matrix_from_ports(ports: Dict[str, Dict[str, object]], n_conductors: int) -> Tuple[np.ndarray, List[str]]:
    if not ports: raise ValueError("No ports defined")
    port_names = sorted(ports.keys())
    cols = [compute_current_pattern(ports[name], n_conductors) for name in port_names]
    return np.stack(cols, axis=1), port_names

def compute_R_L(freq: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, L = np.real(Z), np.zeros_like(Z, dtype=float)
    nz = freq != 0
    # Reshape freq array to (F, 1, 1) to enable broadcasting with (F, N, N) matrix
    L[nz] = np.imag(Z[nz]) / (2 * math.pi * freq[nz, None, None])
    return R, L

def effective_values_from_diag(freq: np.ndarray, R_diag: np.ndarray, L_diag: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_eff, L_eff = np.asarray(R_diag, dtype=float), np.asarray(L_diag, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"): Q = (2 * math.pi * freq * L_eff) / R_eff
    return L_eff, R_eff, Q

def find_resonance(freq: np.ndarray, Zin: np.ndarray) -> float:
    signs = np.sign(np.imag(Zin))
    crossings = np.where(np.diff(signs) != 0)[0]
    if crossings.size == 0: return float("nan")
    idx = crossings[0]
    f1, f2, y1, y2 = freq[idx], freq[idx+1], np.imag(Zin)[idx], np.imag(Zin)[idx+1]
    return float(f1 - y1 * (f2 - f1) / (y2 - y1)) if y2 != y1 else float(f1)

def build_matrix_payload(**kwargs) -> Dict[str, object]:
    for key in ["C_port", "R_port", "L_port"]:
        kwargs["matrices"][key] = np.asarray(kwargs["matrices"][key], dtype=float).tolist()
    kwargs["frequencies_Hz"] = np.asarray(kwargs["freq"], dtype=float).tolist()
    del kwargs["freq"]
    return kwargs

def write_matrix_json(payload: Dict[str, object], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

def get_port_analysis_type(port_name: str) -> str:
    if port_name.startswith('p') or port_name.startswith('s'): return 'transformer'
    if 'Series' in port_name: return 'series_inductor'
    if 'Parallel' in port_name: return 'parallel_inductor'
    return 'unknown'

def process_spiral(spiral: Dict, global_records: List[Dict]):
    """
    Processes a single spiral entry from solver outputs to reduced matrices.
    Raises exceptions on failure.
    """
    spiral_path = spiral.get("path")
    if not spiral_path or not spiral_path.is_dir():
        raise FileNotFoundError(f"Directory not found for spiral {spiral.get('name')}")
    
    spiral_name = spiral_path.name
    
    # This block performs the core scientific computation.
    # It now raises exceptions instead of logging to a file.
    dirs = ensure_analysis_dirs(spiral_path)
    C_trace = load_capacitance_matrix(spiral_path / "FastSolver" / "CapacitanceMatrix.txt")
    freq, Z_trace = load_impedance_and_freq(spiral_path / "FastSolver" / "Zc.mat")
    n = C_trace.shape[0]

    if C_trace.shape[1] != n or Z_trace.shape[1] != n or Z_trace.shape[2] != n:
        raise ValueError(f"Matrix dimension mismatch for {spiral_name}")
    
    ports_config_path = dirs["ports_config"]
    if not ports_config_path.is_file():
        raise FileNotFoundError(f"ports_config.json not found for {spiral_name}. Please generate it first via the main GUI.")
        
    ports = json.loads(ports_config_path.read_text()).get("ports", {})
    if not ports:
        raise ValueError(f"No ports defined in {ports_config_path}")

    R_trace, L_trace = compute_R_L(freq, Z_trace)
    
    ports_by_type = {}
    for name, definition in ports.items():
        analysis_type = get_port_analysis_type(name)
        if analysis_type != 'unknown':
            ports_by_type.setdefault(analysis_type, {})[name] = definition

    # Process each analysis type (e.g., 'transformer', 'inductor')
    for analysis_type, type_ports in ports_by_type.items():
        W, port_names = build_grouping_matrix_from_ports(type_ports, n)
        WT = W.T
        C_port = WT @ C_trace @ W
        R_port = np.stack([WT @ R_trace[k] @ W for k in range(freq.size)])
        L_port = np.stack([WT @ L_trace[k] @ W for k in range(freq.size)])
        payload = build_matrix_payload(
            spiral_name=spiral_name, analysis_type=analysis_type, port_names=port_names, freq=freq,
            matrices={"C_port": C_port, "R_port": R_port, "L_port": L_port},
            source_files={"Zc": str(spiral_path / "FastSolver/Zc.mat"), "CapacitanceMatrix": str(spiral_path / "FastSolver/CapacitanceMatrix.txt")},
        )
        write_matrix_json(payload, dirs["matrices"] / f"{analysis_type}_matrices.json")

    # Process all ports together for the summary records
    W_all, port_names_all = build_grouping_matrix_from_ports(ports, n)
    WT_all = W_all.T
    R_port_all = np.stack([WT_all @ R_trace[k] @ W_all for k in range(freq.size)])
    L_port_all = np.stack([WT_all @ L_trace[k] @ W_all for k in range(freq.size)])

    for p_idx, port_name in enumerate(port_names_all):
        R_diag, L_diag = R_port_all[:, p_idx, p_idx], L_port_all[:, p_idx, p_idx]
        L_eff, R_eff, Q = effective_values_from_diag(freq, R_diag, L_diag)
        Zin = R_eff + 1j * 2 * math.pi * freq * L_eff
        resonance = find_resonance(freq, Zin)
        
        global_records.append({
            "spiral_name": spiral_name, "port_name": port_name, "ref_freq_Hz": REF_FREQ,
            "L_eff_H": float(np.interp(REF_FREQ, freq, L_eff)), "R_eff_ohm": float(np.interp(REF_FREQ, freq, R_eff)),
            "Q": float(np.interp(REF_FREQ, freq, Q)), "first_resonance_Hz": resonance,
            "N_conductors": n, "N_ports": len(port_names_all),
        })


def main():
    parser = argparse.ArgumentParser(description="Process solver outputs into reduced matrices based on a spirals.json file.")
    parser.add_argument("spirals_json_file", type=Path, help="Path to the spirals.json file.")
    args = parser.parse_args()

    spirals_json_file = args.spirals_json_file
    if not spirals_json_file.is_file():
        print(f"Error: Spirals JSON file not found at '{spirals_json_file}'")
        sys.exit(1)
        
    try:
        project_name, spirals = load_spirals_data(spirals_json_file)
        print(f"Loaded {len(spirals)} spirals from project '{project_name}'.")
    except Exception as e:
        print(f"Error: Failed to load or parse {spirals_json_file}: {e}")
        sys.exit(1)

    errors = []
    global_records: List[Dict[str, object]] = []
    processed_count = 0
    skipped_count = 0

    for spiral in spirals:
        status = spiral.get("status", {})
        if not status.get("simulation_run"):
            skipped_count += 1
            continue

        try:
            print(f"Processing: {spiral.get('name', 'N/A')}")
            process_spiral(spiral, global_records)
            status["results_processed"] = True
            processed_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {spiral.get('name', 'N/A')}: {e}")
            error_entry = create_error_entry(
                spiral_name=spiral.get('name', 'N/A'),
                spiral_path=spiral.get('path', Path()),
                message="Failed during plot generation/matrix reduction.",
                exc=e
            )
            errors.append(error_entry)
            status["results_processed"] = False
        spiral["status"] = status
    
    print(f"\nProcessed {processed_count} spirals, skipped {skipped_count}.")

    if errors:
        write_debug_log(__file__, errors, spirals_json_file.parent)
        print(f"Completed with {len(errors)} error(s). See debug log.")
    else:
        print("Completed successfully.")

    # Always save the updated statuses
    try:
        save_spirals_data(spirals_json_file, project_name, spirals)
        print(f"Updated status in {spirals_json_file.name}")
    except Exception as e:
        print(f"[ERROR] Could not save updated status to {spirals_json_file.name}: {e}")


if __name__ == "__main__":
    main()
