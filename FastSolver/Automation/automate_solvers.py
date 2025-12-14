"""Batch runner for FastHenry and FasterCap.

This script reads a list of geometry folders from an Address.txt file and
runs the existing ``run_fasthenry.py`` and ``run_fastercap.py`` utilities
for each entry. It expects each listed folder to contain a ``FastSolver``
subdirectory with the two generated solver inputs:

* ``Wire_Sections.inp`` (for FastHenry)
* ``Wire_Sections_FastCap.txt`` (for FasterCap)

The outputs ``Zc.mat`` and ``CapacitanceMatrix.txt`` are checked after each
run and warnings are printed if anything is missing.
"""
import argparse
import json
import datetime
import os
import sys
from pathlib import Path
from typing import List, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FastSolver.FastHenry.run_fasthenry import run_fasthenry
from FastSolver.FastCap.run_fastercap import run_fastercap
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


def process_geometry_folder(spiral: Dict, eps_r: float = 1.0):
    """
    Runs FastHenry and FasterCap for a single spiral.
    Raises exceptions on failure.
    """
    geometry_root = spiral.get("path")
    if not geometry_root or not geometry_root.is_dir():
        raise FileNotFoundError(f"Root directory not found for {spiral.get('name', 'N/A')}")

    fastsolver_dir = geometry_root / "FastSolver"
    if not fastsolver_dir.is_dir():
        raise FileNotFoundError(f"FastSolver folder missing for {geometry_root.name}")

    fh_input = fastsolver_dir / "geometry.inp"
    fc_input = fastsolver_dir / "geometry_FastCap.txt"

    if not fh_input.is_file():
        raise FileNotFoundError(f"geometry.inp missing in {fastsolver_dir}")
    
    print(f"Running FastHenry for {fh_input}")
    zc_path = run_fasthenry(str(fh_input))
    if not Path(zc_path).is_file():
        raise RuntimeError(f"Zc.mat not found after running FastHenry for {fh_input}")
    print(f"FastHenry output found: {zc_path}")

    if not fc_input.is_file():
        raise FileNotFoundError(f"geometry_FastCap.txt missing in {fastsolver_dir}")

    print(f"Running FasterCap for {fc_input} (eps_r = {eps_r})")
    cap_path = run_fastercap(str(fc_input), eps_r=eps_r)
    if not Path(cap_path).is_file():
        raise RuntimeError(f"CapacitanceMatrix.txt not found after running FasterCap for {fc_input}")
    print(f"FasterCap output found: {cap_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch runner for FastHenry and FasterCap using a spirals.json file.")
    parser.add_argument("spirals_json_file", type=Path, help="Path to the spirals.json file.")
    parser.add_argument("eps_r", type=float, nargs="?", default=1.0, help="Relative permittivity (e.g., 3.5 for FR-4). Defaults to 1.0.")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    spirals_json_file: Path = args.spirals_json_file
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
    processed_count = 0
    skipped_count = 0

    for spiral in spirals:
        print(f"\n--- Processing: {spiral.get('name', 'N/A')} ---")
        status = spiral.get("status", {})
        
        # Skip if solver files weren't created successfully
        if not status.get("solver_files_created"):
            print("Skipping: Solver input files were not created successfully.")
            skipped_count += 1
            continue
        
        try:
            process_geometry_folder(spiral, eps_r=args.eps_r)
            status["simulation_run"] = True
            processed_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to solve {spiral.get('name', 'N/A')}: {e}")
            error_entry = create_error_entry(
                spiral_name=spiral.get('name', 'N/A'),
                spiral_path=spiral.get('path', Path()),
                message="Solver execution failed.",
                exc=e
            )
            errors.append(error_entry)
            status["simulation_run"] = False
        
        spiral["status"] = status

    print("\n--- Batch summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {len(errors)}")

    if errors:
        write_debug_log(__file__, errors, spirals_json_file.parent)
        print(f"Wrote {len(errors)} error(s) to a debug log in {spirals_json_file.parent}")

    try:
        save_spirals_data(spirals_json_file, project_name, spirals)
        print(f"Successfully updated status in {spirals_json_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save updated status to {spirals_json_file.name}: {e}")

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
