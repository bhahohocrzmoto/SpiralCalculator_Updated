"""Interactive helper that batch-converts Wire_Sections.txt files for FastSolver tools.

The script asks the user for the common FastHenry / FastCap parameters and then
creates a ``FastSolver`` sub-folder next to every ``Wire_Sections.txt`` entry
listed inside an ``Address.txt`` file. Both FastHenry (``.inp``) and FastCap
(``_FastCap.txt``) files are generated for each entry.
"""

from __future__ import annotations

import argparse
import json
import datetime
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FastSolver.FastCap import WireSections_to_FastCap_txt as fastcap
from FastSolver.FastHenry import WireSections_to_FastHenry_inp as fasthenry
from main.debug_utils import write_debug_log, create_error_entry


DEFAULT_SEGMENT_WIDTH = 0.25
DEFAULT_SEGMENT_HEIGHT = 0.035
DEFAULT_SIGMA = 58_000.0
DEFAULT_FMIN = 1_000.0
DEFAULT_FMAX = 1_000_000.0
DEFAULT_NDEC = 1.0
DEFAULT_TRACE_WIDTH_MM = 0.25


@dataclass(slots=True)
class ConversionSettings:
    segment_width: float = DEFAULT_SEGMENT_WIDTH
    segment_height: float = DEFAULT_SEGMENT_HEIGHT
    sigma: float = DEFAULT_SIGMA
    fmin: float = DEFAULT_FMIN
    fmax: float = DEFAULT_FMAX
    ndec: float = DEFAULT_NDEC
    trace_width_mm: float = DEFAULT_TRACE_WIDTH_MM
    nhinc: int = 1
    nwinc: int = 1
    rh: float = 2.0
    rw: float = 2.0


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
    # Read the original file to preserve creation date and other metadata
    try:
        with open(spirals_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
    except Exception:
        output_data = {}

    output_data["project_name"] = project_name
    output_data["spirals"] = spirals

    # Convert Path objects to strings for serialization
    for spiral in output_data["spirals"]:
        if isinstance(spiral.get("path"), Path):
            spiral["path"] = str(spiral["path"])

    with open(spirals_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)


def _prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Please enter a valid number.")


def _prompt_path(label: str) -> Path:
    while True:
        raw = input(f"{label}: ").strip()
        if raw:
            return Path(raw.strip("\"").strip("'"))


def _gather_settings(args: argparse.Namespace) -> ConversionSettings:
    interactive = not args.non_interactive
    settings = ConversionSettings()

    def choose(value, default, label):
        if value is not None:
            return value
        if not interactive:
            return default
        return _prompt_float(label, default)

    settings.segment_width = choose(
        args.segment_width, settings.segment_width, "Segment width in mm"
    )
    settings.segment_height = choose(
        args.segment_height, settings.segment_height, "Segment height in mm"
    )
    settings.sigma = choose(args.sigma, settings.sigma, "Sigma (S/m)")
    settings.fmin = choose(args.fmin, settings.fmin, "fmin (Hz)")
    settings.fmax = choose(args.fmax, settings.fmax, "fmax (Hz)")
    settings.ndec = choose(args.ndec, settings.ndec, "Points per decade (ndec)")
    settings.trace_width_mm = choose(
        args.trace_width_mm, settings.trace_width_mm, "Trace width for FastCap (mm)"
    )
    return settings


def _convert_directory(spiral: Dict[str, Any], settings: ConversionSettings) -> None:
    """
    Converts a single spiral's Wire_Sections.txt into FastHenry and FastCap formats.
    Raises exceptions on failure.
    """
    directory = spiral.get("path")
    if not directory or not isinstance(directory, Path) or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found or invalid for spiral: {spiral.get('name', 'N/A')}")

    wire_sections = directory / "geometry.json"
    if not wire_sections.is_file():
        raise FileNotFoundError(f"'geometry.json' not found in {directory}")

    output_dir = directory / "FastSolver"
    output_dir.mkdir(exist_ok=True)

    fasthenry_output = output_dir / f"{wire_sections.stem}.inp"
    fastcap_output = output_dir / f"{wire_sections.stem}_FastCap.txt"

    units, metadata, sections = fasthenry.parse_wire_sections(wire_sections)
    fasthenry_output.write_text(
        fasthenry.build_inp_content(
            units=units,
            metadata=metadata,
            sections=sections,
            segment_width=settings.segment_width,
            segment_height=settings.segment_height,
            sigma=settings.sigma,
            nhinc=settings.nhinc,
            nwinc=settings.nwinc,
            rh=settings.rh,
            rw=settings.rw,
            freq_min=settings.fmin,
            freq_max=settings.fmax,
            ndec=settings.ndec,
        )
    )

    _, cap_sections = fastcap.parse_wire_sections(wire_sections)
    fastcap.write_fastcap_file(fastcap_output, cap_sections, settings.trace_width_mm)

    print(f"[OK] Converted {directory.name} -> {fasthenry_output.name}, {fastcap_output.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch converter for FastHenry/FastCap from a spirals.json file."
    )
    parser.add_argument(
        "spirals_json_file",
        nargs="?",
        type=Path,
        help="Path to spirals.json file.",
    )
    parser.add_argument("--segment-width", type=float, help="Segment width (mm)")
    parser.add_argument("--segment-height", type=float, help="Segment height (mm)")
    parser.add_argument("--sigma", type=float, help="Conductivity sigma (S/m)")
    parser.add_argument("--fmin", type=float, help="Minimum frequency (Hz)")
    parser.add_argument("--fmax", type=float, help="Maximum frequency (Hz)")
    parser.add_argument("--ndec", type=float, help="Points per decade")
    parser.add_argument(
        "--trace-width-mm", type=float, help="Trace width for FastCap panels (mm)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use defaults/CLI options without prompting.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    spirals_json_file: Path
    if args.spirals_json_file is None:
        if args.non_interactive:
            raise SystemExit("spirals.json file must be provided when --non-interactive is used.")
        spirals_json_file = _prompt_path("Path to spirals.json")
    else:
        spirals_json_file = args.spirals_json_file

    if not spirals_json_file.is_file():
        raise SystemExit(f"Spirals JSON file not found: {spirals_json_file}")

    try:
        project_name, spirals = load_spirals_data(spirals_json_file)
        print(f"Loaded {len(spirals)} spirals from '{project_name}'.")
    except Exception as e:
        raise SystemExit(f"Failed to load or parse {spirals_json_file}: {e}")

    settings = _gather_settings(args)
    errors: List[Dict] = []

    for spiral in spirals:
        try:
            _convert_directory(spiral, settings)
            # Update status on success
            if "status" not in spiral: spiral["status"] = {}
            spiral["status"]["solver_files_created"] = True
        except Exception as e:
            print(f"[ERROR] Failed to convert {spiral.get('name', 'N/A')}: {e}")
            # Log error for debug file
            error_entry = create_error_entry(
                spiral_name=spiral.get('name', 'N/A'),
                spiral_path=spiral.get('path', Path()),
                message=f"Failed during solver file conversion.",
                exc=e
            )
            errors.append(error_entry)
            # Update status on failure
            if "status" not in spiral: spiral["status"] = {}
            spiral["status"]["solver_files_created"] = False

    # Write a debug log if any errors occurred
    if errors:
        write_debug_log(__file__, errors, spirals_json_file.parent)
        print(f"\nCompleted with {len(errors)} error(s). See debug log for details.")
    else:
        print("\nCompleted successfully.")

    # Always save the updated statuses back to the file
    try:
        save_spirals_data(spirals_json_file, project_name, spirals)
        print(f"Updated status in {spirals_json_file.name}")
    except Exception as e:
        print(f"[ERROR] Could not save updated status to {spirals_json_file.name}: {e}")

if __name__ == "__main__":
    main()

