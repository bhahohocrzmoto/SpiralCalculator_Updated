#!/usr/bin/env python3
"""
Convert a Wire_Sections.txt file (spiral sections) to a FastCap generic geometry file.

- Input:  Wire_Sections.txt  (like the one exported by your Spiral Drawer / EM workflow)
- Output: FastCap file with quadrilateral panels "Q cond_id x1 y1 z1 ... x4 y4 z4"

ASSUMPTIONS:
-----------
1) First non-empty line is the unit string ("mm", "cm", "m", ...).
   We convert coordinates to meters for FastCap.
2) Second line is a header with parameters (vol_res_cm, coil_res_cm, ...), ignored here.
3) Each following line has the form:
       Section-1,x,y,z,something
   where:
       - "Section-<n>" identifies the conductor (continuous trace)
       - x,y,z are coordinates in the given units (e.g. mm)
       - the last value ("something") is ignored (radius etc. for other tools)
4) Each "Section-*" is a planar trace at constant z. We only build a top surface.
5) The user specifies a constant trace width [mm] for all sections.

USAGE:
------
    python wire_sections_to_fastcap.py Wire_Sections.txt coil_fastcap.txt --width-mm 0.35

If --width-mm is omitted, the script will ask for it interactively.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Type aliases for clarity
Point3D = Tuple[float, float, float]  # (x, y, z) in meters
SectionDict = Dict[str, List[Point3D]]


def detect_length_scale(unit_line: str) -> float:
    """Detect the length scale factor to get meters."""
    u = unit_line.strip().lower()
    if "mm" in u: return 1e-3
    if "cm" in u: return 1e-2
    if "m" == u or "meter" in u or "metre" in u: return 1.0
    print(f"WARNING: Unknown unit '{unit_line.strip()}'. Assuming meters.")
    return 1.0


def _parse_wire_sections_txt(path: Path) -> Tuple[float, SectionDict]:
    """Parse the legacy Wire_Sections.txt file."""
    sections: SectionDict = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError("Input file is empty or only whitespace.")

    unit_line = lines[0]
    scale_to_m = detect_length_scale(unit_line)
    data_lines = lines[2:]

    for ln in data_lines:
        if not ln.startswith("Section-"): continue
        parts = ln.split(",")
        if len(parts) < 4: continue
        sec_name = parts[0].strip()
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        sections.setdefault(sec_name, []).append((x * scale_to_m, y * scale_to_m, z * scale_to_m))

    if not sections:
        raise ValueError("No 'Section-*' lines were found in the file.")
    return scale_to_m, sections

def _parse_geometry_json(path_to_json: Path) -> Tuple[float, SectionDict]:
    """Parse the new geometry.json file."""
    with open(path_to_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    units = data.get("units", "mm")
    scale_to_m = detect_length_scale(units)

    sections_data = data.get("sections", [])
    if not sections_data:
        raise ValueError("No section data found in geometry.json file.")

    sections: SectionDict = {}
    for sec_data in sections_data:
        sec_name = sec_data.get("name")
        if not sec_name: continue
        
        # Apply scaling at the point of creation
        z_level_m = sec_data.get("z_level", 0.0) * scale_to_m
        
        points: List[Point3D] = [
            (v.get('x', 0.0) * scale_to_m, v.get('y', 0.0) * scale_to_m, z_level_m)
            for v in sec_data.get("vertices", [])
        ]
        sections[sec_name] = points

    return scale_to_m, sections

def parse_wire_sections(path: Path) -> Tuple[float, SectionDict]:
    """
    Dispatcher: Parses geometry from either geometry.json or Wire_Sections.txt.
    'path' can be the path to the old Wire_Sections.txt or the directory containing it.
    """
    directory = path.parent if path.is_file() else path
    geometry_json_path = directory / "geometry.json"
    wire_sections_txt_path = directory / "Wire_Sections.txt"
    
    if geometry_json_path.exists():
        return _parse_geometry_json(geometry_json_path)
    elif wire_sections_txt_path.is_file():
        return _parse_wire_sections_txt(wire_sections_txt_path)
    elif path.is_file() and path.name == "Wire_Sections.txt":
        return _parse_wire_sections_txt(path)
    else:
        raise FileNotFoundError(f"No geometry file found in {directory}")


def build_panels_for_section(points: List[Point3D], width_m: float) -> List[List[Point3D]]:
    """
    Build a list of quadrilateral panels approximating a trace from its centerline points.

    Each pair of consecutive points defines one straight segment. For each segment we build
    a rectangle of width 'width_m' centered on that segment, lying (approximately) in the
    same z-plane (planar assumption).

    Parameters
    ----------
    points : list of (x,y,z) in meters
        Centerline points for one Section, in the order they appear in the file.
    width_m : float
        Trace width in meters.

    Returns
    -------
    panels : list of panels
        Each panel is a list of 4 points [P1, P2, P3, P4], each a (x,y,z) tuple.
    """
    panels: List[List[Point3D]] = []
    if len(points) < 2:
        # Not enough points to define any segment
        return panels

    half_w = width_m / 2.0

    for i in range(len(points) - 1):
        x0, y0, z0 = points[i]
        x1, y1, z1 = points[i + 1]

        # Direction vector of the segment (centerline)
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        # For planar PCB traces, z should be (almost) constant per section.
        # We primarily use dx, dy to build a sideways vector.
        seg_len_xy = math.hypot(dx, dy)  # sqrt(dx^2 + dy^2)

        if seg_len_xy < 1e-15:
            # Points are identical or extremely close in XY; skip
            continue

        # Sideways unit vector in XY-plane, perpendicular to the segment direction:
        #   For segment direction (dx, dy), a perpendicular is (-dy, dx).
        ux = -dy / seg_len_xy
        uy = dx / seg_len_xy
        uz = 0.0  # stay in the same z-plane

        # Scale to half the trace width
        ux *= half_w
        uy *= half_w

        # Build 4 corner points for the quadrilateral panel
        # You can choose the order; here we maintain a consistent winding.
        p1 = (x0 + ux, y0 + uy, z0)  # one side at start
        p2 = (x1 + ux, y1 + uy, z1)  # same side at end
        p3 = (x1 - ux, y1 - uy, z1)  # opposite side at end
        p4 = (x0 - ux, y0 - uy, z0)  # opposite side at start

        panels.append([p1, p2, p3, p4])

    return panels


def write_fastcap_file(
    out_path: Path,
    sections: SectionDict,
    trace_width_mm: float,
    title: str = "PCB traces from Wire_Sections"
) -> None:
    """
    Create a FastCap generic geometry file from parsed sections.

    Parameters
    ----------
    out_path : Path
        Output file path for FastCap.
    sections : dict
        { section_name: [ (x,y,z)_meters, ... ] }
    trace_width_mm : float
        Trace width in millimeters.
    title : str
        Title line for the FastCap file.
    """
    width_m = trace_width_mm * 1e-3  # mm -> m

    # Map each section_name to a numeric conductor ID (1,2,3,...)
    section_names = sorted(sections.keys())
    sec_to_id = {name: idx + 1 for idx, name in enumerate(section_names)}

    with out_path.open("w", encoding="utf-8") as f:
        # First line: title, must start with "0"
        f.write(f"0 {title}\n")

        # Optional: add comments mapping conductor IDs to section names
        # (FastCap treats lines starting with '*' as comments)
        f.write(f"* Trace width = {trace_width_mm} mm\n")
        for name, cid in sec_to_id.items():
            f.write(f"* Conductor {cid} corresponds to {name}\n")

        # Now write panels for each section
        for sec_name in section_names:
            cid = sec_to_id[sec_name]
            points = sections[sec_name]

            panels = build_panels_for_section(points, width_m)
            if not panels:
                print(f"WARNING: no panels built for {sec_name} (too few or degenerate points).")
                continue

            for panel in panels:
                # panel is [p1, p2, p3, p4]
                (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = panel

                # FastCap generic quad line:
                #   Q <conductor_id> x1 y1 z1  x2 y2 z2  x3 y3 z3  x4 y4 z4
                # We'll use scientific notation for robustness.
                f.write(
                    "Q {cid}  "
                    "{x1:.8e} {y1:.8e} {z1:.8e}  "
                    "{x2:.8e} {y2:.8e} {z2:.8e}  "
                    "{x3:.8e} {y3:.8e} {z3:.8e}  "
                    "{x4:.8e} {y4:.8e} {z4:.8e}\n".format(
                        cid=cid,
                        x1=x1, y1=y1, z1=z1,
                        x2=x2, y2=y2, z2=z2,
                        x3=x3, y3=y3, z3=z3,
                        x4=x4, y4=y4, z4=z4,
                    )
                )


def main():
    ap = argparse.ArgumentParser(
        description="Convert Wire_Sections.txt (spiral sections) to FastCap geometry."
    )
    ap.add_argument(
        "input",
        type=str,
        help="Path to Wire_Sections.txt"
    )
    ap.add_argument(
        "output",
        type=str,
        help="Output FastCap geometry file (e.g. coil_cap.txt)"
    )
    ap.add_argument(
        "--width-mm",
        type=float,
        default=None,
        help="Trace width in millimeters (if omitted, you will be prompted)."
    )

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")

    # If width not specified on command line, ask the user
    if args.width_mm is None:
        while True:
            try:
                w_str = input("Enter trace width in mm: ").strip()
                width_mm = float(w_str.replace(",", "."))
                break
            except ValueError:
                print("Please enter a valid number (e.g. 0.35)")
    else:
        width_mm = args.width_mm

    print(f"Reading sections from {in_path} ...")
    scale_to_m, sections = parse_wire_sections(in_path)
    print(f"Detected length scale: 1 input unit = {scale_to_m} m")

    print(f"Building panels with trace width = {width_mm} mm ...")
    write_fastcap_file(out_path, sections, width_mm)

    print(f"Done. FastCap geometry written to {out_path}")


if __name__ == "__main__":
    main()
