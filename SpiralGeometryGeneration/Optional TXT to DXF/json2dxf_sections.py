#!/usr/bin/env python3
"""
json2dxf_sections.py

Reads a geometry.json file with the following structure:
{
  "units": "mm",
  "sections": [
    {
      "name": "Section-1",
      "z_level": 0.0,
      "vertices": [
        {"x": 25.5, "y": 0.0},
        ...
      ]
    },
    ...
  ]
}

Outputs a DXF with one polyline per Section.
If `ezdxf` is installed, we use it and set $INSUNITS.
If not, we write a minimal R12 DXF by hand (no dependencies).
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
import json
from typing import Dict, List, Tuple

# ----------------------------- Parsing helpers --------------------------------

SectionName = str
Point3D = Tuple[float, float, float]


def parse_geometry_file(path: Path) -> Tuple[str, Dict[str, List[Point3D]]]:
    """Parses the geometry.json file to extract units and section data."""
    if not path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {path}")

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    unit = data.get("units", "mm")
    sections_data = data.get("sections", [])
    if not sections_data: 
        raise ValueError("No section data found in geometry.json")

    sections: Dict[str, List[Point3D]] = {}
    for sec_data in sections_data:
        sec_name = sec_data.get("name")
        if not sec_name: continue
        z = sec_data.get("z_level", 0.0)
        sections[sec_name] = [(v.get('x', 0.0), v.get('y', 0.0), z) for v in sec_data.get("vertices", [])]
    
    return unit, sections


def any_nonzero_z(points: List[Point3D]) -> bool:
    """Return True if any vertex has |z| > tiny threshold."""
    return any(abs(p[2]) > 1e-15 for p in points)

# ------------------------------ DXF writers -----------------------------------
# ( DXF writer functions remain unchanged )
def write_with_ezdxf(
    out_path: Path,
    sections: Dict[SectionName, List[Point3D]],
    unit: str,
) -> None:
    """
    Write the DXF using ezdxf if available.
    - Sets $INSUNITS so downstream CAD knows the unit.
    - Uses LWPOLYLINE for 2D sections (all z==0) and POLYLINE3D when z≠0 exists.
    """
    import ezdxf  # type: ignore

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    insunits_map = {"in": 1, "ft": 2, "mi": 3, "mm": 4, "cm": 5, "m": 6, "km": 7}
    doc.header["$INSUNITS"] = insunits_map.get(unit, 4)

    for sec_name, pts in sections.items():
        layer = sec_name
        if layer not in doc.layers:
            doc.layers.add(layer)

        if any_nonzero_z(pts):
            msp.add_polyline3d(points=pts, dxfattribs={"layer": layer})
        else:
            xy = [(x, y) for x, y, _ in pts]
            msp.add_lwpolyline(xy, format="xy", dxfattribs={"layer": layer, "const_width": 0.0})

    try:
        all_xy = [(x, y) for pts in sections.values() for (x, y, _) in pts]
        minx = min(p[0] for p in all_xy)
        miny = min(p[1] for p in all_xy)
        msp.add_mtext(f"Unit: {unit.upper()}  •  Sections: {len(sections)}").set_location((minx, miny))
    except Exception:
        pass

    doc.saveas(out_path.as_posix())


def write_r12_minimal(
    out_path: Path,
    sections: Dict[SectionName, List[Point3D]],
    unit: str,
) -> None:
    """Minimalist, dependency-free **DXF R12** writer."""
    try:
        all_xy = [(x, y) for pts in sections.values() for (x, y, _) in pts]
        minx, miny = min(p[0] for p in all_xy), min(p[1] for p in all_xy)
    except Exception:
        minx, miny = (0.0, 0.0)

    chunks: List[str] = ["0\nSECTION\n2\nHEADER\n0\nENDSEC\n", "0\nSECTION\n2\nTABLES\n"]
    chunks.append(f"0\nTABLE\n2\nLAYER\n70\n{len(sections)}\n")
    for layer_name in sections.keys():
        chunks.append(f"0\nLAYER\n2\n{layer_name}\n70\n0\n62\n7\n6\nCONTINUOUS\n")
    chunks.append("0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n")
    chunks.append(f"0\nTEXT\n8\nNOTES\n10\n{minx}\n20\n{miny}\n30\n0\n40\n3.5\n1\nUnit: {unit.upper()}  Sections: {len(sections)}\n")

    for layer_name, pts in sections.items():
        if len(pts) < 2: continue
        chunks.append(f"0\nPOLYLINE\n8\n{layer_name}\n66\n1\n70\n8\n")
        for (x, y, z) in pts:
            chunks.append(f"0\nVERTEX\n8\n{layer_name}\n10\n{x}\n20\n{y}\n30\n{z}\n")
        chunks.append("0\nSEQEND\n")

    chunks.append("0\nENDSEC\n0\nEOF\n")
    out_path.write_text("".join(chunks), encoding="ascii")


# ------------------------------ Main program ----------------------------------

def main():
    if len(sys.argv) > 1:
        in_path = Path(sys.argv[1]).expanduser()
    else:
        # Default to geometry.json in the current folder
        in_path = Path("geometry.json")

    if not in_path.is_file():
        # If a directory is passed, look for geometry.json inside it
        if in_path.is_dir():
            in_path = in_path / "geometry.json"
        if not in_path.is_file():
            sys.stderr.write(f"[error] Input file not found: {in_path}\n")
            sys.stderr.write("Usage: python json2dxf_sections.py /path/to/geometry.json\n")
            sys.exit(1)

    try:
        unit, sections = parse_geometry_file(in_path)
    except Exception as e:
        sys.stderr.write(f"[error] Failed to parse geometry file: {e}\n")
        sys.exit(1)

    out_path = in_path.with_suffix(".dxf")
    try:
        import importlib.util
        has_ezdxf = importlib.util.find_spec("ezdxf") is not None
    except Exception:
        has_ezdxf = False

    if has_ezdxf:
        print(f"[info] Using ezdxf writer (unit={unit}).")
        write_with_ezdxf(out_path, sections, unit)
    else:
        print(f"[info] ezdxf not found -> writing minimal R12 DXF.")
        write_r12_minimal(out_path, sections, unit)

    n_pts = sum(len(v) for v in sections.values())
    print(f"[ok] Wrote {out_path}  •  sections={len(sections)}  •  vertices={n_pts}  •  unit={unit}")



if __name__ == "__main__":
    main()
