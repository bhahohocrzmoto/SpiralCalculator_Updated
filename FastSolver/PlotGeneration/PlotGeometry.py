#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlotGeometry.py
===============
A simple tool to visualize a spiral geometry from a 'geometry.json' file.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_geometry(json_path: Path):
    """
    Reads a geometry.json file and generates a 3D plot of the spiral.
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sections = data.get("sections", [])
    if not sections:
        print("No sections found in the geometry file.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for section in sections:
        vertices = section.get("vertices", [])
        z = section.get("z_level", 0.0)
        if not vertices:
            continue

        x_coords = [v['x'] for v in vertices]
        y_coords = [v['y'] for v in vertices]
        z_coords = [z] * len(vertices)
        
        ax.plot(x_coords, y_coords, z_coords, label=section.get("name"))

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Geometry Plot")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 3D spiral geometry from a geometry.json file.")
    parser.add_argument("json_file", type=Path, help="Path to the geometry.json file.")
    args = parser.parse_args()
    plot_geometry(args.json_file)
