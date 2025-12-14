This is a comprehensive project for the design, simulation, and analysis of multi-layer spiral inductors and transformers for power electronics applications. The workflow is orchestrated through a central graphical user interface (GUI) and is composed of several distinct modules, each responsible for a specific part of the process.

### Workflow Overview

The project follows a five-step workflow, managed by the main GUI (`main/SpiralsMain.py`):

1.  **Geometry Generation:** A batch of spiral variants is generated based on user-defined parameters.
2.  **Solver Preparation:** The generated geometries are converted into input files for the electromagnetic solvers.
3.  **Solver Execution:** The FastHenry and FasterCap solvers are run to calculate the impedance and capacitance matrices of the spirals.
4.  **Result Processing:** The raw solver outputs are processed to calculate reduced multi-port parameters.
5.  **Analysis and Visualization:** Key Performance Indicators (KPIs) are calculated and visualized to compare the performance of the different spiral designs.

### Directory and Script Descriptions

Here is a breakdown of the main directories and the purpose of each script:

#### `main/`

*   `SpiralsMain.py`: The central `tkinter` GUI that orchestrates the entire workflow. It provides a user-friendly interface to guide the user through the five steps of the process.
*   `debug_utils.py`: A utility module that provides functions for writing detailed, timestamped error logs in JSON format. This is used by other scripts to ensure robust error tracking.

#### `SpiralGeometryGeneration/`

*   `Spiral_Batch_Variants_UI_16.11.2025.py`: A `tkinter` GUI for generating a batch of spiral variants. The user can define sweeps for parameters like the number of arms (`K_arms`) and turns (`N_turns`) on a per-layer basis. It uses `Spiral_Drawer_updated.py` to do the actual geometry generation and creates an `Address.txt` file that lists all the generated variants.
*   `Spiral_Drawer_updated.py`: The core engine for generating the spiral geometry. It contains the mathematical functions to create multi-layer, multi-arm Archimedean spirals and writes the geometry to `geometry.json` files. It also has its own GUI for interactive visualization of a single spiral. Notably, in this script, each arm is treated as a separate "Section", which is a key detail for the solver.
*   `Optional TXT to DXF/json2dxf_sections.py`: A utility to convert `geometry.json` files into DXF format, for use in CAD software.

#### `FastSolver/`

This directory contains the scripts for automating the FastHenry and FasterCap electromagnetic solvers.

*   **`Automation/`**
    *   `fast_solver_batch_ui.py`: A command-line tool that prepares the input files for the solvers. It reads the `spirals.json` file and, for each `geometry.json`, it generates a FastHenry input file (`.inp`) and a FasterCap input file (`_FastCap.txt`).
    *   `automate_solvers.py`: This script runs the FastHenry and FasterCap solvers on the generated input files. It uses COM automation to control the solvers on Windows and updates the `status` of each spiral in the `spirals.json` file.

*   **`FastCap/`**
    *   `WireSections_to_FastCap.py`: Converts a `Wire_Sections.txt` file (containing wire geometry) into a format that the FasterCap solver can understand (`coil_cap.txt`), representing the spiral traces as quadrilateral panels.
    *   `run_fastercap.py`: Automates the execution of the FasterCap solver using COM automation, retrieves the resulting capacitance matrix, and saves it to `CapacitanceMatrix.txt`.

*   **`FastHenry/`**
    *   `WireSections_to_FastHenry_inp.py`: Converts a `geometry.json` file into a FastHenry input file (`.inp`), defining the spiral geometry as a series of nodes and segments.
    *   `run_fasthenry.py`: Automates the execution of the FastHenry solver using COM automation. FastHenry then produces a `Zc.mat` file containing the frequency-dependent impedance matrix.

*   **`PlotGeneration/`**
    *   `PlotGeneration.py`: This script processes the raw solver outputs. It reads the `Zc.mat` and `CapacitanceMatrix.txt` files, along with a user-defined port configuration, and performs matrix reduction to calculate per-port parameters like inductance, resistance, and Q-factor. The results are saved in JSON files.
    *   `PlotGeometry.py`: A utility to read a `geometry.json` file and generate a 3D plot for visual inspection.
    *   `MatrixJsonReader.py`: A utility to convert the JSON matrix files into human-readable Excel spreadsheets.

#### `BatchAnalysis/`

*   `design_analyzer.py`: This script is for analyzing the spiral variants as transformers. It reads the processed JSON files, calculates transformer-specific KPIs (like coupling coefficient), and generates a CSV summary and a Pareto plot of coupling coefficient vs. quality factor.
*   `inductor_analyzer.py`: This script is for analyzing the spiral variants as inductors. It calculates inductor-specific KPIs (like effective inductance and SRF) and generates CSV summaries and performance plots of inductance vs. quality factor for both series and parallel-connected inductors.

#### `Bigpicture_Calculator/`

*   `unit_system_calculator.py`: A command-line tool for performing system-level electrical calculations for a modular power converter system. It helps to determine the required voltage, current, and power specifications for the spiral components.
*   `unit_system_calculator_PLOT.py`: An extension of the above script that adds `matplotlib`-based plotting capabilities to visualize the trade-offs in system design choices.

#### `KidCad_PCB_Generation/`

*   `kicad_import_wire_sections_plugin.py`: A plugin for the KiCad PCB design software that can import `geometry.json` files directly into a KiCad layout. It supports automatic mapping of different Z-levels to different copper layers.

### Utilities

*   `summarize_log.py`: A command-line tool to read a `PlotGeneration_Debug.json` log file and print a summarized, human-readable count of the different event types that occurred. This is useful for quickly diagnosing issues during a batch analysis.

This project provides a complete and powerful toolchain for the automated design, simulation, and optimization of spiral inductors and transformers, from high-level system specifications down to detailed electromagnetic analysis and PCB layout integration.
