#!/usr/bin/env python3
"""
Central orchestration GUI for spiral generation, solver automation, and plotting.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter

import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

REPO_ROOT = Path(__file__).resolve().parents[1]
SPIRAL_UI = REPO_ROOT / "SpiralGeometryGeneration" / "Spiral_Batch_Variants_UI_16.11.2025.py"
AUTOMATE = REPO_ROOT / "FastSolver" / "Automation" / "automate_solvers.py"
PLOT_GEN = REPO_ROOT / "FastSolver" / "PlotGeneration" / "PlotGeneration.py"
ANALYSIS_SCRIPT = REPO_ROOT / "BatchAnalysis" / "design_analyzer.py"
INDUCTOR_ANALYSIS_SCRIPT = REPO_ROOT / "BatchAnalysis" / "inductor_analyzer.py"

sys.path.insert(0, str(REPO_ROOT))
from FastSolver.PlotGeneration import PlotGeneration as PG
from FastSolver.PlotGeneration import PlotGeometry
from FastSolver.Automation import fast_solver_batch_ui
from main.debug_utils import write_debug_log, create_error_entry


PHASE_LETTERS = ("A", "B", "C")

def load_spirals_data(spirals_file: Path) -> List[Dict]:
    """Reads, validates, and resolves paths from a spirals.json file."""
    if not spirals_file.is_file():
        raise FileNotFoundError(f"Spirals JSON file not found: {spirals_file}")

    with open(spirals_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "spirals" not in data or not isinstance(data["spirals"], list):
        raise ValueError("JSON is missing top-level 'spirals' list.")

    # Resolve paths to be absolute
    for spiral in data["spirals"]:
        p = Path(spiral.get("path", ""))
        if not p.is_absolute():
            p = spirals_file.parent / p
        spiral["path"] = p.resolve()
    
    return data["spirals"]

def save_spirals_data(spirals_file: Path, project_name: str, spirals: List[Dict]):
    """Saves the updated list of spirals back to the JSON file."""
    # This is a simplified save; a more robust one might merge changes
    output_data = {
        "project_name": project_name,
        "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spirals": spirals
    }
    # Note: path objects need to be converted to strings for JSON serialization
    for spiral in output_data["spirals"]:
        if isinstance(spiral.get("path"), Path):
            spiral["path"] = str(spiral["path"])

    with open(spirals_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

def parse_spiral_folder_name(name: str) -> List[Dict[str, object]]:
    """Extract layer metadata (layer index, K, direction) from a folder name."""
    matches = list(re.finditer(r"L(?P<layer>\d+)_K(?P<K>\d+)_N[^_]+_(?P<dir>CW|CCW)", name))
    info: List[Dict[str, object]] = []
    offset = 0
    for m in matches:
        layer_idx, k, direction = int(m.group("layer")), int(m.group("K")), m.group("dir")
        info.append({"layer": layer_idx, "K": k, "direction": direction, "start": offset})
        offset += k
    return info

def build_sign_vector(active_indices: Sequence[int], total: int) -> List[float]:
    """Build a raw +1 sign vector for the selected conductors."""
    signs = [0.0] * total
    for idx in active_indices:
        if 0 <= idx < total:
            signs[idx] = 1.0
    return signs

def log_subprocess(cmd: List[str], log_widget: tk.Text) -> bool:
    """Runs a command and logs its output to a text widget."""
    log_widget.insert("end", f"\n$ {' '.join(cmd)}\n")
    log_widget.see("end")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        if proc.stdout: log_widget.insert("end", proc.stdout)
        if proc.stderr: log_widget.insert("end", proc.stderr)
        log_widget.see("end")
        return True
    except subprocess.CalledProcessError as exc:
        log_widget.insert("end", exc.stdout or "")
        log_widget.insert("end", exc.stderr or "")
        log_widget.insert("end", f"Command failed: {exc}\n")
        log_widget.see("end")
        messagebox.showerror("Command failed", f"{cmd[0]} exited with status {exc.returncode}")
        return False

class PortsPopup(tk.Toplevel):
    """A popup window for configuring port analyses."""
    def __init__(self, master: tk.Tk, spirals_file: Path, log_widget: tk.Text):
        super().__init__(master)
        self.title("PlotGeneration configuration")
        self.spirals_file = spirals_file
        self.log = log_widget
        self.geometry("1020x640")
        self.transient(master)
        self.grab_set()
        self.spirals_data = self._load_spiral_data()
        self.spiral_paths = [Path(s["path"]) for s in self.spirals_data]
        self.layer_cache: Dict[Path, List[Dict[str, object]]] = {}
        self._build_ui()

    def _log_plot_event(self, log_path: Path, spiral_name: str, stage: str, status: str, detail: str):
        """Appends a structured entry to a JSON log file."""
        entry = {"spiral": spiral_name, "stage": stage, "status": status, "detail": detail}
        try:
            data = json.loads(log_path.read_text(encoding='utf-8')) if log_path.exists() else []
            if not isinstance(data, list): data = []
            data.append(entry)
            log_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            # Fallback to main log if JSON logging fails
            self.log.insert("end", f"Failed to write to debug log: {e}\n")

    def _load_spiral_data(self) -> List[Dict]:
        try:
            spirals = load_spirals_data(self.spirals_file)
        except Exception as exc:
            messagebox.showerror("Spirals JSON read error", str(exc), parent=self)
            error = create_error_entry("N/A", self.spirals_file, "Failed to load spirals.json", exc)
            write_debug_log(__file__, [error], self.spirals_file.parent)
            return []
        
        existing = [s for s in spirals if Path(s.get("path", "")).exists()]
        if not existing:
            messagebox.showwarning("No folders", "No valid, existing folders found in spirals.json.", parent=self)
        return existing

    def _build_ui(self):
        left = ttk.Frame(self); left.pack(side="left", fill="both", expand=False, padx=8, pady=8)
        ttk.Label(left, text="Spiral variations (from spirals.json)").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("name", "conductors"), show="headings", height=20)
        self.tree.heading("name", text="Folder"), self.tree.heading("conductors", text="# conductors")
        self.tree.column("name", width=420), self.tree.column("conductors", width=100, anchor="center")
        self.tree.pack(fill="both", expand=True)
        for spiral in self.spirals_data:
            path = Path(spiral["path"])
            self.tree.insert("", "end", iid=str(path), values=(spiral.get("name", path.name), self._count_conductors(path)))

        right = ttk.Frame(self); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        
        ind_frame = ttk.LabelFrame(right, text="Inductor analysis"); ind_frame.pack(fill="x", pady=6)
        self.var_enable_inductor = tk.BooleanVar(value=True)
        ttk.Checkbutton(ind_frame, text="Enable inductor analysis", variable=self.var_enable_inductor).pack(anchor="w", padx=6, pady=2)
        series_row = ttk.Frame(ind_frame); series_row.pack(fill="x", padx=6, pady=2)
        self.var_series = tk.BooleanVar(value=True)
        ttk.Checkbutton(series_row, text="Series (Port_all_Series)", variable=self.var_series).pack(side="left")
        self.var_parallel = tk.BooleanVar(value=True)
        ttk.Checkbutton(series_row, text="Parallel (Port_all_Parallel)", variable=self.var_parallel).pack(side="left", padx=12)

        tx_frame = ttk.LabelFrame(right, text="Transformer analysis"); tx_frame.pack(fill="x", pady=6)
        self.var_enable_tx = tk.BooleanVar(value=False)
        ttk.Checkbutton(tx_frame, text="Enable transformer analysis", variable=self.var_enable_tx).pack(anchor="w", padx=6, pady=2)
        row1 = ttk.Frame(tx_frame); row1.pack(fill="x", padx=6, pady=2)
        ttk.Label(row1, text="Primary layers (comma separated):").pack(side="left")
        self.var_primary_layers = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.var_primary_layers, width=18).pack(side="left", padx=4)
        row2 = ttk.Frame(tx_frame); row2.pack(fill="x", padx=6, pady=2)
        ttk.Label(row2, text="Secondary layers (comma separated):").pack(side="left")
        self.var_secondary_layers = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.var_secondary_layers, width=18).pack(side="left", padx=4)
        row3 = ttk.Frame(tx_frame); row3.pack(fill="x", padx=6, pady=2)
        ttk.Label(row3, text="Phases per side:").pack(side="left")
        self.var_phase_count = tk.StringVar(value="1")
        ttk.Combobox(row3, values=("1", "2", "3"), textvariable=self.var_phase_count, width=6, state="readonly").pack(side="left", padx=4)
        
        map_frame = ttk.Frame(tx_frame); map_frame.pack(fill="both", padx=6, pady=4)
        ttk.Label(map_frame, text="Optional custom port mapping (format: pA:0,6 | sA:3,9)").pack(anchor="w")
        self.var_custom_ports = tk.Text(map_frame, height=4); self.var_custom_ports.pack(fill="x", expand=True)
        
        self.summary = tk.Text(right, height=10); self.summary.pack(fill="both", expand=True, pady=(6, 0))
        self._refresh_summary()

        action = ttk.Frame(self); action.pack(fill="x", side="bottom", pady=8, padx=10)
        ttk.Button(action, text="Run PlotGeneration", command=self._run_plots).pack(side="right", padx=6)
        ttk.Button(action, text="Cancel", command=self.destroy).pack(side="right")

    def _count_conductors(self, path: Path) -> int:
        # Priority 1: Use the generated capacitance matrix if it exists.
        cap_path = path / "FastSolver" / "CapacitanceMatrix.txt"
        if cap_path.exists():
            try:
                return PG.load_capacitance_matrix(cap_path).shape[0]
            except Exception:
                pass  # Fall through to geometry files

        # Priority 2: Use the new geometry.json file.
        json_path = path / "geometry.json"
        if json_path.exists():
            try:
                with json_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                return len(data.get("sections", []))
            except Exception:
                pass # Fall through to legacy file

        # Priority 3: Fallback to legacy Wire_Sections.txt.
        wire_sections_path = path / "Wire_Sections.txt"
        if wire_sections_path.exists():
            try:
                # Correctly count unique sections instead of lines
                lines = wire_sections_path.read_text(encoding='utf-8', errors='ignore').splitlines()
                section_names = {line.split(',')[0].strip() for line in lines if line.strip().startswith("Section-")}
                return len(section_names)
            except Exception:
                return 0
        
        return 0 # No file found

    def _refresh_summary(self):
        self.summary.delete("1.0", "end")
        self.summary.insert("end", f"Inductor analysis: {'enabled' if self.var_enable_inductor.get() else 'disabled'}\n")
        if self.var_enable_inductor.get(): self.summary.insert("end", f"  - Series: {self.var_series.get()}, Parallel: {self.var_parallel.get()}\n")
        self.summary.insert("end", f"Transformer analysis: {'enabled' if self.var_enable_tx.get() else 'disabled'}\n")
        if self.var_enable_tx.get():
            self.summary.insert("end", f"  - Primary: {self.var_primary_layers.get() or '-'}\n  - Secondary: {self.var_secondary_layers.get() or '-'}\n  - Phases: {self.var_phase_count.get()}\n")
            custom = self.var_custom_ports.get("1.0", "end").strip()
            if custom: self.summary.insert("end", f"  - Custom ports: {custom}\n")
        self.summary.see("end")

    def _parse_layer_selection(self, raw: str) -> List[int]:
        return [int(token) for token in re.split(r"[;,\s]+", raw.strip()) if token.isdigit()]

    def _get_layers_info(self, path: Path) -> List[Dict[str, object]]:
        if path not in self.layer_cache:
            self.layer_cache[path] = parse_spiral_folder_name(path.name)
        return self.layer_cache[path]

    def _validate_series(self, layers: List[Dict[str, object]]) -> Tuple[bool, List[float]]:
        if not layers or len(layers) < 2: return False, []
        if any(int(layer["K"]) != 1 for layer in layers): return False, []
        directions = [str(layer["direction"]) for layer in layers]
        if any(directions[i] == directions[i+1] for i in range(len(directions) - 1)): return False, []
        
        total_k = sum(int(layer["K"]) for layer in layers)
        signs: List[float] = [0.0] * total_k
        dir_to_sign = {"CCW": 1.0, "CW": -1.0}
        for info in layers:
            signs[int(info["start"])] = dir_to_sign.get(str(info["direction"]), 1.0)
        return True, signs

    def _parse_custom_ports(self, text: str, total: int) -> Dict[str, Dict[str, object]]:
        ports = {}
        for line in text.replace("|", "\n").splitlines():
            if ":" not in line: continue
            name, raw_indices = line.split(":", 1)
            indices = [int(val) for val in re.split(r"[Kohls,\s]+", raw_indices.strip()) if val.isdigit()]
            ports[name.strip()] = {"type": "parallel", "signs": build_sign_vector(indices, total), "raw_indices": ",".join(map(str, indices))}
        return ports

    def _build_transformer_ports(
        self, layers: List[Dict[str, object]], primary_layers: List[int], secondary_layers: List[int], phase_count: int
    ) -> Optional[Dict[str, Dict[str, object]]]:
        if not primary_layers and not secondary_layers: return None
        layer_map = {int(info["layer"]): info for info in layers}
        if any(layer not in layer_map for layer in primary_layers + secondary_layers): return None

        def validate_phase_counts(selected: List[int]) -> bool:
            return all(layer_map.get(layer) and int(layer_map[layer]["K"]) % phase_count == 0 for layer in selected)

        if not validate_phase_counts(primary_layers) or not validate_phase_counts(secondary_layers): return None

        total = sum(int(info["K"]) for info in layers)
        custom_text = self.var_custom_ports.get("1.0", "end").strip()
        if custom_text: return self._parse_custom_ports(custom_text, total)

        ports = {}
        letters = PHASE_LETTERS[:phase_count]
        def collect_indices(selected_layers: List[int], phase_idx: int) -> List[int]:
            return [i for layer in selected_layers for i in range(int(layer_map[layer]["start"]) + phase_idx * (int(layer_map[layer]["K"]) // phase_count), int(layer_map[layer]["start"]) + (phase_idx + 1) * (int(layer_map[layer]["K"]) // phase_count))]

        for idx, letter in enumerate(letters):
            if primary_layers:
                p_indices = collect_indices(primary_layers, idx)
                ports[f"p{letter}"] = {"type": "parallel", "signs": build_sign_vector(p_indices, total), "raw_indices": ",".join(map(str, p_indices))}
            if secondary_layers:
                s_indices = collect_indices(secondary_layers, idx)
                ports[f"s{letter}"] = {"type": "parallel", "signs": build_sign_vector(s_indices, total), "raw_indices": ",".join(map(str, s_indices))}
        return ports

    def _run_plots(self):
        self._refresh_summary()
        if not self.spiral_paths:
            messagebox.showwarning("No folders", "No spiral folders were loaded.", parent=self)
            return

        enable_inductor, enable_tx = self.var_enable_inductor.get(), self.var_enable_tx.get()
        if not enable_inductor and not enable_tx:
            messagebox.showwarning("Nothing to run", "Select at least one analysis mode.", parent=self)
            return

        debug_log = self.spirals_file.parent / PG.DEBUG_LOG_NAME
        if debug_log.exists():
            debug_log.unlink()
        
        try:
            phase_count = int(self.var_phase_count.get())
        except ValueError:
            phase_count = 1
        primary_layers = self._parse_layer_selection(self.var_primary_layers.get())
        secondary_layers = self._parse_layer_selection(self.var_secondary_layers.get())

        records: List[Dict[str, object]] = []
        
        # Match spirals_data with spiral_paths
        spirals_map = {str(s['path']): s for s in self.spirals_data}

        for path in self.spiral_paths:
            spiral = spirals_map.get(str(path))
            if not spiral: continue

            layers = self._get_layers_info(path)
            total = sum(int(item["K"]) for item in layers)
            if not total:
                self._log_plot_event(debug_log, spiral_name=path.name, stage="precheck", status="FAILURE", detail="No conductors found")
                continue

            ports: Dict[str, Dict[str, object]] = {}
            if enable_inductor:
                if self.var_parallel.get():
                    if (path / "FastSolver" / "Zc.mat").exists() and (path / "FastSolver" / "CapacitanceMatrix.txt").exists():
                        indices = list(range(total))
                        ports["Port_all_Parallel"] = {"type": "parallel", "signs": build_sign_vector(indices, total), "raw_indices": ",".join(str(i) for i in indices)}
                    else:
                        self._log_plot_event(debug_log, spiral_name=path.name, stage="inductor_parallel", status="FAILURE", detail="Missing Zc.mat or CapacitanceMatrix.txt")
                
                if self.var_series.get():
                    ok_series, signs = self._validate_series(layers)
                    if ok_series:
                        ports["Port_all_Series"] = {"type": "series", "signs": signs, "raw_indices": ",".join(str(i) for i in range(total))}
                    else:
                        self._log_plot_event(debug_log, spiral_name=path.name, stage="inductor_series", status="FAILURE", detail="Series validation failed (K=1 per layer and alternating directions required)")

            if enable_tx:
                tx_ports = self._build_transformer_ports(layers, primary_layers, secondary_layers, phase_count)
                if tx_ports:
                    ports.update(tx_ports)
                else:
                    self._log_plot_event(debug_log, spiral_name=path.name, stage="transformer", status="FAILURE", detail="Transformer port validation failed")
            
            if not ports:
                self._log_plot_event(debug_log, spiral_name=path.name, stage="ports", status="FAILURE", detail="No valid port configurations were generated based on UI settings.")
                continue

            dirs = PG.ensure_analysis_dirs(path)
            system_type = "hybrid" if enable_inductor and enable_tx and len(ports) > 1 else ("transformer" if enable_tx else "inductor")
            dirs["ports_config"].write_text(json.dumps({"ports": ports, "system_type": system_type}, indent=2))
            
            try:
                # PG.process_spiral now takes the spiral dict
                PG.process_spiral(spiral, records)
                spiral.setdefault("status", {})["results_processed"] = True
                self._log_plot_event(debug_log, spiral_name=path.name, stage="processing", status="SUCCESS", detail="Matrix reduction complete.")
            except Exception as e:
                spiral.setdefault("status", {})["results_processed"] = False
                self._log_plot_event(debug_log, spiral_name=path.name, stage="processing", status="FAILURE", detail=str(e))


        # Save the updated statuses back to the main spirals.json file
        try:
            project_name = self.spirals_file.parent.name
            save_spirals_data(self.spirals_file, project_name, self.spirals_data)
            self.log.insert("end", f"Updated status in {self.spirals_file.name}\n")
        except Exception as e:
            self.log.insert("end", f"Failed to save status back to {self.spirals_file.name}: {e}\n")


        if records:
            self.log.insert("end", f"Plot generation complete. Processed {len(records)} ports across all designs.\n")
        else:
            self.log.insert("end", "Plot generation finished. No analyzable folders found.\n")
        self.log.see("end")
        self.destroy()

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spirals Main Panel")
        self.geometry("940x720")
        
        # --- Variables ---
        self.merged_spirals_json_path: Optional[Path] = None
        self.var_eps = tk.StringVar(value="3.5")
        self.var_matrix_json = tk.StringVar()
        self.var_analysis_freq = tk.StringVar()
        
        self.var_freq_min = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_FMIN))
        self.var_freq_max = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_FMAX))
        self.var_points_per_decade = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_NDEC))
        
        self.var_segment_width = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_SEGMENT_WIDTH))
        self.var_segment_height = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_SEGMENT_HEIGHT))
        self.var_sigma = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_SIGMA))
        self.var_trace_width_mm = tk.StringVar(value=str(fast_solver_batch_ui.DEFAULT_TRACE_WIDTH_MM))
        
        self.var_label_mode = tk.StringVar(value="hover")
        self.var_show_plot = tk.BooleanVar(value=False)
        
        self._build_ui()

    def _create_merged_spirals_json(self) -> Optional[Path]:
        paths = self._get_selected_json_paths()
        if not paths:
            messagebox.showerror("File missing", "Select at least one spirals.json file.")
            return None

        all_spirals = []
        project_name = "MergedProject"
        for path in paths:
            try:
                # We need to load the whole data to resolve paths relative to the original file
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                spirals = data.get("spirals", [])
                for spiral in spirals:
                    p = Path(spiral.get("path", ""))
                    if not p.is_absolute():
                        p = path.parent / p
                    spiral["path"] = str(p.resolve())
                all_spirals.extend(spirals)

            except Exception as exc:
                messagebox.showerror("Invalid spirals.json", f"Error in {path.name}:\n{exc}")
                return None

        if not all_spirals:
            messagebox.showwarning("No spirals", "No spirals found in the selected files.")
            return None

        merged_data = {
            "project_name": project_name,
            "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "spirals": all_spirals
        }

        try:
            temp_dir = tempfile.gettempdir()
            merged_file_path = Path(temp_dir) / "merged_spirals.json"
            with open(merged_file_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2)
            
            self.merged_spirals_json_path = merged_file_path
            return merged_file_path
        except Exception as exc:
            messagebox.showerror("Failed to create merged file", str(exc))
            return None

    def _run_pipeline(self):
        if not self._verify_spirals_json(): return
        
        merged_spirals_path = self._create_merged_spirals_json()
        if not merged_spirals_path:
            return

        eps = self.var_eps.get().strip() or "1"
        solver_args = self._collect_solver_args()
        if solver_args is None: return

        # Command for fast_solver_batch_ui.py
        batch_cmd = [
            sys.executable,
            str(REPO_ROOT / "FastSolver" / "Automation" / "fast_solver_batch_ui.py"),
            "--non-interactive",
            str(merged_spirals_path),
        ] + solver_args
        
        if log_subprocess(batch_cmd, self.log):
            # Command for automate_solvers.py
            automate_cmd = [sys.executable, str(AUTOMATE), str(merged_spirals_path), eps]
            if log_subprocess(automate_cmd, self.log):
                messagebox.showinfo("Solvers complete", "FastHenry/FasterCap runs finished.")
                self._open_ports_popup()

    def _build_ui(self):
        # Frame 1: Geometry Generation
        top = ttk.LabelFrame(self, text="1) Geometry generation")
        top.pack(fill="x", padx=10, pady=8)
        ttk.Label(top, text="Use the generator to create spirals and a spirals.json file.").pack(side="left", padx=6)
        ttk.Button(top, text="Open generator", command=self._launch_spiral_ui).pack(side="right", padx=6)

        # Frame 2: JSON & Solver Setup
        mid = ttk.LabelFrame(self, text="2) Project File & Solver Setup")
        mid.pack(fill="x", padx=10, pady=8)

        row = ttk.Frame(mid)
        row.pack(fill="x", pady=4, padx=6)
        ttk.Label(row, text="spirals.json files:").pack(side="left")
        
        self.spirals_listbox = tk.Listbox(row, width=80, height=5)
        self.spirals_listbox.pack(side="left", padx=6)

        btn_frame = ttk.Frame(row)
        btn_frame.pack(side="left")
        ttk.Button(btn_frame, text="Add Files...", command=self._add_spirals_json).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected_json).pack(fill="x", pady=2)
        ttk.Button(row, text="Verify", command=self._verify_spirals_json).pack(side="left", padx=4)

        eps_row = ttk.Frame(mid)
        eps_row.pack(fill="x", pady=4, padx=6)
        ttk.Label(eps_row, text="Permittivity (eps_r):").pack(side="left")
        ttk.Entry(eps_row, textvariable=self.var_eps, width=12).pack(side="left", padx=6)

        # Frame 3: Solve
        solver = ttk.LabelFrame(self, text="3) Solve")
        solver.pack(fill="x", padx=10, pady=8)
        
        freq_row = ttk.Frame(solver)
        freq_row.pack(fill="x", padx=6, pady=4)
        ttk.Label(freq_row, text="Frequency sweep (Hz):").pack(side="left")
        ttk.Entry(freq_row, textvariable=self.var_freq_min, width=14).pack(side="left", padx=(6, 2))
        ttk.Label(freq_row, text="to").pack(side="left")
        ttk.Entry(freq_row, textvariable=self.var_freq_max, width=14).pack(side="left", padx=(2, 6))
        ttk.Label(freq_row, text="Points/decade:").pack(side="left")
        ttk.Entry(freq_row, textvariable=self.var_points_per_decade, width=10).pack(side="left", padx=6)
        
        solver_settings_frame = ttk.LabelFrame(solver, text="Solver Settings")
        solver_settings_frame.pack(fill="x", padx=6, pady=6)
        
        row1 = ttk.Frame(solver_settings_frame)
        row1.pack(fill="x", padx=6, pady=2)
        ttk.Label(row1, text="Segment Width (mm):").pack(side="left")
        ttk.Entry(row1, textvariable=self.var_segment_width, width=10).pack(side="left", padx=4)
        ttk.Label(row1, text="Segment Height (mm):").pack(side="left", padx=10)
        ttk.Entry(row1, textvariable=self.var_segment_height, width=10).pack(side="left", padx=4)
        
        row2 = ttk.Frame(solver_settings_frame)
        row2.pack(fill="x", padx=6, pady=2)
        ttk.Label(row2, text="Sigma (S/m):").pack(side="left")
        ttk.Entry(row2, textvariable=self.var_sigma, width=10).pack(side="left", padx=4)
        ttk.Label(row2, text="Trace Width (mm, for FastCap):").pack(side="left", padx=10)
        ttk.Entry(row2, textvariable=self.var_trace_width_mm, width=10).pack(side="left", padx=4)

        ttk.Button(solver, text="Run conversion + solvers", command=self._run_pipeline).pack(side="left", padx=6, pady=6)
        ttk.Button(solver, text="Configure ports / plots", command=self._open_ports_popup).pack(side="left", padx=6)

        # Frame 4: Matrix Review
        viewer = ttk.LabelFrame(self, text="4) Matrix review")
        viewer.pack(fill="x", padx=10, pady=8)
        row_json = ttk.Frame(viewer)
        row_json.pack(fill="x", pady=4, padx=6)
        ttk.Label(row_json, text="Matrix JSON:").pack(side="left")
        ttk.Entry(row_json, textvariable=self.var_matrix_json, width=70).pack(side="left", padx=6)
        ttk.Button(row_json, text="Browse…", command=self._browse_matrix_json).pack(side="left")

        # Frame 5: Final Analysis
        analysis_frame = ttk.LabelFrame(self, text="5) Final Analysis")
        analysis_frame.pack(fill="x", padx=10, pady=8)
        
        # ... (rest of analysis UI is unchanged)
        freq_row_an = ttk.Frame(analysis_frame); freq_row_an.pack(fill="x", padx=6, pady=(4, 6))
        ttk.Label(freq_row_an, text="Analysis Frequency (Hz):").pack(side="left")
        ttk.Entry(freq_row_an, textvariable=self.var_analysis_freq, width=20).pack(side="left", padx=6)
        ttk.Label(freq_row_an, text="(leave empty for highest)").pack(side="left")
        label_row = ttk.Frame(analysis_frame); label_row.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(label_row, text="Design labels:").pack(side="left")
        ttk.Combobox(label_row, values=("hover", "static", "none"), textvariable=self.var_label_mode, width=10, state="readonly").pack(side="left", padx=4)
        ttk.Checkbutton(label_row, text="Open interactive plot window", variable=self.var_show_plot).pack(side="left", padx=12)
        button_frame = ttk.Frame(analysis_frame); button_frame.pack(fill="x", side="bottom", padx=6, pady=6)
        ttk.Button(button_frame, text="Finalize Inductor Analysis", command=self._run_inductor_analysis).pack(side="left", padx=6)
        ttk.Button(button_frame, text="Finalize Transformer Analysis", command=self._run_full_analysis).pack(side="right", padx=6)

        # Frame 6: Geometry Visualization
        vis_frame = ttk.LabelFrame(self, text="6) Geometry Utilities")
        vis_frame.pack(fill="x", padx=10, pady=8)
        ttk.Button(vis_frame, text="Plot Geometry from JSON", command=self._plot_geometry).pack(side="left", padx=6, pady=6)
        ttk.Button(vis_frame, text="Convert JSON to DXF", command=self._convert_json_to_dxf).pack(side="left", padx=6, pady=6)

        # Log Frame
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = tk.Text(log_frame, wrap="word")
        self.log.pack(fill="both", expand=True)

    def _convert_json_to_dxf(self):
        path = filedialog.askopenfilename(title="Select geometry.json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        
        dxf_script_path = REPO_ROOT / "SpiralGeometryGeneration" / "Optional TXT to DXF" / "json2dxf_sections.py"
        if not dxf_script_path.exists():
            messagebox.showerror("Missing script", f"Cannot find {dxf_script_path}")
            return

        try:
            # We can use log_subprocess here as it's a quick conversion
            cmd = [sys.executable, str(dxf_script_path), path]
            if log_subprocess(cmd, self.log):
                messagebox.showinfo("Conversion Complete", f"DXF file created successfully next to the source JSON.")
        except Exception as exc:
            messagebox.showerror("Conversion failed", str(exc))

    def _plot_geometry(self):
        path = filedialog.askopenfilename(title="Select geometry.json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        
        plot_script_path = REPO_ROOT / "FastSolver" / "PlotGeneration" / "PlotGeometry.py"
        if not plot_script_path.exists():
            messagebox.showerror("Missing script", f"Cannot find {plot_script_path}")
            return

        try:
            proc = subprocess.Popen([sys.executable, str(plot_script_path), path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            self.log.insert("end", f"Launched geometry plotter (pid {proc.pid}).\n")
            self.log.see("end")
            self.after(1200, lambda: self._check_proc(proc))
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))

    def _launch_spiral_ui(self):
        if not SPIRAL_UI.exists():
            messagebox.showerror("Missing script", f"Cannot find {SPIRAL_UI}")
            return
        try:
            proc = subprocess.Popen([sys.executable, str(SPIRAL_UI)], cwd=str(SPIRAL_UI.parent), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return
        self.log.insert("end", f"Launched spiral generator UI (pid {proc.pid}).\n")
        self.log.see("end")
        self.after(1200, lambda: self._check_proc(proc))

    def _check_proc(self, proc):
        if proc.poll() is None: return
        out, err = proc.communicate()
        if proc.returncode != 0: messagebox.showerror("Generator exited", err or out or f"Exited with status {proc.returncode}", parent=self)
        if out or err: self.log.insert("end", (out or "") + (err or "")); self.log.see("end")

    def _add_spirals_json(self):
        paths = filedialog.askopenfilenames(title="Select spirals.json files", filetypes=[("Spiral JSON", "*.json"), ("All files", "*.*")])
        for path in paths:
            if path not in self.spirals_listbox.get(0, "end"):
                self.spirals_listbox.insert("end", path)

    def _remove_selected_json(self):
        selected_indices = self.spirals_listbox.curselection()
        for i in reversed(selected_indices):
            self.spirals_listbox.delete(i)

    def _get_selected_json_paths(self) -> List[Path]:
        return [Path(p) for p in self.spirals_listbox.get(0, "end")]

    def _browse_matrix_json(self):
        path = filedialog.askopenfilename(title="Select matrix JSON", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path: self.var_matrix_json.set(path)

    def _verify_spirals_json(self) -> bool:
        paths = self._get_selected_json_paths()
        if not paths:
            messagebox.showerror("File missing", "Select at least one spirals.json file.")
            return False
        
        total_spirals = 0
        all_valid = True
        for path in paths:
            try:
                spirals = load_spirals_data(path)
                missing = [s for s in spirals if not s.get("path", Path()).exists()]
                if missing:
                    missing_names = [s.get('name', 'N/A') for s in missing]
                    messagebox.showwarning("Missing folders", f"In {path.name}, {len(missing)} folders do not exist:\n" + "\n".join(missing_names))
                    all_valid = False
                total_spirals += len(spirals)
            except Exception as exc:
                messagebox.showerror("Invalid spirals.json", f"Error in {path.name}:\n{exc}")
                error = create_error_entry("N/A", path, "Failed to verify spirals.json", exc)
                write_debug_log(__file__, [error], path.parent)
                all_valid = False

        if all_valid:
            messagebox.showinfo("JSON check", f"{total_spirals} spiral variants found and valid across {len(paths)} file(s).")
        
        return all_valid

    def _collect_solver_args(self) -> Optional[List[str]]:
        # This method remains unchanged
        args: List[str] = []
        for label, var, flag in (
            ("Minimum frequency", self.var_freq_min, "--fmin"),
            ("Maximum frequency", self.var_freq_max, "--fmax"),
            ("Points per decade", self.var_points_per_decade, "--ndec"),
            ("Segment width", self.var_segment_width, "--segment-width"),
            ("Segment height", self.var_segment_height, "--segment-height"),
            ("Sigma", self.var_sigma, "--sigma"),
            ("Trace width", self.var_trace_width_mm, "--trace-width-mm"),
        ):
            raw = var.get().strip()
            if not raw: continue
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                messagebox.showerror("Invalid input", f"{label} must be a number.")
                return None
            if value <= 0 and flag not in ["--segment-width", "--segment-height", "--sigma", "--trace-width-mm"]:
                messagebox.showerror("Invalid input", f"{label} must be greater than zero.")
                return None
            args.extend([flag, str(value)])
        return args



    def _display_log_summary(self, log_path: Path):
        # This method remains unchanged
        if not log_path.is_file(): return
        self.log.insert("end", f"\n\n--- Summary of Plot Generation ---\n")
        try:
            with open(log_path, 'r', encoding='utf-8') as f: logs = json.load(f)
            event_counter = Counter()
            for entry in logs:
                key = (entry.get('stage', 'u'), entry.get('status', 'u'), entry.get('detail', 'u'))
                event_counter[key] += 1
            if not event_counter: self.log.insert("end", "Log is empty.\n")
            else:
                for (stage, status, detail), count in sorted(event_counter.items()):
                    self.log.insert("end", f"\n- Event: [{stage}] / Status: [{status}]\n  Detail: {detail}\n  Occurrences: {count}\n")
        except Exception as e:
            self.log.insert("end", f"Could not read or parse summary log: {e}\n")
        self.log.insert("end", f"--- End of Summary ---\n")
        self.log.see("end")

    def _open_ports_popup(self):
        path_to_use = self.merged_spirals_json_path
        if not path_to_use or not path_to_use.is_file():
            messagebox.showwarning("JSON not found", "Please run the solvers first to create a merged spirals.json file.")
            return

        popup = PortsPopup(self, path_to_use, self.log)
        popup.wait_window()

        debug_log_path = path_to_use.parent / PG.DEBUG_LOG_NAME
        self._display_log_summary(debug_log_path)

    def _run_full_analysis(self):
        path_to_use = self.merged_spirals_json_path
        if not path_to_use or not path_to_use.is_file():
            messagebox.showerror("File missing", "Please run the solvers first to create a merged spirals.json file.")
            return

        freq = self.var_analysis_freq.get().strip()
        if not ANALYSIS_SCRIPT.exists():
            messagebox.showerror("Missing script", f"Cannot find {ANALYSIS_SCRIPT}")
            return
        
        cmd = [sys.executable, str(ANALYSIS_SCRIPT), str(path_to_use), "--label-mode", self.var_label_mode.get()]
        if freq: cmd.extend(["--frequency", freq])
        if self.var_show_plot.get(): cmd.append("--show-plot")
        
        if log_subprocess(cmd, self.log):
            messagebox.showinfo("Analysis Complete", "Transformer KPI analysis finished. Check 'FinalTransformerAnalysis' folder.")

    def _run_inductor_analysis(self):
        path_to_use = self.merged_spirals_json_path
        if not path_to_use or not path_to_use.is_file():
            messagebox.showerror("File missing", "Please run the solvers first to create a merged spirals.json file.")
            return

        freq = self.var_analysis_freq.get().strip()
        if not INDUCTOR_ANALYSIS_SCRIPT.exists():
            messagebox.showerror("Missing script", f"Cannot find {INDUCTOR_ANALYSIS_SCRIPT}")
            return
            
        cmd = [sys.executable, str(INDUCTOR_ANALYSIS_SCRIPT), str(path_to_use), "--label-mode", self.var_label_mode.get()]
        if freq: cmd.extend(["--frequency", freq])
        if self.var_show_plot.get(): cmd.append("--show-plot")
        
        if log_subprocess(cmd, self.log):
            messagebox.showinfo("Analysis Complete", "Inductor KPI analysis finished. Check 'FinalInductorAnalysis' folder.")

def main():
    app = MainApp()
    app.mainloop()

if __name__ == "__main__":
    main()