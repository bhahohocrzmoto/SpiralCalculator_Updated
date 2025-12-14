
import json
import datetime
import traceback
from pathlib import Path
from typing import List, Dict, Any

def write_debug_log(script_name: str, errors: List[Dict[str, Any]], output_dir: Path):
    """
    Writes a list of errors to a timestamped debug JSON file.

    Args:
        script_name: The name of the script generating the error (__file__).
        errors: A list of dictionaries, where each dictionary represents an error.
        output_dir: The directory where the debug log should be saved.
    """
    if not errors:
        return

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Create a unique filename to avoid overwriting logs
    log_file_name = f"{Path(script_name).stem}_debug_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    log_file = output_dir / log_file_name
    
    log_data = {
        "script": Path(script_name).name,
        "timestamp": timestamp,
        "errors": errors
    }
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Wrote {len(errors)} error(s) to {log_file}")

def create_error_entry(spiral_name: str, spiral_path: Path, message: str, exc: Exception = None) -> Dict[str, Any]:
    """Creates a standardized dictionary for an error entry."""
    entry = {
        "spiral_name": spiral_name,
        "spiral_path": str(spiral_path),
        "error_message": message,
    }
    if exc:
        entry["exception"] = str(exc)
        entry["traceback"] = traceback.format_exc()
    return entry
