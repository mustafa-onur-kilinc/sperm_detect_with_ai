"""
Utility function to create a new subdirectory
under runs/train, runs/detect, runs/eval
for recording information about those specific processes seperately for different runs.
"""

import os
from pathlib import Path


def get_next_run_directory(base_path):
    # Ensure the base path exists
    Path(base_path).mkdir(parents=True, exist_ok=True)

    # List all existing runs
    existing_runs = [
        d
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d)) and d.startswith("run_")
    ]

    # Find the highest existing run number
    highest_run = 0
    for run in existing_runs:
        try:
            run_number = int(run.split("_")[1])
            highest_run = max(highest_run, run_number)
        except ValueError:
            # Skip directories that do not follow the expected naming convention
            continue

    # Define the next run directory
    next_run_number = highest_run + 1
    next_run_dir = os.path.join(base_path, f"run_{next_run_number}")

    return next_run_dir
