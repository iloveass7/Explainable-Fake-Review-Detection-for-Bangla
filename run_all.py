#!/usr/bin/env python3
"""
Master runner for Explainable Fake Review Detection for Bangla
Executes all 6 weeks sequentially on Kaggle.
"""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_data_acquisition_audit.ipynb",
    "02_preprocessing_eda.ipynb", 
    "03_feature_engineering.ipynb",
    "04_gold_only_training.ipynb",
    "05_pseudo_labeling.ipynb",
    "06_final_evaluation.ipynb"
]

def run_notebook(nb_path, timeout=3600):
    """Execute a Jupyter notebook via nbconvert."""
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--execute", "--to", "notebook",
        "--output", nb_path,
        "--ExecutePreprocessor.timeout", str(timeout),
        nb_path
    ]
    print(f"\n{'='*60}")
    print(f"Running: {nb_path}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {nb_path} completed successfully")
        return True
    else:
        print(f"✗ {nb_path} FAILED")
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return False

def main():
    base_dir = Path(__file__).parent
    
    # Install requirements first
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                   capture_output=True)
    
    # Run each notebook
    failed = []
    for nb in NOTEBOOKS:
        nb_path = base_dir / nb
        if not nb_path.exists():
            print(f"WARNING: {nb} not found, skipping")
            continue
            
        success = run_notebook(str(nb_path))
        if not success:
            failed.append(nb)
            print(f"Stopping pipeline due to failure in {nb}")
            break
    
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("ALL NOTEBOOKS COMPLETED SUCCESSFULLY!")
        print(f"Results in: {base_dir / 'results'}")
        print(f"Models in: {base_dir / 'models'}")
        print(f"Reports in: {base_dir / 'report'}")

if __name__ == "__main__":
    main()