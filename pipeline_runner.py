"""
pipeline_runner.py — Master Pipeline Entry Point

Usage:
    python pipeline_runner.py            # clean + train models
    python pipeline_runner.py --app      # clean + train + launch Streamlit
"""
import time, sys, subprocess
from clean import run_cleaning
from model import run_models

def run_pipeline():
    print("=" * 55)
    print("  Fraud Detection Pipeline")
    print("=" * 55)
    t0 = time.time()

    print("\n[Step 1] Cleaning data...")
    clean_results = run_cleaning()

    print("\n[Step 2] Training ML models...")
    model_results = run_models()

    elapsed = round(time.time() - t0, 1)
    print("\n" + "=" * 55)
    print(f"  Pipeline complete in {elapsed}s ✓")
    loans = clean_results.get("loans")
    txns  = clean_results.get("transactions")
    if loans is not None:
        print(f"  Loans      : {len(loans):,} rows cleaned")
    if txns is not None:
        print(f"  Transactions: {len(txns):,} rows cleaned")
    print("  Models     : Loan Fraud | Txn Fraud | Customer Risk")
    print("=" * 55)

if __name__ == "__main__":
    run_pipeline()
    if "--app" in sys.argv:
        print("\nLaunching Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "app.py"])
