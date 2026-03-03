"""
Modal app for running v8 prediction model training on Modal (GPU) or locally (CPU).

================================================================================
PRESCRIPTIVE SETUP
================================================================================

1. LOCAL (CPU) — no Modal, runs on your machine
   --------------------------------------------
   pip install -r requirements.txt
   # Place Orders.csv, Payment_History.csv, Test_OrderIDs.csv in project root
   python _run_v8_experiments.py
   python payjoy_model_v8.py
   # Outputs: v8_experiments_*.csv, submission_v8.csv in project root

2. MODAL (GPU) — one-time setup, then run remotely
   ------------------------------------------------
   pip install modal
   modal token set   # or set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET

   # One-time: create volume and upload data (from project root)
   modal volume create payjoy-v8-data
   modal volume put payjoy-v8-data Orders.csv Orders.csv
   modal volume put payjoy-v8-data Payment_History.csv Payment_History.csv
   modal volume put payjoy-v8-data Test_OrderIDs.csv Test_OrderIDs.csv

   # Run (GPU T4, V8_USE_GPU=1 for CatBoost/LightGBM/XGBoost)
   modal run modal_v8.py
   modal run modal_v8.py --grid-search --grid-mode lgb_only --output-suffix lgb

   # Production with best config from experiments
   modal run modal_v8.py --mode production --config-id lgb_only --output-suffix best

   # v5 GPU turbo (CatBoost + LightGBM + XGBoost ensemble)
   modal run modal_v8.py --mode v5_turbo --output-suffix v5_turbo

   # v9 clean build (v5 entity rates + PMT_CORE + v8 Cat+LGB)
   modal run modal_v8.py --mode v9 --output-suffix v9

   # Download results
   modal volume get payjoy-v8-data output/v8_experiments_results.csv ./v8_experiments_results.csv

3. PARALLEL MODAL RUNS — use --output-suffix to avoid overwriting
   ---------------------------------------------------------------
   modal run modal_v8.py --grid-search --grid-mode lgb_only --output-suffix lgb
   modal run modal_v8.py --grid-search --grid-mode cat_only --output-suffix cat
   # Download: output_lgb/, output_cat/

4. GPU vs CPU — Modal uses T4 GPU; local uses CPU
   ------------------------------------------------
   Modal sets V8_USE_GPU=1 so CatBoost, LightGBM, XGBoost use GPU.
   Local runs (python _run_v8_experiments.py) use CPU by default.

5. 60-90 MIN RUNS (detached, no terminal dependency)
   -------------------------------------------------
   ./run_modal_final.sh   # config_sweep + cat + lgb_subset + v5_turbo, all --detach
   ./run_modal_experiments.sh  # full 4-run grid, all --detach
   ./download_modal_outputs.sh --force  # when ready
   python extract_best_config.py  # get best config_id for production
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("payjoy-v8-training")

# Volume for data (input) and outputs
volume = modal.Volume.from_name("payjoy-v8-data", create_if_missing=True)
DATA_MOUNT = "/data"
WORKSPACE = "/workspace"


def _output_dir(suffix: str | None) -> str:
    """Output directory for this run. Use suffix to avoid conflicts when running in parallel."""
    return f"/data/output_{suffix}" if suffix else "/data/output"

# Project root (where modal_v8.py lives)
PROJECT_ROOT = Path(__file__).resolve().parent

# Image: deps + project code (exclude large data files)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "pandas==2.2.0",
        "numpy==1.26.3",
        "scikit-learn==1.4.0",
        "xgboost==2.0.3",
        "lightgbm==4.6.0",
        "catboost>=1.2",
        "scipy>=1.10",
        "matplotlib",
        "seaborn",
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=WORKSPACE,
        ignore=[
            "Orders.csv",
            "Payment_History.csv",
            "*.csv",
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "venv",
        ],
    )
)


def _setup_workspace_and_run(cmd: list[str], output_files: list[str], output_dir: str) -> int:
    """Symlink data from volume, run command, copy outputs to volume."""
    os.makedirs(output_dir, exist_ok=True)

    for f in ["Orders.csv", "Payment_History.csv", "Test_OrderIDs.csv"]:
        src = f"{DATA_MOUNT}/{f}"
        dst = f"{WORKSPACE}/{f}"
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Data file not found: {src}. Upload first: modal volume put payjoy-v8-data {f} {f}"
            )
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(src, dst)

    os.chdir(WORKSPACE)
    sys.path.insert(0, WORKSPACE)

    result = subprocess.run(cmd, capture_output=False)

    for f in output_files:
        src = f"{WORKSPACE}/{f}"
        if os.path.exists(src):
            shutil.copy2(src, f"{output_dir}/{f}")

    volume.commit()
    return result.returncode


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    gpu="T4",
    timeout=7200,  # 2 hours for full grid
    memory=8192,  # 8GB for large datasets
    retries=1,  # Retry once on worker preemption
    env={"V8_USE_GPU": "1"},  # CatBoost/LightGBM/XGBoost use GPU
)
def run_experiments(
    grid_search: bool = False,
    grid_mode: str = "lgb_only",
    config_ids: str | None = None,
    config_range: str | None = None,
    output_suffix: str | None = None,
) -> dict:
    """Run v8 experiments. Data from /data, outputs to /data/output or /data/output_<suffix>."""
    output_dir = _output_dir(output_suffix)
    cmd = [sys.executable, "_run_v8_experiments.py"]
    if grid_search:
        cmd.extend(["--grid-search", "--grid-mode", grid_mode])
    if config_ids:
        cmd.extend(["--config-ids", config_ids])
    elif config_range:
        cmd.extend(["--range", config_range])

    code = _setup_workspace_and_run(
        cmd,
        ["v8_experiments_results.csv", "v8_experiments_params.csv", "v8_experiments.log"],
        output_dir,
    )
    return {"exit_code": code, "output_dir": output_dir}


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    gpu="T4",
    timeout=3600,  # 1 hour for production run
    memory=8192,
    env={"V8_USE_GPU": "1"},
)
def run_production(output_suffix: str | None = None, config_id: str | None = None) -> dict:
    """Run payjoy_model_v8.py for Kaggle submission. Pass config_id from experiments for best config."""
    output_dir = _output_dir(output_suffix)
    cmd = [sys.executable, "payjoy_model_v8.py"]
    if config_id:
        cmd.extend(["--config-id", config_id])
    code = _setup_workspace_and_run(
        cmd,
        ["submission_v8.csv", "run_v8.log"],
        output_dir,
    )
    return {"exit_code": code, "output_dir": output_dir}


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    gpu="T4",
    timeout=3600,  # 1 hour for v5 turbo run
    memory=8192,
    retries=1,
    env={"V8_USE_GPU": "1", "V5_LGB_USE_CPU": "1"},  # LightGBM CPU (Modal T4 has CUDA, not OpenCL)
)
def run_v5_turbo(output_suffix: str | None = "v5_turbo") -> dict:
    """Run v5_gpu_turbo.py (CatBoost + LightGBM + XGBoost GPU ensemble) for Kaggle submission."""
    output_dir = _output_dir(output_suffix)
    cmd = [sys.executable, "v5_gpu_turbo.py"]
    code = _setup_workspace_and_run(
        cmd,
        ["submission_v5_gpu_turbo.csv"],
        output_dir,
    )
    return {"exit_code": code, "output_dir": output_dir}


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    gpu="T4",
    timeout=1800,  # 30 min for v9 single run
    memory=8192,
    retries=1,
    env={"V9_USE_GPU": "1", "V8_USE_GPU": "1", "V5_LGB_USE_CPU": "1"},
)
def run_v9(output_suffix: str | None = "v9") -> dict:
    """Run payjoy_model_v9.py (v5 entity rates + PMT_CORE + v8 Cat+LGB) on GPU."""
    output_dir = _output_dir(output_suffix)
    cmd = [sys.executable, "payjoy_model_v9.py"]
    code = _setup_workspace_and_run(
        cmd,
        ["submission_v9.csv", "run_v9.log"],
        output_dir,
    )
    return {"exit_code": code, "output_dir": output_dir}


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    gpu="T4",
    timeout=1800,
    memory=8192,
    retries=1,
    env={"VIVIAN_DATA_PATH": ".", "V8_USE_GPU": "1", "V5_LGB_USE_CPU": "1"},
)
def run_vivian(output_suffix: str | None = "vivian") -> dict:
    """Run vivian_final_model.py on GPU."""
    output_dir = _output_dir(output_suffix)
    cmd = [sys.executable, "vivian_final_model.py"]
    code = _setup_workspace_and_run(
        cmd,
        ["submission_v5_mismatch.csv", "improved_submission.csv"],
        output_dir,
    )
    return {"exit_code": code, "output_dir": output_dir}


@app.local_entrypoint()
def main(
    mode: str = "experiments",
    grid_search: bool = False,
    grid_mode: str = "lgb_only",
    config_ids: str | None = None,
    config_range: str | None = None,
    output_suffix: str | None = None,
    config_id: str | None = None,
):
    """Entrypoint. Use --output-suffix for parallel runs to avoid overwriting outputs."""
    with modal.enable_output():
        if mode == "production":
            run_production.remote(output_suffix=output_suffix, config_id=config_id)
        elif mode == "v5_turbo":
            run_v5_turbo.remote(output_suffix=output_suffix or "v5_turbo")
        elif mode == "v9":
            run_v9.remote(output_suffix=output_suffix or "v9")
        elif mode == "vivian":
            run_vivian.remote(output_suffix=output_suffix or "vivian")
        else:
            run_experiments.remote(
                grid_search=grid_search,
                grid_mode=grid_mode,
                config_ids=config_ids,
                config_range=config_range,
                output_suffix=output_suffix,
            )
    out = f"output_{output_suffix}" if output_suffix else (
        "output_v5_turbo" if mode == "v5_turbo" else "output_v9" if mode == "v9"
        else "output_vivian" if mode == "vivian" else "output"
    )
    print(f"\nDone. Download outputs with:")
    if mode == "production":
        print(f"  modal volume get payjoy-v8-data {out}/submission_v8.csv ./submission_v8.csv")
        print(f"  modal volume get payjoy-v8-data {out}/run_v8.log ./run_v8.log")
    elif mode == "v5_turbo":
        print(f"  modal volume get payjoy-v8-data {out}/submission_v5_gpu_turbo.csv ./submission_v5_gpu_turbo.csv")
    elif mode == "v9":
        print(f"  modal volume get payjoy-v8-data {out}/submission_v9.csv ./submission_v9.csv")
        print(f"  modal volume get payjoy-v8-data {out}/run_v9.log ./run_v9.log")
    elif mode == "vivian":
        print(f"  modal volume get payjoy-v8-data {out}/submission_v5_mismatch.csv ./submission_vivian.csv")
        print(f"  modal volume get payjoy-v8-data {out}/improved_submission.csv ./improved_submission_vivian.csv")
    else:
        print(f"  modal volume get payjoy-v8-data {out}/v8_experiments_results.csv ./v8_experiments_results.csv")
        print(f"  modal volume get payjoy-v8-data {out}/v8_experiments_params.csv ./v8_experiments_params.csv")
        print(f"  modal volume get payjoy-v8-data {out}/v8_experiments.log ./v8_experiments.log")
