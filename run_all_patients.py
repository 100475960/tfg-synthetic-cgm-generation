"""
Multi-patient orchestrator for the generalisation validation experiment.

Executes timegan_multipatient.ipynb once per (patient, configuration) combination
by injecting parameters into the cell tagged 'parameters' before each run.
Each execution runs in a fresh, independent Python kernel to prevent state leakage
between patients.

REQUIREMENTS:
    pip install nbformat nbclient jupyter ipykernel

CHECKPOINT/RESUME:
    Before launching each combination, the script checks whether
    ResultadosGenerativos/TimeGAN_condicional/patient_{ID}/run_{RUN_NAME}/config.json
    already exists. If it does, the run is assumed complete and skipped.
    This allows interrupting the script and relaunching it without repeating
    completed work.

    If a run is interrupted mid-execution (after config.json has been
    written but before training finishes), the script will incorrectly skip it
    on relaunch. Delete the corresponding run_* folder manually to force re-execution.

OUTPUT:
    - One executed .ipynb per combination, saved to executed_notebooks/
    - A text log (orchestrator_log.txt) with progress and errors
    - Run results in their standard folder (ResultadosGenerativos/.../run_{RUN_NAME}/)
"""

import argparse
import copy
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Batch configuration 
NOTEBOOK_PATH = Path("timegan_multipatient.ipynb")
EXECUTED_DIR  = Path("executed_notebooks")
LOG_PATH      = Path("orchestrator_log.txt")

# OhioT1DM patient IDs
ALL_PATIENTS = [
    "559", "563", "570", "575", "588", "591",   # release 2018
    "540", "544", "552", "567", "584", "596",   # release 2020
]

# Configurations to validate 
ALL_CONFIGS = {
    "norm":          "tanh",   # VAE-Wiener baseline
    "recovery_none": "none",   # recovery without tanh activation
}

# Per-cell timeout in seconds (generous margin for slow hardware)
CELL_TIMEOUT_SECONDS = 7200

# Results base path (must match OUTPUT_BASE as built inside the notebook)
RESULTS_BASE = Path("ResultadosGenerativos") / "TimeGAN_condicional"
WORKING_DIR = Path(__file__).resolve().parent


def build_run_name(patient_id: str, config_suffix: str) -> str:
    """Builds the run name, matching the f-string in the notebook's config cell."""
    return f"vae_wgan_wiener02_{config_suffix}_patient{patient_id}"


def run_already_done(patient_id: str, config_suffix: str) -> bool:
    """Returns True if config.json already exists for this combination."""
    run_name    = build_run_name(patient_id, config_suffix)
    config_path = RESULTS_BASE / f"patient_{patient_id}" / f"run_{run_name}" / "config.json"
    return config_path.exists()


def inject_parameters(nb, patient_id: str, recovery_activation: str, run_name_suffix: str):
    """Returns a deep copy of the notebook with the 'parameters' cell overwritten."""
    nb_copy = copy.deepcopy(nb)

    param_source = (
        f'PATIENT_ID_PARAM           = "{patient_id}"\n'
        f'RECOVERY_ACTIVATION_PARAM  = "{recovery_activation}"\n'
        f'RUN_NAME_SUFFIX_PARAM      = "{run_name_suffix}"\n'
    )

    found = False
    for cell in nb_copy.cells:
        if cell.get("cell_type") == "code" and "parameters" in cell.get("metadata", {}).get("tags", []):
            cell["source"] = param_source
            found = True
            break

    if not found:
        raise RuntimeError(
            "No cell with tag 'parameters' found in the notebook. "
            "Verify that timegan_multipatient.ipynb has the parameters cell correctly tagged."
        )

    return nb_copy


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one_combination(nb_template, patient_id: str, config_suffix: str, recovery_activation: str):
    run_name = build_run_name(patient_id, config_suffix)

    if run_already_done(patient_id, config_suffix):
        log(f"SKIP   patient={patient_id} config={config_suffix} — config.json already exists")
        return "skipped"

    log(f"START  patient={patient_id} config={config_suffix} (RECOVERY_ACTIVATION={recovery_activation})")
    t0 = time.time()

    nb_run = inject_parameters(nb_template, patient_id, recovery_activation, config_suffix)

    client = NotebookClient(
        nb_run,
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(WORKING_DIR)}},
    )

    try:
        client.execute()
        status = "ok"
    except CellExecutionError as e:
        status = "failed"
        log(f"ERROR  patient={patient_id} config={config_suffix} — cell execution error:")
        log(str(e))
    except Exception as e:
        status = "failed"
        log(f"ERROR  patient={patient_id} config={config_suffix} — unexpected exception: {e}")
        log(traceback.format_exc())

    # Save the executed notebook (with outputs) regardless of outcome 
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXECUTED_DIR / f"{run_name}.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb_run, f)

    elapsed_min = (time.time() - t0) / 60
    log(f"{'DONE ' if status == 'ok' else 'FAIL '} patient={patient_id} config={config_suffix} "
        f"— {status} in {elapsed_min:.1f} min. Executed notebook: {out_path}")

    return status


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patients", nargs="+", default=ALL_PATIENTS,
        help="Patient IDs to process (default: all 12 OhioT1DM patients)"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(ALL_CONFIGS.keys()),
        help=f"Configurations to run (default: both — {list(ALL_CONFIGS.keys())})"
    )
    args = parser.parse_args()

    if not NOTEBOOK_PATH.exists():
        print(f"ERROR: {NOTEBOOK_PATH} not found. Run this script from the folder "
              f"containing the notebook, or update NOTEBOOK_PATH.")
        sys.exit(1)

    for cfg in args.configs:
        if cfg not in ALL_CONFIGS:
            print(f"ERROR: unknown config '{cfg}'. Valid options: {list(ALL_CONFIGS.keys())}")
            sys.exit(1)

    n_combinations = len(args.patients) * len(args.configs)
    log("=" * 70)
    log(f"Starting batch: {len(args.patients)} patients x {len(args.configs)} configs "
        f"= {n_combinations} combinations")
    log("=" * 70)

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_template = nbformat.read(f, as_version=4)

    results = {"ok": 0, "failed": 0, "skipped": 0}

    for patient_id in args.patients:
        for config_suffix in args.configs:
            recovery_activation = ALL_CONFIGS[config_suffix]
            status = run_one_combination(nb_template, patient_id, config_suffix, recovery_activation)
            results[status if status in results else "failed"] += 1

    log("=" * 70)
    log(f"Batch complete. OK={results['ok']}  FAILED={results['failed']}  SKIPPED={results['skipped']}")
    if results["failed"] > 0:
        log("Some combinations failed. Check orchestrator_log.txt and the corresponding "
            "notebooks in executed_notebooks/ for details.")
    log("=" * 70)


if __name__ == "__main__":
    main()
