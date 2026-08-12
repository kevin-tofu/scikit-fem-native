from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "memory_preflight_calibration.py"


def test_memory_calibration_runs_case_in_isolated_worker_and_writes_csv(tmp_path: Path):
    output = tmp_path / "memory.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--families", "tet4",
            "--targets", "100",
            "--component-counts", "1",
            "--memory-budget-gib", "1",
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["family"] == "tet4"
    assert row["status"] == "measured"
    assert int(row["actual_dofs"]) > 0
    assert int(row["actual_nnz"]) > 0
    assert float(row["nnz_upper_ratio"]) >= 1.0
    assert int(row["rss_peak"]) >= int(row["rss_before_basis"])
