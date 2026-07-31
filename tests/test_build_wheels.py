from pathlib import Path
import subprocess
import sys


SCRIPT=Path(__file__).parents[1]/"tools"/"build_wheels.py"


def test_build_wheels_dry_run_selects_python_versions(tmp_path):
    result=subprocess.run(
        [
            sys.executable,str(SCRIPT),"--platform","linux",
            "--python","3.10","3.14","--output-dir",str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "--platform linux" in result.stdout
    assert "CIBW_BUILD=cp310-* cp314-*" in result.stdout
