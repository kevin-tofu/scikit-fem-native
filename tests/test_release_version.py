from pathlib import Path
import subprocess
import sys


SCRIPT=Path(__file__).parents[1]/"tools"/"check_release_version.py"


def test_release_version_accepts_optional_v_prefix():
    for tag in ("0.1.0","v0.1.0"):
        result=subprocess.run(
            [sys.executable,str(SCRIPT),tag],
            cwd=SCRIPT.parents[1],capture_output=True,text=True,
        )
        assert result.returncode==0,result.stderr


def test_release_version_rejects_mismatch():
    result=subprocess.run(
        [sys.executable,str(SCRIPT),"v9.9.9"],
        cwd=SCRIPT.parents[1],capture_output=True,text=True,
    )
    assert result.returncode!=0
    assert "does not match" in result.stderr
