from pathlib import Path
import subprocess
import sys
import tomllib


SCRIPT=Path(__file__).parents[1]/"tools"/"check_release_version.py"


def project_version():
    with (SCRIPT.parents[1]/"pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["version"]


def test_release_version_accepts_optional_v_prefix():
    version=project_version()
    for tag in (version,f"v{version}"):
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
