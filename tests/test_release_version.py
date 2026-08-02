from pathlib import Path
import re
import subprocess
import sys


SCRIPT=Path(__file__).parents[1]/"tools"/"check_release_version.py"


def project_version():
    text=(SCRIPT.parents[1]/"pyproject.toml").read_text(encoding="utf-8")
    match=re.search(
        r'(?ms)^\[project\]\s*$.*?^version\s*=\s*["\']([^"\']+)["\']',
        text,
    )
    assert match is not None
    return match.group(1)
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
