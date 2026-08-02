from pathlib import Path
import subprocess
import sys


SCRIPT=Path(__file__).parents[1]/"scripts"/"upgrade_version.py"


def _project(tmp_path):
    path=tmp_path/"pyproject.toml"
    path.write_text(
        '[build-system]\nrequires = []\n\n'
        '[project]\nname = "example"\nversion = "0.1.0"\n\n'
        '[tool.example]\nversion = "unchanged"\n',
        encoding="utf-8",
    )
    return path


def test_upgrade_version_changes_only_project_version(tmp_path):
    project=_project(tmp_path)
    result=subprocess.run(
        [sys.executable,str(SCRIPT),"0.2.0","--pyproject",str(project)],
        capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "0.1.0 -> 0.2.0" in result.stdout
    text=project.read_text(encoding="utf-8")
    assert 'version = "0.2.0"' in text
    assert 'version = "unchanged"' in text


def test_upgrade_version_dry_run_does_not_write(tmp_path):
    project=_project(tmp_path);before=project.read_bytes()
    result=subprocess.run(
        [
            sys.executable,str(SCRIPT),"0.2.0rc1",
            "--pyproject",str(project),"--dry-run",
        ],
        capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert project.read_bytes()==before


def test_upgrade_version_rejects_invalid_or_unchanged_version(tmp_path):
    project=_project(tmp_path)
    for version in ("v0.2.0","0.1.0"):
        result=subprocess.run(
            [sys.executable,str(SCRIPT),version,"--pyproject",str(project)],
            capture_output=True,text=True,
        )
        assert result.returncode!=0
        assert project.read_text(encoding="utf-8").count(
            'version = "0.1.0"'
        )==1
