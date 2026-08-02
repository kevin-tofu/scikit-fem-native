from pathlib import Path


ROOT=Path(__file__).parents[1]


def contents(name):
    return (ROOT/".github"/"workflows"/name).read_text(encoding="utf-8")


def test_fast_ci_cancels_stale_runs_and_skips_result_only_changes():
    workflow=contents("ci.yml")
    assert "cancel-in-progress: true" in workflow
    assert '"benchmarks/**/results/**"' in workflow
    assert "ubuntu-24.04" in workflow
    assert "macos-15" in workflow
    assert "windows-2025" in workflow


def test_release_builds_every_supported_desktop_architecture():
    workflow=contents("workflow.yml")
    for value in (
        "linux-x86_64","macos-arm64","macos-x86_64","windows-amd64",
        "macos-15-intel","arch: AMD64",
    ):
        assert value in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow


def test_full_validation_is_manual_and_includes_wheel_smoke():
    workflow=contents("full-validation.yml")
    assert "workflow_dispatch:" in workflow
    assert "wheel-smoke:" in workflow
    assert "--python 3.14" in workflow
