from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_scikit_fem_gallery_examples_match():
    root=Path(__file__).parents[1]
    result=subprocess.run(
        [sys.executable,"examples/scikit-fem-gallery/run_all.py"],
        cwd=root,capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert result.stdout.count("matrix=")==3
