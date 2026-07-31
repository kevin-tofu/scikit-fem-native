from pathlib import Path
import subprocess
import sys


SCRIPT=(
    Path(__file__).parents[1]
    /"benchmark"/"compare-with-skfem"/"poisson_assembly.py"
)


def test_poisson_benchmark_smoke(tmp_path):
    output=tmp_path/"poisson.csv"
    report=tmp_path/"poisson.md"
    result=subprocess.run(
        [
            sys.executable,str(SCRIPT),"--sizes","2","4",
            "--repeat","1","--warmup","1","--output",str(output),
            "--markdown-output",str(report),
        ],
        cwd=SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "K speedup" in result.stdout
    lines=output.read_text().splitlines()
    assert len(lines)==3
    assert lines[0].startswith("resolution,dofs,elements,")
    contents=report.read_text()
    assert contents.startswith("# skfn vs. scikit-fem")
    assert "Environment:" in contents
