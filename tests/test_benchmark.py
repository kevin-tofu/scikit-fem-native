from pathlib import Path
import subprocess
import sys


SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"compare-with-skfem"/"poisson_assembly.py"
)
OFFICIAL_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"compare-with-skfem"/"official_performance.py"
)
SUPERMESH_PARALLEL_SCRIPT=(
    Path(__file__).parents[1]/"benchmarks"/"supermesh_parallel.py"
)
CUT_ASSEMBLY_SCRIPT=(
    Path(__file__).parents[1]/"benchmarks"/"cutfem"/"cut_assembly.py"
)
IMPLICIT_CROSS_SCRIPT=(
    Path(__file__).parents[1]/"benchmarks"/"cutfem"/"implicit_cross.py"
)
HCURL_SCRIPT=(
    Path(__file__).parents[1]/"benchmarks"/"hcurl_tri_n1_assembly.py"
)


def test_hcurl_tri_n1_benchmark_smoke(tmp_path):
    output=tmp_path/"hcurl.csv"
    result=subprocess.run(
        [
            sys.executable,str(HCURL_SCRIPT),"--resolutions","2","4",
            "--repeat","1","--warmup","0","--output",str(output),
        ],
        cwd=HCURL_SCRIPT.parents[1],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "integrate% scatter%" in result.stdout
    lines=output.read_text().splitlines()
    assert len(lines)==3
    assert lines[0].startswith("resolution,elements,dofs,nnz,")


def test_cut_assembly_benchmark_smoke(tmp_path):
    output=tmp_path/"cut.csv";plot=tmp_path/"cut.png"
    result=subprocess.run(
        [
            sys.executable,str(CUT_ASSEMBLY_SCRIPT),
            "--resolution","3","--fractions",".5",
            "--intorders","1","2","--threads","2","--repeat","1",
            "--output",str(output),"--plot-output",str(plot),
        ],
        cwd=CUT_ASSEMBLY_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "assembly speedup" in result.stdout
    assert len(output.read_text().splitlines())==3
    assert plot.stat().st_size>10_000


def test_implicit_cross_benchmark_smoke(tmp_path):
    output=tmp_path/"cross.csv";plot=tmp_path/"cross.png"
    result=subprocess.run(
        [sys.executable,str(IMPLICIT_CROSS_SCRIPT),"--resolution","3",
         "--intorders","1","2","--threads","2","--repeat","1",
         "--output",str(output),"--plot-output",str(plot)],
        cwd=IMPLICIT_CROSS_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "oracle error" in result.stdout
    assert len(output.read_text().splitlines())==5
    assert plot.stat().st_size>10_000


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
    assert contents.startswith("# skfemntv vs. scikit-fem")
    assert "Environment:" in contents


def test_official_performance_benchmark_smoke(tmp_path):
    output=tmp_path/"official.csv"
    result=subprocess.run(
        [
            sys.executable,str(OFFICIAL_SCRIPT),"--k","6",
            "--output",str(output),
        ],
        cwd=OFFICIAL_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "skfemntv cold speedup" in result.stdout
    assert len(output.read_text().splitlines())==2


def test_supermesh_parallel_benchmark_smoke(tmp_path):
    output=tmp_path/"supermesh-parallel.csv"
    result=subprocess.run(
        [
            sys.executable,str(SUPERMESH_PARALLEL_SCRIPT),
            "--cells","4","--threads","1,2","--repeat","1",
            "--output",str(output),
        ],
        cwd=SUPERMESH_PARALLEL_SCRIPT.parents[1],
        capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "overlap_cells" in result.stdout
    assert "| coupling | 2 |" in result.stdout
    assert len(output.read_text().splitlines())==3
