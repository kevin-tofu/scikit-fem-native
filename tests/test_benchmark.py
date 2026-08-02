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
NONLINEAR_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"nonlinear-assembly"/"neo_hookean.py"
)
J2_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"nonlinear-assembly"/"j2_plasticity.py"
)
J2_HISTORY_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"nonlinear-assembly"/"j2_history.py"
)
SLS_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"nonlinear-assembly"/"standard_linear_solid.py"
)
MESH_ORDER_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"nonlinear-assembly"/"mesh_order_sweep.py"
)
NATIVE_PARALLEL_SCRIPT=(
    Path(__file__).parents[1]
    /"benchmarks"/"nonlinear-assembly"/"native_parallel_scaling.py"
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


def test_nonlinear_assembly_benchmark_smoke(tmp_path):
    output=tmp_path/"neo.csv"
    result=subprocess.run(
        [
            sys.executable,str(NONLINEAR_SCRIPT),"--points","3",
            "--repeat","1","--output",str(output),
        ],
        cwd=NONLINEAR_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "parallel speedup" in result.stdout
    assert len(output.read_text().splitlines())==2


def test_nonlinear_hex_distorted_high_order_benchmark_smoke():
    result=subprocess.run(
        [
            sys.executable,str(NONLINEAR_SCRIPT),
            "--topology","hex","--intorder","4","--distorted",
            "--points","3","--repeat","1",
        ],
        cwd=NONLINEAR_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "| hex | 4 | yes |" in result.stdout


def test_nonlinear_mesh_order_benchmark_smoke(tmp_path):
    output=tmp_path/"mesh-order.csv"
    plot=tmp_path/"mesh-order.png"
    result=subprocess.run(
        [
            sys.executable,str(MESH_ORDER_SCRIPT),
            "--topology","wedge6","--intorder","2","--distorted",
            "--points","2","--repeat","1","--output",str(output),
            "--plot-output",str(plot),
        ],
        cwd=MESH_ORDER_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "skfemntv R t1/t2/t4" in result.stdout
    assert "| wedge6 | 24 |" in result.stdout
    assert len(output.read_text().splitlines())==2
    assert plot.stat().st_size>10_000


def test_native_parallel_scaling_benchmark_smoke(tmp_path):
    output=tmp_path/"native-parallel.csv"
    result=subprocess.run(
        [
            sys.executable,str(NATIVE_PARALLEL_SCRIPT),
            "--topology","hex8","--points","3","--repeat","1",
            "--output",str(output),
        ],
        cwd=NATIVE_PARALLEL_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "colors min/max/eligible" in result.stdout
    assert "| hex8 | 81 | 8 |" in result.stdout
    assert len(output.read_text().splitlines())==2


def test_j2_benchmark_smoke(tmp_path):
    output=tmp_path/"j2.csv"
    result=subprocess.run(
        [
            sys.executable,str(J2_SCRIPT),"--topology","hex",
            "--points","3","--repeat","1","--native-threads","2",
            "--output",str(output),
        ],
        cwd=J2_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "skfem/skfemntv Nt" in result.stdout
    assert "| hex | 2 |" in result.stdout
    assert len(output.read_text().splitlines())==2


def test_j2_history_benchmark_smoke(tmp_path):
    output=tmp_path/"j2-history.csv"
    result=subprocess.run(
        [
            sys.executable,str(J2_HISTORY_SCRIPT),"--points","3",
            "--repeat","1","--native-threads","2","--output",str(output),
        ],
        cwd=J2_HISTORY_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "skfem/skfemntv Nt" in result.stdout
    assert "| tet | 81 |" in result.stdout
    assert len(output.read_text().splitlines())==2


def test_standard_linear_solid_benchmark_smoke(tmp_path):
    output=tmp_path/"sls.csv"
    result=subprocess.run(
        [
            sys.executable,str(SLS_SCRIPT),"--points","3","--repeat","1",
            "--native-threads","2","--output",str(output),
        ],
        cwd=SLS_SCRIPT.parents[2],capture_output=True,text=True,
    )
    assert result.returncode==0,result.stderr
    assert "skfem/skfemntv Nt" in result.stdout
    assert "| 81 |" in result.stdout
    assert len(output.read_text().splitlines())==2
