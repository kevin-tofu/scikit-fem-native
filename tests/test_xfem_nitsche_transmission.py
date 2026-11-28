from pathlib import Path
import runpy


def test_xfem_nitsche_transmission_converges_with_nonzero_jump():
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "xfem_nitsche_transmission.py"
    )
    solve = runpy.run_path(str(example))["solve"]
    coarse_error, coarse_jump_error, coarse_asymmetry = solve(8)
    fine_error, fine_jump_error, fine_asymmetry = solve(16)

    assert fine_error < coarse_error / 3.5
    assert fine_jump_error < coarse_jump_error / 3.5
    assert coarse_asymmetry < 2e-12
    assert fine_asymmetry < 2e-12
