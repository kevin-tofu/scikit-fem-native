from pathlib import Path
import runpy

import skfemntv


def test_experimental_hcurl_vertical_slice_has_small_public_api():
    assert skfemntv.AffineTriN1Basis.__module__=="skfemntv.hcurl_basis"
    assert skfemntv.TriN1Assembler.__module__=="skfemntv.hcurl_assembler"
    assert skfemntv.TriN1LinearAssembler.__module__=="skfemntv.hcurl_assembler"
    assert callable(skfemntv.estimate_tri_n1_assembly_memory)
    assert not hasattr(skfemntv,"NativeTriN1Assembler")
    for name in (
        "AffineTriN1Basis",
        "TriN1Assembler",
        "TriN1LinearAssembler",
        "estimate_tri_n1_assembly_memory",
    ):
        assert name in skfemntv.__all__


def test_experimental_hcurl_example_runs_end_to_end(capsys):
    example=Path(__file__).parents[1]/"examples"/"hcurl_tri_n1_maxwell.py"
    runpy.run_path(example,run_name="__main__")
    output=capsys.readouterr().out
    assert "TriN1 DOFs=" in output
    assert "free residual=" in output
