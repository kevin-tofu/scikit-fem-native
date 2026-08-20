"""Build and test release distributions on the current platform."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


ROOT=Path(__file__).resolve().parents[1]


def run(*command: str,cwd: Path=ROOT) -> None:
    print("+"," ".join(map(str,command)),flush=True)
    subprocess.run(command,cwd=cwd,check=True)


def venv_python(directory: Path) -> Path:
    return directory/("Scripts/python.exe" if sys.platform=="win32" else "bin/python")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--outdir",type=Path,default=ROOT/"dist")
    parser.add_argument(
        "--skip-tests",action="store_true",
        help="only build distributions and check their metadata",
    )
    parser.add_argument(
        "--smoke-only",action="store_true",
        help="install the wheel and import it without running pytest",
    )
    arguments=parser.parse_args()
    if arguments.skip_tests and arguments.smoke_only:
        parser.error("--skip-tests and --smoke-only are mutually exclusive")
    output=arguments.outdir.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    run(sys.executable,"-m","build","--outdir",str(output))
    distributions=sorted(output.iterdir())
    run(sys.executable,"-m","twine","check",*[str(p) for p in distributions])

    wheels=list(output.glob("*.whl"))
    if len(wheels)!=1:
        raise SystemExit(f"expected one wheel, found {len(wheels)}")
    if arguments.skip_tests:
        return

    with tempfile.TemporaryDirectory(prefix="skfemntv-wheel-test-") as temporary:
        environment=Path(temporary)/"venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python=venv_python(environment)
        run(
            str(python),"-m","pip","install",
            str(wheels[0]) if arguments.smoke_only else f"{wheels[0]}[test]",
        )
        if not arguments.smoke_only:
            run(
                str(python),"-m","pytest",str(ROOT/"tests"),"-q",
                cwd=Path(temporary),
            )
        run(
            str(python),"-c",
            "import skfemntv; from importlib.metadata import version; "
            "assert skfemntv.__version__ == version('scikit-fem-native'); "
            "print('installed skfemntv:', skfemntv.__file__, skfemntv.__version__)",
            cwd=Path(temporary),
        )


if __name__=="__main__":
    main()
