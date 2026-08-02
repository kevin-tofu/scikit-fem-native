"""Build and test native wheels with cibuildwheel on the current OS."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--platform",choices=("auto","linux","macos","windows"),
        default="auto",
    )
    parser.add_argument(
        "--python",nargs="+",choices=("3.10","3.11","3.12","3.13","3.14"),
        help="build only the selected CPython versions",
    )
    parser.add_argument(
        "--arch",nargs="+",
        help="cibuildwheel architecture(s), e.g. x86_64, arm64, or AMD64",
    )
    parser.add_argument(
        "--output-dir",type=Path,default=ROOT/"wheelhouse"
    )
    parser.add_argument(
        "--skip-tests",action="store_true",
        help="build wheels without installing and testing each one",
    )
    parser.add_argument("--dry-run",action="store_true")
    arguments=parser.parse_args()

    environment=os.environ.copy()
    if arguments.python:
        environment["CIBW_BUILD"]=" ".join(
            f"cp{version.replace('.','')}-*"
            for version in arguments.python
        )
    if arguments.skip_tests:
        environment["CIBW_TEST_COMMAND"]=""
    command=[
        sys.executable,"-m","cibuildwheel",
        "--platform",arguments.platform,
        "--output-dir",str(arguments.output_dir.resolve()),
    ]
    if arguments.arch:
        command.extend(("--archs",",".join(arguments.arch)))
    print("+"," ".join(command),flush=True)
    if "CIBW_BUILD" in environment:
        print("  CIBW_BUILD="+environment["CIBW_BUILD"],flush=True)
    if arguments.dry_run:
        return
    subprocess.run(command,cwd=ROOT,env=environment,check=True)

    wheels=sorted(arguments.output_dir.resolve().glob("*.whl"))
    if not wheels:
        raise SystemExit("cibuildwheel produced no wheels")
    print("\nBuilt wheels:")
    for wheel in wheels:
        print(f"  {wheel.name}  sha256:{digest(wheel)}")


if __name__=="__main__":
    main()
