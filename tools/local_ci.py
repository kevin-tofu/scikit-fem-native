"""Run the same validation stages as GitHub Actions on the local OS."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+"," ".join(map(str,command)),flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def detected_platform() -> tuple[str,str]:
    system=platform.system()
    machine=platform.machine().lower()
    if system=="Linux":
        return "linux","aarch64" if machine in ("aarch64","arm64") else "x86_64"
    if system=="Darwin":
        return "macos","arm64" if machine=="arm64" else "x86_64"
    if system=="Windows":
        return "windows","ARM64" if machine in ("arm64","aarch64") else "AMD64"
    raise SystemExit(f"unsupported local platform: {system} {machine}")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "stage",choices=("fast","package","wheel","all"),default="fast",
        nargs="?",
    )
    parser.add_argument(
        "--python",nargs="+",choices=("3.10","3.11","3.12","3.13","3.14"),
        help="wheel Python versions; defaults to the running interpreter",
    )
    parser.add_argument("--skip-install",action="store_true")
    arguments=parser.parse_args()
    if arguments.stage in ("fast","all"):
        if not arguments.skip_install:
            run(sys.executable,"-m","pip","install","-e",".[test]","--no-build-isolation")
        run(sys.executable,"-m","pytest")
    if arguments.stage in ("package","all"):
        run(sys.executable,"tools/package_check.py")
    if arguments.stage in ("wheel","all"):
        target,arch=detected_platform()
        versions=arguments.python or [f"{sys.version_info.major}.{sys.version_info.minor}"]
        run(
            sys.executable,"tools/build_wheels.py","--platform",target,
            "--arch",arch,"--python",*versions,
        )


if __name__=="__main__":
    main()
