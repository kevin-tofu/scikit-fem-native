"""Check that a release tag identifies the version in pyproject.toml."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def project_version(path: Path) -> str:
    contents=path.read_text(encoding="utf-8")
    match=re.search(
        r'^\[project\]\s*$.*?^version\s*=\s*["\']([^"\']+)["\']',
        contents,flags=re.MULTILINE|re.DOTALL,
    )
    if match is None:
        raise SystemExit(f"could not find [project] version in {path}")
    return match.group(1)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("tag",help="release tag, with an optional leading v")
    parser.add_argument(
        "--pyproject",type=Path,default=Path("pyproject.toml")
    )
    arguments=parser.parse_args()
    tag=arguments.tag.removeprefix("v")
    version=project_version(arguments.pyproject)
    if tag!=version:
        raise SystemExit(
            f"release tag {arguments.tag!r} does not match project "
            f"version {version!r}"
        )
    print(f"release version: {version}")


if __name__=="__main__":
    main()
