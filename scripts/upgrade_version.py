#!/usr/bin/env python3
"""Update the project version in pyproject.toml safely."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT=ROOT/"pyproject.toml"
VERSION_PATTERN=re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:(a|b|rc)(0|[1-9]\d*))?"
    r"(?:\.post(0|[1-9]\d*))?(?:\.dev(0|[1-9]\d*))?$"
)


def validate_version(value: str) -> str:
    if not VERSION_PATTERN.fullmatch(value):
        raise ValueError(
            "version must be a normalized PEP 440 release such as "
            "1.2.3, 1.2.3rc1, 1.2.3.post1, or 1.2.3.dev1"
        )
    return value


def project_version(path: Path) -> str:
    text=path.read_text(encoding="utf-8")
    project=re.search(r"(?ms)^\[project\]\s*$.*?(?=^\[|\Z)",text)
    if project is None:
        raise ValueError(f"{path} has no [project] table")
    match=re.search(
        r'(?m)^version\s*=\s*["\']([^"\']+)["\']\s*$',
        project.group(0),
    )
    if match is None:
        raise ValueError(f"{path} has no [project].version")
    return match.group(1)


def replace_project_version(text: str,old: str,new: str) -> str:
    project=re.search(r"(?ms)^\[project\]\s*$.*?(?=^\[|\Z)",text)
    if project is None:
        raise ValueError("pyproject.toml has no [project] table")
    section=project.group(0)
    pattern=re.compile(
        rf'(?m)^(version\s*=\s*)["\']{re.escape(old)}["\'](\s*)$'
    )
    replaced,count=pattern.subn(rf'\g<1>"{new}"\g<2>',section)
    if count!=1:
        raise ValueError("expected exactly one [project].version assignment")
    return text[:project.start()]+replaced+text[project.end():]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version",help="new normalized PEP 440 version")
    parser.add_argument(
        "--pyproject",type=Path,default=DEFAULT_PYPROJECT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry-run",action="store_true",
        help="validate and print the change without writing",
    )
    arguments=parser.parse_args()
    try:
        new=validate_version(arguments.version)
        old=project_version(arguments.pyproject)
        if new==old:
            raise ValueError(f"project version is already {new}")
        text=arguments.pyproject.read_text(encoding="utf-8")
        updated=replace_project_version(text,old,new)
    except (OSError,ValueError) as error:
        parser.error(str(error))
    print(f"project version: {old} -> {new}")
    if arguments.dry_run:
        return
    arguments.pyproject.write_text(updated,encoding="utf-8")


if __name__=="__main__":
    main()
