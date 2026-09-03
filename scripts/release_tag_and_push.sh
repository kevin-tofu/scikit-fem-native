#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SKFEMNTV_PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/release_tag_and_push.sh VERSION

Create an annotated release tag (v<VERSION>) and push it to origin.

Arguments:
  VERSION    Version string like 0.3.1 or v0.3.1

Environment:
  SKFEMNTV_PYTHON            Python executable. Default: python
  SKFEMNTV_GITHUB_REPOSITORY GitHub owner/repository for checking remote state.
                             Default: kevin-tofu/scikit-fem-native
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

VERSION="$1"
if [[ "${VERSION}" == v* ]]; then
  TAG="${VERSION}"
else
  TAG="v${VERSION}"
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" tools/check_release_version.py "${TAG}"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "[skfem-native-release] release tag must be created on main" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[skfem-native-release] working tree must be clean" >&2
  exit 1
fi

git fetch origin main --tags
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "[skfem-native-release] main is not synchronized with origin/main" >&2
  exit 1
fi

if git rev-parse --verify --quiet "refs/tags/${TAG}" >/dev/null; then
  echo "[skfem-native-release] tag already exists: ${TAG}" >&2
  exit 1
fi

git tag -a "${TAG}" -m "skfem-native ${TAG}"
git push origin "${TAG}"

echo "[skfem-native-release] pushed ${TAG}"

