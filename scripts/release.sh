#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKFEMNTV_PYTHON="${SKFEMNTV_PYTHON:-python}"
REPOSITORY="${SKFEMNTV_GITHUB_REPOSITORY:-kevin-tofu/scikit-fem-native}"

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh VERSION [--skip-build]

Validate an already committed and pushed release version, create and push its
annotated tag, and publish a GitHub Release.  The Release workflow publishes
the wheels and sdist to PyPI through Trusted Publishing.

The script requires a clean main branch synchronized with origin/main and an
authenticated GitHub CLI (`gh auth status`).

Environment:
  SKFEMNTV_PYTHON             Python executable. Default: python
  SKFEMNTV_GITHUB_REPOSITORY  GitHub owner/repository.
                              Default: kevin-tofu/scikit-fem-native
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

VERSION="$1"
shift
SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[skfem-native-release] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${ROOT_DIR}"
TAG="v${VERSION}"
"${SKFEMNTV_PYTHON}" tools/check_release_version.py "${TAG}"

if [[ "$(git branch --show-current)" != "main" ]]; then
  echo "[skfem-native-release] release must be created from main" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[skfem-native-release] worktree must be clean" >&2
  exit 1
fi

git fetch origin main --tags
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "[skfem-native-release] main must be synchronized with origin/main" >&2
  exit 1
fi
if git rev-parse --verify --quiet "refs/tags/${TAG}" >/dev/null; then
  echo "[skfem-native-release] tag already exists: ${TAG}" >&2
  exit 1
fi
if ! command -v gh >/dev/null; then
  echo "[skfem-native-release] GitHub CLI (gh) is required" >&2
  exit 1
fi
gh auth status

if [[ "${SKIP_BUILD}" == 0 ]]; then
  scripts/build.sh
fi

git tag -a "${TAG}" -m "skfem-native ${TAG}"
git push origin "${TAG}"
gh release create "${TAG}" \
  --repo "${REPOSITORY}" \
  --verify-tag \
  --title "skfem-native ${TAG}" \
  --generate-notes

echo "[skfem-native-release] created ${TAG}; PyPI publish workflow has started" >&2
