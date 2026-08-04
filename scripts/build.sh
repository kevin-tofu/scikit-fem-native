#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKFEMNTV_PYTHON="${SKFEMNTV_PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/build.sh [--smoke-only | --skip-tests]

Build the skfem-native sdist and native wheel, validate their metadata, install
the wheel into a temporary clean environment, and run its tests.

Options:
  --smoke-only  Install and import the wheel without running the full test suite.
  --skip-tests  Build distributions and validate metadata only.

Environment:
  SKFEMNTV_PYTHON  Python executable used for the build. Default: python
EOF
}

PACKAGE_CHECK_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-only|--skip-tests)
      PACKAGE_CHECK_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[skfem-native-build] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#PACKAGE_CHECK_ARGS[@]} -gt 1 ]]; then
  echo "[skfem-native-build] --smoke-only and --skip-tests are mutually exclusive" >&2
  exit 2
fi

cd "${ROOT_DIR}"
echo "[skfem-native-build] python=${SKFEMNTV_PYTHON}" >&2
"${SKFEMNTV_PYTHON}" -c 'import build, twine' 2>/dev/null || {
  echo "[skfem-native-build] install build tools first:" >&2
  echo "  ${SKFEMNTV_PYTHON} -m pip install --upgrade build twine" >&2
  exit 1
}
"${SKFEMNTV_PYTHON}" tools/package_check.py "${PACKAGE_CHECK_ARGS[@]}"

