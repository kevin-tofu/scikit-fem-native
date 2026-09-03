#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY="${SKFEMNTV_GITHUB_REPOSITORY:-kevin-tofu/scikit-fem-native}"

usage() {
  cat <<'EOF'
Usage:
  scripts/release_with_gh.sh TAG [--title "Title"] [--notes NOTE_FILE] [--draft] [--latest]

Create a GitHub Release for an existing tag.

Arguments:
  TAG           Release tag. Example: v0.3.1

Options:
  --title TEXT         Custom release title (default: "skfem-native <TAG>")
  --notes NOTE_FILE    Path to release notes file passed to --notes
  --draft              Create a draft release
  --latest             Mark this release as latest

Environment:
  SKFEMNTV_GITHUB_REPOSITORY GitHub owner/repository.
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

TAG="$1"
shift

TITLE="skfem-native ${TAG}"
NOTES_ARGS=()
DRAFT_FLAG=""
LATEST_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)
      if [[ $# -lt 2 ]]; then
        echo "[skfem-native-gh-release] --title requires a value" >&2
        exit 2
      fi
      TITLE="$2"
      shift 2
      ;;
    --notes)
      if [[ $# -lt 2 ]]; then
        echo "[skfem-native-gh-release] --notes requires a file path" >&2
        exit 2
      fi
      NOTES_ARGS+=("--notes-file" "$2")
      shift 2
      ;;
    --draft)
      DRAFT_FLAG="--draft"
      shift
      ;;
    --latest)
      LATEST_FLAG="--latest"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[skfem-native-gh-release] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${ROOT_DIR}"

if ! command -v gh >/dev/null; then
  echo "[skfem-native-gh-release] GitHub CLI (gh) is required" >&2
  exit 1
fi

gh auth status --hostname github.com

git rev-parse --verify --quiet "refs/tags/${TAG}" >/dev/null 2>&1 || {
  echo "[skfem-native-gh-release] tag not found locally: ${TAG}" >&2
  exit 1
}

if gh release view "${TAG}" --repo "${REPOSITORY}" >/dev/null 2>&1; then
  echo "[skfem-native-gh-release] release already exists: ${TAG}" >&2
  exit 1
fi

gh release create "${TAG}" \
  --repo "${REPOSITORY}" \
  --verify-tag \
  --title "${TITLE}" \
  "${NOTES_ARGS[@]}" \
  ${DRAFT_FLAG} ${LATEST_FLAG} \
  --generate-notes

echo "[skfem-native-gh-release] created release for ${TAG}"
