# GitHub CLI usage

This note records the `gh` commands used to inspect CI and publish a release of
`skfem-native`.  Run the commands from the repository root.

## Installation and authentication

Confirm that GitHub CLI is available:

```bash
command -v gh
gh --version
```

On this workstation the executable may be installed at `/snap/bin/gh`.  If it
is not found through `PATH`, invoke that path directly or add `/snap/bin` to
`PATH`.

Check authentication without printing the token itself:

```bash
gh auth status
```

Authenticate interactively when necessary:

```bash
gh auth login
```

Select `github.com`, SSH for Git operations, and browser-based authentication.
Do not put a GitHub token in this repository, command history, notes, or logs.

The examples below specify the repository explicitly so that they also work
outside the local checkout:

```text
kevin-tofu/skfem-native
```

## Inspecting Actions

List recent workflow runs:

```bash
gh run list --repo kevin-tofu/skfem-native --limit 10
```

Limit the result to one workflow:

```bash
gh run list \
  --repo kevin-tofu/skfem-native \
  --workflow ci.yml \
  --limit 10

gh run list \
  --repo kevin-tofu/skfem-native \
  --workflow workflow.yml \
  --limit 10
```

`ci.yml` is the regular test workflow.  `workflow.yml` builds release
artifacts and publishes them to PyPI after a GitHub Release is published.

Inspect one run, where `RUN_ID` is the numeric ID shown by `gh run list`:

```bash
gh run view RUN_ID --repo kevin-tofu/skfem-native
```

Open it in a browser:

```bash
gh run view RUN_ID --repo kevin-tofu/skfem-native --web
```

Monitor it until completion and return a nonzero exit status on failure:

```bash
gh run watch RUN_ID \
  --repo kevin-tofu/skfem-native \
  --interval 10 \
  --exit-status
```

Get a compact machine-readable status, including every job:

```bash
gh run view RUN_ID \
  --repo kevin-tofu/skfem-native \
  --json status,conclusion,jobs,url
```

## Diagnosing a failed run

Print only failed-step logs:

```bash
gh run view RUN_ID \
  --repo kevin-tofu/skfem-native \
  --log-failed
```

Print all logs when the short output is insufficient:

```bash
gh run view RUN_ID \
  --repo kevin-tofu/skfem-native \
  --log
```

Rerun only failed jobs after pushing a fix:

```bash
gh run rerun RUN_ID \
  --repo kevin-tofu/skfem-native \
  --failed
```

Rerunning an old release job does not change its source revision.  If the
release code itself changed, bump the version and publish a new tag and Release
instead of rebuilding a different artifact under an existing version.

## Preparing a release

First update all version declarations using the repository script:

```bash
./scripts/upgrade_version.py 0.1.3
```

Run the tests and commit/push the version change.  Confirm that CI for the
release commit succeeds before tagging it.

Create and push an annotated tag:

```bash
git tag -a v0.1.3 -m "skfem-native v0.1.3"
git push origin v0.1.3
```

Verify that the local version matches the release tag:

```bash
python tools/check_release_version.py v0.1.3
```

Do not reuse or move a published version tag.  PyPI versions are immutable.

## Publishing a GitHub Release and PyPI package

Publish the Release from an existing, verified tag:

```bash
gh release create v0.1.3 \
  --repo kevin-tofu/skfem-native \
  --verify-tag \
  --title "skfem-native v0.1.3" \
  --generate-notes
```

Alternatively, supply curated notes using `--notes` or `--notes-file`.

Publishing the GitHub Release triggers `.github/workflows/workflow.yml`.  That
workflow builds the source distribution and Linux, Windows, macOS arm64, and
macOS Intel wheels.  Its final publish job uses PyPI Trusted Publishing; no
long-lived PyPI API token should be added to GitHub secrets.

Find and monitor the resulting run:

```bash
gh run list \
  --repo kevin-tofu/skfem-native \
  --workflow workflow.yml \
  --limit 3

gh run watch RUN_ID \
  --repo kevin-tofu/skfem-native \
  --interval 10 \
  --exit-status
```

Check the published Release:

```bash
gh release view v0.1.3 --repo kevin-tofu/skfem-native
```

Finally, verify the package independently from GitHub:

```bash
python -m pip index versions skfem-native
```

PyPI indexing can lag briefly after a successful upload.  Test installation in
a fresh virtual environment rather than an editable development environment:

```bash
python -m venv /tmp/skfem-native-release-check
/tmp/skfem-native-release-check/bin/python -m pip install skfem-native==0.1.3
/tmp/skfem-native-release-check/bin/python -c "import skfemntv; print(skfemntv.__version__)"
```

## Useful release queries

List Releases:

```bash
gh release list --repo kevin-tofu/skfem-native
```

Inspect the current Release and its assets:

```bash
gh release view v0.1.3 \
  --repo kevin-tofu/skfem-native \
  --json tagName,name,isDraft,isPrerelease,publishedAt,assets,url
```

Download artifacts attached to a Release:

```bash
gh release download v0.1.3 \
  --repo kevin-tofu/skfem-native \
  --dir /tmp/skfem-native-v0.1.3
```

Workflow artifacts and GitHub Release assets are different.  `gh run download`
downloads artifacts retained by Actions, while `gh release download` downloads
files permanently attached to a Release.
