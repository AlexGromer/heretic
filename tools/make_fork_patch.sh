#!/bin/sh
# Builds the patch a runner needs to turn an upstream Heretic checkout into this fork, and
# prints its provenance (base commit, fork commit, sha256). The runner vendors the patch
# file and applies it to the upstream base; nothing of the runner lives in this repository.
#
#   tools/make_fork_patch.sh [output.patch] [upstream base commit]
#
# The default base is the upstream commit this fork branched from, read from the merge base
# with the recorded upstream ref when it exists, else UPSTREAM_BASE below.
set -eu

OUT="${1:-heretic-fork-src.patch}"
UPSTREAM_BASE="${2:-a2e626f}"

cd "$(dirname "$0")/.."

if ! git rev-parse --verify --quiet "$UPSTREAM_BASE" >/dev/null; then
  echo "STOP: base commit $UPSTREAM_BASE is not in this repository" >&2
  exit 1
fi

HEAD_SHA="$(git rev-parse HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: the working tree has uncommitted changes; the patch is built from HEAD" >&2
fi

# Only src/ travels: the runner supplies its own configs, and the tests stay here.
git diff "$UPSTREAM_BASE..HEAD" -- src > "$OUT"

SUM="$(sha256sum "$OUT" | cut -d' ' -f1)"
FILES="$(git diff --name-only "$UPSTREAM_BASE..HEAD" -- src | wc -l | tr -d ' ')"
LINES="$(wc -l < "$OUT" | tr -d ' ')"

cat <<EOT
$OUT written.

| field          | value |
|----------------|-------|
| fork commit    | $HEAD_SHA |
| upstream base  | $(git rev-parse "$UPSTREAM_BASE") |
| files          | $FILES under src/ |
| patch lines    | $LINES |
| sha256         | $SUM |

Apply it on an upstream checkout at the base commit:

  git checkout $UPSTREAM_BASE && git apply --check $OUT && git apply $OUT

Settings the fork adds are documented in config.default.toml; a runner that generates its
own config must emit the keys it wants (everything else keeps the upstream default).
EOT
