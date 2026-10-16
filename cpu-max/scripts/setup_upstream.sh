#!/usr/bin/env bash
# Build the unmodified upstream engine (QuixiAI/embeddinggemma.c at d8f9e33, the
# commit this fork is based on) as the baseline. It is checked out from this
# repository's own history as a git worktree in cpu-max/third_party/upstream
# (git-ignored), so no second clone or network access is needed.
#   cpu-max/scripts/setup_upstream.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=d8f9e33
DIR="$PWD/third_party/upstream"
if [ ! -e "$DIR/.git" ]; then
  mkdir -p third_party
  git worktree add --detach "$DIR" "$BASE"
fi
make -C "$DIR" -j"$(nproc)" build/embeddinggemma
echo "upstream baseline: $DIR/build/embeddinggemma ($(git -C "$DIR" rev-parse --short HEAD))"
