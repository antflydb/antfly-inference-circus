#!/usr/bin/env bash
set -euo pipefail

# Nightly termite benchmark: latest tag vs tip of main
# Intended to run via cron with no agent involvement.
#
# Usage:
#   ./scripts/nightly-bench.sh                  # run + print results
#   ./scripts/nightly-bench.sh --post-zulip     # run + post to Zulip
#
# Cron example (2:05 AM Pacific nightly):
#   5 2 * * * cd ~/rowan/termite-circus && ./scripts/nightly-bench.sh --post-zulip 2>&1 >> ~/.termite-circus-nightly.log

CIRCUS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ZIG_DIR="$CIRCUS_DIR/termite-zig-clean"
RESULTS="$CIRCUS_DIR/results/nightly.md"
ZIG_BUILD_FLAGS="-Doptimize=ReleaseFast -Donnx=false -Dpjrt=false -Dcoreml=false"

# Load credentials from .env if present
if [ -f "$CIRCUS_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$CIRCUS_DIR/.env"
fi

TAG_BIN="$CIRCUS_DIR/.nightly/tag/termite"
MAIN_BIN="$CIRCUS_DIR/.nightly/main/termite"

log() { echo "$(date +%H:%M:%S) $*" >&2; }

# ─── Fetch latest code ───────────────────────────────────────────
log "fetching termite-zig..."
git -C "$ZIG_DIR" fetch --tags --quiet

LATEST_TAG=$(git -C "$ZIG_DIR" describe --tags --abbrev=0 origin/main 2>/dev/null || echo "")
MAIN_SHA=$(git -C "$ZIG_DIR" rev-parse --short origin/main)

if [ -z "$LATEST_TAG" ]; then
    log "no tags found, benchmarking tip of main only"
    LATEST_TAG=""
fi

# ─── Build at latest tag ─────────────────────────────────────────
if [ -n "$LATEST_TAG" ]; then
    TAG_SHA=$(git -C "$ZIG_DIR" rev-parse --short "$LATEST_TAG")
    log "building $LATEST_TAG ($TAG_SHA)..."
    git -C "$ZIG_DIR" checkout --quiet "$LATEST_TAG"
    (cd "$ZIG_DIR" && zig build $ZIG_BUILD_FLAGS -Dgit-commit="$TAG_SHA" 2>&1 | tail -1)
    mkdir -p "$(dirname "$TAG_BIN")"
    cp "$ZIG_DIR/zig-out/bin/termite" "$TAG_BIN"
fi

# ─── Build at tip of main ────────────────────────────────────────
log "building main ($MAIN_SHA)..."
git -C "$ZIG_DIR" checkout --quiet origin/main
(cd "$ZIG_DIR" && zig build $ZIG_BUILD_FLAGS -Dgit-commit="$MAIN_SHA" 2>&1 | tail -1)
mkdir -p "$(dirname "$MAIN_BIN")"
cp "$ZIG_DIR/zig-out/bin/termite" "$MAIN_BIN"

# ─── Restore clean state ─────────────────────────────────────────
git -C "$ZIG_DIR" checkout --quiet main 2>/dev/null || true

# ─── Run benchmark ────────────────────────────────────────────────
log "running benchmark..."
BENCH_ARGS=(
    -model "$HOME/.termite/models/generators/ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
    -model-dir "$HOME/.termite/models/generators/ggml-org/gemma-4-E4B-it-GGUF"
    -llama-cli "llama-cli"
    -zig-clean "$MAIN_BIN"
    -zig-clean-dir "$ZIG_DIR"
    -output "$RESULTS"
)

# Use the tag build as "experimentation" slot (it's the second zig engine)
if [ -n "$LATEST_TAG" ]; then
    BENCH_ARGS+=(
        -zig-exp "$TAG_BIN"
        -zig-exp-dir "$ZIG_DIR"
    )
fi

cd "$CIRCUS_DIR"
go run ./cmd/bench "${BENCH_ARGS[@]}" 2>&1

# ─── Fixup engine names in output ────────────────────────────────
# The bench tool uses generic names; replace with specific versions.
if [ -n "$LATEST_TAG" ]; then
    sed -i '' \
        -e "s/termite-zig (clean)/termite-zig main ($MAIN_SHA)/g" \
        -e "s/termite-zig (exp)/termite-zig $LATEST_TAG ($TAG_SHA)/g" \
        "$RESULTS"
fi

log "results written to $RESULTS"
cat "$RESULTS"

# ─── Post to Zulip ───────────────────────────────────────────────
if [[ "${1:-}" == "--post-zulip" ]]; then
    if [ -n "${ZULIP_BOT:-}" ] && [ -n "${ZULIP_KEY:-}" ] && [ -n "${ZULIP_SITE:-}" ]; then
        log "posting to Zulip..."
        curl -sf -u "$ZULIP_BOT:$ZULIP_KEY" \
            "$ZULIP_SITE/api/v1/messages" \
            --data-urlencode type=stream \
            --data-urlencode to=eval-runs \
            --data-urlencode topic=termite-circus-nightly \
            --data-urlencode "content@$RESULTS" \
            > /dev/null
        log "posted."
    else
        log "warning: ZULIP_BOT/ZULIP_KEY/ZULIP_SITE not set (see .env.example), skipping post"
    fi
fi
