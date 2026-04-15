#!/usr/bin/env bash
set -euo pipefail

# Nightly termite benchmark: local main vs origin/main
# Intended to run non-interactively from launchd/cron.
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

STABLE_BIN="$CIRCUS_DIR/.nightly/stable/termite"
LATEST_BIN="$CIRCUS_DIR/.nightly/latest/termite"

log() { echo "$(date +%H:%M:%S) $*" >&2; }

fetch_termite_zig() {
    # Match the interactive shell's GitHub auth path, but refuse prompts so the
    # daemon fails fast if the token/config is unavailable.
    GIT_TERMINAL_PROMPT=0 git \
        -c core.askPass= \
        -c credential.interactive=never \
        -c 'credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential' \
        -C "$ZIG_DIR" \
        fetch --tags --quiet origin
}

escape_sed_replacement() {
    printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

build_termite() {
    local ref="$1"
    local sha="$2"
    local out_bin="$3"

    git -C "$ZIG_DIR" checkout --quiet "$ref"
    (cd "$ZIG_DIR" && zig build $ZIG_BUILD_FLAGS -Dgit-commit="$sha" 2>&1 | tail -1)
    mkdir -p "$(dirname "$out_bin")"
    cp "$ZIG_DIR/zig-out/bin/termite" "$out_bin"
}

# ─── Fetch latest code ───────────────────────────────────────────
log "fetching termite-zig..."
if ! fetch_termite_zig; then
    log "warning: fetch failed; continuing with cached origin refs/tags"
fi

STABLE_SHA=$(git -C "$ZIG_DIR" rev-parse --short main)
STABLE_DATE=$(git -C "$ZIG_DIR" log -1 --format=%cs main)
LATEST_SHA=$(git -C "$ZIG_DIR" rev-parse --short origin/main)
LATEST_DATE=$(git -C "$ZIG_DIR" log -1 --format=%cs origin/main)
STABLE_LABEL="termite-zig stable ($STABLE_SHA, $STABLE_DATE)"
LATEST_LABEL="termite-zig latest ($LATEST_SHA, $LATEST_DATE)"

# ─── Build local main and latest origin/main ─────────────────────
log "building local main ($STABLE_SHA)..."
build_termite "main" "$STABLE_SHA" "$STABLE_BIN"

log "building origin/main ($LATEST_SHA)..."
build_termite "origin/main" "$LATEST_SHA" "$LATEST_BIN"

# ─── Run benchmark ────────────────────────────────────────────────
log "running benchmark..."
BENCH_ARGS=(
    -model "$HOME/.termite/models/generators/ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
    -model-dir "$HOME/.termite/models/generators/ggml-org/gemma-4-E4B-it-GGUF"
    -llama-cli "llama-cli"
    -zig-clean "$STABLE_BIN"
    -zig-clean-dir "$ZIG_DIR"
    -zig-exp "$LATEST_BIN"
    -zig-exp-dir "$ZIG_DIR"
    -output "$RESULTS"
)

cd "$CIRCUS_DIR"
go run ./cmd/bench "${BENCH_ARGS[@]}" 2>&1

# ─── Fixup engine names in output ────────────────────────────────
STABLE_LABEL_ESCAPED="$(escape_sed_replacement "$STABLE_LABEL")"
LATEST_LABEL_ESCAPED="$(escape_sed_replacement "$LATEST_LABEL")"
sed -i '' \
    -e "s/termite-zig (clean)/$STABLE_LABEL_ESCAPED/g" \
    -e "s/termite-zig (exp)/$LATEST_LABEL_ESCAPED/g" \
    -e "s/- $STABLE_LABEL_ESCAPED: \`[^\`]*\`/- $STABLE_LABEL_ESCAPED: \`$STABLE_SHA\`/g" \
    -e "s/- $LATEST_LABEL_ESCAPED: \`[^\`]*\`/- $LATEST_LABEL_ESCAPED: \`$LATEST_SHA\`/g" \
    "$RESULTS"

log "results written to $RESULTS"

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
