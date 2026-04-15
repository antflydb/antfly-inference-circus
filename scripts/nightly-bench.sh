#!/usr/bin/env bash
set -euo pipefail

# Nightly termite benchmark: antfly-zig/pkg/termite stable vs latest
# Intended to run non-interactively from launchd/cron.
#
# Usage:
#   ./scripts/nightly-bench.sh                  # run + print results
#   ./scripts/nightly-bench.sh --post-zulip     # run + post to Zulip
#
# Cron example (2:05 AM Pacific nightly):
#   5 2 * * * cd ~/rowan/termite-circus && ./scripts/nightly-bench.sh --post-zulip 2>&1 >> ~/.termite-circus-nightly.log

CIRCUS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ANTFLY_DIR="$CIRCUS_DIR/antfly-zig"
RESULTS="$CIRCUS_DIR/results/nightly.md"
ANTFLY_STABLE_COMMIT="${ANTFLY_STABLE_COMMIT:-8caea6fc21abc4b8a96dd43a1cc7c49ae9faf188}"
TERMITE_BUILD_FLAGS="-Doptimize=ReleaseFast -Dshared-lib-root=../.."

# Load credentials from .env if present
if [ -f "$CIRCUS_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$CIRCUS_DIR/.env"
fi

STABLE_DIR="$CIRCUS_DIR/.nightly/stable"
LATEST_DIR="$CIRCUS_DIR/.nightly/latest"
STABLE_BIN="$STABLE_DIR/termite"
LATEST_BIN="$LATEST_DIR/termite"
PROMPT_FILE="$CIRCUS_DIR/.nightly/prompts.txt"

log() { echo "$(date +%H:%M:%S) $*" >&2; }

fetch_antfly_zig() {
    # Match the interactive shell's GitHub auth path, but refuse prompts so the
    # daemon fails fast if the token/config is unavailable.
    GIT_TERMINAL_PROMPT=0 git \
        -c core.askPass= \
        -c credential.interactive=never \
        -c 'credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential' \
        -C "$ANTFLY_DIR" \
        fetch --tags --quiet origin
}

escape_sed_replacement() {
    printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

write_prompts() {
    mkdir -p "$CIRCUS_DIR/.nightly"
    cat >"$PROMPT_FILE" <<'EOF'
In two sentences, explain why the peppered moth's dark form declined in Britain after the mid-20th century.
In two sentences, explain why monarch butterfly populations in North America have declined in recent decades.
In two sentences, explain why the spotted lanternfly is considered a serious invasive pest in the eastern United States.
In two sentences, explain why the emerald ash borer kills ash trees so effectively.
In two sentences, explain why bed bugs have resurged in major cities since the late 20th century.
In two sentences, explain why mountain pine beetle outbreaks can devastate western forests.
In two sentences, explain why Varroa mites are such a major threat to honey bee colonies.
EOF
}

build_termite() {
    local ref="$1"
    local sha="$2"
    local out_dir="$3"
    local out_bin="$out_dir/termite"

    git -C "$ANTFLY_DIR" checkout --quiet "$ref"
    (cd "$ANTFLY_DIR/pkg/termite" && zig build $TERMITE_BUILD_FLAGS -Dgit-commit="$sha" 2>&1 | tail -1)
    mkdir -p "$out_dir"
    cp "$ANTFLY_DIR/pkg/termite/zig-out/bin/termite" "$out_bin"
}

# ─── Fetch latest code ───────────────────────────────────────────
log "fetching antfly-zig..."
if ! fetch_antfly_zig; then
    log "warning: fetch failed; continuing with cached origin refs/tags"
fi

STABLE_SHA=$(git -C "$ANTFLY_DIR" rev-parse --short "$ANTFLY_STABLE_COMMIT")
STABLE_DATE=$(git -C "$ANTFLY_DIR" log -1 --format=%cs "$ANTFLY_STABLE_COMMIT")
LATEST_SHA=$(git -C "$ANTFLY_DIR" rev-parse --short origin/main)
LATEST_DATE=$(git -C "$ANTFLY_DIR" log -1 --format=%cs origin/main)
STABLE_LABEL="termite-zig stable ($STABLE_DATE)"
LATEST_LABEL="termite-zig latest ($LATEST_DATE)"

# ─── Build pinned stable and latest origin/main ──────────────────
log "building pinned stable antfly-zig ($STABLE_SHA)..."
build_termite "$ANTFLY_STABLE_COMMIT" "$STABLE_SHA" "$STABLE_DIR"

log "building latest antfly-zig origin/main ($LATEST_SHA)..."
build_termite "origin/main" "$LATEST_SHA" "$LATEST_DIR"

# ─── Run benchmark ────────────────────────────────────────────────
log "running benchmark..."
write_prompts
BENCH_ARGS=(
    -model "$HOME/.termite/models/generators/ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
    -model-dir "$HOME/.termite/models/generators/ggml-org/gemma-4-E4B-it-GGUF"
    -llama-cli "llama-cli"
    -prompt-file "$PROMPT_FILE"
    -zig-clean "$STABLE_BIN"
    -zig-clean-dir "$ANTFLY_DIR"
    -zig-exp "$LATEST_BIN"
    -zig-exp-dir "$ANTFLY_DIR"
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

# Leave antfly-zig checked out at the latest benchmarked ref for inspection.
git -C "$ANTFLY_DIR" checkout --quiet origin/main

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
