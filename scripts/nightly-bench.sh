#!/usr/bin/env bash
set -euo pipefail

# Nightly termite benchmark: antfly-zig/pkg/termite stable vs latest, plus public Antfly repo Termite
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
ANTFLY_MAIN_REPO_URL="${ANTFLY_MAIN_REPO_URL:-https://github.com/antflydb/antfly}"
RESULTS="$CIRCUS_DIR/results/nightly.md"
ANTFLY_STABLE_COMMIT="${ANTFLY_STABLE_COMMIT:-8caea6fc21abc4b8a96dd43a1cc7c49ae9faf188}"
TERMITE_BUILD_FLAGS_BASE="-Doptimize=ReleaseSafe -Dshared-lib-root=../.. -Dskip-openapi=true"
MODEL_REPO="${MODEL_REPO:-ggml-org/gemma-4-E4B-it-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-E4B-it-Q4_K_M.gguf}"
POST_ZULIP=false
if [[ "${1:-}" == "--post-zulip" ]]; then
    POST_ZULIP=true
fi

# Load credentials from .env if present
if [ -f "$CIRCUS_DIR/.env" ]; then
    # shellcheck disable=SC1091
    source "$CIRCUS_DIR/.env"
fi

MLX_ROOT="${MLX_ROOT:-${HOMEBREW_PREFIX:-$HOME/.homebrew}}"

STABLE_DIR="$CIRCUS_DIR/.nightly/stable"
LATEST_DIR="$CIRCUS_DIR/.nightly/latest"
ANTFLY_MAIN_SRC_DIR="${ANTFLY_MAIN_SRC_DIR:-$CIRCUS_DIR/.nightly/antfly-main-src}"
ANTFLY_MAIN_DIR="$CIRCUS_DIR/.nightly/antfly-main"
STABLE_BIN="$STABLE_DIR/termite"
LATEST_BIN="$LATEST_DIR/termite"
ANTFLY_MAIN_BIN="$ANTFLY_MAIN_DIR/termite"
PROMPT_FILE="$CIRCUS_DIR/.nightly/prompts.txt"
MODEL_DIR="$HOME/.termite/models/generators/$MODEL_REPO"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
LOG_PATH="${LOG_PATH:-$HOME/.termite-circus-nightly.log}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }

post_zulip_file() {
    local file="$1"
    if [ -n "${ZULIP_BOT:-}" ] && [ -n "${ZULIP_KEY:-}" ] && [ -n "${ZULIP_SITE:-}" ]; then
        curl -sf -u "$ZULIP_BOT:$ZULIP_KEY" \
            "$ZULIP_SITE/api/v1/messages" \
            --data-urlencode type=stream \
            --data-urlencode to=eval-runs \
            --data-urlencode topic=termite-circus-nightly \
            --data-urlencode "content@$file" \
            > /dev/null
    else
        return 1
    fi
}

notify_failure() {
    local status=$?
    local msg
    trap - ERR
    log "ERROR: nightly failed with status $status"

    if $POST_ZULIP; then
        msg="$(mktemp)"
        {
            echo "# Termite nightly benchmark failed"
            echo
            echo "**Date:** $(date '+%Y-%m-%d %H:%M')"
            echo "**Host:** $(hostname)"
            echo "**Exit status:** $status"
            echo
            echo '```spoiler recent log'
            tail -80 "$LOG_PATH" 2>/dev/null || true
            echo '```'
        } >"$msg"
        if post_zulip_file "$msg"; then
            log "posted failure notice."
        else
            log "warning: failed to post failure notice"
        fi
        rm -f "$msg"
    fi

    exit "$status"
}

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

git_fetch() {
    local dir="$1"
    GIT_TERMINAL_PROMPT=0 git \
        -c core.askPass= \
        -c credential.interactive=never \
        -c 'credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential' \
        -C "$dir" \
        fetch --tags --quiet origin
}

ensure_antfly_main_repo() {
    if [ -d "$ANTFLY_MAIN_SRC_DIR/.git" ]; then
        git_fetch "$ANTFLY_MAIN_SRC_DIR"
        return
    fi

    rm -rf "$ANTFLY_MAIN_SRC_DIR"
    mkdir -p "$(dirname "$ANTFLY_MAIN_SRC_DIR")"
    GIT_TERMINAL_PROMPT=0 git \
        -c core.askPass= \
        -c credential.interactive=never \
        -c 'credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential' \
        clone --quiet "$ANTFLY_MAIN_REPO_URL" "$ANTFLY_MAIN_SRC_DIR"
}

ensure_model() {
    if [ -f "$MODEL_PATH" ]; then
        return
    fi

    log "model missing at $MODEL_PATH; downloading $MODEL_REPO/$MODEL_FILE..."
    run_with_timeout 7200 go run "$CIRCUS_DIR/cmd/model-dl" -repo "$MODEL_REPO" -file "$MODEL_FILE"
}

escape_sed_replacement() {
    printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

write_prompts() {
    mkdir -p "$CIRCUS_DIR/.nightly"
    local all_prompts=(
        "In two sentences, explain why the peppered moth's dark form declined in Britain after the mid-20th century."
        "In two sentences, explain why monarch butterfly populations in North America have declined in recent decades."
        "In two sentences, explain why the spotted lanternfly is considered a serious invasive pest in the eastern United States."
        "In two sentences, explain why the emerald ash borer kills ash trees so effectively."
        "In two sentences, explain why bed bugs have resurged in major cities since the late 20th century."
        "In two sentences, explain why mountain pine beetle outbreaks can devastate western forests."
        "In two sentences, explain why Varroa mites are such a major threat to honey bee colonies."
    )
    # Pick one prompt based on day of week (0=Sun .. 6=Sat)
    local dow
    dow=$(date +%w)
    echo "${all_prompts[$dow]}" >"$PROMPT_FILE"
}

run_with_timeout() {
    local timeout_s="$1"; shift
    local timed_out
    timed_out="$(mktemp)"
    rm -f "$timed_out"

    "$@" &
    local pid=$!
    (
        sleep "$timeout_s"
        if kill -0 "$pid" 2>/dev/null; then
            touch "$timed_out"
            kill_tree "$pid"
        fi
    ) &
    local watchdog=$!
    local status=0
    wait "$pid" 2>/dev/null || status=$?
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true

    if [ -f "$timed_out" ]; then
        rm -f "$timed_out"
        return 124
    fi
    rm -f "$timed_out"
    return "$status"
}

kill_tree() {
    local pid="$1"
    local child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        kill_tree "$child"
    done
    kill "$pid" 2>/dev/null || true
}

zig_for_current_ref() {
    local version
    version="$(sed -n 's/.*minimum_zig_version = "\([^"]*\)".*/\1/p' "$ANTFLY_DIR/pkg/termite/build.zig.zon" | head -1)"
    if [ -n "$version" ] && command -v zigup >/dev/null 2>&1; then
        printf 'zigup run %s' "$version"
    else
        printf 'zig'
    fi
}

termite_build_flags_for_current_ref() {
    local flags="$TERMITE_BUILD_FLAGS_BASE"
    local build_zig="$ANTFLY_DIR/pkg/termite/build.zig"
    if [ -f "$MLX_ROOT/lib/libmlxc.dylib" ] && grep -q '"mlx"' "$build_zig"; then
        flags="$flags -Dmlx=true"
        if grep -q '"mlx-root"' "$build_zig"; then
            flags="$flags -Dmlx-root=$MLX_ROOT"
        fi
    fi
    printf '%s' "$flags"
}

termite_build_flags_without_mlx() {
    local flags="$TERMITE_BUILD_FLAGS_BASE"
    local build_zig="$ANTFLY_DIR/pkg/termite/build.zig"
    if grep -q '"mlx"' "$build_zig"; then
        flags="$flags -Dmlx=false"
    fi
    printf '%s' "$flags"
}

build_termite() {
    local ref="$1"
    local sha="$2"
    local out_dir="$3"
    local out_bin="$out_dir/termite"
    local build_log="$out_dir/build.log"
    local zig_cmd
    local build_flags

    git -C "$ANTFLY_DIR" checkout --quiet "$ref"
    mkdir -p "$out_dir"
    zig_cmd="$(zig_for_current_ref)"
    build_flags="$(termite_build_flags_for_current_ref)"
    log "using $zig_cmd for $ref ($sha)"
    if ! run_with_timeout 1800 bash -c \
        "cd '$ANTFLY_DIR/pkg/termite' && $zig_cmd build $build_flags -Dgit-commit='$sha' >'$build_log' 2>&1"; then
        if [[ "$build_flags" == *"-Dmlx=true"* ]]; then
            log "warning: MLX build failed for $ref ($sha); retrying without MLX"
            build_flags="$(termite_build_flags_without_mlx)"
            if run_with_timeout 1800 bash -c \
                "cd '$ANTFLY_DIR/pkg/termite' && $zig_cmd build $build_flags -Dgit-commit='$sha' >'$build_log' 2>&1"; then
                cp "$ANTFLY_DIR/pkg/termite/zig-out/bin/termite" "$out_bin"
                return 0
            fi
        fi
        log "ERROR: build timed out or failed for $ref ($sha); tail of $build_log:"
        tail -80 "$build_log" >&2 || true
        return 1
    fi
    if [ ! -x "$ANTFLY_DIR/pkg/termite/zig-out/bin/termite" ]; then
        log "ERROR: build for $ref ($sha) did not produce zig-out/bin/termite; tail of $build_log:"
        tail -80 "$build_log" >&2 || true
        return 1
    fi
    cp "$ANTFLY_DIR/pkg/termite/zig-out/bin/termite" "$out_bin"
}

build_antfly_main_termite() {
    local sha="$1"
    local out_dir="$2"
    local out_bin="$out_dir/termite"
    local build_log="$out_dir/build.log"

    mkdir -p "$out_dir"
    git -C "$ANTFLY_MAIN_SRC_DIR" checkout --quiet --force origin/main
    log "go testing Antfly main lib/termite ($sha)..."
    if ! run_with_timeout 600 bash -c \
        "cd '$ANTFLY_MAIN_SRC_DIR' && GOEXPERIMENT=simd go test ./lib/termite >'$build_log' 2>&1"; then
        log "ERROR: go test failed for Antfly main lib/termite ($sha); tail of $build_log:"
        tail -80 "$build_log" >&2 || true
        return 1
    fi

    log "building Termite (public antfly repo) CLI ($sha)..."
    if ! run_with_timeout 1800 bash -c \
        "cd '$ANTFLY_MAIN_SRC_DIR/pkg/termite' && GOEXPERIMENT=simd go build -o '$out_bin' ./cmd >>'$build_log' 2>&1"; then
        log "ERROR: go build failed for Termite (public antfly repo) CLI ($sha); tail of $build_log:"
        tail -80 "$build_log" >&2 || true
        return 1
    fi
    if [ ! -x "$out_bin" ]; then
        log "ERROR: Termite (public antfly repo) build did not produce $out_bin; tail of $build_log:"
        tail -80 "$build_log" >&2 || true
        return 1
    fi
}

trap notify_failure ERR

# ─── Fetch latest code ───────────────────────────────────────────
log "fetching antfly-zig..."
if ! fetch_antfly_zig; then
    log "warning: fetch failed; continuing with cached origin refs/tags"
fi

log "fetching Antfly main..."
ensure_antfly_main_repo

ensure_model

STABLE_SHA=$(git -C "$ANTFLY_DIR" rev-parse --short "$ANTFLY_STABLE_COMMIT")
STABLE_DATE=$(git -C "$ANTFLY_DIR" log -1 --format=%cs "$ANTFLY_STABLE_COMMIT")
LATEST_SHA=$(git -C "$ANTFLY_DIR" rev-parse --short origin/main)
LATEST_DATE=$(git -C "$ANTFLY_DIR" log -1 --format=%cs origin/main)
ANTFLY_MAIN_SHA=$(git -C "$ANTFLY_MAIN_SRC_DIR" rev-parse --short origin/main)
ANTFLY_MAIN_DATE=$(git -C "$ANTFLY_MAIN_SRC_DIR" log -1 --format=%cs origin/main)
STABLE_LABEL="termite-zig stable ($STABLE_DATE)"
LATEST_LABEL="termite-zig latest ($LATEST_DATE)"
ANTFLY_MAIN_LABEL="Termite (public antfly repo)"

# ─── Build pinned stable, latest origin/main, and Antfly main ────
log "building pinned stable antfly-zig ($STABLE_SHA)..."
build_termite "$ANTFLY_STABLE_COMMIT" "$STABLE_SHA" "$STABLE_DIR"

log "building latest antfly-zig origin/main ($LATEST_SHA)..."
build_termite "origin/main" "$LATEST_SHA" "$LATEST_DIR"

log "building Termite (public antfly repo) ($ANTFLY_MAIN_SHA)..."
build_antfly_main_termite "$ANTFLY_MAIN_SHA" "$ANTFLY_MAIN_DIR"

# ─── Run benchmark ────────────────────────────────────────────────
log "running benchmark..."
write_prompts
BENCH_ARGS=(
    -model "$MODEL_PATH"
    -model-dir "$MODEL_DIR"
    -llama-cli "llama-cli"
    -prompt-file "$PROMPT_FILE"
    -zig-clean "$STABLE_BIN"
    -zig-clean-dir "$ANTFLY_DIR"
    -zig-exp "$LATEST_BIN"
    -zig-exp-dir "$ANTFLY_DIR"
    -termite "$ANTFLY_MAIN_BIN"
    -termite-dir "$ANTFLY_MAIN_SRC_DIR"
    -termite-name "$ANTFLY_MAIN_LABEL"
    -output "$RESULTS"
)

cd "$CIRCUS_DIR"
go run ./cmd/bench "${BENCH_ARGS[@]}" 2>&1

# ─── Fixup engine names in output ────────────────────────────────
STABLE_LABEL_ESCAPED="$(escape_sed_replacement "$STABLE_LABEL")"
LATEST_LABEL_ESCAPED="$(escape_sed_replacement "$LATEST_LABEL")"
ANTFLY_MAIN_LABEL_ESCAPED="$(escape_sed_replacement "$ANTFLY_MAIN_LABEL")"
sed -i '' \
    -e "s/termite-zig (clean)/$STABLE_LABEL_ESCAPED/g" \
    -e "s/termite-zig (exp)/$LATEST_LABEL_ESCAPED/g" \
    -e "s/- $STABLE_LABEL_ESCAPED: \`[^\`]*\`/- $STABLE_LABEL_ESCAPED: \`$STABLE_SHA\`/g" \
    -e "s/- $LATEST_LABEL_ESCAPED: \`[^\`]*\`/- $LATEST_LABEL_ESCAPED: \`$LATEST_SHA\`/g" \
    -e "s/- $ANTFLY_MAIN_LABEL_ESCAPED: \`[^\`]*\`/- $ANTFLY_MAIN_LABEL_ESCAPED: \`$ANTFLY_MAIN_SHA\`/g" \
    "$RESULTS"

log "results written to $RESULTS"

# Leave antfly-zig checked out at the latest benchmarked ref for inspection.
git -C "$ANTFLY_DIR" checkout --quiet origin/main

# ─── Post to Zulip ───────────────────────────────────────────────
if $POST_ZULIP; then
    log "posting to Zulip..."
    if post_zulip_file "$RESULTS"; then
        log "posted."
    else
        log "warning: Zulip credentials missing or post failed; skipping post"
    fi
fi
