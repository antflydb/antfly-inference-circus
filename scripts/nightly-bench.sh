#!/usr/bin/env bash
set -Eeuo pipefail

# Nightly termite benchmark: brew-installed antfly (stable) vs antfly built
# from antfly main with the latest zig (latest, runnable as `antfly inference`).
# Intended to run non-interactively from launchd/cron.
#
# Usage:
#   ./scripts/nightly-bench.sh                  # run + print results
#   ./scripts/nightly-bench.sh --post-zulip     # run + post to Zulip

CIRCUS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ANTFLY_REPO_URL="${ANTFLY_REPO_URL:-https://github.com/antflydb/antfly}"
RESULTS="$CIRCUS_DIR/results/nightly.md"
TERMITE_BUILD_FLAGS_BASE="-Doptimize=ReleaseSafe -Dedition=full"
BENCH_TOKENS="${BENCH_TOKENS:-128}"
DEFAULT_NIGHTLY_MODELS="$(cat <<'MODELS'
qwen3.6-a3b|unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf|28672
gemma4-e4b-qat|unsloth/gemma-4-E4B-it-qat-GGUF|gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf|20480
MODELS
)"
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

# Source layout. Prefer a pre-existing checkout at $CIRCUS_DIR/antfly (which
# is .gitignored), fall back to cloning into .nightly/antfly-src on demand.
DEFAULT_LOCAL_ANTFLY="$CIRCUS_DIR/antfly"
if [ -z "${ANTFLY_SRC_DIR:-}" ]; then
    if [ -d "$DEFAULT_LOCAL_ANTFLY/.git" ]; then
        ANTFLY_SRC_DIR="$DEFAULT_LOCAL_ANTFLY"
    else
        ANTFLY_SRC_DIR="$CIRCUS_DIR/.nightly/antfly-src"
    fi
fi

LATEST_DIR="$CIRCUS_DIR/.nightly/latest"
LATEST_BIN="$LATEST_DIR/antfly"
PROMPT_FILE="$CIRCUS_DIR/.nightly/prompts.txt"
LOG_PATH="${LOG_PATH:-$HOME/.termite-circus-nightly.log}"

STABLE_BIN="${ANTFLY_STABLE_BIN:-$(command -v antfly || true)}"

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

git_fetch() {
    local dir="$1"
    GIT_TERMINAL_PROMPT=0 git \
        -c core.askPass= \
        -c credential.interactive=never \
        -c 'credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential' \
        -C "$dir" \
        fetch --force --tags --quiet origin
}

ensure_antfly_repo() {
    if [ -d "$ANTFLY_SRC_DIR/.git" ]; then
        if ! git_fetch "$ANTFLY_SRC_DIR"; then
            log "warning: fetch failed for $ANTFLY_SRC_DIR; continuing with cached refs/tags"
        fi
        return
    fi

    rm -rf "$ANTFLY_SRC_DIR"
    mkdir -p "$(dirname "$ANTFLY_SRC_DIR")"
    log "cloning antfly into $ANTFLY_SRC_DIR..."
    GIT_TERMINAL_PROMPT=0 git \
        -c core.askPass= \
        -c credential.interactive=never \
        -c 'credential.https://github.com.helper=!/opt/homebrew/bin/gh auth git-credential' \
        clone --quiet "$ANTFLY_REPO_URL" "$ANTFLY_SRC_DIR"
}

ensure_model() {
    local model_repo="$1"
    local model_file="$2"
    local model_path="$HOME/.termite/models/generators/$model_repo/$model_file"

    if [ -f "$model_path" ]; then
        return
    fi

    log "model missing at $model_path; downloading $model_repo/$model_file..."
    run_with_timeout 7200 go run "$CIRCUS_DIR/cmd/model-dl" -repo "$model_repo" -file "$model_file"
}

escape_sed_replacement() {
    printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

write_prompts() {
    mkdir -p "$CIRCUS_DIR/.nightly"
    local all_prompts=(
        "Write a concise field-note about peppered moths on soot-darkened birch trees, explaining how camouflage, predation, and cleaner air changed the frequency of light and dark forms over time."
        "Explain how a moth's wing pattern can be shaped by nocturnal predators, resting surfaces, and industrial pollution, then include one reason biologists still treat the peppered moth story carefully."
        "Describe a night survey in which a researcher compares pale and dark moth morphs across woodland edges, city parks, and restored forests, focusing on what evidence would make the result convincing."
        "Summarize why moths are useful indicators of environmental change, using peppered moths as the central example and contrasting them with one other insect affected by human land use."
        "Explain the difference between a memorable classroom version of the peppered moth story and a rigorous evolutionary study, keeping the answer grounded in selection pressure and habitat context."
        "Write a short natural-history explanation of why moth coloration, tree bark, bird vision, and air quality can interact to produce measurable evolutionary change within a few human generations."
        "Describe how a museum collection of moth specimens could help reconstruct environmental change across decades, including what metadata would be needed to avoid overconfident conclusions."
    )
    # Pick one prompt based on day of week (0=Sun .. 6=Sat)
    local dow
    dow=$(date +%w)
    echo "${all_prompts[$dow]}" >"$PROMPT_FILE"
}

nightly_model_specs() {
    if [ -n "${NIGHTLY_MODELS:-}" ]; then
        printf '%s\n' "$NIGHTLY_MODELS"
        return
    fi
    if [ -n "${MODEL_REPO:-}" ] || [ -n "${MODEL_FILE:-}" ]; then
        printf 'custom|%s|%s|%s\n' \
            "${MODEL_REPO:-ggml-org/gemma-4-E4B-it-GGUF}" \
            "${MODEL_FILE:-gemma-4-E4B-it-Q4_K_M.gguf}" \
            "${BACKEND_BUDGET_MB:-20480}"
        return
    fi
    printf '%s\n' "$DEFAULT_NIGHTLY_MODELS"
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

# antfly main keeps its zig sources under `zig/`. Earlier layouts used
# `pkg/termite` as the build root; tolerate both.
antfly_zig_build_dir() {
    local src_dir="${1:-$ANTFLY_SRC_DIR}"
    if [ -f "$src_dir/zig/build.zig" ]; then
        printf '%s\n' "$src_dir/zig"
        return 0
    fi
    if [ -f "$src_dir/pkg/termite/build.zig" ]; then
        printf '%s\n' "$src_dir/pkg/termite"
        return 0
    fi
    log "ERROR: unable to find a zig build root under $src_dir"
    return 1
}

zig_for_build_dir() {
    local build_dir="$1"
    local zon="$build_dir/build.zig.zon"
    local version=""
    if [ -f "$zon" ]; then
        version="$(sed -n 's/.*minimum_zig_version = "\([^"]*\)".*/\1/p' "$zon" | head -1)"
    fi
    if [ -n "$version" ] && command -v zigup >/dev/null 2>&1; then
        printf 'zigup run %s' "$version"
    else
        printf 'zig'
    fi
}

termite_build_flags_for_build_dir() {
    local build_dir="$1"
    local flags="$TERMITE_BUILD_FLAGS_BASE"
    local build_zig="$build_dir/build.zig"
    if [ -f "$MLX_ROOT/lib/libmlxc.dylib" ] && grep -q '"mlx"' "$build_zig" 2>/dev/null; then
        flags="$flags -Dmlx=true"
        if grep -q '"mlx-root"' "$build_zig" 2>/dev/null; then
            flags="$flags -Dmlx-root=$MLX_ROOT"
        fi
    fi
    printf '%s' "$flags"
}

termite_build_flags_without_mlx() {
    local build_dir="$1"
    local flags="$TERMITE_BUILD_FLAGS_BASE"
    local build_zig="$build_dir/build.zig"
    if grep -q '"mlx"' "$build_zig" 2>/dev/null; then
        flags="$flags -Dmlx=false"
    fi
    printf '%s' "$flags"
}

build_latest_antfly() {
    local sha="$1"
    local out_dir="$2"
    local out_bin="$out_dir/antfly"
    local build_log="$out_dir/build.log"
    local build_dir
    local zig_cmd
    local build_flags

    git -C "$ANTFLY_SRC_DIR" checkout --quiet --force origin/main
    build_dir="$(antfly_zig_build_dir "$ANTFLY_SRC_DIR")"
    mkdir -p "$out_dir"
    zig_cmd="$(zig_for_build_dir "$build_dir")"
    build_flags="$(termite_build_flags_for_build_dir "$build_dir")"
    log "using $zig_cmd for antfly origin/main ($sha) in $build_dir"
    if ! run_with_timeout 1800 bash -c \
        "cd '$build_dir' && $zig_cmd build $build_flags -Dantfly-version='$sha' >'$build_log' 2>&1"; then
        if [[ "$build_flags" == *"-Dmlx=true"* ]]; then
            log "warning: MLX build failed for antfly main ($sha); retrying without MLX"
            build_flags="$(termite_build_flags_without_mlx "$build_dir")"
            if ! run_with_timeout 1800 bash -c \
                "cd '$build_dir' && $zig_cmd build $build_flags -Dantfly-version='$sha' >'$build_log' 2>&1"; then
                log "ERROR: build timed out or failed for antfly main ($sha); tail of $build_log:"
                tail -80 "$build_log" >&2 || true
                return 1
            fi
        else
            log "ERROR: build timed out or failed for antfly main ($sha); tail of $build_log:"
            tail -80 "$build_log" >&2 || true
            return 1
        fi
    fi
    if [ ! -x "$build_dir/zig-out/bin/antfly" ]; then
        log "ERROR: build for antfly main ($sha) did not produce zig-out/bin/antfly; tail of $build_log:"
        tail -80 "$build_log" >&2 || true
        return 1
    fi
    cp "$build_dir/zig-out/bin/antfly" "$out_bin"
}

if [[ "${NIGHTLY_BENCH_LIB_ONLY:-false}" == "true" ]]; then
    return 0 2>/dev/null || exit 0
fi

trap notify_failure ERR

# ─── Stable: brew-installed antfly ───────────────────────────────
if [ -z "${STABLE_BIN:-}" ]; then
    log "ERROR: no stable antfly binary found on PATH (override with ANTFLY_STABLE_BIN)"
    exit 1
fi
if [ ! -x "$STABLE_BIN" ]; then
    log "ERROR: stable antfly binary at $STABLE_BIN is not executable"
    exit 1
fi
STABLE_VERSION="$($STABLE_BIN --version 2>/dev/null | head -1 | awk '{print $NF}' || echo unknown)"
STABLE_LABEL="antfly (brew $STABLE_VERSION)"

# ─── Latest: build antfly from antfly/zig ────────────────────────
log "fetching antfly main into $ANTFLY_SRC_DIR..."
ensure_antfly_repo

LATEST_SHA=$(git -C "$ANTFLY_SRC_DIR" rev-parse --short origin/main)
LATEST_DATE=$(git -C "$ANTFLY_SRC_DIR" log -1 --format=%cs origin/main)
LATEST_LABEL="antfly latest ($LATEST_DATE)"

log "building latest antfly from $ANTFLY_SRC_DIR ($LATEST_SHA)..."
build_latest_antfly "$LATEST_SHA" "$LATEST_DIR"

# ─── Run benchmark ────────────────────────────────────────────────
log "running benchmark..."
write_prompts
mkdir -p "$(dirname "$RESULTS")"
{
    echo "# Termite nightly benchmark"
    echo
    echo "**Date:** $(date '+%Y-%m-%d %H:%M')"
    echo "**Host:** $(hostname)"
    echo "**Max tokens:** $BENCH_TOKENS"
    echo
} >"$RESULTS"

cd "$CIRCUS_DIR"
while IFS='|' read -r model_name model_repo model_file backend_budget_mb; do
    [ -n "$model_name" ] || continue
    model_dir="$HOME/.termite/models/generators/$model_repo"
    model_path="$model_dir/$model_file"
    model_results="$CIRCUS_DIR/results/nightly-$model_name.md"

    ensure_model "$model_repo" "$model_file"
    log "running benchmark for $model_name..."
    BENCH_ARGS=(
        -model "$model_path"
        -model-dir "$model_dir"
        -llama-cli "llama-cli"
        -prompt-file "$PROMPT_FILE"
        -n "$BENCH_TOKENS"
        -backend-budget-mb "$backend_budget_mb"
        -antfly-stable "$STABLE_BIN"
        -antfly-stable-name "$STABLE_LABEL"
        -antfly-latest "$LATEST_BIN"
        -antfly-latest-dir "$ANTFLY_SRC_DIR"
        -antfly-latest-name "$LATEST_LABEL"
        -output "$model_results"
    )

    go run ./cmd/bench "${BENCH_ARGS[@]}" 2>&1

    # Stable has no source dir so the bench harness can't compute a git hash.
    # Inject the brew version so the Versions block stays useful.
    STABLE_LABEL_ESCAPED="$(escape_sed_replacement "$STABLE_LABEL")"
    sed -i '' \
        -e "/\*\*Versions:\*\*/a\\
- $STABLE_LABEL_ESCAPED: \`$STABLE_VERSION\`
" \
        "$model_results" 2>/dev/null || true

    {
        echo
        cat "$model_results"
    } >>"$RESULTS"
done < <(nightly_model_specs)

log "results written to $RESULTS"

# Leave the antfly source checkout at origin/main for inspection.
git -C "$ANTFLY_SRC_DIR" checkout --quiet origin/main

# ─── Post to Zulip ───────────────────────────────────────────────
if $POST_ZULIP; then
    log "posting to Zulip..."
    if post_zulip_file "$RESULTS"; then
        log "posted."
    else
        log "warning: Zulip credentials missing or post failed; skipping post"
    fi
fi
