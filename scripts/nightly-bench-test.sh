#!/usr/bin/env bash
set -euo pipefail

CIRCUS_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
NIGHTLY_BENCH_LIB_ONLY=true source "$CIRCUS_DIR/scripts/nightly-bench.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_build_dir() {
    local src_dir="$1"
    local want_build_dir="$2"
    local got_build_dir

    got_build_dir="$(antfly_zig_build_dir "$src_dir")"
    [ "$got_build_dir" = "$want_build_dir" ] || fail "build dir: got $got_build_dir, want $want_build_dir"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

new_layout="$tmp/new"
mkdir -p "$new_layout/zig"
touch "$new_layout/zig/build.zig"
assert_build_dir "$new_layout" "$new_layout/zig"

old_layout="$tmp/old"
mkdir -p "$old_layout/pkg/termite"
touch "$old_layout/pkg/termite/build.zig"
assert_build_dir "$old_layout" "$old_layout/pkg/termite"

missing_layout="$tmp/missing"
mkdir -p "$missing_layout"
if antfly_zig_build_dir "$missing_layout" >/dev/null 2>&1; then
    fail "missing layout unexpectedly resolved"
fi

echo "nightly-bench layout tests passed"
