#!/usr/bin/env bash
set -euo pipefail

CIRCUS_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
NIGHTLY_BENCH_LIB_ONLY=true source "$CIRCUS_DIR/scripts/nightly-bench.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_layout() {
    local src_dir="$1"
    local want_test_dir="$2"
    local want_test_pkgs="$3"
    local want_build_dir="$4"
    local got_test_dir
    local got_test_pkgs
    local got_build_dir

    IFS=$'\t' read -r got_test_dir got_test_pkgs got_build_dir < <(antfly_main_termite_layout "$src_dir")

    [ "$got_test_dir" = "$want_test_dir" ] || fail "test dir: got $got_test_dir, want $want_test_dir"
    [ "$got_test_pkgs" = "$want_test_pkgs" ] || fail "test packages: got $got_test_pkgs, want $want_test_pkgs"
    [ "$got_build_dir" = "$want_build_dir" ] || fail "build dir: got $got_build_dir, want $want_build_dir"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

new_layout="$tmp/new"
mkdir -p "$new_layout/go/pkg/termite/cmd"
touch "$new_layout/go/pkg/termite/go.mod"
assert_layout \
    "$new_layout" \
    "$new_layout/go/pkg/termite" \
    ". ./cmd/..." \
    "$new_layout/go/pkg/termite"

old_layout="$tmp/old"
mkdir -p "$old_layout/lib/termite" "$old_layout/pkg/termite/cmd"
assert_layout \
    "$old_layout" \
    "$old_layout" \
    "./lib/termite" \
    "$old_layout/pkg/termite"

missing_layout="$tmp/missing"
mkdir -p "$missing_layout"
if antfly_main_termite_layout "$missing_layout" >/dev/null 2>&1; then
    fail "missing layout unexpectedly resolved"
fi

echo "nightly-bench layout tests passed"
