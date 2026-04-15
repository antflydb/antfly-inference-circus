#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PLIST="$ROOT_DIR/launchd/com.termite-circus.nightly-bench.plist"
TARGET_PLIST="/Library/LaunchDaemons/com.termite-circus.nightly-bench.plist"
LABEL="com.termite-circus.nightly-bench"

if [[ ! -f "$SOURCE_PLIST" ]]; then
    echo "missing plist template: $SOURCE_PLIST" >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 1
fi

install -o root -g wheel -m 0644 "$SOURCE_PLIST" "$TARGET_PLIST"

if launchctl print "system/$LABEL" >/dev/null 2>&1; then
    launchctl bootout system "$TARGET_PLIST" >/dev/null 2>&1 || true
fi

launchctl bootstrap system "$TARGET_PLIST"
launchctl enable "system/$LABEL"
launchctl print "system/$LABEL"

echo
echo "installed: $TARGET_PLIST"
echo "logs: /Users/stinkbug/.termite-circus-nightly.log"
