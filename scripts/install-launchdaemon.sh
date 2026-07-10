#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PLIST="$ROOT_DIR/launchd/com.antfly-inference.nightly-bench.plist"
TARGET_PLIST="/Library/LaunchDaemons/com.antfly-inference.nightly-bench.plist"
LABEL="com.antfly-inference.nightly-bench"

if [[ ! -f "$SOURCE_PLIST" ]]; then
    echo "missing plist template: $SOURCE_PLIST" >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 1
fi

# Unload if already loaded
if launchctl print "system/$LABEL" >/dev/null 2>&1; then
    echo "unloading existing job..."
    launchctl bootout "system/$LABEL" 2>/dev/null || true
fi

install -o root -g wheel -m 0644 "$SOURCE_PLIST" "$TARGET_PLIST"
# macOS may add provenance/quarantine xattrs (e.g. if written by a sandboxed
# process) which silently prevent launchctl from loading the plist.
xattr -c "$TARGET_PLIST"

launchctl bootstrap system "$TARGET_PLIST"
launchctl enable "system/$LABEL"

echo "installed and loaded: $LABEL"
echo "next run: 2:00 AM Monday/Wednesday/Friday"
echo "logs: ~/.antfly-inference-nightly.log"
launchctl print "system/$LABEL" 2>&1 | head -5
