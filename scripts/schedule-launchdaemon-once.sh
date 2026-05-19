#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: sudo $0 YYYY-MM-DD HH:MM" >&2
    exit 1
fi

DATE_ARG="$1"
TIME_ARG="$2"
LABEL="com.termite-circus.nightly-bench"
TARGET_PLIST="/Library/LaunchDaemons/$LABEL.plist"

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo $0 $DATE_ARG $TIME_ARG" >&2
    exit 1
fi

IFS=- read -r YEAR MONTH DAY <<<"$DATE_ARG"
IFS=: read -r HOUR MINUTE <<<"$TIME_ARG"

for value in "$YEAR" "$MONTH" "$DAY" "$HOUR" "$MINUTE"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "invalid date/time: $DATE_ARG $TIME_ARG" >&2
        exit 1
    fi
done

if [[ ! -f "$TARGET_PLIST" ]]; then
    echo "missing installed plist: $TARGET_PLIST" >&2
    echo "run sudo ./scripts/install-launchdaemon.sh first" >&2
    exit 1
fi

# Unload, patch to one-shot time, reload
if launchctl print "system/$LABEL" >/dev/null 2>&1; then
    launchctl bootout "system/$LABEL" 2>/dev/null || true
fi

/usr/libexec/PlistBuddy -c "Delete :StartCalendarInterval" "$TARGET_PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval dict" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Year integer $YEAR" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Month integer $MONTH" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Day integer $DAY" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Hour integer $HOUR" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Minute integer $MINUTE" "$TARGET_PLIST"

xattr -c "$TARGET_PLIST"
launchctl bootstrap system "$TARGET_PLIST"
launchctl enable "system/$LABEL"

echo "scheduled one-shot run for $DATE_ARG $TIME_ARG"
echo "NOTE: re-run sudo ./scripts/install-launchdaemon.sh afterwards to restore the nightly 2:05 AM schedule"
launchctl print "system/$LABEL" 2>&1 | head -5
