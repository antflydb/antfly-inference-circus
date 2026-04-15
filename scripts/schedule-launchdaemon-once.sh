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
    exit 1
fi

/usr/libexec/PlistBuddy -c "Delete :StartCalendarInterval" "$TARGET_PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval dict" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Year integer $YEAR" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Month integer $MONTH" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Day integer $DAY" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Hour integer $HOUR" "$TARGET_PLIST"
/usr/libexec/PlistBuddy -c "Add :StartCalendarInterval:Minute integer $MINUTE" "$TARGET_PLIST"

launchctl bootout system "$TARGET_PLIST" >/dev/null 2>&1 || true
launchctl bootstrap system "$TARGET_PLIST"
launchctl print "system/$LABEL"

echo
echo "scheduled one-shot run for $DATE_ARG $TIME_ARG"
