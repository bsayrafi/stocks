#!/bin/bash
# install_report_server.sh - start report_server.py automatically at login (macOS launchd) and right now.
#
#   source .venv/bin/activate          # (or however you activate the venv the screener runs in)
#   ./install_report_server.sh         # uses that venv's python3
#   ./install_report_server.sh /path/to/venv/bin/python3     # or name the python explicitly
#   ./install_report_server.sh --uninstall
#
# Output of the server goes to logs/report_server.log next to this script.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.screener.reportserver"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [[ "${1:-}" == "--uninstall" ]]; then
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed: the report server no longer starts at login (and is stopped)."
    exit 0
fi

PY="${1:-$(command -v python3)}"
[[ -x "$PY" ]] || { echo "python not found: $PY"; exit 1; }
# absolute path WITHOUT resolving symlinks: a venv's bin/python3 is a symlink, and following it would lose the venv
PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
[[ -f "$DIR/report_server.py" ]] || { echo "report_server.py not found in $DIR"; exit 1; }
[[ -f "$DIR/run_trend_entry.py" ]] || { echo "run_trend_entry.py not found in $DIR"; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents" "$DIR/logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PY</string>
        <string>$DIR/report_server.py</string>
    </array>
    <key>WorkingDirectory</key><string>$DIR</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>30</integer>
    <key>StandardOutPath</key><string>$DIR/logs/report_server.log</string>
    <key>StandardErrorPath</key><string>$DIR/logs/report_server.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>$(dirname "$PY"):/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true      # reinstall: stop the old one first
launchctl bootstrap "$DOMAIN" "$PLIST"
sleep 2
if curl -fsS -o /dev/null "http://127.0.0.1:${REPORT_SERVER_PORT:-8765}/ping.gif"; then
    echo "OK: report server running on http://127.0.0.1:${REPORT_SERVER_PORT:-8765} (python: $PY)"
    echo "It starts by itself at every login. Log: $DIR/logs/report_server.log"
else
    echo "Installed, but the server does not answer yet - look at $DIR/logs/report_server.log"
    echo "If it says 'Operation not permitted': the project is in Documents/Desktop/Downloads, which macOS protects."
    echo "Then either move the project folder elsewhere, or give that python Full Disk Access"
    echo "(System Settings > Privacy & Security > Full Disk Access > + > $PY)."
fi
