#!/usr/bin/env bash
# tui-uat.sh — tmux-based TUI acceptance testing harness for mold
#
# Usage:
#   scripts/tui-uat.sh launch [--local] [--host URL]  # Start TUI in tmux session
#   scripts/tui-uat.sh capture                          # Capture current screen
#   scripts/tui-uat.sh send <keys...>                   # Send keystrokes
#   scripts/tui-uat.sh view <1|2|3|4|name>              # Navigate to a view reliably
#   scripts/tui-uat.sh wait-for <pattern> [timeout_s]   # Wait for text on screen
#   scripts/tui-uat.sh assert <pattern>                  # Assert text is on screen
#   scripts/tui-uat.sh quit                              # Quit TUI and tear down
#   scripts/tui-uat.sh status                            # Check if session is running
#
# The session name is "mold-tui-uat". Only one session at a time.
#
# Key names for send: Enter, Tab, C-c, C-g, C-m, C-r, etc.
# Literal characters: 'q', '2', 'j', 'k', etc.
# Use the `view` command for reliable view switching (handles Escape internally).

set -euo pipefail

SESSION="mold-tui-uat"
WIDTH=140
HEIGHT=45
MOLD_BIN="${MOLD_BIN:-./target/debug/mold}"

# ── Helpers ─��────────────────────────────────────────────────────────

tmux_cmd() {
    command tmux "$@"
}

session_exists() {
    tmux_cmd has-session -t "$SESSION" 2>/dev/null
}

require_session() {
    if ! session_exists; then
        echo "ERROR: No active session '$SESSION'. Run '$0 launch' first." >&2
        exit 1
    fi
}

capture() {
    tmux_cmd capture-pane -t "$SESSION" -p 2>/dev/null
}

send_key() {
    tmux_cmd send-keys -t "$SESSION" "$1"
    sleep 0.15
}


# ── Commands ─────────────────────────────────────────────────���───────

cmd_launch() {
    if session_exists; then
        echo "ERROR: Session '$SESSION' already running. Run '$0 quit' first." >&2
        exit 1
    fi

    if [ ! -x "$MOLD_BIN" ]; then
        echo "ERROR: mold binary not found at $MOLD_BIN" >&2
        echo "Run 'cargo build -p mold-cli --features tui' first, or set MOLD_BIN." >&2
        exit 1
    fi

    local args=("tui")
    args+=("$@")

    tmux_cmd new-session -d -s "$SESSION" -x "$WIDTH" -y "$HEIGHT" \
        "$MOLD_BIN" "${args[@]}"

    # Set escape-time to 0 as a server option (-s) so Escape (C-[) is sent
    # immediately. The default 500ms delay causes tmux to wait for a following
    # character (interpreting Escape+key as Alt+key), which swallows bare
    # Escape keypresses. The tmux server restarts when the last session is
    # killed, so we set this on every launch.
    tmux_cmd set-option -s escape-time 0

    # Wait for TUI to render
    local timeout=10
    for i in $(seq 1 $((timeout * 10))); do
        if capture 2>/dev/null | grep -q "mold"; then
            echo "OK: TUI launched in tmux session '$SESSION' (${WIDTH}x${HEIGHT})"
            return 0
        fi
        sleep 0.1
    done

    echo "ERROR: TUI did not render within ${timeout}s" >&2
    cmd_quit 2>/dev/null || true
    exit 1
}

cmd_capture() {
    require_session
    capture
}

# Render the current screen as a styled HTML file and optionally take a
# PNG screenshot via a local HTTP server + Playwright (if available).
# Usage: scripts/tui-uat.sh screenshot [output.png]
cmd_screenshot() {
    require_session
    local out="${1:-tui-screenshot.png}"
    local html="/tmp/mold-tui-capture.html"

    # Capture with ANSI escape codes and convert to styled HTML
    capture_ansi | nix run nixpkgs#aha -- --black | sed '
s|<head>|<head>\
<style>\
  html, body { margin: 0; padding: 0; background: #1e1e2e; width: fit-content; height: fit-content; }\
  pre { font-family: "JetBrains Mono", "Menlo", "Monaco", "Cascadia Code", "Fira Code", "Consolas", monospace; font-size: 13px; line-height: 1.3; margin: 0; padding: 12px 16px; letter-spacing: 0; white-space: pre; }\
</style>|
s|<body style="color:white; background-color:black">|<body style="color:#cdd6f4; background-color:#1e1e2e">|
' > "$html"

    echo "OK: HTML written to $html"
    echo "Screenshot: open $html in a browser, or use playwright-cli to capture '$out'"
}

# Capture with ANSI escape sequences preserved (for HTML conversion)
capture_ansi() {
    tmux_cmd capture-pane -t "$SESSION" -e -p 2>/dev/null
}

cmd_send() {
    require_session
    if [ $# -eq 0 ]; then
        echo "ERROR: No keys specified. Usage: $0 send <key> [key...]" >&2
        exit 1
    fi
    for key in "$@"; do
        send_key "$key"
    done
    # Allow render time
    sleep 0.3
}

# Reliable view navigation.
#
# All views (Generate, Gallery, Models, Settings) handle number keys 1-4
# for view switching. However, in Generate view with prompt focused, number
# keys type into the prompt instead. We detect this and use C-[ (Escape)
# to enter nav mode first. C-[ from non-Generate views switches to Generate,
# so we never use it from those views.
#
# The Generate nav-mode transition also has a quirk: the first key after
# Escape can be swallowed by the Unfocus action's event cycle. We retry
# the key up to 4 times with polling.
cmd_view() {
    require_session
    local target="$1"
    local key landmark

    # Landmarks must be unique to each view's content area, NOT the tab header.
    # The tab header ("1 Generate  2 Gallery  3 Models  4 Settings") is on every screen.
    case "$target" in
        1|generate|Generate)  key="1"; landmark="Parameters";;
        2|gallery|Gallery)    key="2"; landmark="┌ Gallery";;
        3|models|Models)      key="3"; landmark="┌ Installed\|┌ Available";;
        4|settings|Settings)  key="4"; landmark="┌ Settings";;
        *)
            echo "ERROR: Unknown view '$target'. Use 1-4 or generate/gallery/models/settings." >&2
            exit 1
            ;;
    esac

    # Already on the target view?
    if capture | grep -q "$landmark"; then
        echo "OK: Already on view $target"
        return 0
    fi

    # Navigation strategy:
    # - From Generate prompt focus: Tab (→Parameters) + Escape (→Nav) + key
    # - From other views: key directly (1-4 are handled in all views)
    # - The first character key after entering Nav mode is swallowed by a
    #   crossterm/tmux timing quirk, so we send the key multiple times.
    #
    # On each attempt:
    # 1. If prompt-focused (footer has "Esc Nav" but NOT "q Quit"),
    #    navigate to Nav mode via Tab → Escape
    # 2. Send the view key
    # 3. Check if the view switched
    local attempts=5
    for attempt in $(seq 1 $attempts); do
        local footer
        footer=$(capture | tail -1)

        # Prompt-focused: use Tab→Escape to reach Nav mode
        if echo "$footer" | grep -q "Esc Nav" && ! echo "$footer" | grep -q "q Quit"; then
            send_key Tab
            sleep 0.3
            send_key Escape
            sleep 0.5
        fi

        send_key "$key"
        sleep 1

        if capture | grep -q "$landmark"; then
            echo "OK: Switched to view $target"
            return 0
        fi
    done

    echo "FAIL: Could not switch to view '$target' after $attempts attempts" >&2
    echo "--- Screen dump ---" >&2
    capture >&2
    echo "---" >&2
    return 1
}

cmd_wait_for() {
    require_session

    local pattern="$1"
    local timeout="${2:-5}"

    for i in $(seq 1 $((timeout * 10))); do
        if capture | grep -q "$pattern"; then
            echo "OK: Found '$pattern'"
            return 0
        fi
        sleep 0.1
    done

    echo "FAIL: '$pattern' not found within ${timeout}s" >&2
    echo "--- Screen dump ---" >&2
    capture >&2
    echo "---" >&2
    return 1
}

cmd_assert() {
    require_session

    local pattern="$1"
    if capture | grep -q "$pattern"; then
        echo "PASS: '$pattern' found on screen"
        return 0
    else
        echo "FAIL: '$pattern' not found on screen" >&2
        echo "--- Screen dump ---" >&2
        capture >&2
        echo "---" >&2
        return 1
    fi
}

cmd_quit() {
    if session_exists; then
        # Try graceful quit first
        tmux_cmd send-keys -t "$SESSION" C-c 2>/dev/null || true
        sleep 0.5
        # Kill session regardless
        tmux_cmd kill-session -t "$SESSION" 2>/dev/null || true
        echo "OK: Session '$SESSION' terminated"
    else
        echo "OK: No session to quit"
    fi
}

cmd_status() {
    if session_exists; then
        echo "RUNNING: Session '$SESSION' is active"
        echo "--- Current screen (first 5 lines) ---"
        capture | head -5
    else
        echo "STOPPED: No active session '$SESSION'"
    fi
}

# ── Main ─────────────────────────────────────────────────────────────

case "${1:-help}" in
    launch)
        shift
        cmd_launch "$@"
        ;;
    capture)
        cmd_capture
        ;;
    screenshot)
        shift
        cmd_screenshot "$@"
        ;;
    send)
        shift
        cmd_send "$@"
        ;;
    view)
        shift
        cmd_view "$@"
        ;;
    wait-for)
        shift
        cmd_wait_for "$@"
        ;;
    assert)
        shift
        cmd_assert "$@"
        ;;
    quit)
        cmd_quit
        ;;
    status)
        cmd_status
        ;;
    help|*)
        echo "Usage: $0 {launch|capture|send|view|wait-for|assert|quit|status}"
        echo ""
        echo "Commands:"
        echo "  launch [--local] [--host URL]  Start TUI in tmux"
        echo "  capture                         Print current screen"
        echo "  send <key> [key...]             Send keystrokes"
        echo "  view <1-4|name>                 Navigate to view reliably"
        echo "  wait-for <pattern> [timeout]    Wait for text (default 5s)"
        echo "  assert <pattern>                Check text is on screen"
        echo "  quit                            Tear down session"
        echo "  status                          Check if running"
        echo ""
        echo "Views: 1=Generate, 2=Gallery, 3=Models, 4=Settings"
        echo ""
        echo "Environment:"
        echo "  MOLD_BIN    Path to mold binary (default: ./target/debug/mold)"
        echo ""
        echo "Key names: Enter, Tab, C-c, C-g, C-[ (Escape), etc."
        echo "Literal chars: 'q', '2', 'j', 'k', etc."
        ;;
esac
