#!/usr/bin/env bash
# tui-uat.sh — Ghostty-native TUI acceptance testing harness for mold
#
# Uses Ghostty 1.3+ AppleScript API for native terminal control:
# real font rendering, true colors, and pixel-perfect screenshots.
#
# Usage:
#   scripts/tui-uat.sh launch [--local] [--host URL]  # Start TUI in a Ghostty window
#   scripts/tui-uat.sh capture                          # Capture current screen (plain text)
#   scripts/tui-uat.sh screenshot [output.png]          # Native Ghostty screenshot (PNG)
#   scripts/tui-uat.sh send <keys...>                   # Send keystrokes
#   scripts/tui-uat.sh view <1|2|3|4|name>              # Navigate to a view reliably
#   scripts/tui-uat.sh wait-for <pattern> [timeout_s]   # Wait for text on screen
#   scripts/tui-uat.sh assert <pattern>                  # Assert text is on screen
#   scripts/tui-uat.sh quit                              # Quit TUI and tear down
#   scripts/tui-uat.sh status                            # Check if session is running
#
# Only one UAT session at a time. State is tracked in /tmp/mold-tui-uat.state.
#
# Key names for send:
#   Special keys: enter, escape, tab, space, up, down, left, right,
#                 backspace, delete, home, end, page_up, page_down,
#                 f1-f12
#   With modifiers: ctrl+c, ctrl+g, ctrl+m, ctrl+r, alt+1, shift+tab
#   Literal text:   any other string is sent as text input

set -euo pipefail

STATE_FILE="/tmp/mold-tui-uat.state"
MOLD_BIN="${MOLD_BIN:-./target/debug/mold}"

# ── Helpers ─────────────────────────────────────────────────────────

save_state() {
    local terminal_id="$1"
    echo "$terminal_id" > "$STATE_FILE"
}

load_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: No active UAT session. Run '$0 launch' first." >&2
        exit 1
    fi
    cat "$STATE_FILE"
}

session_exists() {
    [ -f "$STATE_FILE" ] || return 1
    local term_id
    term_id=$(cat "$STATE_FILE")
    # Verify the terminal still exists in Ghostty
    osascript -e "
        tell application \"Ghostty\"
            try
                set term to terminal id \"$term_id\"
                return true
            on error
                return false
            end try
        end tell
    " 2>/dev/null | grep -q "true"
}

require_session() {
    if ! session_exists; then
        rm -f "$STATE_FILE"
        echo "ERROR: No active UAT session. Run '$0 launch' first." >&2
        exit 1
    fi
}

# Capture screen text via Ghostty's write_screen_file action.
# Writes to a temp file and returns the content.
capture() {
    local term_id
    term_id=$(load_state)

    # Save current clipboard, perform action, read file path from clipboard
    local prev_clip
    prev_clip=$(pbpaste 2>/dev/null || true)

    osascript -e "
        tell application \"Ghostty\"
            set term to terminal id \"$term_id\"
            perform action \"write_screen_file:copy,plain\" on term
        end tell
    " >/dev/null 2>&1

    # Small delay for clipboard to update
    sleep 0.1

    local file_path
    file_path=$(pbpaste 2>/dev/null)

    if [ -f "$file_path" ]; then
        cat "$file_path"
    else
        echo "ERROR: Could not read screen capture file: $file_path" >&2
        return 1
    fi

    # Restore clipboard
    if [ -n "$prev_clip" ]; then
        echo -n "$prev_clip" | pbcopy 2>/dev/null || true
    fi
}

# Map key names to Ghostty AppleScript send key format.
# Returns "key:<name>" for special keys, "text:<string>" for literal text,
# or "mod:<modifiers>:<key>" for modified keys.
map_key() {
    local input="$1"

    # Handle modifier combinations: ctrl+x, alt+x, shift+x, cmd+x
    if [[ "$input" =~ ^(ctrl|control|alt|option|shift|cmd|command)\+(.+)$ ]]; then
        local mod="${BASH_REMATCH[1]}"
        local key="${BASH_REMATCH[2]}"
        # Normalize modifier names
        case "$mod" in
            ctrl|control) mod="control" ;;
            alt|option)   mod="option" ;;
            shift)        mod="shift" ;;
            cmd|command)  mod="command" ;;
        esac
        echo "mod:$mod:$key"
        return
    fi

    # Handle tmux-style C-x notation for backward compat
    if [[ "$input" =~ ^C-(.+)$ ]]; then
        echo "mod:control:${BASH_REMATCH[1]}"
        return
    fi

    # Special key names (case-insensitive matching)
    case "${input,,}" in
        enter|return)     echo "key:return" ;;
        escape|esc)       echo "key:escape" ;;
        tab)              echo "key:tab" ;;
        space)            echo "key:space" ;;
        backspace)        echo "key:backspace" ;;
        delete)           echo "key:delete" ;;
        up)               echo "key:up" ;;
        down)             echo "key:down" ;;
        left)             echo "key:left" ;;
        right)            echo "key:right" ;;
        home)             echo "key:home" ;;
        end)              echo "key:end" ;;
        page_up|pageup)   echo "key:page_up" ;;
        page_down|pagedn) echo "key:page_down" ;;
        f[0-9]|f1[0-2])  echo "key:${input,,}" ;;
        *)
            # Single character or literal text
            echo "text:$input"
            ;;
    esac
}

send_one_key() {
    local term_id="$1"
    local input="$2"
    local mapped
    mapped=$(map_key "$input")

    local type="${mapped%%:*}"
    local rest="${mapped#*:}"

    case "$type" in
        key)
            osascript -e "
                tell application \"Ghostty\"
                    set term to terminal id \"$term_id\"
                    send key \"$rest\" to term
                end tell
            " >/dev/null 2>&1
            ;;
        mod)
            local modifier="${rest%%:*}"
            local key="${rest#*:}"
            osascript -e "
                tell application \"Ghostty\"
                    set term to terminal id \"$term_id\"
                    send key \"$key\" modifiers \"$modifier\" to term
                end tell
            " >/dev/null 2>&1
            ;;
        text)
            osascript -e "
                tell application \"Ghostty\"
                    set term to terminal id \"$term_id\"
                    input text \"$rest\" to term
                end tell
            " >/dev/null 2>&1
            ;;
    esac
    sleep 0.15
}

# Get the CGWindowID for the UAT Ghostty window (needed for screencapture).
get_cg_window_id() {
    local term_id
    term_id=$(load_state)

    # Get the window name that contains our terminal
    local win_name
    win_name=$(osascript -e "
        tell application \"Ghostty\"
            set term to terminal id \"$term_id\"
            -- Walk up: terminal is in a tab, tab is in a window
            repeat with w in every window
                repeat with t in every tab of w
                    repeat with trm in every terminal of t
                        if id of trm is \"$term_id\" then
                            return name of w
                        end if
                    end repeat
                end repeat
            end repeat
            return \"\"
        end tell
    " 2>/dev/null)

    if [ -z "$win_name" ]; then
        echo "ERROR: Could not find window for terminal $term_id" >&2
        return 1
    fi

    # Find CGWindowID via Swift CGWindowListCopyWindowInfo
    swift -e "
import CoreGraphics
let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
for w in windows {
    if let owner = w[kCGWindowOwnerName as String] as? String, owner == \"Ghostty\",
       let name = w[kCGWindowName as String] as? String, name == \"$win_name\",
       let layer = w[kCGWindowLayer as String] as? Int, layer == 0,
       let wid = w[kCGWindowNumber as String] as? Int {
        print(wid)
        break
    }
}
" 2>/dev/null
}


# ── Commands ────────────────────────────────────────────────────────

cmd_launch() {
    if session_exists; then
        echo "ERROR: UAT session already running. Run '$0 quit' first." >&2
        exit 1
    fi
    rm -f "$STATE_FILE"

    if [ ! -x "$MOLD_BIN" ]; then
        echo "ERROR: mold binary not found at $MOLD_BIN" >&2
        echo "Run 'cargo build -p mold-cli --features tui' first, or set MOLD_BIN." >&2
        exit 1
    fi

    # Build the mold tui command with arguments
    local cmd
    cmd="$MOLD_BIN tui $*"

    # Create a new Ghostty window running the TUI
    local term_id
    term_id=$(osascript -e "
        tell application \"Ghostty\"
            activate
            set cfg to new surface configuration
            set command of cfg to \"$cmd\"
            set wait after command of cfg to true
            set initial working directory of cfg to \"$(pwd)\"
            set w to new window with configuration cfg
            delay 0.5
            set term to focused terminal of selected tab of w
            return id of term
        end tell
    " 2>&1)

    if [ -z "$term_id" ] || [[ "$term_id" == *"error"* ]]; then
        echo "ERROR: Failed to create Ghostty window: $term_id" >&2
        exit 1
    fi

    save_state "$term_id"

    # Wait for TUI to render
    local timeout=10
    for _ in $(seq 1 $((timeout * 4))); do
        if capture 2>/dev/null | grep -q "mold"; then
            echo "OK: TUI launched in Ghostty window (terminal: $term_id)"
            return 0
        fi
        sleep 0.25
    done

    echo "ERROR: TUI did not render within ${timeout}s" >&2
    cmd_quit 2>/dev/null || true
    exit 1
}

cmd_capture() {
    require_session
    capture
}

cmd_screenshot() {
    require_session
    local out="${1:-tui-screenshot.png}"

    local wid
    wid=$(get_cg_window_id)

    if [ -z "$wid" ]; then
        echo "ERROR: Could not determine CGWindowID for screenshot" >&2
        return 1
    fi

    # -o: no shadow, -l: capture specific window by ID
    screencapture -o -l"$wid" "$out" 2>&1

    if [ -f "$out" ]; then
        local size
        size=$(stat -f%z "$out" 2>/dev/null || stat --printf=%s "$out" 2>/dev/null)
        echo "OK: Screenshot saved to $out ($(( size / 1024 ))KB, window ID: $wid)"
    else
        echo "ERROR: Screenshot failed" >&2
        return 1
    fi
}

cmd_send() {
    require_session
    if [ $# -eq 0 ]; then
        echo "ERROR: No keys specified. Usage: $0 send <key> [key...]" >&2
        exit 1
    fi
    local term_id
    term_id=$(load_state)
    for key in "$@"; do
        send_one_key "$term_id" "$key"
    done
    # Allow render time
    sleep 0.3
}

# Reliable view navigation.
#
# All views (Generate, Gallery, Models, Settings) handle number keys 1-4
# for view switching. However, in Generate view with prompt focused, number
# keys type into the prompt instead. We detect this and send Escape to
# enter nav mode first.
cmd_view() {
    require_session
    local target="$1"
    local key landmark
    local term_id
    term_id=$(load_state)

    # Landmarks must be unique to each view's content area, NOT the tab header.
    # The tab header ("1 Generate  2 Gallery  3 Models  4 Settings") is on every screen,
    # so we use box-drawing prefixes (┌) to match section headers instead.
    case "$target" in
        1|generate|Generate)  key="1"; landmark="┌ Parameters\|┌ Prompt";;
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
    # - From Generate prompt focus: Tab (to Parameters) + Escape (to Nav) + key
    # - From other views: key directly (1-4 work in all views)
    local attempts=5
    for attempt in $(seq 1 $attempts); do
        local footer
        footer=$(capture | tail -1)

        # Prompt-focused: use Tab + Escape to reach Nav mode
        if echo "$footer" | grep -q "Esc Nav" && ! echo "$footer" | grep -q "q Quit"; then
            send_one_key "$term_id" tab
            sleep 0.3
            send_one_key "$term_id" escape
            sleep 0.5
        fi

        send_one_key "$term_id" "$key"
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

    for _ in $(seq 1 $((timeout * 4))); do
        if capture | grep -q "$pattern"; then
            echo "OK: Found '$pattern'"
            return 0
        fi
        sleep 0.25
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
    if [ -f "$STATE_FILE" ]; then
        local term_id
        term_id=$(cat "$STATE_FILE")
        # Try to close the terminal via AppleScript
        osascript -e "
            tell application \"Ghostty\"
                try
                    set term to terminal id \"$term_id\"
                    close term
                end try
            end tell
        " >/dev/null 2>&1 || true
        rm -f "$STATE_FILE"
        echo "OK: UAT session terminated"
    else
        echo "OK: No session to quit"
    fi
}

cmd_status() {
    if session_exists; then
        local term_id
        term_id=$(cat "$STATE_FILE")
        echo "RUNNING: UAT session active (terminal: $term_id)"
        echo "--- Current screen (first 5 lines) ---"
        capture | head -5
    else
        rm -f "$STATE_FILE" 2>/dev/null || true
        echo "STOPPED: No active UAT session"
    fi
}

# ── Main ────────────────────────────────────────────────────────────

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
        cmd_screenshot "${1:-}"
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
        echo "Usage: $0 {launch|capture|screenshot|send|view|wait-for|assert|quit|status}"
        echo ""
        echo "Commands:"
        echo "  launch [--local] [--host URL]  Start TUI in a Ghostty window"
        echo "  capture                         Print current screen (plain text)"
        echo "  screenshot [output.png]         Native Ghostty screenshot (PNG)"
        echo "  send <key> [key...]             Send keystrokes"
        echo "  view <1-4|name>                 Navigate to view reliably"
        echo "  wait-for <pattern> [timeout]    Wait for text (default 5s)"
        echo "  assert <pattern>                Check text is on screen"
        echo "  quit                            Close UAT window"
        echo "  status                          Check if running"
        echo ""
        echo "Views: 1=Generate, 2=Gallery, 3=Models, 4=Settings"
        echo ""
        echo "Environment:"
        echo "  MOLD_BIN    Path to mold binary (default: ./target/debug/mold)"
        echo ""
        echo "Key names: enter, escape, tab, space, up, down, left, right,"
        echo "           backspace, delete, home, end, page_up, page_down, f1-f12"
        echo "Modifiers: ctrl+c, ctrl+g, alt+1, shift+tab"
        echo "Text:      any other string is sent as literal text input"
        echo ""
        echo "Requires: Ghostty 1.3+ with macos-applescript=true (default)"
        ;;
esac
