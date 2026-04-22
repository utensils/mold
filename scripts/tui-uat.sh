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
DEFAULT_MOLD_BIN="./target/dev-fast/mold"
if [ ! -x "$DEFAULT_MOLD_BIN" ]; then
    DEFAULT_MOLD_BIN="./target/debug/mold"
fi
MOLD_BIN="${MOLD_BIN:-$DEFAULT_MOLD_BIN}"

# ── Helpers ─────────────────────────────────────────────────────────

# State file format (one key=value per line):
#   terminal=<ghostty terminal id>
#   mold_home=<absolute path>    (empty if not isolated)
#   db_path=<absolute path>      (empty if not explicitly set)
# Legacy state (plain terminal id, no key=) is still accepted by
# `load_state` for backward compatibility — `mold_home` / `db_path`
# are reported empty in that case.

save_state() {
    local terminal_id="$1"
    local mold_home="${2:-}"
    local db_path="${3:-}"
    {
        printf 'terminal=%s\n' "$terminal_id"
        printf 'mold_home=%s\n' "$mold_home"
        printf 'db_path=%s\n' "$db_path"
    } > "$STATE_FILE"
}

load_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "ERROR: No active UAT session. Run '$0 launch' first." >&2
        exit 1
    fi
    # Legacy single-line format: a bare terminal ID with no `=`. Newer
    # format is `terminal=<id>\nmold_home=<path>\ndb_path=<path>`.
    local first
    first=$(head -1 "$STATE_FILE")
    if [[ "$first" != *=* ]]; then
        echo "$first"
        return 0
    fi
    awk -F= '$1=="terminal"{print $2; exit}' "$STATE_FILE"
}

# Print the session's MOLD_HOME (or empty string for non-isolated launches).
load_mold_home() {
    [ -f "$STATE_FILE" ] || return 0
    awk -F= '$1=="mold_home"{print $2; exit}' "$STATE_FILE"
}

# Print the session's MOLD_DB_PATH if set, else the default path derived
# from MOLD_HOME (for isolated sessions) or `$HOME/.mold/mold.db`.
load_db_path() {
    [ -f "$STATE_FILE" ] || return 0
    local explicit home
    explicit=$(awk -F= '$1=="db_path"{print $2; exit}' "$STATE_FILE")
    if [ -n "$explicit" ]; then
        echo "$explicit"
        return 0
    fi
    home=$(load_mold_home)
    if [ -n "$home" ]; then
        echo "$home/mold.db"
        return 0
    fi
    # Not isolated — point at the user's real DB. Callers that would
    # mutate should check this and refuse, to avoid clobbering real
    # gallery state from a smoke-test gone wrong.
    echo "$HOME/.mold/mold.db"
}

session_exists() {
    [ -f "$STATE_FILE" ] || return 1
    local term_id
    term_id=$(load_state)
    [ -n "$term_id" ] || return 1
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

    # Save current clipboard and ensure it's restored on all exit paths.
    # Use a named-export default so strict-mode `set -u` + the RETURN
    # trap body don't abort the shell when the trap body runs during
    # early exit (bash evaluates expansions inside trap strings lazily).
    local prev_clip=""
    prev_clip=$(pbpaste 2>/dev/null || echo "")
    trap 'if [ -n "${prev_clip:-}" ]; then echo -n "$prev_clip" | pbcopy 2>/dev/null || true; fi' RETURN

    osascript -e "
        tell application \"Ghostty\"
            set term to terminal id \"$term_id\"
            perform action \"write_screen_file:copy,plain\" on term
        end tell
    " >/dev/null 2>&1

    # Poll for clipboard to update (write_screen_file is async)
    local file_path=""
    local poll_attempts=20
    for _ in $(seq 1 $poll_attempts); do
        file_path=$(pbpaste 2>/dev/null)
        # The action writes a temp file path; detect it changed from prev_clip
        if [ "$file_path" != "$prev_clip" ] && [ -f "$file_path" ]; then
            break
        fi
        sleep 0.05
    done

    if [ -f "$file_path" ]; then
        cat "$file_path"
    else
        echo "ERROR: Could not read screen capture file: $file_path" >&2
        return 1
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
    # NOTE: Ghostty's AppleScript API uses "enter", not "return" —
    # sending "return" errors silently with `Unknown key name`.
    case "${input,,}" in
        enter|return)     echo "key:enter" ;;
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

    # Escape window name for Swift string literal
    local escaped_name
    escaped_name="${win_name//\\/\\\\}"
    escaped_name="${escaped_name//\"/\\\"}"

    # Find CGWindowID via Swift CGWindowListCopyWindowInfo
    swift -e "
import CoreGraphics
let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
let target = \"$escaped_name\"
for w in windows {
    if let owner = w[kCGWindowOwnerName as String] as? String, owner == \"Ghostty\",
       let name = w[kCGWindowName as String] as? String, name == target,
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

    # Parse UAT-specific flags (not passed to mold). Everything else
    # flows through to `mold tui`.
    local fresh_home=""
    local -a env_pairs=()
    local -a passthrough=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --fresh)
                fresh_home=$(mktemp -d "${TMPDIR:-/tmp}/mold-uat.XXXXXXXX")
                shift
                ;;
            --env)
                if [ $# -lt 2 ]; then
                    echo "ERROR: --env requires KEY=VALUE" >&2
                    exit 1
                fi
                env_pairs+=("$2")
                shift 2
                ;;
            --)
                shift
                passthrough+=("$@")
                break
                ;;
            *)
                passthrough+=("$1")
                shift
                ;;
        esac
    done

    # Build the env-prefix for the spawned shell. Env vars set in this
    # shell don't propagate through Ghostty's AppleScript command path —
    # we need to inline them via `/usr/bin/env KEY=VAL ... mold tui …`
    # so the TUI process actually sees them.
    local env_prefix="/usr/bin/env"
    if [ -n "$fresh_home" ]; then
        env_pairs+=("MOLD_HOME=$fresh_home")
    fi
    local mold_home_for_state=""
    local db_path_for_state=""
    for pair in "${env_pairs[@]}"; do
        # Single-quote for shell safety; mold args and env values don't
        # contain single quotes in any known scenario.
        env_prefix="$env_prefix '$pair'"
        case "$pair" in
            MOLD_HOME=*)    mold_home_for_state="${pair#MOLD_HOME=}" ;;
            MOLD_DB_PATH=*) db_path_for_state="${pair#MOLD_DB_PATH=}" ;;
        esac
    done

    # Assemble the full shell command. Binary + mold's own CLI args
    # quoted individually so paths with spaces survive.
    local mold_args=""
    for a in "${passthrough[@]}"; do
        # shellcheck disable=SC1003
        mold_args+=" $(printf %q "$a")"
    done
    local shell_cmd
    shell_cmd="$env_prefix $(printf %q "$MOLD_BIN") tui$mold_args"

    # AppleScript string escaping (backslash + double quote only — the
    # command is a plain shell string Ghostty hands to `/bin/sh -c`).
    local cmd_escaped="${shell_cmd//\\/\\\\}"
    cmd_escaped="${cmd_escaped//\"/\\\"}"

    local cwd
    cwd="$(pwd)"
    cwd="${cwd//\\/\\\\}"
    cwd="${cwd//\"/\\\"}"

    # Create a new Ghostty window running the TUI
    local term_id
    term_id=$(osascript -e "
        tell application \"Ghostty\"
            activate
            set cfg to new surface configuration
            set command of cfg to \"$cmd_escaped\"
            set wait after command of cfg to true
            set initial working directory of cfg to \"$cwd\"
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

    save_state "$term_id" "$mold_home_for_state" "$db_path_for_state"

    # Wait for TUI to render
    local timeout=10
    for _ in $(seq 1 $((timeout * 4))); do
        if capture 2>/dev/null | grep -q "mold"; then
            local isolation_msg=""
            if [ -n "$mold_home_for_state" ]; then
                isolation_msg=" (isolated MOLD_HOME=$mold_home_for_state)"
            fi
            echo "OK: TUI launched in Ghostty window (terminal: $term_id)$isolation_msg"
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
        1|generate|Generate)  key="1"; landmark="┌ Parameters|┌ Prompt";;
        2|gallery|Gallery)    key="2"; landmark="┌ Gallery";;
        3|models|Models)      key="3"; landmark="┌ Installed|┌ Available";;
        4|queue|Queue)        key="4"; landmark="┌ Queue";;
        5|settings|Settings)  key="5"; landmark="┌ Appearance|┌ Configuration";;
        *)
            echo "ERROR: Unknown view '$target'. Use 1-5 or generate/gallery/models/queue/settings." >&2
            exit 1
            ;;
    esac

    # Already on the target view?
    if capture | grep -Eq "$landmark"; then
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

        if capture | grep -Eq "$landmark"; then
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
        local term_id home db
        term_id=$(load_state)
        home=$(load_mold_home)
        db=$(load_db_path)
        echo "RUNNING: UAT session active (terminal: $term_id)"
        if [ -n "$home" ]; then
            echo "  MOLD_HOME: $home"
        else
            echo "  MOLD_HOME: (not isolated — uses user's real \$HOME/.mold/)"
        fi
        echo "  DB path:   $db"
        echo "--- Current screen (first 5 lines) ---"
        capture | head -5
    else
        rm -f "$STATE_FILE" 2>/dev/null || true
        echo "STOPPED: No active UAT session"
    fi
}

# Print the session's resolved MOLD_HOME / DB path. Useful from
# follow-up shell commands that want to sqlite3 or inspect files.
cmd_env() {
    require_session
    local home db
    home=$(load_mold_home)
    db=$(load_db_path)
    if [ -n "$home" ]; then
        printf 'MOLD_HOME=%s\n' "$home"
    fi
    printf 'MOLD_DB_PATH=%s\n' "$db"
}

# Is this session writing to the user's real ~/.mold/?
session_is_isolated() {
    [ -n "$(load_mold_home)" ]
}

# Run `sqlite3 "$db" <sql>` against the session's DB. Read-only by
# default; writes require --write to make accidents obvious. Refuses
# to write against the user's real DB without --force.
cmd_db() {
    require_session
    local write=0 force=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --write) write=1; shift ;;
            --force) force=1; shift ;;
            --) shift; break ;;
            *) break ;;
        esac
    done
    if [ $# -eq 0 ]; then
        echo "ERROR: $0 db [--write] [--force] <sql>" >&2
        exit 1
    fi
    local db_path
    db_path=$(load_db_path)
    if [ ! -f "$db_path" ]; then
        echo "ERROR: DB not found at $db_path" >&2
        exit 1
    fi
    if [ "$write" -eq 1 ] && ! session_is_isolated && [ "$force" -eq 0 ]; then
        echo "ERROR: refusing to write against user's real DB at $db_path." >&2
        echo "       Launch with '--fresh' for isolation, or pass --force to override." >&2
        exit 1
    fi
    sqlite3 "$db_path" "$*"
}

# Read a single settings key (e.g. `tui.theme`) and print its value.
cmd_db_get() {
    require_session
    if [ $# -lt 1 ]; then
        echo "ERROR: $0 db-get <key>" >&2
        exit 1
    fi
    local db_path
    db_path=$(load_db_path)
    sqlite3 "$db_path" "SELECT value FROM settings WHERE key='$1' LIMIT 1;"
}

# Assert that the settings row for `key` equals `value`. Non-zero exit
# on mismatch (suitable for test scripts).
cmd_db_assert() {
    require_session
    if [ $# -lt 2 ]; then
        echo "ERROR: $0 db-assert <key> <value>" >&2
        exit 1
    fi
    local key="$1" expect="$2" actual
    actual=$(cmd_db_get "$key")
    if [ "$actual" = "$expect" ]; then
        echo "PASS: settings.$key = $expect"
        return 0
    fi
    echo "FAIL: settings.$key = '$actual' (expected '$expect')" >&2
    return 1
}

# List every tui.*/model_prefs row with updated-at timestamps. Handy
# for spotting stale theme/per-model rows when the visible TUI doesn't
# match what the DB actually has.
cmd_db_dump() {
    require_session
    local db_path
    db_path=$(load_db_path)
    echo "── settings (tui.* / expand.* / generate.*) ──"
    sqlite3 -header -column "$db_path" \
        "SELECT key, value, value_type, datetime(updated_at_ms/1000, 'unixepoch', 'localtime') AS updated
         FROM settings ORDER BY key;"
    echo ""
    echo "── model_prefs ──"
    sqlite3 -header -column "$db_path" \
        "SELECT model, width, height, steps, guidance, scheduler, lora_path,
                datetime(updated_at_ms/1000, 'unixepoch', 'localtime') AS updated
         FROM model_prefs ORDER BY model;"
}

# Move Settings-view focus to `appearance` or `configuration`. Must be
# called with the TUI already on the Settings view.
cmd_settings_focus() {
    require_session
    local target="${1:-}"
    if [ -z "$target" ]; then
        echo "ERROR: $0 settings-focus <appearance|configuration>" >&2
        exit 1
    fi
    local term_id
    term_id=$(load_state)
    # Ensure the Settings view is active.
    if ! capture | grep -Eq "┌ (Appearance|Settings)"; then
        echo "ERROR: must be on Settings view before settings-focus. Run 'view settings' first." >&2
        exit 1
    fi
    case "$target" in
        appearance|Appearance)
            # Walk off the top of the Configuration list → focus flips
            # to Appearance. 50 presses is safely more than the longest
            # row list we ship.
            for _ in $(seq 1 50); do
                send_one_key "$term_id" k
            done
            ;;
        configuration|Configuration)
            # One Down from Appearance flips focus back.
            send_one_key "$term_id" j
            ;;
        *)
            echo "ERROR: unknown focus '$target'. Use 'appearance' or 'configuration'." >&2
            exit 1
            ;;
    esac
    sleep 0.2
    echo "OK: settings focus → $target"
}

# Switch the active model via the Parameters → Model → Enter picker
# path. We deliberately avoid Ctrl+M because AppleScript's
# `send key "m" modifiers "control"` gets aliased to CR (Enter) by
# many terminals — the app never sees the modifier, and the filter
# text gets typed into whatever handler was active, including 'q' →
# Quit. The Parameters-focused Enter path is boring and reliable.
cmd_model() {
    require_session
    local name="${1:-}"
    if [ -z "$name" ]; then
        echo "ERROR: $0 model <name-or-filter>" >&2
        exit 1
    fi
    local term_id
    term_id=$(load_state)
    cmd_view generate >/dev/null
    # Reach Parameters reliably: Escape → Nav mode → Shift+Tab (FocusPrev
    # from Nav lands directly on Parameters, regardless of whether the
    # Negative pane is visible). Then up-key to row 0 (Model) and Enter
    # to open the picker.
    send_one_key "$term_id" escape
    sleep 0.2
    send_one_key "$term_id" shift+tab
    sleep 0.2
    local _
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        send_one_key "$term_id" k
    done
    sleep 0.2
    send_one_key "$term_id" enter
    sleep 0.4
    # Type the filter text character-by-character so each char is an
    # `input text` event (not a `send key`, which would pass modifier
    # state through).
    local i ch
    for (( i=0; i<${#name}; i++ )); do
        ch="${name:$i:1}"
        send_one_key "$term_id" "$ch"
        sleep 0.02
    done
    sleep 0.2
    send_one_key "$term_id" enter
    sleep 0.4
    # Escape regex metacharacters in the model name (`:` is fine in
    # BRE/ERE, but `\.`-style escapes or `+`-like chars would break).
    # Use plain `fgrep` semantics via `grep -F` on the row prefix and a
    # secondary `grep -F "$name"` check to avoid regex surprises.
    if capture | grep -F '│Model' | grep -F "$name" >/dev/null; then
        echo "OK: model switched to $name"
    else
        echo "NOTE: switch attempted; current Model row:" >&2
        capture | grep -F '│Model' | head -1 >&2
        return 1
    fi
}

# Assert a single `model_prefs` column for a given model. Example:
#   db-model-assert flux-dev:q4 width 1024
cmd_db_model_assert() {
    require_session
    if [ $# -lt 3 ]; then
        echo "ERROR: $0 db-model-assert <model> <column> <value>" >&2
        exit 1
    fi
    local model="$1" column="$2" expect="$3"
    local db_path
    db_path=$(load_db_path)
    # Column name must be an allow-listed field; don't interpolate raw
    # input into SQL. This keeps the helper from growing an injection
    # hole if someone paths user input in from a CI script.
    case "$column" in
        width|height|steps|guidance|scheduler|seed_mode|batch|format|\
lora_path|lora_scale|expand|offload|strength|control_scale|frames|fps|\
last_prompt|last_negative) ;;
        *) echo "ERROR: unknown model_prefs column '$column'." >&2; exit 1 ;;
    esac
    local actual
    actual=$(sqlite3 "$db_path" "SELECT IFNULL($column,'') FROM model_prefs WHERE model='$model';")
    if [ "$actual" = "$expect" ]; then
        echo "PASS: model_prefs[$model].$column = $expect"
        return 0
    fi
    echo "FAIL: model_prefs[$model].$column = '$actual' (expected '$expect')" >&2
    return 1
}

# Set the TUI theme by slug. Navigates to Appearance, then cycles the
# current selection until the desired preset is active. Asserts on the
# visible ✓ marker to confirm success.
cmd_theme_set() {
    require_session
    local slug="${1:-}"
    if [ -z "$slug" ]; then
        echo "ERROR: $0 theme-set <mocha|latte|ristretto|gruvbox|tokyo|nord|dracula>" >&2
        exit 1
    fi
    # Presets in display order — must match `ThemePreset::ALL` in
    # crates/mold-tui/src/ui/theme.rs.
    local -a presets=(mocha latte ristretto gruvbox tokyo nord dracula)
    local want_idx=-1 i
    for i in "${!presets[@]}"; do
        if [ "${presets[$i]}" = "$slug" ]; then
            want_idx=$i
            break
        fi
    done
    if [ $want_idx -lt 0 ]; then
        echo "ERROR: unknown theme '$slug'. Valid: ${presets[*]}" >&2
        exit 1
    fi

    # Make sure we're on Settings + Appearance focus.
    cmd_view settings >/dev/null
    cmd_settings_focus appearance >/dev/null

    # Find the currently-active theme from the header (`theme · <slug>`)
    # and cycle forward to the target.
    local cur_slug cur_idx
    cur_slug=$(capture | grep -o 'theme · [a-z]*' | head -1 | awk '{print $3}')
    if [ -z "$cur_slug" ]; then
        echo "ERROR: couldn't read current theme from Appearance header." >&2
        return 1
    fi
    for i in "${!presets[@]}"; do
        if [ "${presets[$i]}" = "$cur_slug" ]; then
            cur_idx=$i
            break
        fi
    done
    local total=${#presets[@]}
    local delta=$(( (want_idx - cur_idx + total) % total ))
    local term_id
    term_id=$(load_state)
    for _ in $(seq 1 $delta); do
        send_one_key "$term_id" "+"
        sleep 0.05
    done
    sleep 0.2
    if capture | grep -q "theme · $slug"; then
        echo "OK: theme set to $slug"
    else
        echo "FAIL: theme did not settle on $slug; header reads:" >&2
        capture | grep 'theme ·' | head -1 >&2
        return 1
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
    env)
        cmd_env
        ;;
    db)
        shift
        cmd_db "$@"
        ;;
    db-get)
        shift
        cmd_db_get "$@"
        ;;
    db-assert)
        shift
        cmd_db_assert "$@"
        ;;
    db-dump)
        cmd_db_dump
        ;;
    settings-focus)
        shift
        cmd_settings_focus "$@"
        ;;
    theme-set)
        shift
        cmd_theme_set "$@"
        ;;
    model)
        shift
        cmd_model "$@"
        ;;
    db-model-assert)
        shift
        cmd_db_model_assert "$@"
        ;;
    help|*)
        cat <<'USAGE'
Usage: tui-uat.sh <command> [args]

Lifecycle:
  launch [--fresh] [--env KEY=VAL]* [-- mold tui args…]
                                  Start the TUI in a Ghostty window.
                                  --fresh creates a tmp MOLD_HOME for
                                  full isolation; --env injects extra
                                  vars into the spawned process.
  quit                            Close UAT window
  status                          Active session summary (+ MOLD_HOME)
  env                             Print MOLD_HOME / MOLD_DB_PATH in
                                  a form suitable for `eval`

Screen I/O:
  capture                         Print current screen (plain text)
  screenshot [output.png]         Native Ghostty screenshot (PNG)
  send <key>...                   Send keystrokes (see KEYS below)
  view <1-5|name>                 Navigate to view (1=Generate,
                                  2=Gallery, 3=Models, 4=Queue,
                                  5=Settings)
  wait-for <pattern> [timeout]    Wait up to N seconds for text
  assert <pattern>                Fail if text missing from screen

DB / persistence helpers:
  db [--write] [--force] <sql>    Run SQL against the session's DB
                                  (read-only unless --write; --force
                                  required to touch the user's real
                                  ~/.mold/mold.db)
  db-get <key>                    Print settings row value for <key>
  db-assert <key> <value>         Pass/fail a single-key equality
  db-dump                         Human-readable `settings` +
                                  `model_prefs` tables

Settings helpers:
  settings-focus <pane>           Move Settings-view focus to
                                  'appearance' or 'configuration'
  theme-set <slug>                Cycle to a named theme — one of
                                  mocha, latte, ristretto, gruvbox,
                                  tokyo, nord, dracula

KEYS
  Special:  enter, escape, tab, space, up, down, left, right,
            backspace, delete, home, end, page_up, page_down, f1-f12
  Modified: ctrl+c, alt+5, shift+tab, cmd+q
  Literal:  anything else is typed as text

ENVIRONMENT
  MOLD_BIN      Path to mold binary (default: ./target/dev-fast/mold
                or ./target/debug/mold if dev-fast is missing)

Requires: Ghostty 1.3+ with macos-applescript=true (default).
USAGE
        ;;
esac
