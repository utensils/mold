- **Notification severity is color-coded green / yellow / red.** The
  notifications bell and the toast shelf on desktop and web now tint ordinary
  notices and successes green, warnings yellow, and errors red, with the
  severity also named for screen readers and the unread bell badge taking the
  worst unread entry's color — so a bell holding only notices reads green rather
  than alarming red. `--warning` gained proper light-mode and Mold-family values
  so the yellow stays readable in every theme.
- **Unreachable machines say they are reconnecting.** Desktop and web already
  keep polling a machine that drops, so it comes back on its own; that is now
  visible. Losing a machine raises a yellow "Can't reach `<machine>` — retrying
  automatically" notice instead of a red error, the Machines card marks the row
  _reconnecting…_, and when the machine answers again the warning is withdrawn
  and a green "Reconnected to `<machine>`" confirms it.
- **Severity is announced, not just tinted.** Warnings now share the error
  toast's assertive live region — the sticky "your machine is gone" notice no
  longer waits for a screen-reader user to go idle — and every surface reads one
  shared severity table, so the same event can no longer show one glyph in a
  toast and a different one in the bell.
- **Notifications can be copied.** Every row in the notifications bell has a
  Copy button that puts the message, its full untruncated body, and the
  machine/time line on the clipboard, and the notification text itself is
  selectable again even though the surrounding app chrome is not.
