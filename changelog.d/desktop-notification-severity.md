- **Notification severity is color-coded green / yellow / red.** The
  notifications bell and the toast shelf on desktop and web now tint success
  green, warnings yellow, and errors red (plain information stays neutral), with
  the severity also named for screen readers and the unread badge taking the
  worst unread entry's color. `--warning` gained proper light-mode and Mold-family
  values so the yellow stays readable in every theme.
- **Unreachable machines say they are reconnecting.** Desktop and web already
  keep polling a machine that drops, so it comes back on its own; that is now
  visible. Losing a machine raises a yellow "Can't reach `<machine>` — retrying
  automatically" notice instead of a red error, the Machines card marks the row
  _reconnecting…_, and when the machine answers again the warning is withdrawn
  and a green "Reconnected to `<machine>`" confirms it.
