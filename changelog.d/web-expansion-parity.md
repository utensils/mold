- **Write more for me rewrites the prompt in place on the web.** The browser's
  Create composer now does what the desktop app and the phone do: one rewrite
  on the machine, installed straight into the prompt bed with an `expanded ·
undo` chip beside it, and a live line naming the machine while it writes. The
  prompt-expansion dialog is gone, and with it its "Enable expansion before
  submit" checkbox, its 1/3/5 variation count (five prompts for a one-print
  render was a server error) and its model-family override. A rewrite that
  lands after the prompt, style, conditioning or machine changed is refused by
  name instead of silently replacing what you typed.
- **The web no longer asks the machine to expand at generate time.** Web was
  the only client sending the request's `expand` flag, and the rewrite it
  produced never appeared in the composer. A saved draft or starter that still
  carries the old setting loads and drops it.
- **Why reviewed prompt work went stale reads the same on every screen.** The
  rule is shared now, so both apps say "Style changed" and "Machine selection
  changed" in one wording.
