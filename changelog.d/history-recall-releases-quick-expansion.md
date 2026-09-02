### Fixed

- Pressing ↑/↓ to recall a prompt from history after Expand or Remix no longer raises the "Expanded prompt changed after it was prepared" banner or blocks Generate on the web and desktop Create surfaces. A history recall replaces the whole prompt, so it now releases the prepared rewrite outright, restoring the style chip and negative fragments the expansion baked in, and the recalled prompt submits as your own; hand edits to the rewrite keep the existing recovery actions.
