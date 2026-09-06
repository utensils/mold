- **A long clip survives a page reload again.** Reloading the browser while
  the machine was rendering a long clip reported "server progress lost" while
  the machine happily carried on stitching it. The page now picks the same
  render back up where it left off
  ([#1621](https://github.com/utensils/mold/issues/1621)).
- **A clip paused by a restart now says so and stops.** When a machine parks a
  long clip at shutdown it keeps every finished piece, ready to resume — but
  the desktop app sat on that row forever, checking every few seconds and never
  saying anything. It now tells you the clip is paused and points you at
  Resume in the queue
  ([#1622](https://github.com/utensils/mold/issues/1622)).
