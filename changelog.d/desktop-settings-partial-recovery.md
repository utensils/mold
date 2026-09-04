- **Desktop settings survive a build that cannot read one value.** When a
  `settings.json` key holds a value the running build does not understand
  (an update channel or connection mode introduced by a newer nightly, then
  read by an older build or a dev build sharing the same app data), only that
  key falls back to its default; the update channel, saved machines, panel
  widths, and every other preference stay, and a list keeps every entry that
  still reads. Previously the whole file was replaced by defaults on the next
  save, which silently moved a Nightly install back to the Stable channel and
  forgot every remembered machine. A file that is not a JSON object is kept
  beside the store as `settings.json.invalid` instead of being overwritten.
