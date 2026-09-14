- **Removed the interactive terminal UI.** `mold tui`, the `tui` build feature
  and `mold library grid` are gone. A source build that passes `--features tui`
  now fails with an unknown-feature error — drop `tui` from the list and the
  build is otherwise unchanged. Every other `mold` command behaves exactly as
  before, and the desktop, web and phone apps are the graphical surfaces
  ([#1717](https://github.com/utensils/mold/issues/1717)).
- **The CLI's last-used model moved to `generate.last_model`.** The row is
  copied from `tui.last_model` on first launch, so `mold run` with no model
  still resolves the model you used last. Terminal-app-only settings, including
  remembered machines and their stored API keys, are removed from `mold.db`;
  downgrading to 0.29 or earlier will not find them. Existing
  `<file>.thumb.png` thumbnail twins are swept from the cache
  ([#1717](https://github.com/utensils/mold/issues/1717)).
