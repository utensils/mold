# Native macOS layout audit

Scope: all nine Settings panes and the Generate, Library grid/viewer, Queue,
Models and Machines destinations. User screenshots were visual evidence.

## Repairs

- Removed the hidden label and fixed outer width from the shared machine picker;
  Accounts now uses the same picker as the other machine-scoped panes.
- Advanced assigns compact Source/Reset columns, leaves space for editable
  values, and makes truncated keys available in full on hover.
- Text defaults have bordered fields and an actual placeholder; secret fields
  no longer repeat their status as a placeholder and adjacent label.
- Main window minimum: 1,080 points (220 sidebar + 336 inspector + 524 for the
  canvas, toolbar and dividers), enforced through content minimum sizing.
- Sidebar machine names truncate with full-name help. WrappingHStack remeasures
  oversized children against the available width, so tag/collection chips fit.
- The inspector does not repeat a lone prose label already used as its heading.

## Validation

- Clean Debug build passed. An incremental build crashed in SwiftUI's
  OnSubmitModifier while opening Advanced; a clean rebuild and repeated pane
  navigation did not reproduce it.
- MoldClient: 940 tests passed.
- Native app: 769 tests passed on rerun. The first run failed the untouched
  asynchronous DefaultMachineTests preference-observer check; rerun passed.
- MoldStyle: 7 tests passed, including actual ImageRenderer measurement of a
  270-character chip at 180 points and wrapping the following tag to a new row.
- Architecture, semantic-color and accessibility lints passed (existing size
  advisories remain).
- Real app UAT used a separately identified remote-only Debug bundle and
  disposable application preferences. Read live Plato data; no generation,
  config edits, queue operations, or gallery mutations were submitted.
- Inspected all nine Settings tabs; short and long machine names; light and
  dark appearance; Advanced key/value/source alignment; visible empty text fields.
- Inspected Library grid and open viewer at the default size and a narrow
  macOS Window > Move & Resize > Left size, with sidebar and inspector toggles.
  The sidebar text stayed inside its column, inspector content wrapped, the
  viewer preserved image aspect ratio, and toolbar controls stayed reachable.
- Checked Queue, Models and Machines at the narrow size. Long chip regression
  uses a rendered test rather than adding tags to the user's gallery.
- Independent GPT-5.6 Sol peer review found no blocking correctness issues.

Native AppKit-backed window chrome was checked in the running app. ImageRenderer
coverage is restricted to the custom flow layout, which it can render faithfully.
