# Native Mac performance diagnostics

Nightly versions enable lightweight native diagnostics automatically. Stable
versions leave them off. Set `MOLD_PERFORMANCE_DIAGNOSTICS=1` in the launch
environment to enable, or `0` to disable, regardless of version. The process
reads the switch once at launch. It does not change engine logging settings.
For a direct diagnostic launch after quitting normally:

```sh
MOLD_PERFORMANCE_DIAGNOSTICS=1 '/Applications/Mold Studio.app/Contents/MacOS/Mold'
```

Enabled diagnostics use Apple's unified log, subsystem `io.mold.native`, category
`Performance`. A background utility queue checks responsiveness once a second,
with at most one outstanding main-queue acknowledgment. A main-thread delay of
at least two seconds emits `main_thread_unresponsive`; repeated warnings are
limited to once every 30 seconds. Monitoring stops while the app is inactive or
quitting, and timer scheduling gaps over ten seconds reset the pending clock to
avoid reporting system sleep as work. This is a responsiveness signal, not a
backtrace or proof that a particular operation caused a hang.

Sync runs have `NativeOperation` signpost intervals for Instruments' Points of
Interest. Other callers can measure expensive cache misses or whole catalog
loads using `PerformanceDiagnostics.shared.begin(.sourceInventory)` and
`defer { PerformanceDiagnostics.shared.end(interval) }`. Do not instrument every
row or cache hit. Supported operation names are a fixed enum: no prompts, URLs,
filenames, identifiers, credentials or media bytes enter these events.
Operations taking at least 100 ms produce a `slow_operation` duration, limited
to one log per operation category per 30 seconds. Sync elapsed time includes
network waits and should not be interpreted as CPU time.

Watch live events without launching another app:

```sh
log stream --style compact --level info --predicate 'subsystem == "io.mold.native" AND category == "Performance"'
```

When the existing app becomes slow, find its PID in Activity Monitor, then run
from the repository root:

```sh
apps/macos/scripts/capture-performance.sh PID 10
```

The script samples the existing process for 1–60 seconds (10 by default), records
CPU/resident-memory snapshots, and collects only this native Performance category
from the last ten minutes. It creates a private directory in the system temporary
folder and prints its path. No restart, engine action, upload or privileged system
trace occurs. Stack samples can contain local usernames and paths: review before
sharing; never commit raw samples. Captures also work against older builds,
although those builds do not emit the new events. For deeper diagnosis, attach
Instruments' Time Profiler and SwiftUI instruments to the existing PID, correlate
main-thread work with the signpost intervals, then compare a second capture after
the fix using the same library and operation.

Automated policy tests cover opt-in/nightly selection, slow-event thresholds,
independent category rate limits, the bounded heartbeat queue and sleep gaps.
Runtime log emission still requires exercising an enabled build; a build or
unit test alone is not evidence of an actual performance improvement.

The October 18, 2026 live sync investigation sampled the installed app for eight
seconds without restarting it. Of 3,075 main-thread sample stacks, approximately
1,300 included the Library subtitle's baseline count/merge path, 548 included
source-access privacy projection, and 319 included sidebar counts. These are
inclusive stack counts, not separate elapsed-time measurements. The fixes cache
Library counts by row/query/privacy revision and calculate source access only
while a source browser or print detail needs it. A 10,000-row regression checks
that 100 unchanged redraw requests derive the count once. Comparing a rebuilt
app against the same live library remains the appropriate performance UAT;
the old installed process does not acquire these fixes from a source build.

An isolated executable built from the diagnostics sources also exercised actual
unified-log emission: startup, a 151 ms sync interval, and a deliberately blocked
main queue produced the expected `diagnostics_started`, `slow_operation` and
`main_thread_unresponsive` events. This validates the plumbing independently of
the user's running app; it is not a before/after Library responsiveness result.
