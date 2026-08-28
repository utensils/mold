- **A queued job now tells you what it is.** Selecting a queue row on desktop,
  web, or iPhone opens the whole job: its prompt, the settings it was submitted
  with, its live place in line, when it was submitted, whether it survives a
  restart, how many times a worker has claimed it, the scheduler's lane and
  estimate, a parked job's complete reason and error behind a Copy control, and
  a running job's live denoise preview — with **Reuse settings**, **Cancel**,
  and **Retry** routed to that exact machine. Web and iPhone had no queue detail
  at all, and desktop's said the host was too old to share the job's settings.
  It is not: the durable queue lists a row before the machine loads its request,
  so the panel now says that in the machine's own terms, and desktop fills the
  gap from its own copy of the request it submitted.
- **iPhone and Android queue rows swipe.** Dragging a row right to left reveals
  a 44pt **Cancel** and, where the machine supports reordering, a
  non-destructive **To back**; a full swipe commits the cancel. Revealing the
  tray is the first step and the tap or full swipe is the second, so nothing is
  cancelled by one flick, and every action stays reachable from the row's
  **Actions** button for VoiceOver and hardware keyboards. The horizontal pan is
  scoped to the row, so the list scroll, the Library grid's column pinch, and
  the gallery viewer's swipe are unaffected.
