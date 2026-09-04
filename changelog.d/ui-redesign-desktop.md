- **Mold Studio desktop redesign.** The desktop app moves onto the new Mold
  Studio design system: six named themes (Mocha, Safelight, Blueprint,
  Graphite, Porcelain, Nebula) with a Match-system pairing, each a complete
  token map with its own type and radius scale and bundled OFL fonts; a new
  shell with a unified toolbar, a sidebar that keeps the queue under the target
  machine's card, and a status bar that always answers which machine, how full,
  and how deep the queue is; and the plain-English lexicon on every door — New
  image, Queue, My images, Styles, Machines, with technical truth beside each
  plain label in mono. The Queue is its own view (⌘2). Styles opens on Ready to
  use | Browse more with a disk-used-by-styles meter, a download banner that
  reads as a sentence with the CLI progress line beside it, and Get it as the
  one acquisition verb. Machines is a master/detail list beside the machine's
  pane (Right now tiles, Loaded and ready, Waiting on this machine, Storage,
  Downloads here) with Connect a machine as one dialog and Rent a GPU stating
  its cost. Settings is a jump nav over always-open sections (Look, Defaults
  for new images, Write more for me, …), with a new Styles & disk section
  holding where styles are kept, where finished pictures are written, and how
  full that disk is; each theme card shows a band of its own surfaces, and
  style licences are one row apiece with their state and a single action. The
  command palette shows each command's shortcut beside it and groups rows the
  way the app is organized, with Generate from these words, Pause the queue,
  and Download a style among them. The lightbox,
  toasts, dialogs, and menus take the same anatomy, the clip timeline frames
  scenes under the canvas, and the inspector says Repeat this look, Keep |
  Surprise me, and Add-on looks. Saved appearance settings migrate to the
  nearest theme. `docs/design/` is the new package.
- **The desktop queue says what it is waiting on, and lets you hold the
  line.** A waiting row's ⋯ menu offers Pause and Resume where the machine
  supports per-job pause, the card for the image being made carries a pause
  beside its stop, and Space pauses or resumes the queue from anywhere outside
  a text field (My images keeps Space for Quick Look). A parked row now reads
  "Paused after restart", "Held", or "Getting a style ready · 42%" in the
  sidebar instead of a bare "Waiting", a downloading row draws its meter in the
  warning tone, and where the machine predicts a finish the queue says how long
  is left. Done today counts the day's prints from your images rather than the
  session, and the card for the image being made states its place in a batch.
- **The lexicon reaches web and the phone where the policy is shared.** The
  img2img strength control is now "How much to change it" on every surface —
  the one label policy already served web, desktop, and iPhone, and it kept
  saying "Denoise strength". The machine-detail utility card reads "Machine
  utility", and the phone-pairing panels now paint from the theme's own tokens
  instead of falling back to hard-coded greys, so they follow the six themes
  and stop failing contrast on the light ones.
