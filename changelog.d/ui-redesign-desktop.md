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
  for new images, Write more for me, …). The command palette, lightbox,
  toasts, dialogs, and menus take the same anatomy, the clip timeline frames
  scenes under the canvas, and the inspector says Repeat this look, Keep |
  Surprise me, and Add-on looks. Saved appearance settings migrate to the
  nearest theme. `docs/design/` is the new package.
- **The lexicon reaches web and the phone where the policy is shared.** The
  img2img strength control is now "How much to change it" on every surface —
  the one label policy already served web, desktop, and iPhone, and it kept
  saying "Denoise strength". The machine-detail utility card reads "Machine
  utility", and the phone-pairing panels now paint from the theme's own tokens
  instead of falling back to hard-coded greys, so they follow the six themes
  and stop failing contrast on the light ones.
