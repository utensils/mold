- **The browser's New image page follows the web design.** The composer now
  sticks to the bottom of the column, so Generate never scrolls away on a long
  page, and it carries the style, shape and Make controls beside the words they
  apply to ([#1700](https://github.com/utensils/mold/issues/1700)).
- **A finished picture stays on the canvas, with its own actions.** The
  browser used to drop a finished print back to the empty canvas a moment
  after it rendered; it now stays until the next one runs, and Download, Copy
  link and Make 4 variations sit over it. Copy link yields an address that
  opens that exact print in My images, which is something the desktop app has
  nothing to copy ([#1700](https://github.com/utensils/mold/issues/1700)).
- **The settings column says which machine the tab is talking to, first.** The
  machine card leads the column with a way to change it, followed by a
  Draft / Good / Best quality ladder built from the style's own recommended
  passes, the two sliders, and plain-language rows for starting from a photo,
  add-on looks, repeating a look, starters, filing and everything else
  ([#1700](https://github.com/utensils/mold/issues/1700)).
- **A narrow browser window is now shorter than a wide one, not longer.** Below
  900px the settings column leaves the page and opens as one sheet instead of
  being stacked into the composer alongside a second settings sheet
  ([#1700](https://github.com/utensils/mold/issues/1700)).
