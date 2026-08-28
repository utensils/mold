- **MiniMax H3 reference crop.** Ref2VA image references can be cropped before
  they are sent: the reference row's new **Crop** action on web, desktop, and
  iPhone opens a drag rectangle with Free / 1:1 / 4:3 / 3:2 / 16:9 presets, a
  64 px minimum, and a live vision-pad cost hint. The crop is applied
  client-side at the photograph's original resolution before digest and
  upload, recorded as additive `references[].provenance.crop` provenance
  (validated by the server, kept in the print's metadata), and restored by
  Reuse settings when the same original is reattached. It never changes the
  print's size or fits the reference to the canvas.
