- **A trashed print shows its own pixels everywhere.** The gallery media
  routes take `?view=trash`, so a Trash row on any machine reads the trashed
  file even after a new print took the same name
  ([#1597](https://github.com/utensils/mold/issues/1597)).
- **Licence rows say what they unlock.** Settings ▸ Style licences leads each
  row with the styles the licence gates, in plain words, over the licence's
  name; the id moves to the tooltip.
- **The compact MiniMax H3 recipe offers Draft / Good / Best.** Its
  `steps.recommended` ladder now has three rungs like every other adjustable
  recipe (private UAT feature).
- **Rent this GPU says what it costs.** The GPU list and the "billing begins
  now" confirm state RunPod's own hourly rate for the chosen cloud, and
  `mold runpod gpus` gains `$/hr` columns.
- **A machine's Storage card says what the pictures take.** `/api/status`
  carries the gallery's bytes and print count, live and trashed, summed from
  the host's own records.
