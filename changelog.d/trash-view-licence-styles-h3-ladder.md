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
