- **MiniMax H3 base tags now floor steps at the reviewed schedule.** The
  undistilled `comfy-pruned-int8` FL2VA and Ref2VA tags advertise and enforce
  21–50 terminal-inclusive grid points instead of 2–50; below 21 the print
  flashes once per latent frame, and 4- and 8-step renders belong to the Turbo
  tags, which stay pinned. The web, desktop, and phone apps show the reason
  under the Steps control
  ([#1435](https://github.com/utensils/mold/issues/1435)).
- **The MiniMax H3 model page links the prompting guide** and explains why a
  one-line prompt under-performs
  ([#1675](https://github.com/utensils/mold/issues/1675)).
