- **One canonical Shape and Size control across Create.** Attaching a source
  image now selects the closest model-valid size with the source's shape and
  keeps following it across model and pipeline changes until you choose a
  canvas yourself, and the `Source` badge can no longer appear next to a canvas
  the source did not produce. Shape chips are canonical families (1:1, 5:4,
  4:3, 3:2, 16:9, 21:9 plus portrait twins, and Source) instead of each model's
  gcd-reduced buckets, so LTX-2's `19:11`, `30:17` and `20:11` are one 16:9
  family; the sizes under a family are the model's own authored pixels, labelled
  `1216×704` with megapixels and any authored mark rather than a positional
  `Small`/`Standard`/`Native` or `Recommended`/`Max` tier. Web, desktop, and
  iPhone render one resolver result, so the header chip, shape chips, size
  pills, badge, and status sentence cannot disagree
  ([#1166](https://github.com/utensils/mold/issues/1166)).
