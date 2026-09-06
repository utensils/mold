- **Legacy LTX-Video no longer auto-splits into repeated clips.** One-shot
  requests stay one render up to the engine ceiling, manually submitted
  context-free ephemeral chains are refused, custom legacy models reject
  unsupported source images instead of ignoring them, and the guidance points
  image-conditioned work to LTX-2.3 or LTX-2.5
  ([#1575](https://github.com/utensils/mold/issues/1575)).
