- **Qwen Image 2.1 in Mold Studio for Mac.** The native macOS app now numbers
  up to ten ordered reference images, lets the last one set the canvas shape
  while the size is still the model's default (rounded half-to-even on the
  32 px grid, exactly as the engine does), sends references as the original
  bytes so a transparent PNG is never flattened, offers a **Transparent
  background** toggle and WebP stills wherever the recipe advertises them,
  draws a checkerboard behind prints that carry alpha in the Library, the
  viewer and the Generate result, restores the toggle on Use These Settings,
  and shows the Qwen Research licence before any Qwen Image 2.1 download —
  including one a render or a held Queue row would start
  ([#1768](https://github.com/utensils/mold/issues/1768)).
- **Transparent prints reach Photos intact on iPhone and Android.** Photos
  auto-save and multi-select save now include WebP stills, which they skipped
  before, and iPhone hands Photos the original file instead of re-encoding it,
  so a transparent PNG or WebP keeps its alpha. An animated WebP is still never
  saved as a photo ([#1769](https://github.com/utensils/mold/issues/1769)).
