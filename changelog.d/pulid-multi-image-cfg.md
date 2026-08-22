- **Several identity photographs, averaged.** `mold run --id-image` now repeats
  up to four times (and the API gains `id_images`), conditioning one render on
  several references of the same person. Each photograph is extracted
  independently and the identity token sets are averaged, matching
  `cubiq/PuLID_ComfyUI`; two or three angles usually hold a likeness better
  across poses than one does. Saved metadata records every photograph's name and
  digest in request order
  ([#1226](https://github.com/utensils/mold/issues/1226)).
- **True classifier-free guidance for identity renders.** `--true-cfg` (with
  `--cfg-start-step`) restores a real negative branch on FLUX's otherwise
  guidance-distilled path, so `--negative-prompt` finally does something on an
  identity render. Upstream's advice is to drop `--guidance` to `1.0` when you
  turn it on. It roughly doubles denoise time, which admission now accounts for;
  the default `1.0` is inert and renders bit-identically to before
  ([#1226](https://github.com/utensils/mold/issues/1226)).
- **Identity request shapes are capability-gated.** `GET /api/capabilities` now
  reports `identity: { multi_photo, max_photos, true_cfg }`, and `mold run`
  refuses several photographs or `--true-cfg` against a server that does not
  advertise them instead of submitting fields that server would silently drop
  ([#1226](https://github.com/utensils/mold/issues/1226)).
