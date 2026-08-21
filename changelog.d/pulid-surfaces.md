- **Identity photos in Create on web and desktop.** Create gains an
  **Identity** photo well beside the source-image wells, with **Identity
  strength** (0.0–3.0, default 1.0) and **Identity start step** in Advanced.
  The whole control is capability-gated by the server's own
  `supports_identity` authority, so an unqualified checkpoint or an older host
  hides it rather than offering work the server would refuse; a photo already
  attached is parked, not discarded, and comes back when a qualified model is
  selected again. On a qualified model, attaching a photo alongside a LoRA or
  a source image, or leaving a knob set with no photo, reports the reason
  inline beside the control — never a toast — and blocks Generate. The photo is never cropped or fitted to the canvas: it is a
  face reference, and the picked bytes travel untouched. Batch siblings and
  prepared variations inherit it
  ([#1224](https://github.com/utensils/mold/issues/1224)).
- **The Library shows where a print's likeness came from.** A print rendered
  with an identity photo records its name, digest, strength, and start step,
  and the Lightbox aside now shows them. **Reuse settings** restores the two
  values and re-attaches the photo itself when this device still holds it,
  saying so plainly when it does not — saved metadata never contains the face
  bytes ([#1224](https://github.com/utensils/mold/issues/1224)).
