- **Identity photos on iPhone.** Create gains an **Identity** photo well in the
  primary form beside the source wells, with **Identity strength** (0.0–3.0,
  default 1.0) and **Identity start step** in the full-screen Advanced sheet,
  where they count toward its active badge and clear on Reset. Picking uses the
  native photo/camera picker, and the photo is never cropped or fitted to the
  canvas — it is a face reference, so the picked bytes travel untouched. The
  whole control is capability-gated by the host's own `supports_identity`
  authority: an unqualified checkpoint or an older machine hides it rather than
  offering work the server would refuse, and a photo already attached is parked
  — kept in the draft, left off the wire, Develop still enabled — and comes back
  the moment a qualified model is selected again. Every refusal (a photo
  alongside a LoRA or a source image, a knob set with no photo, an oversized or
  unsupported file) reads inline beside the control and blocks Develop, never as
  a toast. Prepared Batch N siblings inherit the partition
  ([#1231](https://github.com/utensils/mold/issues/1231)).
- **iPhone Library shows where a print's likeness came from.** The viewer's
  **Info** sheet lists the identity photo's name, short digest, effective
  strength, and start step for any print that carried one, and **Use as prompt**
  restores both knobs and re-attaches the photo when this device still holds it
  — saying so in the persistent inline status line when it does not. Saved
  metadata records the digest, never the face bytes
  ([#1231](https://github.com/utensils/mold/issues/1231)).
- **A changed identity photo stales reviewed prompt work** on every surface,
  exactly like a changed source image: the client-only conditioning fingerprint
  now reads `id_image`, so a remix or prepared batch reviewed against one face
  is never submitted against another
  ([#1231](https://github.com/utensils/mold/issues/1231)).
