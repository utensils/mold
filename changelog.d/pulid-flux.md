- **Face-identity conditioning: `mold run --id-image`.** Give mold a reference
  photograph and FLUX keeps that person's face across arbitrary prompts. This is
  PuLID-FLUX ported natively to Rust — SCRFD detection, ArcFace embedding,
  EVA02-CLIP-L-14-336, and the IDFormer resampler feed twenty cross-attention
  modules injected between the FLUX transformer blocks. `--id-weight` (0.0–3.0,
  default 1.0) sets the strength and `--id-start-step` delays it; `--id-weight 0`
  is completely inert and renders byte-identically to the same seed with no
  identity at all. Qualified for `flux-dev:q4` and `flux-dev:q8`, on CUDA and
  Metal, over the CLI, the HTTP API, and forced-local
  ([#1223](https://github.com/utensils/mold/issues/1223)).
- **The PuLID bundle is a licence-gated pull.** `mold pull pulid-flux
  --accept-license insightface-antelopev2` fetches the four auxiliary artifacts
  (~2.1 GB). The two InsightFace face models are pretrained weights licensed for
  non-commercial research only, so mold prints the terms and refuses to download
  them until acceptance is recorded on that machine; `mold licenses` lists what
  has been accepted, and `mold rm pulid-flux` now also deletes the vision tower
  mold derived on first use
  ([#1223](https://github.com/utensils/mold/issues/1223)).
- **The identity is extracted once per request, not once per print.** Admission
  resolves the reference photograph into an immutable 32x2048 embedding before
  batch fan-out and freezes it into the prepared plan, so every sibling — on
  every device — conditions on the identical value, and a re-prepared batch child
  never re-extracts. The whole extraction runs on the CPU before a GPU is even
  leased, so it can never compete with the text encoders for memory
  ([#1223](https://github.com/utensils/mold/issues/1223)).
- **Identity provenance is recorded.** A print rendered with a reference
  photograph records that photograph's file name and SHA-256 plus the applied
  weight and start step, in embedded metadata and the gallery — never the
  photograph itself, and never your directory layout
  ([#1223](https://github.com/utensils/mold/issues/1223)).
- **PuLID's memory budget is measured rather than declared.** Identity now
  charges 1.25 GB of VRAM instead of the 2.3 GB placeholder, because the
  detector, the recognizer, and the vision tower run on the host at admission and
  are charged there; only the cross-attention adapter is resident on the
  generation device ([#1223](https://github.com/utensils/mold/issues/1223)).
- **Face identity works in the desktop app.** The desktop app is its own build
  and had never compiled the feature, so its **This device** engine never
  advertised identity and the photo well simply never appeared — a silent
  absence with nothing in the interface to explain it. It now ships in every
  desktop recipe ([#1223](https://github.com/utensils/mold/issues/1223)).
- **Your reference photograph is never written to disk.** A face photo is
  biometric data supplied for one render, so it is excluded from the durable
  generation queue and redacted from batch recovery manifests. The cost is
  honest: a queued identity print is reported as not-resumable rather than
  quietly replayed, and a batch interrupted mid-flight is refused rather than
  re-rendered with the face missing
  ([#1223](https://github.com/utensils/mold/issues/1223)).
- **Mold tells you when it had to choose a face.** If a reference photograph
  contains several faces, mold conditions on the largest one and now says so —
  in the CLI, over HTTP, and in the browser — instead of only writing it to the
  server log ([#1223](https://github.com/utensils/mold/issues/1223)).
