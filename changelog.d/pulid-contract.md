- **Face-identity request contract (PuLID-FLUX, milestone 1).** `GenerateRequest`
  gains additive `id_image` (base64 PNG/JPEG), `id_image_name`, `id_weight`
  (`0.0..=3.0`, default 1.0), and `id_start_step` (`< steps`, default 0); saved
  metadata records the reference's name, SHA-256, and effective values, never
  the face bytes. `/api/models[].supports_identity` and
  `generation_profile.capabilities.supports_identity` advertise the
  identity-qualified checkpoints — `flux-dev:q4` and `flux-dev:q8` on a build
  with the off-by-default `pulid` feature. Every other model, any LoRA or
  img2img combination, an identity field without `id_image`, an oversized or
  non-PNG/JPEG reference, and any identity request on a build without the
  feature are refused at admission with a named reason — never silently
  ignored ([#1220](https://github.com/utensils/mold/issues/1220)).
- **PuLID identity assets are a first-class install.** `pulid-flux` is a
  hidden auxiliary bundle covering the PuLID-FLUX v0.9.1 adapter, the
  EVA02-CLIP-L-14-336 vision tower, and the InsightFace antelopev2 face
  detector and recognizer — every file SHA-256 pinned. `mold pull`,
  incomplete-pull repair, installed-state reporting, and `mold rm` all handle
  it, and it is never offered as a checkpoint or picked as a default model
  ([#1220](https://github.com/utensils/mold/issues/1220)).
- **Restricted model licenses must be accepted before download.** The
  InsightFace antelopev2 weights are licensed for non-commercial research only,
  so Mold refuses to fetch them — from the CLI, the server's auto-pull, or any
  automatic client pull — until you record acceptance with
  `mold pull pulid-flux --accept-license insightface-antelopev2`. The record
  lives in an owner-only `$MOLD_HOME/license-acceptances.json` and is bound to
  the exact license text, so changed terms require accepting again
  ([#1220](https://github.com/utensils/mold/issues/1220)).
- **Identity requests plan and materialize the PuLID bundle themselves.** A
  request that conditions on a face resolves `pulid-flux` through ordinary
  dependency preparation: `POST /api/generate/placement-preview` lists whatever
  is missing under `pending_downloads` without fetching anything, admission
  materializes it after the InsightFace license gate, and the four exact paths
  are frozen into the execution plan the worker dispatches. An `id_weight` of 0
  is completely inert — no assets planned, nothing downloaded, no memory
  charged ([#1220](https://github.com/utensils/mold/issues/1220)).
- **Identity assets are verified against their manifest pins before use.**
  Hugging Face `main` is a mutable branch, so every PuLID file Mold
  materializes is hashed against the SHA-256 the manifest pinned — after the
  download, and again for any copy already sitting in the models directory. The
  bytes are read through a retained no-follow descriptor, so a symlink or path
  swap cannot substitute different content, and the `.sha256-verified` sidecar
  is never accepted as proof: model roots are allowed to be group-writable, so
  anyone who can drop weights there could drop an attestation for them too. A
  mismatch names the file and both digests, removes the rejected bytes, and
  fails the job instead of freezing unverified weights into an execution plan.
  Each file is read once per process, so repeat admissions cost a stat rather
  than a re-hash ([#1220](https://github.com/utensils/mold/issues/1220)).
- **License acceptance works against a remote server.**
  `mold pull <model> --accept-license <id>` now records the acceptance on
  whichever machine runs the pull: the id travels to `MOLD_HOST` and the server writes it
  into its own `$MOLD_HOME`, instead of the client accepting on its own behalf
  and the server refusing anyway. New `GET /api/licenses` lists each license
  with its pinned terms, `accepted`, and `required_by`; `POST /api/downloads`
  and `POST /api/models/pull` take an additive `accept_licenses` array; a gated
  download without one is a `403` with code `LICENSE_NOT_ACCEPTED` and a
  structured `license` object so web, desktop, and iPhone can offer acceptance
  in-app. Servers advertise `capabilities.licenses`, and `mold licenses` shows
  the state along with which machine it read
  ([#1220](https://github.com/utensils/mold/issues/1220)).
