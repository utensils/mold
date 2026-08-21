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
