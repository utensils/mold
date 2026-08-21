# PuLID-FLUX user acceptance testing (#1223)

The record of what was actually run on real hardware to close the
"PuLID-FLUX: functional" milestone, with the exact commands, digests, and
measured numbers. Everything below is reproducible; where a check could not be
run, this says so rather than omitting it.

## Hosts

| Host | Backend | GPU | Build | Mold home |
| --- | --- | --- | --- | --- |
| plato | CUDA 12.9 | 4x NVIDIA L40S (48 GB) | `cargo build --release -p mold-ai --features cuda,pulid` | `/storage/mold` |
| halcyon | Metal | Apple Silicon | not run — see [Metal](#metal-halcyon) | `/Volumes/ExternalStorage/mold2` |

Tier: `flux-dev:q4` (the FLUX.1-dev Q4 GGUF), 1024x1024, 25 steps, guidance 3.5.
Reference photographs are the public-domain NASA astronaut portraits committed
at `crates/mold-inference/testdata/pulid/faces/`.

Rendered outputs are kept at `/Volumes/ExternalStorage/pulid-dev/uat/` and named
by the check that produced them.

## Results

| # | Check | Host | Tier | Result |
| --- | --- | --- | --- | --- |
| 1 | Missing licence acceptance refuses the pull | plato | — | **PASS** |
| 2 | `mold pull pulid-flux --accept-license insightface-antelopev2` | plato | — | **PASS** — 2.1 GB, acceptance recorded |
| 3 | Zero-weight byte identity (three-way SHA) | plato | q4 | **PASS** — A == B, C differs |
| 4 | Positive identity render, forced-local | plato | q4 | **PASS** — ArcFace cosine **0.6055** |
| 5 | Positive identity render, second face | plato | q4 | **PASS** — ArcFace cosine **0.7016** |
| 6 | Remote render over HTTP | plato | q4 | **PASS** — byte-identical to the forced-local render |
| 7 | Refusals (unqualified model, invalid start step) | plato | mixed | **PASS** |
| 8 | `supports_identity` advertised for exactly the qualified tiers | plato | — | **PASS** |
| 9 | Metal path | halcyon | — | **NOT RUN** — no FLUX checkpoint obtainable |

## 1. Missing acceptance refuses the pull

```console
$ mold pull pulid-flux
● Pulling pulid-flux on http://127.0.0.1:7680
error: server error: pulid-flux includes files under a license that must be accepted before download.

  InsightFace pretrained models (antelopev2)
  InsightFace pretrained models (antelopev2: scrfd_10g_bnkps, glintr100) are licensed for non-commercial research purposes only.
  Terms: https://raw.githubusercontent.com/deepinsight/insightface/master/README.md

Review the terms, then accept explicitly:

  mold pull pulid-flux --accept-license insightface-antelopev2
```

Not a byte moved. Note this ran against plato's *existing* 0.23.3 server, so the
refusal is the shipped behaviour rather than something this branch introduced.

## 2. The accepted pull

```console
$ mold pull pulid-flux --accept-license insightface-antelopev2
  InsightFace pretrained models (antelopev2)
  Terms (pinned): https://raw.githubusercontent.com/deepinsight/insightface/7fadd420c2351d0ffa8cac403421c1a3ed733365/README.md
  Project terms:  https://github.com/deepinsight/insightface#license
✓ recorded acceptance of insightface-antelopev2 on this machine
● Pulling pulid-flux (2.1GB to download)
✓ pulid-flux is ready!

$ du -sh /storage/mold/models/shared/pulid
2.2G    /storage/mold/models/shared/pulid
```

The pinned-commit terms URL is what #1252 added; the acceptance is bound to the
text that was displayed.

## 3. Zero-weight byte identity

The three-way check from `tmp/sdcpp/docs/pulid.md`, seed 42, same prompt
(`"a candid photograph of a person on a beach at sunset"`), 1024x1024, 25 steps:

| Run | Flags | SHA-256 |
| --- | --- | --- |
| A | none | `751b91287c6511d13d217af2f38e2c0f2ec3361b8f4b2f9938368d1f29efa497` |
| B | `--id-image … --id-weight 0.0` | `751b91287c6511d13d217af2f38e2c0f2ec3361b8f4b2f9938368d1f29efa497` |
| C | `--id-image … --id-weight 1.0` | `a37cb3dc050a215682a2845a1545f8172696e0ade2f5201bd6b3d948e71d71cc` |

**A == B, C differs from both.** Exactly the expected relation.

**These runs used `--no-metadata`, and that matters.** With metadata embedded, A
and B differ — but only in the `mold:parameters` chunk, because B *correctly*
records that a reference photograph was supplied at weight 0 (`id_image_name`,
`id_image_sha256`, `id_weight: 0.0`). The pixels are identical either way. The
first run of this check compared metadata-bearing PNGs and looked like a
failure; it was the provenance working. Anyone repeating this must compare
pixel-bearing bytes only.

Files: `flux-dev-q4-nometa-{A,B,C}.png`. The metadata-bearing triple is retained
as `flux-dev-q4-{A-baseline,B-weight-zero,C-weight-one}.png`.

### A note on the prompt

The first positive-render attempt reused this beach prompt and produced a
distant back-lit silhouette — a face a few pixels across, which no detector will
find and no cosine can score. That is a prompt problem, not a pipeline problem,
and `flux-dev-q4-C-weight-one.png` is kept as the illustration. The fidelity
check below uses an explicit close-up portrait prompt.

## 4. Positive identity render and the fidelity gate

Prompt: `"a close-up portrait photograph of a person looking straight at the
camera, soft studio lighting, sharp focus"`, seed 7, 1024x1024, 25 steps,
`--id-weight 1.0`, `--id-start-step 0` (default).

| Run | Output | SHA-256 |
| --- | --- | --- |
| baseline (no identity) | `flux-dev-q4-portrait-baseline.png` | `df0021e86b77d6dea2fc15482ff3a75fe2c1db56b3537db74cdb80f456ecd431` |
| identity (Frank Rubio) | `flux-dev-q4-portrait-id-frank-rubio-official-portrait.png` | `accbd724bf81e2866db1cf6b94fcf7006bc52572f0109172c19fb633481727b5` |

ArcFace cosine, measured with the same `glintr100` graph the conditioning used
(`crates/mold-inference/tests/pulid_identity_fidelity.rs`):

```
PASS  cosine 0.6055  (threshold 0.28)  flux-dev-q4-portrait-id-frank-rubio-official-portrait.png
```

**0.6055** against the reference photograph. The gate's threshold is
InsightFace's own same-person decision value for this recognizer (0.28); PuLID's
paper reports face similarity in the 0.6–0.8 band, so this lands squarely inside
upstream's own range rather than merely clearing the floor.

The seed-matched baseline renders a completely different person, as it must —
and SCRFD finds no face in it at all, because that render crops the face past
the edges of the frame. Not a defect: it is why a *conditioned* render is the
one being scored.

Reproduce:

```bash
MOLD_TEST_PULID_ASSETS=/Volumes/ExternalStorage/pulid-dev \
MOLD_TEST_IDENTITY_REFERENCE=$PWD/crates/mold-inference/testdata/pulid/faces/frank-rubio-official-portrait.jpg \
MOLD_TEST_IDENTITY_RENDER=/Volumes/ExternalStorage/pulid-dev/uat/flux-dev-q4-portrait-id-frank-rubio-official-portrait.png \
  cargo test -p mold-ai-inference --features pulid \
  --test pulid_identity_fidelity -- --ignored --nocapture
```

## 5. Second reference face

Same prompt, same seed, same flags, a different public-domain portrait
(Kayla Barron):

| Output | SHA-256 | Cosine |
| --- | --- | --- |
| `flux-dev-q4-portrait-id-kayla-barron.png` | `fe148b0714a4cd68c06d618a48c940ae0fe500a0346418a6ba64680dc0824454` | **0.7016** |

Two faces, two independent identities, both inside PuLID's own 0.6-0.8 band. A
single sample could have been the prompt; two different people from the same
seed and prompt cannot be.

## 6. Remote render over HTTP

This branch's binary was started as a server (`mold serve --bind 127.0.0.1
--port 7681`, `MOLD_API_KEY` set) and the same render was submitted over HTTP:

```bash
MOLD_HOST=http://127.0.0.1:7681 MOLD_API_KEY=uat-1223 \
  mold run flux-dev:q4 "<the portrait prompt>" \
  --seed 7 --steps 25 --width 1024 --height 1024 \
  --id-image frank-rubio-official-portrait.jpg --id-weight 1.0
```

| Path | Output | SHA-256 |
| --- | --- | --- |
| forced-local (`--local`) | `flux-dev-q4-portrait-id-frank-rubio-official-portrait.png` | `accbd724bf81e2866db1cf6b94fcf7006bc52572f0109172c19fb633481727b5` |
| remote (HTTP + SSE) | `flux-dev-q4-remote-id-frank-rubio.png` | `accbd724bf81e2866db1cf6b94fcf7006bc52572f0109172c19fb633481727b5` |

**Byte-identical, embedded provenance included.** That is the strongest form of
the local/remote parity the issue asked for: not "the two requests look the
same" but "the two renders ARE the same", which is only possible if the same
`IdentityOptions` produced the same `GenerateRequest`, the same extraction
produced the same embedding, and the same adapter consumed it.

## 7. Refusals and capability advertisement

**Unqualified model, over HTTP:**

```console
$ MOLD_HOST=http://127.0.0.1:7681 mold run z-image-turbo:q4 "…" --id-image face.jpg
error: z-image-turbo:q4 does not support face-identity conditioning; identity is qualified only for flux-dev:q4 and flux-dev:q8
```

**Invalid start step, raw API:**

```console
$ curl -X POST …/api/generate -d '{… "id_start_step": 99, "steps": 4}'
{"error":"id_start_step (99) must be less than steps (4)","code":"VALIDATION_ERROR"}
```

**`/api/models[].supports_identity`** — advertised for exactly the two qualified
tiers and nothing else, on a `pulid` build:

```
supports_identity == true for: ['flux-dev:q8', 'flux-dev:q4']
  flux-dev:bf16 supports_identity= False  downloaded= False
  flux-dev:q8   supports_identity= True   downloaded= False
  flux-dev:q6   supports_identity= False  downloaded= False
  flux-dev:q4   supports_identity= True   downloaded= True
```

Note `flux-dev:q8` advertises `true` while `downloaded: false`: the capability
is a property of the checkpoint, not of whether it is installed, which is what
lets a client offer the control before the pull.

## Metal (halcyon)

**Not run.** Two independent blockers, both environmental:

- `flux-dev:q4` is a **gated** Hugging Face repository and no HF token is
  configured on this machine — not in the environment, not in
  `$MOLD_HOME/catalog-credentials.json`, and not in `~/.cache/huggingface/token`.
- The external volume holding `MOLD_HOME` has ~18 GB free against a 16.7 GB
  install plus the 2.1 GB PuLID bundle, and was already at 100% during this
  work.

What *was* verified on Apple Silicon: the whole workspace compiles and its tests
pass with `--features pulid`, `cargo clippy -p mold-ai --features
metal,…,pulid --all-targets -- -D warnings` is clean (the CI job this branch
added runs exactly that), and the extraction half — SCRFD, ArcFace, the EVA
tower, the IDFormer, and the ArcFace cosine gate above — was executed on this
machine against the real 2.3 GB bundle, because that half is CPU-only by design.

What is therefore **unverified on Metal**: the adapter's twenty cross-attention
injections inside the FLUX transformer, and the Metal VRAM accounting for them.
The adapter is dtype-generic and shares its code path with CUDA, and #1221's
adapter tests run on CPU on this machine, so the risk is low — but it is a real
gap and should be closed by pulling `flux-dev:q4` on a machine with an HF token
and repeating checks 3 and 4.

## Not covered here

- `flux-dev:q8` — the second qualified tier. Same code path, same adapter shape;
  it differs only in the transformer's quantization, which the adapter does not
  read.
- Dependency auto-pull during a generate (as opposed to the explicit
  `mold pull`). The two share `identity_dependencies::materialize_identity_assets`
  and the licence gate above is enforced there; the server-side unit tests cover
  the branch.
- Krea, block offload, LoRA, and img2img alongside an identity. All are refused
  by the request contract in milestone 1 and belong to a milestone-2
  qualification pass.
