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
| 9 | `mold rm pulid-flux` removes the derived tower | plato | — | **PASS** |
| 10 | Dependency auto-pull during a conditioned generate | plato | q4 | **PASS** — byte-identical render after a full wipe |
| 11 | Metal path | halcyon | — | **NOT RUN** — no FLUX checkpoint obtainable |

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

## 9 and 10. Removal, and auto-pull back

The bundle was removed and then re-acquired by an ordinary conditioned
generate, which is the same sequence a user hits the first time they render a
face on a fresh machine that has already accepted the licence.

```console
$ mold rm pulid-flux --force
pulid-flux
  Delete:   pulid_flux_v0.9.1.safetensors (1.1 GB)
  Delete:   EVA02_CLIP_L_336_psz14_s6B.pt (816.8 MB)
  Delete:   scrfd_10g_bnkps.onnx (16.1 MB)
  Delete:   glintr100.onnx (248.6 MB)
  Delete:   eva02_clip_l_336_vision.safetensors (580.8 MB)
  Delete:   eva02_clip_l_336_vision.json (348 B)

Removed pulid-flux (freed 2.1 GB)
```

The last two lines are what #1223 added: they are mold's own conversion output,
invisible to anything that walks the manifest, and before this change they
survived removal as a 580 MB orphan. `shared/pulid/` is now empty and gone.

Licence acceptance is deliberately NOT removed — it is a record of what the
human agreed to, not an artifact:

```console
$ mold licenses --local
  insightface-antelopev2   accepted   InsightFace pretrained models (antelopev2)
                           needed by: pulid-flux
```

Which is what lets the next conditioned generate re-acquire everything with no
flag:

```console
$ mold run --local flux-dev:q4 "<the portrait prompt>" --seed 7 \
    --id-image frank-rubio-official-portrait.jpg --id-weight 1.0
  ✓ Denoising (25 steps) [54.3s]
✓ Done — flux-dev:q4 in 74.3s (seed: 7)

real  2m55s
```

2m55s wall for a 2.1 GB download, an 856 MB pickle read converted to a 609 MB
safetensors, and the render — of which the render itself was 74.3 s, the same
as the pre-removal run.

And the output:

```
accbd724bf81e2866db1cf6b94fcf7006bc52572f0109172c19fb633481727b5  flux-dev-q4-autopull-id-frank-rubio.png
accbd724bf81e2866db1cf6b94fcf7006bc52572f0109172c19fb633481727b5  flux-dev-q4-portrait-id-frank-rubio-official-portrait.png
```

**Byte-identical to the render from before the wipe.** The derived conversion is
deterministic in practice as well as by construction, and the whole identity
pipeline reproduces exactly across a complete asset wipe and re-acquisition.

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
- Krea, block offload, LoRA, and img2img alongside an identity. All are refused
  by the request contract in milestone 1 and belong to a milestone-2
  qualification pass.

## Follow-up UAT (q8 / Metal)

The two checks #1223 deferred, run against `main` at `90d68ef5` (the merged
#1270; the branch recording this sits on `cf116aff`, a docs-only commit that
changes no identity code). Same method as above: the tree rsynced to
`/home/jamesbrink/mold-1223` on plato, built there inside the Nix devshell with
`cargo build --release -p mold-ai --features cuda,pulid`, and the scratch build
deleted afterwards. Outputs are at `/Volumes/ExternalStorage/pulid-dev/uat/`
under `q8-` names.

| # | Check | Host | Tier | Result |
| --- | --- | --- | --- | --- |
| 12 | Zero-weight byte identity (three-way SHA) | plato | q8 | **PASS** — A == B, C differs |
| 13 | Positive identity render, forced-local (Frank Rubio) | plato | q8 | **PASS** — ArcFace cosine **0.5993** |
| 14 | Positive identity render, second face (Kayla Barron) | plato | q8 | **PASS** — ArcFace cosine **0.7241** |
| 15 | Identity refused by a server built without `pulid` | plato | q8 | **PASS** — HTTP 422, surfaced verbatim, nothing rendered |
| 16 | Remote render over HTTP against this build | plato | q8 | **PASS** — byte-identical to the forced-local render |
| 17 | `supports_identity` advertised for exactly the qualified tiers | plato | — | **PASS** — `['flux-dev:q4', 'flux-dev:q8']`, and `false` for all 125 on the non-`pulid` server |
| 18 | Metal path | halcyon | — | **NOT RUN** — `flux-dev:q4` needs a gated Hugging Face repo and no token exists on this machine |

## 12. Zero-weight byte identity on q8

The same three-way check as #3, on `flux-dev:q8`: seed 42, the beach prompt,
1024x1024, 25 steps, `--no-metadata` (see #3 for why the metadata chunk has to
be excluded — B correctly records that a photograph was supplied at weight 0).

| Run | Flags | SHA-256 |
| --- | --- | --- |
| A | none | `0a92b67c8d9be30038d74f32e7a2cf4f2d43678607d4506690c78058d5e31952` |
| B | `--id-image … --id-weight 0.0` | `0a92b67c8d9be30038d74f32e7a2cf4f2d43678607d4506690c78058d5e31952` |
| C | `--id-image … --id-weight 1.0` | `c194a5671a33e21d468cca984465208c184b3031a6a97a7da27f585e267688c5` |

**A == B, C differs.** Files: `q8-nometa-{A,B,C}.png`.

The wall clocks say the same thing independently: A and B both reported
`✓ Done — flux-dev:q8 in 73.6s`, to the tenth of a second, while C took 75.0s.
Weight zero did not merely produce the same pixels — it did not do the work.

## 13 and 14. Fidelity on q8

Prompt, seed, and flags exactly as #4: the close-up portrait prompt, seed 7,
1024x1024, 25 steps, `--id-weight 1.0`, default `--id-start-step`.

| Run | Output | SHA-256 | Cosine |
| --- | --- | --- | --- |
| baseline (no identity) | `q8-portrait-baseline.png` | `cc1b26ba068996f8980f8b8604b58db4b8669ee4ae323a918167542cd6350d23` | — |
| identity (Frank Rubio) | `q8-portrait-id-frank-rubio.png` | `200e0645dbb1a128f7b09afecc5c3aef4686e3e90c3fc307fd0d5effc51864c0` | **0.5993** |
| identity (Kayla Barron) | `q8-portrait-id-kayla-barron.png` | `11b7dd8cfacb0e0af2c12f43b2966521120afb7fbeefd4bdef08d892919964a6` | **0.7241** |

Measured with `crates/mold-inference/tests/pulid_identity_fidelity.rs` against
the same `glintr100` graph the conditioning used, with
`MOLD_TEST_PULID_ASSETS=/storage/mold/models/shared/pulid`:

```
PASS  cosine 0.5993  (threshold 0.28)  q8-portrait-id-frank-rubio.png
PASS  cosine 0.7241  (threshold 0.28)  q8-portrait-id-kayla-barron.png
```

Both land in PuLID's own 0.6-0.8 band, and both sit within 0.03 of the q4
numbers (0.6055 / 0.7016), in opposite directions — which is the expected
result for a change that only alters the transformer's quantization. The
adapter does not read it.

The seed-matched unconditioned baseline was scored as a control and SCRFD found
**no face in it at all** — the same outcome the q4 baseline produced at this
seed, and the reason a *conditioned* render is the one being measured:

```
no face in q8-portrait-baseline.png: no face was detected in the identity image
```

## 15. A server without `pulid` refuses, and never accepts-and-ignores

plato's long-running production server on `:7680` is an ordinary 0.23.3 build
with no `pulid` feature. The identity request went to it as raw HTTP, so the
server's own words are on the record:

```console
$ curl -X POST http://127.0.0.1:7680/api/generate -d '{… "id_image": "…", "id_weight": 1.0}'
HTTP 422
{"error":"this server was built without PuLID face-identity support; remove id_image (and any id_weight, id_start_step, or id_image_name) or use a server built with the `pulid` feature","code":"VALIDATION_ERROR"}
```

and through the CLI, which surfaces it verbatim and exits non-zero without
rendering anything:

```console
$ MOLD_HOST=http://127.0.0.1:7680 mold run flux-dev:q8 "<the portrait prompt>" \
    --seed 7 --id-image frank-rubio-official-portrait.jpg --id-weight 1.0
error: this server was built without PuLID face-identity support; remove id_image …
--- exit=1
```

**Use `127.0.0.1`, not the host's own Tailscale address, when the client runs on
the server.** The first attempt used `http://100.105.134.43:7680` from plato
itself, which does not hairpin — `curl` returns `000` from plato while the same
URL is fine from another machine. `classify_generate_error` reads a connection
failure as `FallbackLocal`, so `mold run` quietly rendered on the local GPU and
printed `Using local GPU inference`. That is the documented remote-to-local
fallback behaving correctly on an unreachable host, not identity being ignored —
but in a log it looks exactly like an accept-and-ignore, so anyone repeating
this check should confirm the host is reachable before reading the result.

## 16. Remote render over HTTP

This build was served on a second port with its own `MOLD_HOME`. The production
`/storage/mold/output` is owned by the `mold` user, and a scratch server running
as a different user cannot take the gallery batch-authority lock there — it
exits with `Permission denied` on `.mold-batch-parent-*.lock`:

```bash
MOLD_HOME=/home/jamesbrink/uat-q8-home \
MOLD_MODELS_DIR=/storage/mold/models MOLD_OUTPUT_DIR=$MOLD_HOME/output \
MOLD_API_KEY=uat-1223 mold serve --bind 127.0.0.1 --port 7681
```

| Path | Output | SHA-256 |
| --- | --- | --- |
| forced-local (`--local`) | `q8-portrait-id-frank-rubio.png` | `200e0645dbb1a128f7b09afecc5c3aef4686e3e90c3fc307fd0d5effc51864c0` |
| remote (HTTP + SSE) | `q8-remote-id-frank-rubio.png` | `200e0645dbb1a128f7b09afecc5c3aef4686e3e90c3fc307fd0d5effc51864c0` |

**Byte-identical**, exactly as on q4.

## 17. `supports_identity` per build

The same 125-model listing, from two builds' `/api/models`:

| Server | `supports_identity == true` |
| --- | --- |
| `:7681`, this build (`cuda,pulid`) | `['flux-dev:q4', 'flux-dev:q8']` |
| `:7680`, stock 0.23.3 (no `pulid`) | none — all 125 report `false` |

The field is present on both, so a client that reads it (rather than guessing
from a version) gets the right answer from either. On the `pulid` build both
qualified tiers report `downloaded: true` and the two unqualified `flux-dev`
tiers report `false`, so the capability tracks the checkpoint and not the
install state.

## 18. Metal (halcyon) — still not run

Unchanged from the [Metal](#metal-halcyon) section above, and re-checked rather
than assumed. The disk objection has since cleared (167 GB free on the external
volume at the time of this run), so the **only** remaining blocker is the token:

```console
$ echo "$HF_TOKEN" "$HUGGING_FACE_HUB_TOKEN"          # both empty
$ ls ~/.cache/huggingface/token                        # No such file or directory
$ ls $MOLD_HOME/catalog-credentials.json ~/.mold/catalog-credentials.json
                                                       # No such file or directory
$ MOLD_HOME=/Volumes/ExternalStorage/mold2 mold config list | grep -i hf
                                                       # nothing
```

And the reason a token is required is now recorded concretely rather than
inferred. `flux-dev:q4`'s own transformer is ungated (`city96/FLUX.1-dev-gguf`),
but `shared_flux_files()` in `crates/mold-core/src/manifest.rs` takes the VAE
from a BFL repo that is not — `ae.safetensors`, `gated: true`:

```console
$ curl -sI https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
HTTP/2 401
x-error-code: GatedRepo
x-error-message: Access to model black-forest-labs/FLUX.1-schnell is restricted. …
```

So no FLUX tier is obtainable on this machine without a Hugging Face account
that has accepted BFL's terms. The gap is therefore unchanged, and unchanged in
kind: the adapter's twenty cross-attention injections and the Metal VRAM
accounting for them are still unverified on Apple Silicon, while the CPU
extraction half continues to run there (it is what scored the cosines in #1222).
Closing it needs a token on a Metal host and nothing more — the commands are
#12 and #13 above with `--width 512 --height 512`.
