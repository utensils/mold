# MiniMax H3 conformance evidence

Mold's H3 implementation is developed against one revision-locked numerical
contract. The checked-in harness contains no MiniMax weights, generated media,
or real-checkpoint activations. It gives ordinary CI a deterministic synthetic
fixture while keeping real evidence outside the repository and behind the
[authorization gate](https://github.com/utensils/mold/issues/831).

The manifest at
`tests/fixtures/minimax_h3/conformance-manifest.json` pins the official code and
model, Diffusers numerical oracle, ComfyUI deployment layout, and the SGLang and
vLLM-Omni performance references. It also content-addresses the official model
and component index files. Diffusers BF16/FP32 mixed execution is authoritative
for numerical comparisons; approximate or non-bit-stable acceleration is
excluded from ground-truth capture.

Here, model-repository pins and component hashes refer only to small public
text metadata. They do not establish that a binary checkpoint shard,
safetensors header range, or generated output was downloaded or opened. No such
artifact is part of the current evidence.

Validate the repository contract and weight-free fixture:

```bash
python3 scripts/minimax-h3-conformance.py check-contract
python3 scripts/tests/minimax-h3-conformance-contract.py
```

Compare the checked-in, weight-free per-layer producer pair:

```bash
python3 scripts/minimax-h3-conformance.py compare \
  --oracle tests/fixtures/minimax_h3/synthetic-oracle-v1.json \
  --mold tests/fixtures/minimax_h3/synthetic-mold-v1.json
```

The two documents conform to
`docs/qualification/minimax-h3-layer-output.schema.json`. Each document names
one case and primitive layer, the producer role and exact source revision, the
adapter command and tensor-hash encoding, the input/component fingerprints,
the execution environment, and a keyed tensor set. Tensor records carry exact
shape, dtype, content hash, min/max/mean/std, and coordinate-addressed stable
samples. Only the oracle declares comparison policy, so a Mold producer cannot
weaken its own acceptance threshold.

The comparator requires the same case, layer, authority tier, input, output
keys, sample coordinates, shapes, dtypes, and hash encoding. It rejects every
NaN or infinity before schema comparison. Numeric statistics and samples pass
only when
`abs(mold - oracle) <= absolute + relative * abs(oracle)`. An `exact` hash
policy makes a content-hash difference fatal; `record-only` retains both hashes
as an explicit diagnostic while numerical tolerances remain authoritative.
Failures aggregate the missing/extra keys and every comparable shape, dtype,
hash, statistic, sample-coordinate, and tolerance discrepancy.

The synthetic pair deliberately includes a small within-tolerance delta and a
record-only hash difference. Contract mutation tests deterministically prove
shape, dtype, output-key, sample-key, missing/extra, NaN, infinity, exact-hash,
and tolerance failures without loading H3 artifacts. Recreate either checked
document for review with:

```bash
python3 scripts/minimax-h3-conformance.py print-synthetic-output --role oracle
python3 scripts/minimax-h3-conformance.py print-synthetic-output --role mold
```

Both documents serialize the fixture's actual coupled Euler outputs rather
than placeholder vectors. The ordinary CPU sampler test executes that update
through Candle and compares its video and audio tensors directly with the
schema-bound oracle samples, while the Python contract independently verifies
their provenance, hashes, statistics, and declared tolerances.

`verify-sources` can verify pinned code checkouts and metadata-only model
checkouts without executing a checkpoint:

```bash
python3 scripts/minimax-h3-conformance.py verify-sources \
  --source minimax-official-code=/absolute/path/to/MiniMax-H3-code \
  --source minimax-official-model=/absolute/path/to/MiniMax-H3-model \
  --source diffusers=/absolute/path/to/diffusers \
  --source transformers=/absolute/path/to/transformers \
  --source comfyui=/absolute/path/to/ComfyUI \
  --source comfy-checkpoints=/absolute/path/to/Comfy-H3-model \
  --source sglang=/absolute/path/to/sglang \
  --source vllm-omni=/absolute/path/to/vllm-omni
```

This record does not claim that command was run against a populated model
checkout. While the authorization gate is closed, model-repository paths must
contain only Git metadata and the named small text files, with Git LFS smudge
disabled. Do not run `git lfs pull`, fetch checkpoint objects, open
`.safetensors` files, or populate these paths merely to satisfy the verifier.
If that metadata-only precondition cannot be proven, defer `verify-sources`
until checkpoint access is authorized.

## External real-checkpoint fixtures

Real checkpoint execution and fixture capture are restricted to an explicitly
approved scope. Evidence must be stored under an absolute fixture root outside
the Mold checkout. The authorization record must also live outside the
repository and content-address the reviewed source document. The validator is
an accidental-bypass guard: it does not authenticate the issuer or replace
legal review.

Everything in this section is a deferred runbook. No authorization record,
real-checkpoint fixture bundle, generated H3 output, or passing licensed UAT is
reported by this document.

The external authorization record has this exact shape:

```json
{
  "schema_version": "mold.minimax-h3.authorization.v1",
  "family": "minimax-h3",
  "decision": "approved",
  "license_revision": "bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
  "license_sha256": "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44",
  "approved_scopes": [
    "checkpoint-execution",
    "fixture-capture",
    "generated-output-retention"
  ],
  "source_document_path": "/external/compliance/reviewed-authorization-evidence.bin",
  "source_document_sha256": "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d",
  "review_reference": "<external review identifier>"
}
```

Validate a captured bundle and every referenced evidence hash:

```bash
python3 scripts/minimax-h3-conformance.py validate-bundle \
  --fixture-root "$MOLD_H3_FIXTURE_ROOT" \
  --bundle "$MOLD_H3_FIXTURE_ROOT/bundle.json" \
  --authorization-record "$MOLD_H3_AUTHORIZATION_RECORD"
```

After authorization and capture, compare any external per-layer pair under the
same approved root:

```bash
python3 scripts/minimax-h3-conformance.py compare \
  --oracle "$MOLD_H3_FIXTURE_ROOT/layers/<case>/oracle.json" \
  --mold "$MOLD_H3_FIXTURE_ROOT/layers/<case>/mold.json" \
  --fixture-root "$MOLD_H3_FIXTURE_ROOT" \
  --authorization-record "$MOLD_H3_AUTHORIZATION_RECORD"
```

Without those last two arguments, `compare` accepts only the exact checked-in
synthetic pair. Any other paths—even files labeled synthetic—must be inside the
external root and pass the existing authorization gate. Non-synthetic producer
documents must additionally bind the authorization source-document hash. The
comparison command consumes adapter JSON only; it never opens checkpoints or
media itself.

The bundle schema is
`docs/qualification/minimax-h3-fixture-bundle.schema.json`. Every fixture names
its primitive layer, authority tier, component-index authorities, environment,
shape/dtype/statistics/sampled values, evidence hash, and numerical tolerance.
Quantized Comfy results use structural, temporal, and audio quality metrics;
they are never mislabeled as bit-identical full-precision evidence.

### Tokenizer and processor capture producer

The first production capture slice is
scripts/capture-minimax-h3-conditioner.py tokenizer-processor. It is an opt-in
evidence producer, not a Mold runtime command. It has no Cargo feature, binary
entry point, release dependency, automatic workflow trigger, or model download
path. Its Python ML imports occur only after the authorization, external-root,
source, and component preflights pass.

The producer fixes one reproducible conditioner case:

- a UTF-8 raw prompt containing both non-ASCII text and a literal Qwen vision
  token spelling, with no chat template and no added special tokens;
- one generated 256 by 256 RGB image, chosen to meet the pinned processor
  minimum without a resize;
- one generated 37-frame, 24 fps, 64 by 64 RGB video, sampled through the
  pinned Diffusers helper at 2 fps into source frames 0, 12, 24, and 36 and
  paired into timestamped Qwen vision blocks;
- the exact combined image/video Ref2VA presentation from the pinned Diffusers
  implementation.

It records five manifest-required measurements. Token IDs are length-prefixed
raw and multimodal presentations. Special-token evidence is a length-prefixed
set of the required token IDs, H3 row tags, Qwen modality type IDs, sampled
source indices, and millisecond timestamps. Processor shapes use stable numeric
field IDs followed by rank and dimensions. `processor-grids` records the actual
flattened image and video `grid_thw` integer values, not only their tensor
shapes, so token counts and multimodal rotary geometry are exact evidence.
Processor pixel values are
concatenated after an actual CUDA BF16 copy and hashed as canonical typed
little-endian bytes. The raw generated pixels are not retained; only their
input hashes and strict tensor summaries leave the process.

Capture requires clean external checkouts at these exact revisions:

- Diffusers: 9c6a68c32b3b2a64db91800b624d33cec6e25ab8
- Transformers: 42f189ded85d18d00b51161d694cafd325e32b91
- MiniMaxAI/MiniMax-H3 snapshot:
  bfc8ed0353f5a9733be73e6b2c98ec0948195b86

The Transformers revision is the main-tree companion available when the
pinned Diffusers H3 integration landed and includes the exact
create_mm_token_type_ids authority that integration calls. The producer
resolves both imported implementations back into those clean checkouts. For
every model metadata file, it verifies both the manifest SHA-256 and the
Hugging Face local-directory metadata revision. A manually assembled directory
without revision metadata fails closed even if its visible files happen to
hash correctly.

The manifest also pins the complete oracle runtime identity used by this
producer: Python 3.13.13, PyTorch 2.13.0+cu130, NumPy 2.5.1, CUDA 13.0, and the
full Transformers revision above. The capture script itself is a
manifest-pinned SHA-256 authority whose repository-relative implementation path
is traversal-, symlink-, and checkout-containment checked before hashing. Both
identities are repeated in structured layer and bundle evidence and enforced
again by the protected runner. Bundles carry a canonical per-layer
`oracle_adapters` list derived from the manifest contracts for the fixture
layers they contain. Missing, extra, duplicate, reordered, or cross-wired
adapter records fail closed, which permits later capture producers to add their
own adapter contract without weakening or special-casing this one.

Before importing either source or opening model configuration at execution
time, the producer creates an owner-only temporary directory under the external
fixture root. It exports each exact Git revision with `git archive` and copies
each required model component from one no-follow read whose bytes match the
manifest hash. Runtime imports and `from_pretrained` calls use only this staged
snapshot. The complete staged file set, sizes, and hashes are revalidated after
execution and immediately before evidence is written; the original checkouts
and model paths are never reopened as runtime authority.

Prepare the external paths, including both tokenizer/ and processor/ metadata
directories from the exact model snapshot, then run:

    export MOLD_H3_FIXTURE_ROOT=/external/h3/evidence
    export MOLD_H3_AUTHORIZATION_RECORD=/external/h3/compliance/authorization.json
    export MOLD_H3_OFFICIAL_MODEL=/external/h3/models/MiniMax-H3
    export MOLD_H3_DIFFUSERS_CHECKOUT=/external/h3/src/diffusers
    export MOLD_H3_TRANSFORMERS_CHECKOUT=/external/h3/src/transformers
    python3 scripts/capture-minimax-h3-conditioner.py \
      tokenizer-processor --device cuda:0

The producer disables TF32, flash and memory-efficient SDPA, cuDNN SDPA,
non-deterministic algorithms, and every manifest-excluded acceleration before
the CUDA copy. Environment variables that configure an excluded acceleration
are rejected before any ML package import. It writes mode-0600 oracle.json and
oracle-bundle.json files under a new, input-addressed directory below
MOLD_H3_FIXTURE_ROOT and refuses to overwrite an existing capture.

The emitted partial bundle is independently valid under
mold.minimax-h3.fixture-bundle.v1, and its layer is accepted by the exact
protected measurement validator. It is intentionally not a complete campaign:
the protected runner still requires paired oracle and Mold evidence for all
exact-full-bf16 layers.

### Exact-BF16 Qwen layer-50 capture

`scripts/capture-minimax-h3-qwen-layer50.py` is the paired producer for the
next conditioner boundary. It records two cases under the strict
`qwen-layer-50` contract:

- a raw text-only presentation; and
- a representative presentation containing one deterministic image and one
  deterministically sampled video.

Both roles consume the same input-addressed token IDs, three-axis Qwen
positions, processor grids, and BF16 processor values. The oracle calls the
pinned Diffusers `get_qwen3vl_prompt_embeds` adapter at
`text_encoder_layer=50`; the Mold role compiles and invokes the isolated
`h3_qwen_layer50_capture` binary from the exact clean checkout. That binary
uses only the private exact-BF16 streamed loader over the complete official
checkpoint and returns the unnormalized state after language layer 49. The
oracle retains the full BF16 state in host RAM and moves one official language
layer at a time to CUDA through pinned Transformers module hooks. Mold keeps
the embedding and vision modules resident, mmap-loads exactly one of the 50
selected language layers for each production forward step, synchronizes before
dropping it, and records a peak of one resident language layer. The streamed
design is intended to fit a 46 GiB device without changing BF16 equations or
using a quantized checkpoint; only real paired CUDA capture establishes the
total peak including resident modules, activations, and allocator state. The
deployment quantized conditioner is not an oracle and is not reachable from
this adapter.

Use the same five external environment paths shown above. Run the roles
separately so each full model is released before the other is loaded:

    python3 scripts/capture-minimax-h3-qwen-layer50.py \
      capture --role oracle --device cuda:0
    python3 scripts/capture-minimax-h3-qwen-layer50.py \
      capture --role mold --device cuda:0

Preflight at least 96 GiB of available host RAM and 12 GiB of free device
memory; do not start capture when either bound fails. Before the oracle loads,
the producer authenticates all 14 official text
encoder shards against revision-bound Hugging Face metadata and their LFS
SHA-256 values. The Mold loader independently hashes and structurally accounts
for the same full checkpoint. Both roles import Diffusers and Transformers only
from owner-only, read-only Git archive snapshots, stage the small official
tokenizer and processor authorities the same way, record the exact Python,
PyTorch, NumPy, CUDA, and Transformers oracle runtime identity in oracle
evidence, verify that same pinned preparation runtime for both roles, and
revalidate staged and full-checkpoint identities after inference. The Mold role
additionally builds from an owner-only Git archive of the exact clean Mold
revision and uses `cargo run --locked --release` with
`dev-bins,h3-private-uat,cuda`; its Cargo target directory lives below
`MOLD_H3_FIXTURE_ROOT`, never in the repository. The binary is not part of any
shipping feature set, and published-binary verification rejects its private
claim marker as an independent release-isolation check.

Each role writes two owner-only layer documents and one partial bundle into a
shared input-addressed directory below
`MOLD_H3_FIXTURE_ROOT/captures/qwen-layer-50/`. The Mold role additionally
retains its exact request and raw BF16 response there. All writes use
create-new semantics and refuse replacement; request and response files are
sealed read-only before they become evidence, and raw output is capped at 128
MiB. The documents include shape, dtype, a full-activation content hash and
statistics, every activation coordinate/value in canonical row-major order,
and the reviewed magnitude-aware BF16 policy. After the released precision
boundaries were matched, the complete 102,400-element text and 583,680-element
multimodal L40S evidence measured no ordinary-value violation under
`48 + abs(oracle) / 64`. Seven multimodal values had oracle magnitude at least
1,024; their worst observed relative difference was `5/13`, and the tightest
reviewed high-magnitude envelope is `48 + 3 * abs(oracle) / 8`. Protected
validation requires those exact two piecewise bounds, the 1,024 threshold,
complete ordered coordinates, and record-only retention of both content
hashes; neither producer can widen the policy. A real capture has not passed
merely because these producer contracts and CUDA typechecks pass; only a clean
paired protected external campaign may establish numerical parity.

### FP32/FP16 visual-VAE capture

`scripts/capture-minimax-h3-visual-vae.py` is the paired producer for the
`visual-vae` boundary. It uses one deterministic
`visual-vae-320x320x22-seed42-v1` case. That geometry exercises the released
shortest round-trip temporal path: two 17-frame encoder chunks produce seven
latent frames and decode back to 22 frames. It also exercises a real two-by-two
spatial tile grid. The input identity binds the 256-pixel tile, 64-pixel minimum
overlap, effective seam boundaries at pixels 64 and 256, and representative
seam frames 0, 10, and 21.

The oracle is the pinned Diffusers `AutoencoderKLMiniMaxH3` loaded from the
official three-shard FP32 checkpoint. The Mold role compiles the isolated
`h3_visual_vae_capture` binary from a sealed source archive. Both roles record,
in manifest order, FP32 ImageNet-normalized pixels, FP32 posterior moments, the
fresh CPU seed-42 FP32 noise, the FP32 posterior sample, the actual FP16
round-trip tensor, FP32 normalized latents, FP32 decoded unit-RGB frames, and
the complete bounded FP32 seam-probe vector. The independently generated
seed-42 noise streams use the same PyTorch MT19937 recipe but permit the
reviewed 2e-6 absolute plus 1e-6 relative FP32 scalar-math bound; their hashes
remain record evidence. Numerical model records use the checked 1/64
elementwise tolerance and likewise retain their hashes as record evidence.

Use the same five external environment paths as the conditioner captures and
run each role separately:

    python3 scripts/capture-minimax-h3-visual-vae.py \
      capture --role oracle --device cuda:0
    python3 scripts/capture-minimax-h3-visual-vae.py \
      capture --role mold --device cuda:0

Before either model executes, the producer authenticates the official VAE
config, index, and all three shards against exact revision-bound hashes, plus
revision metadata, aggregate payload size, and a canonical artifact-set hash. It pins the Python, PyTorch, NumPy, CUDA,
Transformers, Diffusers, adapter, and source identities; disables TF32 and
approximate attention; and revalidates every protected identity after capture.
The Mold adapter independently rejects non-FP32 storage, non-FP32 encode,
non-FP16 CUDA decode, non-math attention, Comfy weights, non-CUDA devices,
permissive output directories, mutable requests, and in-repository evidence.
Its Cargo target and all retained request/raw/layer/bundle evidence remain
under the owner-only external fixture root. No checkpoint, raw tensor, or
decoded media is committed to the repository.

### Exact-FP32 AudioVAE capture

`scripts/capture-minimax-h3-audio-vae.py` captures the manifest's `audio-vae`
boundary as paired authorization-bound evidence. The oracle loads the official
Diffusers AudioVAE checkpoint; the Mold role builds the isolated
`h3_audio_vae_capture` binary from an owner-only archive of the exact clean
checkout and loads the reviewed Comfy folded FP32 checkpoint through Mold's
validated AudioVAE loader. Both paths are CUDA-only and FP32-only. The binary
requires `dev-bins,h3-private-uat,cuda`, is absent from every shipping feature
set, and retains a marker that published-binary verification rejects.

The deterministic case is a two-item stereo batch with four distinct channels
and 2,401 samples per channel. It binds channel-major packing into the official
`[batch * 2, 1, samples]` mono contract, padding to 3,200 samples, four latent
rows at 40 Hz, and restoration to `[batch, 2, samples]`. Evidence retains the
complete decoded FP32 PCM and normalized latent payloads, input and decoded
waveform statistics, lagged phase/polarity correlations, the exact packing
map, and the complete typed content hashes. The two checkpoints have different
reviewed hashes, so continuous AudioVAE records use bounded numerical
tolerances with `record-only` hashes; packing and timeline records remain
zero-tolerance and hash-exact. Pair comparison requires the official checkpoint
hash for the oracle role and the reviewed Comfy folded checkpoint hash for the
Mold role; any other role/checkpoint pairing fails before numerical comparison.

Prepare the five external paths and run each role in a fresh process so GPU
memory from one implementation cannot affect the other:

    export MOLD_H3_FIXTURE_ROOT=/external/h3/evidence
    export MOLD_H3_AUTHORIZATION_RECORD=/external/h3/compliance/authorization.json
    export MOLD_H3_OFFICIAL_MODEL=/external/h3/models/MiniMax-H3
    export MOLD_H3_DIFFUSERS_CHECKOUT=/external/h3/src/diffusers
    export MOLD_H3_COMFY_MODEL=/external/h3/models/Comfy-H3
    python3 scripts/capture-minimax-h3-audio-vae.py \
      capture --role oracle --device cuda:0
    python3 scripts/capture-minimax-h3-audio-vae.py \
      capture --role mold --device cuda:0

The oracle environment must provide Python 3, NumPy, PyTorch with CUDA,
Safetensors, Hugging Face Hub, Accelerate, and the pinned Diffusers checkout on
its import path; the producer imports Diffusers only from its sealed Git archive
snapshot. The Mold role additionally requires the Rust 1.93 toolchain, Cargo,
a CUDA toolkit, and the native build dependencies already used by Mold's CUDA
CI. Its Cargo target directory is created below `MOLD_H3_FIXTURE_ROOT`.

Before inference, the producer authenticates the official config and checkpoint
and the Comfy checkpoint by exact revision, size, and SHA-256, validates the
external authorization record and clean source checkouts, disables every
manifest-excluded acceleration, and seals source snapshots read-only. After
inference it revalidates every protected input. Runtime, checkpoint, and adapter
implementation hashes are bound into structured evidence and provenance.
Create-new owner-only writes refuse overwrite and keep raw request, response,
weights, and evidence outside the repository. Passing contract tests or CUDA
typechecks alone is not evidence that this real-checkpoint capture ran.

### Paired token-refiner and transformer-block capture

`scripts/capture-minimax-h3-transformer.py` produces the remaining paired
evidence for `token-refiner` and `transformer-block`. Run FL2VA and Ref2VA as
distinct cases: their deterministic tensor geometry is intentionally equal,
but their input identities, task labels, selected checkpoint directories, and
raw adapter receipts are not interchangeable. Both layer documents bind the
complete reviewed 32-record authority set: config, index, and all fourteen
official shards for each task checkpoint.

The oracle imports the first official Diffusers token-refiner block, first
main block, final normalization, and both output heads from the selected task.
It reads only the tensors those modules need rather than materializing complete
multi-gigabyte shards. The Mold role builds the private
`h3_transformer_capture` adapter from an owner-only archive of the exact clean
checkout and passes the manifest from that same archive. The adapter retains
authenticated descriptors for all 32 artifacts, loads the selected task only,
converts grouped QKV and the official value/gate fused FFN order into Mold's
production layout, observes the production equations, and rehashes every
descriptor after synchronized CUDA execution.

Use the same five external paths as the conditioner capture. The official
model root must contain both complete `transformer/` and `transformer_ref/`
directories at the pinned model revision. Preflight free host RAM and device
memory before each command. Run one role per process; each command captures the
FL2VA task first, releases its model, then captures Ref2VA:

    python3 scripts/capture-minimax-h3-transformer.py \
      capture --role oracle --device cuda:0
    python3 scripts/capture-minimax-h3-transformer.py \
      capture --role mold --device cuda:0

The deterministic record covers token-refiner output shape/statistics, Q/K RMS,
both Q and K leading multimodal RoPE axes, AdaLN parameters, and the FP32 video
and audio heads. Every compared transformer coordinate is retained in canonical
order; both producer hashes remain evidence, but floating hashes are record-only
because the pinned PyTorch and Candle CUDA math paths do not promise byte-identical
BF16 accumulation. Integer and policy records remain exact.

The reviewed elementwise policies add `1/64 * abs(oracle)` to these absolute
caps: 8 for the high-magnitude token-refiner output, `1/16` for partial MM-RoPE,
the existing `1/64` for Q/K RMS, token statistics, and AdaLN, 1.25 for the FP32
video head, and 0.5 for the FP32 audio head. The real two-task L40S comparison
that established those bounds measured normalized RMSE of 0.23–0.39% across
the BF16 records and 0.82–1.09% across the derived FP32 heads. Protected
validation independently caps each named policy, requires complete ordered
coordinate coverage for both roles, and rejects any out-of-bound scalar. The
protected reader admits layer documents only as no-follow regular files up to
64 MiB; the complete transformer-block records are approximately 32.7 MiB.

TF32, approximate attention, compilation, FP8, and non-deterministic algorithms
remain disabled. Raw adapter output and evidence use create-new owner-only files
below the external fixture root, and raw output is parsed from the exact
descriptor-retained bytes before revalidation. Every published layer and bundle
binds the reviewed authorization source-evidence hash carried by the validated
wrapper record, never the wrapper file's own hash. The producer and its CI
contract do not claim that the real CUDA campaign has passed until a protected
pair using these reviewed policies succeeds.

### Opt-in protected GPU validation

The manual `MiniMax H3 private conformance` workflow validates an approved,
already-captured real-checkpoint campaign on an access-controlled self-hosted
CUDA runner. It has no push, pull-request, schedule, or chained-workflow
trigger. Dispatch it from `main` with the exact reviewed commit as
`expected_source_sha`; its preflight rejects every other ref or source SHA, and
the protected runner independently checks its checkout before reading external
evidence.

The following are **unchecked administrator prerequisites**. Verify them in
GitHub before every dispatch; neither the workflow nor this repository can
attest repository settings:

- Create the `minimax-h3-private-uat` Environment, add required reviewers and
  the intended branch/ref protection, and configure its four path secrets.
  Referencing a missing Environment can create one without the intended
  protection, so the name in this workflow is not evidence that approval is
  enforced.
- Create the dedicated `minimax-h3-private-conformance` runner group, restrict
  it to selected workflows, and allow exactly
  `utensils/mold/.github/workflows/minimax-h3-private-conformance.yml@refs/heads/main`.
  Provision only an ephemeral, isolated runner in that group. Do not register
  a persistent public-repository runner for this campaign.
- Give the runner the `self-hosted`, `linux`, `x64`, `cuda`, and
  `minimax-h3-private-uat` labels. Labels select compatible capacity; they are
  routing metadata, not an access-control boundary. Runner-group workflow/ref
  restriction and Environment reviewers provide the administrative boundary.

A missing matching runner can leave the job queued, and missing secrets are
rejected by the validator, but neither behavior proves that the Environment or
runner-group restrictions exist. Each secret value is an absolute path visible
only to the validator step:

- `MOLD_H3_FIXTURE_ROOT`: the external root containing all retained evidence.
- `MOLD_H3_AUTHORIZATION_RECORD`: the external authorization JSON described
  above.
- `MOLD_H3_ORACLE_BUNDLE`: a Diffusers bundle captured at the manifest's exact
  pinned revision.
- `MOLD_H3_MOLD_BUNDLE`: a distinct Mold bundle captured at the dispatched
  source SHA.

Both bundles use `mold.minimax-h3.fixture-bundle.v1`; their referenced files use
`mold.minimax-h3.layer-output.v1`. Bundle and layer capture environments must
declare a `cuda:` device and the canonical `bfloat16` execution dtype, and their
attention backend must not name an acceleration excluded by the manifest. Both
scopes also carry `acceleration_policy.enabled` and `.disabled`: identifiers
must be canonical and duplicate-free, `disabled` must equal the complete
manifest exclusion set, `enabled` must contain none of it, and each layer must
repeat its bundle's capture policy. The manifest separately pins reviewed
aliases for every exclusion, so names such as precision formats or abbreviated
attention backends cannot bypass the enabled/backend checks. This is structured
capture attestation; the runner does not execute the recorded command or
inspect the original process.
Non-synthetic documents must bind the already reviewed authorization evidence
SHA-256 shown above. The protected runner requires a `provenance` record whose
keys exactly equal each manifest layer's `required_provenance`; canonical
source, component, device, dtype, attention-backend, and capture-command values
must also agree with the document's structured producer, input, environment,
and adapter fields. Each layer must carry the exact manifest-selected component
authority records, including every member of a multi-component layer; the
singular component hash remains only a compatibility summary of the first
manifest-selected record. Manifest-declared role-invariant provenance must
also match between the oracle and Mold documents. The runner likewise requires
output and oracle-policy keys to exactly equal `required_measurements`, and
pins `tensor_hash_encoding` to `canonical-typed-le-v1`. The additive component
authority and provenance fields remain optional in the base schemas only so
the older checked synthetic contract stays compatible; they are mandatory for
this protected path.

The tokenizer and processor provenance hashes are separately authority-pinned,
not merely compared between roles. Each is SHA-256 over canonical UTF-8 JSON
with schema `mold.minimax-h3.component-authority-set.v1` and a sorted
`components` array of `{id,sha256}` records. The tokenizer set binds its JSON,
configuration, merges, and vocabulary; the processor set binds its image and
video configuration, chat template, and processor-local tokenizer files. All
of those raw files are independently content-addressed at the official model
revision and must also belong to the layer's component authority set.

The runner uses an explicit dtype and comparison policy for every required
measurement. Captured floating activations are `bfloat16`. Computed metric and
statistical summaries are serialized as `float64` so reductions and derived
quality values retain stable host-side precision; this does not claim that the
captured model tensors ran in float64. Discrete tokenizer, shape, layout,
ordering, allocation, polarity, and hash records are signed `int64`; every
minimum, maximum, and sampled value must remain within the signed 64-bit range.
They use zero absolute and relative tolerance and require an exact content hash.
For `bfloat16` records, every minimum, maximum, and sample must round-trip
exactly through BF16; means and standard deviations must be finite, exactly
representable binary64 values. Every `float64` metric statistic and sample, and
every oracle tolerance, must likewise be finite and exactly representable in
binary64. Floating records require an exact content hash and retain absolute
and relative tolerances independently capped at `1/64`. Any protected
exact-tier hash mismatch is fatal,
including when the recorded summaries and samples still agree. Oracle and Mold
outputs must agree on measurement keys, dtype, content hashes, and declared
role-invariant provenance; the protected policy is derived by the runner rather
than accepted from a relabeled fixture.

The three end-to-end layers use an explicit
`mold.minimax-h3.e2e-input.v1` descriptor. `input.sha256` is SHA-256 of its UTF-8
JSON with keys sorted recursively, comma/colon separators, non-ASCII text
emitted directly rather than `\u`-escaped, no insignificant whitespace, and no
non-finite numbers; sampler decimal values are canonical strings so the digest
does not depend on language-specific float formatting. The descriptor binds
exact prompt bytes,
the absence of a negative prompt, ordered raw conditioning-byte digests, an
unsigned 64-bit seed, dimensions, frame count, FPS, and both sampler settings.
The runner enforces the H3 32-pixel dimension grid, pixel/aspect limits,
124–345-frame `17k+5` grid, 24 FPS, equal step counts of at least two, zero
guidance, rectified-flow Euler schedules, and video/audio shifts 12/3. T2VA has
no conditioning, FL2VA has canonical first/last-frame ordering, and Ref2VA has
a non-empty ordered reference list. Oracle and Mold input digests must match;
the end-to-end component sets also bind tokenizer, processor, transformer, VAE,
and both official scheduler configurations.

No fixed prompt, media set, seed, dimensions, frame count, or step count is yet
pinned as the single approved UAT case. This contract proves each supplied
descriptor is canonical and that both roles ran the same semantically valid H3
case; a later reviewed campaign digest is required to mandate one particular
case.

The runner derives the authority-tier map from the validated manifest and
requires paired oracle and Mold evidence for all eleven `exact-full-bf16`
layers; it excludes the manifest's synthetic sampler instead of relabeling it.
Synthetic and quantized-structural authority remain available only to their
separate, explicitly labeled qualification paths.

Each bundle's duplicated tensor summary must exactly mirror the document's
first output. The oracle bundle tolerance must exactly mirror that output's
validated policy, and the Mold bundle must repeat the same policy summary. A
mismatch is rejected rather than silently trusting bundle metadata. Mold layer
documents cannot declare a second comparison policy; the runner applies the
validated oracle policy to both producers, preventing a Mold-side tolerance or
hash-policy override.

The workflow neither executes the adapter `command` strings nor captures new
evidence. Those fields remain provenance from separately reviewed capture
steps. It also does not upload artifacts or copy external evidence into the
checkout; detailed evidence stays on the protected runner, and failures expose
only redacted contract context in CI logs. Evidence bytes are hashed, parsed,
validated, and retained from one read; numerical comparison uses those same
in-memory documents, so a later path replacement cannot change the compared
values. Adding this validation
infrastructure does not by itself claim that a real checkpoint was captured or
that licensed GPU UAT passed.

## Day-zero frame decision

Mold caps the advertised duration at the largest valid `17k+5` value that does
not exceed 15 seconds: 345 frames at 24 fps, or 14.375 seconds. The pinned
Diffusers path aligns 360 to 362 and then rejects it for exceeding 15 seconds.
The fixture preserves that discrepancy and Mold's compatible 345-frame policy.
