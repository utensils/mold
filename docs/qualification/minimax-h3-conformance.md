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
124–362-frame `17k+5` grid, 24 FPS, equal step counts of at least two, zero
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

Mold intentionally accepts the advertised nominal 15-second result after H3
alignment: 362 frames at 24 fps, or 15.0833 seconds. The pinned Diffusers path
currently aligns 360 to 362 and then rejects it for exceeding 15 seconds. The
synthetic fixture preserves both the discrepancy and Mold's explicit decision
so later refactors cannot inherit the rejection accidentally.
