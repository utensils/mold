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

Real checkpoint execution and fixture capture remain unavailable until the
license gate is cleared. After that decision, evidence must be stored under an
absolute fixture root outside the Mold checkout. The authorization record must
also live outside the repository and content-address the reviewed source
document. The validator is an accidental-bypass guard: it does not authenticate
the issuer or replace legal review.

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
  "source_document_path": "/external/compliance/minimax-h3-authorization.pdf",
  "source_document_sha256": "<sha256>",
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
its primitive layer, authority tier, component-index hash, environment,
shape/dtype/statistics/sampled values, evidence hash, and numerical tolerance.
Quantized Comfy results use structural, temporal, and audio quality metrics;
they are never mislabeled as bit-identical full-precision evidence.

## Day-zero frame decision

Mold intentionally accepts the advertised nominal 15-second result after H3
alignment: 362 frames at 24 fps, or 15.0833 seconds. The pinned Diffusers path
currently aligns 360 to 362 and then rejects it for exceeding 15 seconds. The
synthetic fixture preserves both the discrepancy and Mold's explicit decision
so later refactors cannot inherit the rejection accidentally.
