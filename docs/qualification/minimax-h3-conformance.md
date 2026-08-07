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

Validate the repository contract and weight-free fixture:

```bash
python3 scripts/minimax-h3-conformance.py check-contract
python3 scripts/tests/minimax-h3-conformance-contract.py
```

Verify pinned local metadata/source clones without executing a checkpoint:

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

## External real-checkpoint fixtures

Real checkpoint execution and fixture capture remain unavailable until the
license gate is cleared. After that decision, evidence must be stored under an
absolute fixture root outside the Mold checkout. The authorization record must
also live outside the repository and content-address the reviewed source
document. The validator is an accidental-bypass guard: it does not authenticate
the issuer or replace legal review.

The external authorization record has this exact shape:

```json
{
  "schema_version": "mold.minimax-h3.authorization.v1",
  "family": "minimax-h3",
  "decision": "approved",
  "license_revision": "bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
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
