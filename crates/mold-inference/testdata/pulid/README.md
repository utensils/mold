# PuLID parity goldens

Fixtures captured from upstream `ToTheBeginning/PuLID` so mold's candle ports
can be falsified against the reference implementation rather than against
another reading of it.

The face-extraction goldens (#1222) live in `faces/`, with their own
`README.md` and `capture_goldens.py`; this file covers the adapter.

## `ca_goldens.safetensors` — `PerceiverAttentionCA` (#1221)

The cross-attention module the FLUX adapter is twenty copies of
(`pulid/encoders_transformer.py:29-72`). Compared by
`crates/mold-inference/tests/pulid_adapter_parity.rs`.

| | |
| --- | --- |
| Capture script | `capture_ca_goldens.py` (imports upstream's module; it does not restate it) |
| Weights | `pulid_flux_v0.9.1.safetensors`, sha256 `92c41c3af322b02e58e1b32842e4601e08c8f16ec1fe80089dbe957df510f51d`, from [`guozinan/PuLID`](https://huggingface.co/guozinan/PuLID) |
| Upstream commit | `ToTheBeginning/PuLID` `main` (shallow clone, 2026-08-21) |
| torch | 2.13.0, CPU, float32 |
| Geometry | `dim=3072`, `dim_head=128`, `heads=16`, `kv_dim=2048` |
| Inputs | image tokens `[1, 64, 3072]` (seed `0x50554C4944434149`), identity `[1, 32, 2048]` (seed `0x50554C4944434144`) |
| Modules sampled | 0, 5, 9, 10, 15, 19 — the ends of both index ranges (double 0–9, single 10–19) plus one interior module each |

Inputs are **not** committed. Both the script and the Rust test generate them
from the same `xorshift64*` stream, so a fixture of any size costs nothing in
the repository; the Rust test pins the stream so a drift in either copy is a
test failure rather than a silent comparison against different inputs.

Each module contributes two arrays:

- `ca{i}.probe` — 512 scattered elements of the `[1, 64, 3072]` output, at flat
  indices drawn from seed `0x50554C4944434150`.
- `ca{i}.stats` — mean, sample standard deviation, and max absolute value over
  the whole output, so a change that misses every probe index still shows up.

`ca_goldens.json` records the same provenance in machine-readable form.

### Measured agreement

mold's port against these goldens, CPU f32:

| module | max abs | max rel |
| --- | --- | --- |
| 0 | 9.91e-7 | 9.91e-7 |
| 5 | 2.27e-6 | 2.27e-6 |
| 9 | 6.97e-6 | 6.97e-6 |
| 10 | 4.05e-6 | 3.73e-6 |
| 15 | 1.68e-5 | 1.26e-5 |
| 19 | 2.29e-5 | 1.12e-5 |

The budgets in the test are `1e-4` absolute and `5e-5` relative — a little
above the worst measurement so an attention-path change is a visible regression
rather than a flake, and far below the values themselves (`absmax` reaches 39.4
on `pulid_ca.19`), so a wrong port still fails.

The error grows with depth because the later modules have larger weights, not
because anything accumulates: each module is evaluated independently on the
same input. The test computes its summary statistics in `f64` — a naive `f32`
sum over 196 608 elements loses more precision than the port does, which would
make the assertion a measurement of the harness.

### Regenerating

```sh
python crates/mold-inference/testdata/pulid/capture_ca_goldens.py \
  --pulid-weights /path/to/pulid_flux_v0.9.1.safetensors \
  --pulid-repo tmp/PuLID
```

Needs only `torch` and `safetensors`. `tmp/` is gitignored and is where the
upstream clone lives.
