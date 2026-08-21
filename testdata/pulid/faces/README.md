# PuLID face-extraction fixtures (#1222)

Four public-domain portrait photographs and the InsightFace goldens captured
from them. Consumed by `crates/mold-inference/tests/pulid_face_parity.rs`.

The images are committed with `git add -f` because the repository ignores
`*.jpg` and `*.png` wholesale. Total committed size is ~2.1 MB.

## Licenses and sources

Every image is **public domain** (a NASA work). `sources.json` in this
directory carries the machine-readable record — Commons title, license, credit,
description URL, and the exact source URL — and is regenerated alongside the
files.

| file | subject | credit | license | Commons page |
| --- | --- | --- | --- | --- |
| `frank-rubio-official-portrait.jpg` | Frank Rubio | Bill Stafford (NASA) | Public domain | [File:Frank Rubio official portrait.jpg](https://commons.wikimedia.org/wiki/File:Frank_Rubio_official_portrait.jpg) |
| `kayla-barron-official-portrait.jpg` | Kayla Barron | Bill Stafford (NASA) | Public domain | [File:Kayla Barron official portrait.jpg](https://commons.wikimedia.org/wiki/File:Kayla_Barron_official_portrait.jpg) |
| `mae-jemison-official-portrait-of-1987-astronaut-candidate.jpg` | Mae Jemison | NASA | Public domain | [File:Mae Jemison - Official portrait of 1987 astronaut candidate.jpg](https://commons.wikimedia.org/wiki/File:Mae_Jemison_-_Official_portrait_of_1987_astronaut_candidate.jpg) |
| `raja-chari-official-portrait.jpg` | Raja Chari | Bill Stafford (NASA) | Public domain | [File:Raja Chari official portrait.jpg](https://commons.wikimedia.org/wiki/File:Raja_Chari_official_portrait.jpg) |

Each was downscaled to 800 px wide and re-encoded as JPEG under 300 KB by
`../fetch_faces.py`, which refuses any Commons file whose license is not public
domain or CC0.

## What is committed per face

| suffix | contents |
| --- | --- |
| `.jpg` | the source photograph |
| `.golden.json` | landmarks, bbox, score, the raw 512-d ArcFace embedding, `m112`, `m512` (cv2 LMEDS), `m512_skimage` |
| `.arcface112.png` | `cv2.warpAffine` 112×112 ArcFace crop |
| `.eva512.png` | facexlib's 512×512 crop |

`../onnx-inventory.json` holds the op/attribute inventory of both ONNX graphs
with their SHA-256 digests, so the Step-0 op gate runs without the weights.

## Capture

The InsightFace `antelopev2` weights are **not** committed — they are
non-commercial-research-only pretrained models (see `THIRD_PARTY_NOTICES.md`).
Regenerate the goldens with a scratch venv holding `insightface onnxruntime
opencv-python-headless numpy scikit-image pillow`:

```bash
python3 testdata/pulid/fetch_faces.py testdata/pulid/faces
python3 testdata/pulid/capture_goldens.py \
    --assets /path/to/antelopev2 --faces testdata/pulid/faces
cargo run --release -p mold-ai-inference --features dev-bins,pulid \
    --bin pulid_face_probe -- inventory /path/to/antelopev2 \
    --write testdata/pulid/onnx-inventory.json
```

Captured 2026-08-21 on macOS aarch64 with onnxruntime 1.29.0, OpenCV 5.0.0,
insightface 1.0.1, scikit-image ≥ 0.26, against `scrfd_10g_bnkps.onnx`
`5838f7fe…b5b91` and `glintr100.onnx` `4ab1d643…4cdf` — the SHA-256 pins in
`crates/mold-core/src/manifest.rs`.

## Tolerances, and the numbers that earned them

Set in `crates/mold-inference/tests/pulid_face_parity.rs`. Each is the measured
worst case across these four faces plus headroom, so a resampler or candle
change shows up as a regression rather than a flake.

| check | tolerance | measured worst |
| --- | --- | --- |
| `m112` vs skimage `SimilarityTransform` | 1e-4 | 1.74e-5 |
| `m512` vs skimage | 1e-4 | 1.74e-5 |
| `m512` vs `cv2.estimateAffinePartial2D(LMEDS)` | 1e-4 | 1.14e-5 |
| 112 crop, mean abs channel delta | 0.6 / 255 | 0.229 |
| 112 crop, p99.9 abs channel delta | 4 LSB | 2 |
| 512 crop, mean abs channel delta | 0.6 / 255 | 0.190 |
| 512 crop, p99.9 abs channel delta | 4 LSB | 2 |
| landmark position (weight-gated) | 1.0 px | 0.232 px |
| bbox corner (weight-gated) | 2.0 px | inside |
| detection score (weight-gated) | 0.02 | inside |
| ArcFace cosine (weight-gated) | ≥ 0.99 | 0.999384 |

Per-face ArcFace cosine: Frank Rubio 0.999384, Kayla Barron 0.999773, Mae
Jemison 0.999871, Raja Chari 0.999774.

Template fit residuals (RMS, source landmarks onto the template), recorded so a
transcription slip in either template is visible: 112 → 4.48–6.20 px, 512 →
9.05–14.86 px.

Full context: `docs/architecture/pulid-face-extraction.md`.
