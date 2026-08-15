# LTX-2 image-preprocess golden fixtures

Golden inputs and expected outputs captured from the **official Lightricks/LTX-2 Python
implementation**, for the image-conditioning preprocess parity work in issue #1055.

Upstream revision: **`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`**
(<https://github.com/Lightricks/LTX-2>, merge of PR #273).

Everything here was produced by running upstream's own functions, loaded directly from
their source files — no reimplementation:

| Function | Upstream file | Lines |
| --- | --- | --- |
| `decode_image` | `packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/decode.py` | 139–170 |
| `preprocess` | same | 413–435 |
| `encode_single_frame` | same | 386–400 |
| `decode_single_frame` | same | 403–410 |
| `resize_and_center_crop` | `.../media_io/resize.py` | 41–73 |

`crf = 33` is upstream's `DEFAULT_IMAGE_CRF`
(`packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:36`). LTX-2.4 uses
`LTX_2_4_IMAGE_CRF = 18` (`constants.py:37`); only 33 is captured here.

## Regenerating

The capture script is `tmp/capture_1055_fixtures.py` (gitignored, deliberately not
committed — it depends on a network clone of upstream and on nixpkgs' Python set).

```bash
# 1. Shallow-clone upstream at the exact revision
mkdir -p tmp && git init tmp/LTX-2-fd4ded7 && cd tmp/LTX-2-fd4ded7
git remote add origin https://github.com/Lightricks/LTX-2.git
git fetch --depth 1 origin fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
git checkout FETCH_HEAD && cd ../..

# 2. Fetch the Display-P3 ICC profile used by display_p3.png (see below)
curl -sSL -o tmp/DisplayP3-v4.icc \
  https://raw.githubusercontent.com/saucecontrol/Compact-ICC-Profiles/master/profiles/DisplayP3-v4.icc

# 3. Run the capture (NixOS: manylinux torch/av wheels will not run — use nix)
nix-shell -p 'python3.withPackages (ps: with ps; [ torch pillow numpy av einops ])' \
  --run 'python3 tmp/capture_1055_fixtures.py'
```

`decode.py` imports `OpenImageIO`, `ltx_core.*`, and the EXR/HDR/range-map helpers, none
of which the SDR still-image path touches. The script stubs exactly those modules in
`sys.modules` and then `exec`s the real `decode.py` and `resize.py` files, so the code
under capture is upstream's, byte for byte.

### Toolchain used for this capture

| Component | Version |
| --- | --- |
| Python | 3.13.12 |
| PyTorch | 2.9.1 |
| NumPy | 2.3.4 |
| Pillow | 12.2.0 |
| PyAV | 16.0.1 |
| einops | 0.8.1 |
| libavcodec | 62.11.100 |
| libavutil | 60.8.100 |
| libavformat | 62.3.100 |
| libswscale | 9.1.100 |
| H.264 encoder | `libx264` (bundled with nixpkgs' PyAV) |

## File format

Each expected output is a raw little-endian `.bin` plus a `.json` sidecar:

```json
{ "shape": [H, W, 3], "dtype": "u8" | "f32", "desc": "...", "crf": 33 }
```

`u8` is one byte per sample; `f32` is IEEE-754 little-endian. Layout is always
interleaved HWC RGB (row-major, `H * W * 3` samples).

Inputs are the `.png` / `.jpg` files in this directory. They are generated
deterministically (fixed seeds, no clock or unseeded RNG), so a re-run reproduces them.

## Inputs

| File | Size (W×H) | Contents |
| --- | --- | --- |
| `gradient_96x64.png` | 96×64 | Horizontal R ramp, vertical G ramp, flat B, plus a 16×16 two-pixel-cell checkerboard at (8, 8) and a hard diagonal edge — high-frequency detail so the codec has real work to do. |
| `photo_like_128x96.png` | 128×96 | Seeded (`20260814`) gaussian noise, box-blurred three times to photo-like low-frequency statistics, re-stretched, plus fine grain, a solid red rectangle, a solid blue disc, and a 2px bright vertical bar. |
| `oddsize_97x63.png` | 97×63 | Same generator as `gradient`, at odd dimensions in both axes. |
| `portrait_exif6.jpg` | stored 96×64, upright 64×96 | Asymmetric scene (red square top-left, blue circle bottom-right) stored rotated 90° CCW with EXIF `Orientation = 6` (tag `0x0112`), JPEG quality 95. A correct decoder yields the upright 64×96 scene. |
| `display_p3.png` | 64×64 | Four mid-saturation quadrants with an embedded Display-P3 ICC profile. |

### Note on `portrait_exif6.jpg`

Upstream maps orientation 6 to a **270° `PIL.Image.rotate` with `expand=True`**
(`decode.py:32`, `decode.py:143-145`). PIL's `rotate` is counter-clockwise, so 270° CCW
is the 90° clockwise rotation EXIF orientation 6 calls for. The stored raster is
therefore the upright scene rotated 90° CCW. `decoded_portrait_exif6.bin` asserts the
red square lands top-left and the blue circle bottom-right.

### Note on `display_p3.png`

The embedded profile is **`DisplayP3-v4.icc` from
[saucecontrol/Compact-ICC-Profiles](https://github.com/saucecontrol/Compact-ICC-Profiles)**
(480 bytes, ICC v4.2, RGB→XYZ matrix-shaper, internal description `sP3`, CC0 licensed,
`sha256:cb51de38e482ee974c0c76b9689e16aad04bad16e226fed2f30c842d15ff3a3d`). No
Display-P3 profile ships in nixpkgs — `colord` provides AdobeRGB1998, Rec709, sRGB and
friends but not P3 — so it is fetched from that repository. No fallback profile was
needed. The profile itself is not vendored here; the PNG carries it inline, which is
all the fixture needs.

The quadrant colors are deliberately **mid-saturation, not primaries**. Fully saturated
P3 primaries convert to out-of-gamut sRGB values that clamp straight back to the same
bytes, making the ICC transform read as a no-op (measured: mean-abs 0.67 for
red/green/blue/orange primaries). The four chosen colors each move 10–15 per channel
with no channel clamping at 0 or 255:

| Quadrant | Source (tagged P3) | `decode_image` output (sRGB) |
| --- | --- | --- |
| top-left | `(204, 102, 51)` | `(219, 94, 31)` |
| top-right | `(96, 176, 128)` | `(60, 178, 124)` |
| bottom-left | `(90, 150, 200)` | `(67, 152, 205)` |
| bottom-right | `(210, 150, 60)` | `(221, 147, 30)` |

Overall mean-abs shift versus reading the same PNG with the profile ignored: **13.25**.
A port that skips ICC handling fails this fixture unambiguously.

## Expected outputs

### Decode

| Fixture | Shape (H, W, C) | dtype | What it pins |
| --- | --- | --- | --- |
| `decoded_portrait_exif6` | (96, 64, 3) | u8 | `decode_image` applies EXIF orientation 6 before returning. |
| `decoded_display_p3` | (64, 64, 3) | u8 | `decode_image` converts a non-sRGB ICC profile to sRGB via `ImageCms.profileToProfile`. |

### CRF re-compression (`preprocess(image, crf=33)`)

| Fixture | Source shape | Output shape | dtype |
| --- | --- | --- | --- |
| `crf33_gradient_96x64` | (64, 96, 3) | (64, 96, 3) | u8 |
| `crf33_photo_like_128x96` | (96, 128, 3) | (96, 128, 3) | u8 |
| `crf33_oddsize_97x63` | (63, 97, 3) | **(62, 96, 3)** | u8 |

### Resize (`resize_and_center_crop`)

Values stay in the **0..255 domain** — `normalize_images` is *not* applied, so these
isolate the geometry. Upstream returns `(1, C, 1, H, W)`; the fixtures are squeezed to
HWC. Input is the decoded source image cast to `float32` — the CRF step is deliberately
*not* applied, so these test resize alone.

| Fixture | Target (H×W) | Arithmetic |
| --- | --- | --- |
| `resized_gradient_96x64_to_64x64` | 64×64 | `scale = max(64/64, 64/96) = 1.0` → 64×96, `crop_left = (96-64)//2 = 16`. Pure center crop, no interpolation change. |
| `resized_photo_like_128x96_to_96x96` | 96×96 | `scale = max(96/96, 96/128) = 1.0` → 96×128, `crop_left = (128-96)//2 = 16`. Pure center crop. |
| `resized_oddsize_97x63_to_64x48` | 48×64 | `scale = max(48/63, 64/97) = 0.761905` → `ceil` → 48×74, `crop_left = (74-64)//2 = 5`. Exercises bilinear interpolation and the `math.ceil` guard at `resize.py:63-64`. |

## Measured CRF-33 degradation

PSNR and mean absolute difference between the **decoded original** and the upstream
round-tripped output. For `oddsize` the original is cropped to the output's even
dimensions before comparison. This calibrates how destructive libx264 at CRF 33 actually
is, so mold's openh264 stand-in can be judged against a comparable envelope rather than
an invented threshold.

| Fixture | PSNR (dB) | mean abs diff | max abs diff |
| --- | --- | --- | --- |
| `crf33_gradient_96x64` | 24.33 | 6.279 | 117 |
| `crf33_photo_like_128x96` | 25.14 | 10.107 | 159 |
| `crf33_oddsize_97x63` | 24.28 | 6.410 | 116 |

CRF 33 is genuinely lossy: ~24–25 dB PSNR, mean error of 6–10 levels per channel, and
worst-case single-sample errors past 100. A mold implementation landing in this envelope
is behaving like upstream; one landing at 40 dB is under-compressing and conditioning the
model on the wrong statistics.

## Reproducibility caveats for consumers

The three groups of fixtures have very different tolerance expectations:

1. **`resized_*` (f32)** — deterministic arithmetic (`torch.nn.functional.interpolate`,
   `mode="bilinear"`, `align_corners=False`). A correct port should match to float
   rounding; a tight tolerance (~1e-3 in the 0..255 domain) is reasonable.
2. **`decoded_*` (u8)** — depends on libjpeg-turbo (for the EXIF fixture) and lcms2 (for
   the ICC fixture). Expect near-exact, but allow ±1–2 per sample rather than demanding
   byte equality across decoder implementations.
3. **`crf33_*` (u8)** — **not a byte-exact target.** These bytes are specific to
   libx264 at `preset="veryfast"` (`decode.py:389`) with the RGB→`yuv420p` conversion
   at `decode.py:396`, and will shift with the x264/libswscale version. mold uses
   openh264, which will never reproduce them exactly. Use them for the statistical
   envelope in the table above (PSNR band, mean/max error), and for the *shape* contract
   — especially the odd-dimension flooring — not for equality.

## Upstream behaviour confirmed during capture

- **Odd dimensions are floored to even, inside `encode_single_frame`, not `preprocess`.**
  `decode.py:391-393` computes `shape // 2 * 2` and slices the array before setting the
  stream size, because H.264 `yuv420p` needs even dimensions. A 97×63 input therefore
  comes back 96×62 — the last row and column are **dropped, not resampled**. Confirmed:
  `(63, 97, 3) -> (62, 96, 3)`.
- **`crf == 0` is an exact identity.** `decode.py:425-426` returns the input array
  untouched — no encode, no `yuv420p` round trip. Confirmed.
- **Images with a dimension below 2 are returned untouched.** `decode.py:427-428`, the
  guard for inputs the codec cannot represent. Confirmed with a 1×8 array.
- **`crf=None` raises rather than defaulting.** `decode.py:420-424`. Upstream treats the
  CRF as a property of the model generation that must be resolved against the checkpoint
  (`ImageConditioner.resolve_crf` / `detect_params(...).default_image_crf`), so a port
  must not invent a code-level default either.
- **ICC survives EXIF rotation, on this Pillow.** `decode.py:147` reads
  `icc_profile` off the image *after* the orientation rotation at `decode.py:143-145`.
  `Image.rotate` returns a new image, so whether the profile is still attached is a
  Pillow implementation detail. On Pillow 12.2.0 it is preserved (verified directly:
  `icc_profile` present, 480 bytes, before and after `rotate(270, expand=True)`), and a
  JPEG carrying both orientation 6 and the P3 profile is still color-converted (mean-abs
  2.998 versus the rotated-but-unconverted pixels). Worth re-checking if the pinned
  Pillow ever moves — an older Pillow that drops `.info` here would silently skip the
  color conversion for rotated images only.
- **`resize_and_center_crop` never letterboxes.** `resize.py:60` takes `max` of the two
  scales, so the source always fills the target and the overflow is center-cropped away;
  `resize.py:63-64` uses `math.ceil` specifically so float rounding cannot produce a
  negative crop offset. The reflect-pad variant (`resize.py:106-143`) is a separate mode
  that the still-image conditioning path does not use.
- **Both `decode_image` conversion branches end at `outputMode="RGB"`.** An RGBA input
  is flattened to RGB *before* the ICC transform (`decode.py:149-152`), and a failed
  profile conversion falls back to a plain `convert("RGB")` with a warning rather than
  propagating (`decode.py:164-166`).

## Measured mold parity (openh264 constant-QP 33, rounded BT.601 conversion)

Recorded at implementation time (`ltx2/preprocess.rs`; regenerate with the
`qp_sweep_measurement` ignored test):

| Comparison | gradient | photo_like |
| --- | --- | --- |
| mold round-trip vs original (PSNR) | 29.21 dB | 24.71 dB |
| upstream libx264 CRF-33 vs original (PSNR) | 24.33 dB | 25.14 dB |
| mold round-trip vs upstream round-trip (PSNR) | 25.92 dB | 28.59 dB |
| decoded EXIF fixture vs golden (mean-abs) | — | 0.172 (max 4) |
| decoded Display-P3 fixture vs golden (mean-abs) | — | 0.083 (max 1) |

Photo-statistics content lands within 0.43 dB of upstream's degradation
envelope; the synthetic gradient stays ~4.9 dB lighter because libx264's
psychovisual RD deliberately spends extra distortion on smooth gradients and
openh264 has no equivalent knob. A QP sweep (32–38) does not close that gap
(the gradient's mold-vs-original PSNR moves only 29.98→28.44 dB across the
sweep), so the nominal QP 33 — which is also the best cross-similarity match —
is pinned rather than an offset invented from one synthetic fixture.
