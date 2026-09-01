#!/usr/bin/env python3
"""Pre-frame a Hunyuan3D source image the way mold's own letterbox would.

Why this exists
---------------
mold and ComfyUI disagree about CONDITIONING, not about the network. Given a
raw background-removed cutout, mold applies Tencent's `ImageProcessorV2.recenter`
(crates/mold-inference/src/hunyuan3d/dino2.rs, `letterbox_square`): it takes the
bounding box of the non-transparent pixels and rescales it to fill 85 % of a
square frame. ComfyUI's `CLIPVisionEncode` with `crop: center` does no such
thing — it centre-crops the picture as given. The two networks therefore see
different pictures, and a comparison of their meshes measures the framing
policy rather than the port.

Feeding BOTH the same pre-framed picture removes that variable: mold's own
letterbox becomes (very nearly) the identity, because the subject already fills
85 % of an already-square frame, and ComfyUI's centre crop of a square is a
no-op. What is left to measure is the networks.

The port
--------
Mirrors `letterbox_square` exactly, including the details that are easy to get
wrong:

  * the bounding box maxima are EXCLUSIVE, matching upstream's
    `image[x_min:x_max, y_min:y_max]` slice, so the content extent is
    `max - min` and not `max - min + 1`;
  * `side = max(width, height)` and `desired = int(side * 0.85)` both truncate;
  * `scale = desired / max(content_h, content_w)` is applied to BOTH axes and
    the scaled extents truncate again;
  * placement is `(side - scaled) // 2` on each axis;
  * the subject is composited over WHITE, and numpy's
    `clip(0, 255).astype(uint8)` truncates rather than rounds.

One deliberate difference: `letterbox_square` returns RGB, having thrown the
alpha away. This writes RGBA, keeping the subject's alpha and leaving alpha 0
everywhere else. That matters because mold re-runs its letterbox on whatever it
is given: an opaque square would have the whole frame as its bounding box and
would be shrunk to 85 % a second time, whereas the retained alpha makes the
second pass a no-op. The RGB channels are already white-composited, so the RGB
a ComfyUI sees is exactly the square mold's letterbox produces.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

BORDER_RATIO = 0.15


class FramingError(RuntimeError):
    """The image has nothing to frame."""


def alpha_bounding_box(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """Return (row_min, row_max, column_min, column_max) of the non-zero mask.

    The maxima are inclusive pixel indices; callers take `max - min` to get the
    extent, which is what upstream's exclusive slice amounts to.
    """
    rows = np.flatnonzero(alpha.any(axis=1))
    columns = np.flatnonzero(alpha.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise FramingError("input image is empty: every pixel is fully transparent")
    return int(rows[0]), int(rows[-1]), int(columns[0]), int(columns[-1])


def frame_plan(width: int, height: int, alpha: np.ndarray) -> dict:
    """Everything the framing does, as numbers, so a test can assert on it."""
    row_min, row_max, column_min, column_max = alpha_bounding_box(alpha)
    content_height = row_max - row_min
    content_width = column_max - column_min
    if content_height == 0 or content_width == 0:
        raise FramingError(
            f"input image is empty: opaque content is {content_width}x{content_height}"
        )

    side = max(width, height)
    desired = int(side * (1.0 - BORDER_RATIO))
    scale = desired / max(content_height, content_width)
    scaled_height = int(content_height * scale)
    scaled_width = int(content_width * scale)
    if scaled_height <= 0 or scaled_width <= 0:
        raise FramingError(
            f"letterboxed content collapsed to {scaled_width}x{scaled_height}"
        )

    return {
        "row_min": row_min,
        "row_max": row_max,
        "column_min": column_min,
        "column_max": column_max,
        "content_height": content_height,
        "content_width": content_width,
        "side": side,
        "desired": desired,
        "scale": scale,
        "scaled_height": scaled_height,
        "scaled_width": scaled_width,
        "top": (side - scaled_height) // 2,
        "left": (side - scaled_width) // 2,
    }


def frame_image(image):
    """Return (framed RGBA image, plan) for a PIL image."""
    from PIL import Image

    has_alpha = "A" in image.getbands()
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width == 0 or height == 0:
        raise FramingError("source image is empty")

    array = np.asarray(rgba, dtype=np.uint8)
    if has_alpha:
        mask = array[..., 3] != 0
    else:
        # An image with no alpha channel is treated as fully opaque, which
        # makes the bounding box the whole frame and this a plain letterbox.
        mask = np.ones((height, width), dtype=bool)

    plan = frame_plan(width, height, mask)

    crop = rgba.crop(
        (
            plan["column_min"],
            plan["row_min"],
            plan["column_min"] + plan["content_width"],
            plan["row_min"] + plan["content_height"],
        )
    )
    scaled = crop.resize((plan["scaled_width"], plan["scaled_height"]), Image.BILINEAR)
    scaled_array = np.asarray(scaled, dtype=np.uint8)
    if not has_alpha:
        scaled_array = scaled_array.copy()
        scaled_array[..., 3] = 255

    # Composite over white exactly as the Rust does, truncating rather than
    # rounding, then keep the subject's alpha instead of discarding it.
    alpha = scaled_array[..., 3:4].astype(np.float32) / 255.0
    composited = scaled_array[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    composited = np.clip(composited, 0.0, 255.0).astype(np.uint8)

    side = plan["side"]
    canvas = np.zeros((side, side, 4), dtype=np.uint8)
    canvas[..., :3] = 255
    top = plan["top"]
    left = plan["left"]
    canvas[top : top + plan["scaled_height"], left : left + plan["scaled_width"], :3] = (
        composited
    )
    canvas[top : top + plan["scaled_height"], left : left + plan["scaled_width"], 3] = (
        scaled_array[..., 3]
    )
    return Image.fromarray(canvas, mode="RGBA"), plan


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Pre-frame a Hunyuan3D source image")
    parser.add_argument("--input", help="source image (a background-removed cutout)")
    parser.add_argument("--output", help="framed RGBA PNG to write")
    parser.add_argument(
        "--self-test", action="store_true", help="check the framing arithmetic and exit"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.input or not args.output:
        parser.error("--input and --output are required unless --self-test is given")

    from PIL import Image

    with Image.open(args.input) as image:
        image.load()
        framed, plan = frame_image(image)
    framed.save(args.output, format="PNG")
    print(
        f"framed {args.input} -> {args.output} "
        f"({plan['side']}x{plan['side']}, subject "
        f"{plan['scaled_width']}x{plan['scaled_height']} at "
        f"{plan['left']},{plan['top']})"
    )
    return 0


def self_test() -> int:
    from PIL import Image

    # A 400x300 frame whose subject is an opaque rectangle at rows 50..149 and
    # columns 100..199. The maxima are exclusive, so the content is 99x99, not
    # 100x100 — the single detail most likely to drift from the Rust.
    width, height = 400, 300
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[..., :3] = 30
    array[50:150, 100:200] = (200, 40, 60, 255)
    source = Image.fromarray(array, mode="RGBA")

    framed, plan = frame_image(source)
    assert plan["row_min"] == 50 and plan["row_max"] == 149, plan
    assert plan["column_min"] == 100 and plan["column_max"] == 199, plan
    assert plan["content_height"] == 99 and plan["content_width"] == 99, plan
    assert plan["side"] == 400, plan
    assert plan["desired"] == 340, plan
    assert abs(plan["scale"] - 340 / 99) < 1e-9, plan
    assert plan["scaled_height"] == 340 and plan["scaled_width"] == 340, plan
    assert plan["top"] == 30 and plan["left"] == 30, plan
    assert framed.size == (400, 400), framed.size
    assert framed.mode == "RGBA", framed.mode

    out = np.asarray(framed, dtype=np.uint8)
    assert (out[0, 0] == (255, 255, 255, 0)).all(), out[0, 0].tolist()
    assert (out[200, 200] == (200, 40, 60, 255)).all(), out[200, 200].tolist()
    assert out[..., 3].max() == 255
    rows = np.flatnonzero(out[..., 3].any(axis=1))
    columns = np.flatnonzero(out[..., 3].any(axis=0))
    assert (int(rows[0]), int(rows[-1])) == (30, 369), (rows[0], rows[-1])
    assert (int(columns[0]), int(columns[-1])) == (30, 369), (columns[0], columns[-1])

    # Re-framing an already-framed picture must be a near no-op: that is the
    # whole point of writing RGBA rather than an opaque square.
    reframed, second = frame_image(framed)
    assert second["side"] == 400, second
    assert abs(second["scaled_width"] - 340) <= 2, second
    assert abs(second["left"] - 30) <= 2, second
    assert reframed.size == (400, 400), reframed.size

    # An image with no alpha channel is a plain letterbox of the whole frame.
    opaque = Image.fromarray(array[..., :3], mode="RGB")
    _, opaque_plan = frame_image(opaque)
    assert opaque_plan["content_height"] == 299, opaque_plan
    assert opaque_plan["content_width"] == 399, opaque_plan
    assert opaque_plan["desired"] == 340, opaque_plan
    assert opaque_plan["scaled_width"] == 340, opaque_plan

    print("hunyuan3d-frame-source self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
