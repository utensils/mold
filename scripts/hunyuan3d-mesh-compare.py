#!/usr/bin/env python3
"""Compare two Hunyuan3D meshes (mold vs a ComfyUI reference) on Apple Metal.

Deliberately dependency-light: numpy, scipy and Pillow only, which is exactly
what the retained ComfyUI virtualenv already has. There is no trimesh and no
pygltflib, so this file carries its own minimal binary-glTF reader and writer.

Outputs a JSON report (bounds, counts, symmetric vertex Chamfer distance) and
a side-by-side orthographic point-splat PNG so a human can see both meshes.

Both meshes arrive in the same coordinate frame, which is what makes a
per-axis extent comparison meaningful at all: ComfyUI reaches `ShapeVAE.decode`
through the generic `comfy.sd.VAE.decode` wrapper, whose channels-last
`movedim(1,-1)` (`comfy/sd.py:1277`) sits on top of the VAE's own
`movedim(-2,-1)`, and `VoxelToMesh`'s `fliplr` then cancels both — so the
vertices of either implementation are the raw query `(x, y, z)`. An axis
mismatch here is a real regression, never a convention difference.

Pass criteria
-------------
* every bounding-box extent within 10 % of the reference's,
* triangle count within +/-35 % of the reference's,
* normalised symmetric vertex Chamfer at or below `--chamfer-max`, default
  0.02.

The Chamfer gate is an ABSOLUTE number, not a multiple of the seed-to-seed
noise floor, and the three measurements from the Apple Metal captures on
2026-09-01 are why:

    seed-to-seed noise floor (mold, same rung, one seed apart)   ~0.002-0.0025
    mold vs ComfyUI, both fed the same pre-framed picture             ~0.0103
    gate                                                                0.02

The middle number is roughly five times the floor and will not come down,
because the two references genuinely disagree about conditioning resolution:
for the mini tier mold encodes at 1022 px, following Tencent's
`hunyuan3d-dit-v2-mini-turbo/config.yaml` `image_processor.size`, while
ComfyUI always encodes at 518. A "twice the noise floor" gate would therefore
fail a correct port forever. 0.02 sits comfortably above the observed
framing-matched gap and far below the 0.030 seen when the two are fed
DIFFERENT pictures, which is the regression this is here to catch. Feed both
sides the same pre-framed image (`scripts/hunyuan3d-frame-source.py`) or the
gap reopens for reasons that have nothing to do with the port.

The noise floor is still computed and reported when `--floor-a`/`--floor-b`
are given. It is information about sampler variance, not a gate.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

EXTENT_TOLERANCE = 0.10
FACE_COUNT_TOLERANCE = 0.35
# See the module docstring: floor ~0.002, framing-matched gap ~0.010, gate 0.02.
DEFAULT_CHAMFER_MAX = 0.02


class GlbError(RuntimeError):
    """The bytes handed in are not a binary glTF this reader understands."""


def read_glb_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices (N,3) float32, faces (M,3) int64) from GLB bytes.

    Only what a Hunyuan3D mesh actually uses is supported: the first mesh's
    first primitive, a float32 VEC3 POSITION accessor and an unsigned integer
    SCALAR index accessor. byteOffset is honoured on both the accessor and the
    buffer view; byteStride is honoured for interleaved vertex data.
    """
    if len(data) < 12:
        raise GlbError("file is shorter than a GLB header")
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GlbError("missing glTF magic")
    if version != 2:
        raise GlbError(f"unsupported GLB version: {version}")
    if total_length > len(data):
        raise GlbError("GLB header length exceeds the file")

    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset + 8 <= total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        payload = data[offset : offset + chunk_length]
        if len(payload) < chunk_length:
            raise GlbError("truncated GLB chunk")
        if chunk_type == CHUNK_JSON and json_chunk is None:
            json_chunk = payload
        elif chunk_type == CHUNK_BIN and not bin_chunk:
            bin_chunk = payload
        offset += chunk_length
    if json_chunk is None:
        raise GlbError("GLB has no JSON chunk")

    gltf = json.loads(json_chunk.decode("utf-8"))
    meshes = gltf.get("meshes") or []
    if not meshes:
        raise GlbError("GLB has no meshes")
    primitives = meshes[0].get("primitives") or []
    if not primitives:
        raise GlbError("first mesh has no primitives")
    primitive = primitives[0]

    attributes = primitive.get("attributes") or {}
    if "POSITION" not in attributes:
        raise GlbError("first primitive has no POSITION attribute")

    def read_accessor(index: int) -> np.ndarray:
        accessor = gltf["accessors"][index]
        dtype = COMPONENT_DTYPES.get(accessor["componentType"])
        if dtype is None:
            raise GlbError(f"unsupported componentType {accessor['componentType']}")
        components = TYPE_COMPONENTS.get(accessor["type"])
        if components is None:
            raise GlbError(f"unsupported accessor type {accessor['type']}")
        count = int(accessor["count"])
        view_index = accessor.get("bufferView")
        if view_index is None:
            raise GlbError("sparse or bufferless accessors are not supported")
        view = gltf["bufferViews"][view_index]
        if view.get("buffer", 0) != 0:
            raise GlbError("only the embedded GLB buffer is supported")
        base = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
        item_size = np.dtype(dtype).itemsize * components
        stride = int(view.get("byteStride", 0)) or item_size
        if stride == item_size:
            flat = np.frombuffer(
                bin_chunk, dtype=dtype, count=count * components, offset=base
            )
            return flat.reshape(count, components)
        rows = np.empty((count, components), dtype=dtype)
        for i in range(count):
            start = base + i * stride
            rows[i] = np.frombuffer(bin_chunk, dtype=dtype, count=components, offset=start)
        return rows

    positions = read_accessor(attributes["POSITION"]).astype(np.float32, copy=False)
    if positions.shape[1] != 3:
        raise GlbError("POSITION accessor is not VEC3")

    index_accessor = primitive.get("indices")
    if index_accessor is None:
        count = positions.shape[0] - positions.shape[0] % 3
        faces = np.arange(count, dtype=np.int64).reshape(-1, 3)
    else:
        flat = read_accessor(index_accessor).astype(np.int64, copy=False).reshape(-1)
        faces = flat[: flat.size - flat.size % 3].reshape(-1, 3)
    return positions, faces


def read_glb(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        return read_glb_bytes(handle.read())


def write_glb_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Minimal GLB writer, used only by --self-test to exercise the reader."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)

    def pad4(buffer: bytes, filler: bytes = b"\x00") -> bytes:
        return buffer + filler * ((4 - len(buffer) % 4) % 4)

    vertex_bytes = pad4(vertices.tobytes())
    index_bytes = pad4(faces.tobytes())
    binary = vertex_bytes + index_bytes
    gltf = {
        "asset": {"version": "2.0", "generator": "hunyuan3d-mesh-compare self-test"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": vertices.nbytes, "target": 34962},
            {
                "buffer": 0,
                "byteOffset": len(vertex_bytes),
                "byteLength": faces.nbytes,
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5125,
                "count": int(faces.size),
                "type": "SCALAR",
            },
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_bytes = pad4(json.dumps(gltf).encode("utf-8"), b" ")
    header = struct.pack("<III", GLB_MAGIC, 2, 12 + 8 + len(json_bytes) + 8 + len(binary))
    return (
        header
        + struct.pack("<II", len(json_bytes), CHUNK_JSON)
        + json_bytes
        + struct.pack("<II", len(binary), CHUNK_BIN)
        + binary
    )


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict:
    if vertices.size == 0:
        return {
            "vertex_count": 0,
            "face_count": int(faces.shape[0]),
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [0.0, 0.0, 0.0],
            "extents": [0.0, 0.0, 0.0],
            "bbox_diagonal": 0.0,
        }
    lo = vertices.min(axis=0).astype(float)
    hi = vertices.max(axis=0).astype(float)
    extents = (hi - lo).astype(float)
    return {
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "extents": extents.tolist(),
        "bbox_diagonal": float(np.linalg.norm(extents)),
    }


def chamfer(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Symmetric mean nearest-neighbour vertex distance, normalised by scale."""
    from scipy.spatial import cKDTree

    if a.size == 0 or b.size == 0:
        return float("inf")
    forward = cKDTree(b).query(a, k=1)[0]
    backward = cKDTree(a).query(b, k=1)[0]
    raw = 0.5 * (float(forward.mean()) + float(backward.mean()))
    if scale <= 0.0:
        return float("inf") if raw > 0.0 else 0.0
    return raw / scale


def splat_view(vertices: np.ndarray, size: int, lo: np.ndarray, span: float) -> np.ndarray:
    """Orthographic point splat down -Z with a z-buffer, returned as uint8."""
    image = np.full((size, size), 255, dtype=np.uint8)
    if vertices.size == 0 or span <= 0.0:
        return image
    margin = max(2, size // 32)
    usable = size - 2 * margin
    xs = ((vertices[:, 0] - lo[0]) / span * usable + margin).astype(np.int64)
    # Image rows grow downward; flip Y so the mesh is not rendered upside down.
    ys = (size - margin - (vertices[:, 1] - lo[1]) / span * usable).astype(np.int64)
    zs = vertices[:, 2].astype(np.float64)

    z_lo = float(zs.min())
    z_hi = float(zs.max())
    z_span = max(z_hi - z_lo, 1e-9)
    shade = (40.0 + 180.0 * (1.0 - (zs - z_lo) / z_span)).astype(np.uint8)

    # A 3x3 splat per point, expanded once and resolved with a single sorted
    # scatter: numpy keeps the LAST write for a duplicated index, so sorting
    # far-to-near makes the nearest fragment the one that survives. That is a
    # z-buffer without a Python loop over half a million vertices.
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    all_y = np.concatenate([ys + dy for dy, _ in offsets])
    all_x = np.concatenate([xs + dx for _, dx in offsets])
    all_z = np.tile(zs, len(offsets))
    all_shade = np.tile(shade, len(offsets))

    inside = (all_y >= 0) & (all_y < size) & (all_x >= 0) & (all_x < size)
    all_y = all_y[inside]
    all_x = all_x[inside]
    all_z = all_z[inside]
    all_shade = all_shade[inside]
    if all_z.size == 0:
        return image

    order = np.argsort(-all_z, kind="stable")
    flat = all_y[order] * size + all_x[order]
    image.reshape(-1)[flat] = all_shade[order]
    return image


def render_side_by_side(
    left: tuple[np.ndarray, str], right: tuple[np.ndarray, str], path: str, size: int = 480
) -> None:
    from PIL import Image, ImageDraw

    stacked = [p for p in (left[0], right[0]) if p.size]
    if stacked:
        joined = np.concatenate(stacked, axis=0)
        lo = joined.min(axis=0).astype(float)
        hi = joined.max(axis=0).astype(float)
        span = float(max(hi - lo).item()) if np.any(hi - lo) else 1.0
    else:
        lo = np.zeros(3)
        span = 1.0

    label_height = 24
    canvas = Image.new("L", (size * 2 + 12, size + label_height), color=255)
    for column, (vertices, label) in enumerate((left, right)):
        tile = Image.fromarray(splat_view(vertices, size, lo, span), mode="L")
        canvas.paste(tile, (column * (size + 12), label_height))
        ImageDraw.Draw(canvas).text((column * (size + 12) + 6, 6), label, fill=0)
    canvas.convert("RGB").save(path, format="PNG")


def compare(mold_path: str, comfy_path: str) -> dict:
    mold_vertices, mold_faces = read_glb(mold_path)
    comfy_vertices, comfy_faces = read_glb(comfy_path)
    mold = mesh_stats(mold_vertices, mold_faces)
    comfy = mesh_stats(comfy_vertices, comfy_faces)

    extent_ratios = []
    for axis in range(3):
        reference = comfy["extents"][axis]
        if reference <= 0.0:
            extent_ratios.append(float("inf") if mold["extents"][axis] > 0.0 else 0.0)
        else:
            extent_ratios.append(abs(mold["extents"][axis] - reference) / reference)

    if comfy["face_count"] > 0:
        face_ratio = abs(mold["face_count"] - comfy["face_count"]) / comfy["face_count"]
    else:
        face_ratio = float("inf") if mold["face_count"] > 0 else 0.0

    return {
        "mold": dict(mold, path=mold_path),
        "comfy": dict(comfy, path=comfy_path),
        "extent_relative_differences": extent_ratios,
        "face_count_relative_difference": face_ratio,
        "chamfer_normalized": chamfer(mold_vertices, comfy_vertices, comfy["bbox_diagonal"]),
        "_vertices": (mold_vertices, comfy_vertices),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mold", help="mold-produced .glb")
    parser.add_argument("--comfy", help="ComfyUI reference .glb")
    parser.add_argument("--floor-a", help="noise-floor mesh A (mold, seed 1)")
    parser.add_argument("--floor-b", help="noise-floor mesh B (mold, seed 2)")
    parser.add_argument("--out", help="JSON report path")
    parser.add_argument("--png", help="side-by-side PNG path")
    parser.add_argument(
        "--chamfer-max",
        type=float,
        default=DEFAULT_CHAMFER_MAX,
        help="maximum normalised symmetric vertex Chamfer distance "
        f"(default {DEFAULT_CHAMFER_MAX})",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="exercise the reader/writer and exit"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    for required in ("mold", "comfy", "out"):
        if not getattr(args, required):
            parser.error(f"--{required} is required unless --self-test is given")

    result = compare(args.mold, args.comfy)
    mold_vertices, comfy_vertices = result.pop("_vertices")

    reasons = []
    for axis, ratio in enumerate(result["extent_relative_differences"]):
        if not ratio <= EXTENT_TOLERANCE:
            reasons.append(
                f"extent axis {axis} differs by {ratio:.3f}, tolerance {EXTENT_TOLERANCE}"
            )
    if not result["face_count_relative_difference"] <= FACE_COUNT_TOLERANCE:
        reasons.append(
            "face count differs by "
            f"{result['face_count_relative_difference']:.3f}, "
            f"tolerance {FACE_COUNT_TOLERANCE}"
        )

    if not result["chamfer_normalized"] <= args.chamfer_max:
        reasons.append(
            f"chamfer {result['chamfer_normalized']:.5f} exceeds "
            f"the {args.chamfer_max} ceiling"
        )

    # Reported, never gated. The floor says how much of the difference is the
    # sampler; the gate above says how much difference is acceptable at all.
    floor = None
    if args.floor_a and args.floor_b:
        floor_a_vertices, floor_a_faces = read_glb(args.floor_a)
        floor_b_vertices, floor_b_faces = read_glb(args.floor_b)
        floor_scale = mesh_stats(floor_b_vertices, floor_b_faces)["bbox_diagonal"]
        floor_value = chamfer(floor_a_vertices, floor_b_vertices, floor_scale)
        floor = {
            "floor_a": args.floor_a,
            "floor_b": args.floor_b,
            "chamfer_normalized": floor_value,
            "gated": False,
            "chamfer_over_floor": (
                result["chamfer_normalized"] / floor_value if floor_value > 0.0 else None
            ),
        }
        _ = floor_a_faces  # counts are not gated for the floor pair

    png_path = None
    if args.png:
        render_side_by_side(
            (mold_vertices, "mold"), (comfy_vertices, "comfyui"), args.png
        )
        png_path = args.png

    report = {
        "schema_version": "mold.hunyuan3d.mesh-compare.v1",
        "mold": result["mold"],
        "comfy": result["comfy"],
        "extent_relative_differences": result["extent_relative_differences"],
        "extent_tolerance": EXTENT_TOLERANCE,
        "face_count_relative_difference": result["face_count_relative_difference"],
        "face_count_tolerance": FACE_COUNT_TOLERANCE,
        "chamfer_normalized": result["chamfer_normalized"],
        "chamfer_max": args.chamfer_max,
        "noise_floor": floor,
        "png_path": png_path,
        "pass": not reasons,
        "reasons": reasons,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"pass": report["pass"], "reasons": reasons}))
    return 0 if report["pass"] else 1


def unit_cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def self_test() -> int:
    vertices, faces = unit_cube()
    round_tripped_vertices, round_tripped_faces = read_glb_bytes(
        write_glb_bytes(vertices, faces)
    )
    assert round_tripped_vertices.shape == vertices.shape, "vertex shape lost in round trip"
    assert np.allclose(round_tripped_vertices, vertices), "vertex values lost in round trip"
    assert round_tripped_faces.shape == faces.shape, "face shape lost in round trip"
    assert np.array_equal(round_tripped_faces, faces.astype(np.int64)), "face indices lost"

    stats = mesh_stats(round_tripped_vertices, round_tripped_faces)
    assert stats["vertex_count"] == 8, stats
    assert stats["face_count"] == 12, stats
    assert np.allclose(stats["extents"], [2.0, 2.0, 2.0]), stats

    identical = chamfer(round_tripped_vertices, vertices, stats["bbox_diagonal"])
    assert identical == 0.0, f"identical meshes must have zero chamfer, got {identical}"

    shifted = vertices + np.float32(0.5)
    moved = chamfer(shifted, vertices, stats["bbox_diagonal"])
    assert moved > 0.0, "a translated mesh must have a positive chamfer"

    # A second, differently sized mesh proves the reader is not returning the
    # first one's buffers by accident.
    big_vertices, big_faces = unit_cube()
    big_vertices = big_vertices * np.float32(3.0)
    big_read_vertices, big_read_faces = read_glb_bytes(
        write_glb_bytes(big_vertices, big_faces)
    )
    big_stats = mesh_stats(big_read_vertices, big_read_faces)
    assert np.allclose(big_stats["extents"], [6.0, 6.0, 6.0]), big_stats

    # The Chamfer gate is absolute, so it must be exercised end to end rather
    # than inferred: identical meshes pass, a mesh nudged well past the ceiling
    # fails, and supplying a noise-floor pair changes the report but never the
    # verdict.
    import contextlib
    import io
    import tempfile

    # `main` prints its own verdict per invocation; the self-test has one
    # verdict of its own, so swallow theirs. Assertion messages still surface,
    # because they travel out on stderr.
    with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()):
        def write(name: str, points: np.ndarray) -> str:
            path = f"{scratch}/{name}.glb"
            with open(path, "wb") as handle:
                handle.write(write_glb_bytes(points, faces))
            return path

        reference = write("reference", vertices)
        same = write("same", vertices)
        # 8 % of the bbox diagonal: inside the extent and face-count
        # tolerances, far outside a 0.02 Chamfer ceiling.
        nudged = write("nudged", vertices + np.float32(0.1 * stats["bbox_diagonal"] / 3.0))
        floor_b = write("floor-b", vertices + np.float32(1e-4))

        report_path = f"{scratch}/report.json"
        assert main(["--mold", same, "--comfy", reference, "--out", report_path]) == 0
        with open(report_path, encoding="utf-8") as handle:
            passing = json.load(handle)
        assert passing["pass"] is True, passing
        assert passing["chamfer_max"] == DEFAULT_CHAMFER_MAX, passing
        assert passing["noise_floor"] is None, passing

        assert main(["--mold", nudged, "--comfy", reference, "--out", report_path]) == 1
        with open(report_path, encoding="utf-8") as handle:
            failing = json.load(handle)
        assert failing["pass"] is False, failing
        assert any("chamfer" in reason for reason in failing["reasons"]), failing

        # A generous ceiling makes the same pair pass: the gate is the number,
        # not the meshes.
        assert (
            main(
                [
                    "--mold", nudged, "--comfy", reference,
                    "--chamfer-max", "10", "--out", report_path,
                ]
            )
            == 0
        )

        # A noise floor is reported and explicitly not gated: a floor far
        # smaller than the measured Chamfer must still leave the verdict at
        # pass, which is exactly what the old 2x-floor rule got wrong.
        assert (
            main(
                [
                    "--mold", same, "--comfy", reference,
                    "--floor-a", same, "--floor-b", floor_b,
                    "--out", report_path,
                ]
            )
            == 0
        )
        with open(report_path, encoding="utf-8") as handle:
            floored = json.load(handle)
        assert floored["pass"] is True, floored
        assert floored["noise_floor"]["gated"] is False, floored
        assert floored["noise_floor"]["chamfer_normalized"] > 0.0, floored

    print("hunyuan3d-mesh-compare self-test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
