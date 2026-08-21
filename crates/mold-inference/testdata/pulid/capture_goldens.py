#!/usr/bin/env python3
"""Capture the #1222 parity goldens from the pinned InsightFace models.

Provenance only: this is committed so the fixtures can be regenerated and
audited. It is NOT run by the test suite, and mold ships no Python.

    python3 crates/mold-inference/testdata/pulid/capture_goldens.py \\
        --assets /path/to/antelopev2 \\
        --faces crates/mold-inference/testdata/pulid/faces

Requires a scratch venv with `insightface onnxruntime opencv-python-headless
numpy` (see `docs/architecture/pulid-face-extraction.md`). The antelopev2 ONNX
files are InsightFace pretrained models, licensed for non-commercial research
use only; they are never committed and never redistributed by mold.

For each face it records, per `<name>.golden.json`:

* `landmarks`      SCRFD's five keypoints in source pixels
* `bbox` / `score` the largest detection, as PuLID selects it
* `embedding`      the RAW 512-d `glintr100` output, exactly the value
                   `PuLID/pulid/pipeline_flux.py:130` conditions on (NOT
                   L2-normalized -- see `identity/arcface.rs`)
* `m112`           `face_align.estimate_norm` (skimage SimilarityTransform)
* `m512`           `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)` against
                   facexlib's FFHQ 512 template
* `m512_skimage`   the same fit done with skimage's SimilarityTransform, so the
                   LMEDS-vs-least-squares deviation mold takes is measured
                   rather than assumed

and writes `<name>.arcface112.png` / `<name>.eva512.png` warp goldens.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils import face_align
from skimage import transform as skimage_transform

# `facexlib/utils/face_restoration_helper.py:73-74`, at face_size=512.
FACEXLIB_FFHQ_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)
# `face_restoration_helper.py:258-259` -- BGR, on a cv2-decoded image.
FACEXLIB_BORDER_BGR = (135, 133, 132)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="directory holding the antelopev2 ONNX files")
    parser.add_argument("--faces", default="crates/mold-inference/testdata/pulid/faces")
    args = parser.parse_args()

    detector = SCRFD(os.path.join(args.assets, "scrfd_10g_bnkps.onnx"))
    detector.prepare(-1, det_size=(640, 640))
    recognizer = ArcFaceONNX(os.path.join(args.assets, "glintr100.onnx"))
    recognizer.prepare(-1)
    print("onnxruntime", onnxruntime.__version__, "| opencv", cv2.__version__)

    # The face list comes from `sources.json`, never from a directory glob.
    #
    # A glob is not idempotent here: this script WRITES `.arcface112.png` and
    # `.eva512.png` beside its inputs, so the second run would treat its own
    # output as a face to capture — producing goldens of goldens. It would also
    # pick up `.exif6.jpg`, the orientation twin that deliberately has no
    # golden of its own (it is checked against the upright fixture's).
    #
    # `sources.json` is written by `fetch_faces.py` and names exactly the
    # downloaded source photographs, so rerunning this script any number of
    # times converges on the same fixture set.
    sources_path = os.path.join(args.faces, "sources.json")
    if not os.path.exists(sources_path):
        print(f"missing {sources_path}; run fetch_faces.py first", file=sys.stderr)
        return 1
    with open(sources_path) as handle:
        names = sorted(entry["file"] for entry in json.load(handle))
    if not names:
        print("sources.json lists no fixture faces", file=sys.stderr)
        return 1
    missing = [n for n in names if not os.path.exists(os.path.join(args.faces, n))]
    if missing:
        print(f"sources.json names absent files: {missing}", file=sys.stderr)
        return 1

    for name in names:
        path = os.path.join(args.faces, name)
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"could not read {path}", file=sys.stderr)
            return 1
        bboxes, kpss = detector.detect(image_bgr, input_size=(640, 640))
        if bboxes.shape[0] == 0:
            print(f"NO FACE in {name}", file=sys.stderr)
            return 1
        # `PuLID/pulid/pipeline_flux.py:127-129`: the largest bbox wins.
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best = int(np.argmax(areas))
        landmarks = kpss[best].astype(np.float64)

        crop112 = face_align.norm_crop(image_bgr, landmark=kpss[best], image_size=112)
        embedding = recognizer.get_feat(crop112).flatten().astype(np.float64)

        m112 = face_align.estimate_norm(kpss[best], 112).astype(np.float64)
        m512 = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32), FACEXLIB_FFHQ_512, method=cv2.LMEDS
        )[0].astype(np.float64)
        tform = skimage_transform.SimilarityTransform()
        tform.estimate(landmarks.astype(np.float64), FACEXLIB_FFHQ_512.astype(np.float64))
        m512_skimage = tform.params[0:2, :].astype(np.float64)
        crop512 = cv2.warpAffine(
            image_bgr,
            m512,
            (512, 512),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=FACEXLIB_BORDER_BGR,
        )

        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(args.faces, f"{stem}.arcface112.png"), crop112)
        cv2.imwrite(os.path.join(args.faces, f"{stem}.eva512.png"), crop512)
        golden = {
            "image": name,
            "detector": "scrfd_10g_bnkps.onnx",
            "recognizer": "glintr100.onnx",
            "det_size": [640, 640],
            "faces_detected": int(bboxes.shape[0]),
            "bbox": bboxes[best][:4].astype(float).tolist(),
            "score": float(bboxes[best][4]),
            "landmarks": landmarks.tolist(),
            "m112": m112.tolist(),
            "m512": m512.tolist(),
            "m512_skimage": m512_skimage.tolist(),
            "embedding": embedding.tolist(),
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
        with open(os.path.join(args.faces, f"{stem}.golden.json"), "w") as handle:
            json.dump(golden, handle, indent=1)
            handle.write("\n")
        delta = float(np.abs(m512 - m512_skimage).max())
        print(
            f"{name}: {bboxes.shape[0]} face(s), score {golden['score']:.4f}, "
            f"|embedding| {golden['embedding_norm']:.3f}, "
            f"max|LMEDS - skimage| on m512 = {delta:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
