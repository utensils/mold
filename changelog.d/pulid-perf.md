- **PuLID face extraction now runs as resident candle modules, and the whole
  extraction is measured for the first time.** SCRFD and `glintr100` are built
  once from the same SHA-pinned ONNX files instead of re-materializing 278 MB of
  initializers on every conditioned request; the port is parity-exact (SCRFD
  bit-identical to the evaluator it replaces, ArcFace cosine 1.0) and is what
  makes a future device path possible at all. `pulid_face_probe bench` gained
  `--full`, `--compare`, and `--regress-against`, which together show that the
  re-materialization was worth ~4% and that the EVA02-CLIP vision tower — which
  had no number anywhere in the repository — is 79% of a real extraction
  ([#1227](https://github.com/utensils/mold/issues/1227)).
