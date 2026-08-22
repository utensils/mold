- **PuLID face extraction runs resident, and the whole extraction is finally
  measured.** SCRFD and `glintr100` are now ordinary `candle` modules built once
  from the same SHA-pinned ONNX files, instead of re-materializing 278 MB of
  initializers on every conditioned request. `pulid_face_probe bench` gained
  `--full`, `--compare`, and `--regress-against`, and reports the EVA02-CLIP
  tower and the IDFormer — which had no number anywhere in the repository even
  though they run on every identity request
  ([#1227](https://github.com/utensils/mold/issues/1227)).
