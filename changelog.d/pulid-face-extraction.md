- **PuLID face detection and identity embedding, in pure Rust.** Builds under
  the `pulid` feature now detect a face with InsightFace's SCRFD, align it to
  the ArcFace and FFHQ templates, and produce the 512-d `glintr100` identity
  embedding plus the 512×512 crop PuLID's vision tower conditions on — all
  through candle, with no ONNX runtime, no Python, and no new native
  dependency. Several faces in one photo picks the largest and says so; a
  photo with no face is refused with a clear error instead of a meaningless
  embedding. Measured against the upstream Python pipeline on four
  public-domain portraits: landmarks within 0.24 px and ArcFace cosine at or
  above 0.9993
  ([#1222](https://github.com/utensils/mold/issues/1222)).
