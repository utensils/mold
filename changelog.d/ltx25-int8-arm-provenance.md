- **LTX-2 INT8 ConvRot prints record their execution arm.** Renders from INT8
  ConvRot checkpoints carry an additive `int8_arm` on `VideoData` and saved
  `OutputMetadata` (`native-w8a8`, `dequant-cuda`, `dequant-metal`,
  `dequant-host`), mirroring `attention_path`, so provenance shows which
  quantized arm produced the pixels.
