### Added

- LTX-2 prints rendered from INT8 ConvRot checkpoints record which execution arm produced them: additive `int8_arm` on `VideoData` and saved `OutputMetadata` (`native-w8a8`, `dequant-cuda`, `dequant-metal`, `dequant-host`), mirroring `attention_path`.
