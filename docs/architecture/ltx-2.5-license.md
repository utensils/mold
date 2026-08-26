# LTX-2.5 license boundary

Reviewed on 2026-08-25 against the official
[LTX-2.x Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md)
(license date 2026-08-11) and the official
[LTX-2.5 model card](https://huggingface.co/Lightricks/LTX-2.5).

Mold's source code remains MIT licensed. LTX-2.5 model weights are separate
third-party assets governed by the LTX-2.x Community License Agreement and its
acceptable-use restrictions. Mold does not redistribute those weights: `mold
pull` downloads the pinned files from Lightricks' Hugging Face repository, whose
gated access remains authoritative.

The reviewed agreement requires an entity with at least USD 10 million in
annual revenue to obtain a paid commercial-use license for uses outside the
agreement's non-commercial exception. It also imposes notice, license-copy,
attribution, and use-restriction obligations when LTX-2.5 or a derivative is
redistributed. Users are responsible for determining whether their use and any
redistribution comply with the current upstream terms.

Release handling:

- keep every LTX-2.5 manifest file marked `gated` and sourced from the official
  `Lightricks/LTX-2.5` repository;
- link the current upstream terms from user-facing model documentation and keep
  model assets distinct from Mold's MIT license;
- re-review this decision if Lightricks changes the license, model repository,
  distribution path, or USD 10 million threshold;
- do not bundle LTX-2.5 weights or derivatives in Mold release artifacts without
  a separate distribution review that satisfies the agreement's pass-through
  obligations.

This record documents the repository's shipping boundary; it is not legal
advice or a substitute for reviewing the current upstream agreement.
