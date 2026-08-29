- **Fix four bugs in the LTX-2.5 CUDA qualification harness.** The `.sha256-verified`
  marker read now takes only the first line before stripping whitespace, so a
  marker file with trailing content no longer corrupts the comparison. `--seal`
  no longer passes the full per-row result array as a single `jq --argjson`
  command-line argument — at real matrix sizes this exceeded Linux's
  `MAX_ARG_STRLEN` (128 KiB per argument, independent of `ulimit -s`); it now
  reads the payload via `--slurpfile` from a temp file. The audio sample-rate
  check is now per-row (`expect.audio_sample_rate`, defaulting to 48000)
  instead of hardcoding LTX-2.5's 48 kHz for every checkpoint — LTX-2 19B's
  audio is genuinely 24 kHz. The text-to-audio (`t2a`) row's expectation no
  longer assumes video-shaped provenance/metadata it never produces.
