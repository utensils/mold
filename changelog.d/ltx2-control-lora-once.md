- **A control render records and merges the caller's LoRA once.** `mold run --lora`
  on an LTX-2 model fills both the legacy `lora` field and the `loras` stack with
  the same adapter, and the three places that prepend a built-in `--ic-lora-control`
  adapter concatenated the two wells — so the print's provenance listed the
  caller's adapter twice and the engine merged it twice, at double its requested
  scale. The two fields are alternatives everywhere else (`loras` wins, `lora` is
  the fallback), and now they are here too; where they disagreed, the stack wins
  instead of resurrecting a singular adapter no engine would have applied.
