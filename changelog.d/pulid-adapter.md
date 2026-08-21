- **PuLID cross-attention runs on every FLUX transformer variant.** The
  twenty-module identity adapter is ported to candle and wired into all four
  FLUX transformer paths — dense BF16, quantized GGUF, mold's bypass-LoRA GGUF,
  and block-offloaded — through one shared injection policy, so a face-identity
  render behaves the same whichever route a machine picks
  ([#1221](https://github.com/utensils/mold/issues/1221)). An effective
  `id_weight` of 0, and every step before `id_start_step`, run no identity
  arithmetic at all: they take the exact transformer call an ordinary request
  takes, so the render is bit-identical to one that never asked for a face. The
  adapter stays fully resident rather than streaming with the offloaded blocks,
  and is dropped as soon as a request stops conditioning on a face. Extracting
  the identity from a portrait lands separately; until then a face-conditioned
  request is refused with a message that says so instead of quietly rendering a
  stranger.
