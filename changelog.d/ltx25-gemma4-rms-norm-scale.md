- **LTX-2.5 renders follow the prompt again.** The Gemma 4 prompt encoder read
  its RMSNorm weights as Gemma 3-style offsets (`1 + w`) when the LTX-2.5
  checkpoint stores them as absolute scales, so every normalization left the
  channels the checkpoint suppresses at full gain. Conditioning collapsed onto
  a per-seed attractor and video came out well-formed but unrelated to the
  prompt; it now matches the reference encoder to BF16 rounding.
