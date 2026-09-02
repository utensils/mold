- **MiniMax H3 dialogue tags now reach the model as real tokens.** H3's official
  `<d>…</d>` dialogue delimiters, along with `<|cutoff|>`, `<|lyrics_start|>`,
  `<|lyrics_end|>`, `<|caption_start|>` and `<|caption_end|>`, were tokenized as
  ordinary text, so byte-level BPE merged each tag into the words around it and
  no token marked where speech began. mold now registers them at the ids the
  model was released with, matching the tokenizer the official pipelines build
  ([#1430](https://github.com/utensils/mold/issues/1430)).
