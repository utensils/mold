- **MiniMax H3 dialogue tags now reach the model as real tokens.** H3's official
  `<d>…</d>` dialogue delimiters, along with `<|cutoff|>`, `<|lyrics_start|>`,
  `<|lyrics_end|>`, `<|caption_start|>` and `<|caption_end|>`, were being
  tokenized as ordinary text, so nothing marked where speech began and lip-sync
  and dialogue timing suffered. mold now registers them at the ids the model was
  released with
  ([#1430](https://github.com/utensils/mold/issues/1430)).
