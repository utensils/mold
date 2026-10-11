- **Shorter, clearer model names everywhere, from one place.** Every built-in
  model now has a short title, a variant (precision or speed tier) and a
  one-line reason to pick it, published on `/api/models` and generated into
  the web, desktop, phone and native Swift clients from one Rust authority.
  MiniMax H3 reads as **MiniMax H3 Frames** / **MiniMax H3 References** instead
  of FL2VA / Ref2VA, and family labels agree across the CLI, Discord and every
  app. Civitai and Hugging Face models keep their own titles.
