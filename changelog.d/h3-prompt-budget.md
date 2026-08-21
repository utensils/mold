- **Fixed: reviewed MiniMax H3 renders accept real prompts.** The reviewed
  compact/Turbo FL2VA envelope budgeted only about forty prompt tokens, so any
  prompt longer than a sentence was refused as differing from the reviewed
  envelope — after ninety seconds of artifact verification. The reviewed
  conditioner budget now carries roughly a thousand prompt tokens, an
  over-budget prompt is refused immediately and names the budget it has, and an
  envelope refusal names every axis it differs on instead of one unhelpful
  sentence ([#1245](https://github.com/utensils/mold/issues/1245)).
