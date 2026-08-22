- **`/api/models` now advertises MiniMax H3's synchronized audio.** Every
  reviewed H3 identity reported `supports_audio: null`, so a client had to know
  from the family name that H3 always renders audio; the FL2VA rows only looked
  right because the private-runtime bridge set the flag separately. The row and
  its generation-profile recipes are now derived from the family's own
  `synchronized_audio` declaration — including the Ref2VA row and cold,
  not-yet-downloaded tiers — so a third-party client reads the capability
  instead of guessing it
  ([#841](https://github.com/utensils/mold/issues/841)).
