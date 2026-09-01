# MiniMax H3 prompting

Manifest family: `minimax-h3`.

H3 generates synchronized stereo audio and video. Use the official three
sections in order: `integrated_multimodal_description`, `overall_soundscape`,
and `non_diegetic_music`. Put dialogue in the visual timeline, assign a stable
speaker ID such as `(S1)`, describe voice traits outside the dialogue tag, and
place only the language tag plus verbatim words inside `<d>...</d>`. Do not
repeat dialogue in the soundscape; use `N/A` when no score is wanted.

Generic `mold expand` does not produce H3 Context-IR. Check the selected model's
live runtime availability before promising speech or lip sync.

Read exactly one direct task leaf from `SKILL.md`: base-modes for FL2VA
identities, or Ref2VA for reference-conditioned identities.
