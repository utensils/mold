# MiniMax H3 prompting

Manifest family: `minimax-h3`.

## Prompt style

Write the three core fields in order: `integrated_multimodal_description`,
`overall_soundscape`, `non_diegetic_music`. Every detail must be visible or
audible. Open `[Shot 1]` with the visual style and initial composition, then
subjects, scene, and actions. Prefer concrete detail over abstract words.
Write in English, preserving the original language of dialogue, lyrics, and
visible text. Stay within {{word_limit}} words.

## Syntax

`[Shot 1]` takes no timestamp. Later shots open `[Shot n] At MM:SS.mmm,` with
a strictly increasing cut time inside the duration. Write camera motion as
natural English in the shot: type, amplitude, speed. The types are Zoom
In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake
Slightly/Strongly, POV, and Roll Clockwise/Counterclockwise. Omit amplitude
and speed when medium and normal. Give each vocal source a stable id such as
`(S1)`, with its identifying phrase and delivery outside the tag. Inside
`<d>[English] ...</d>` put only the language tag and the verbatim words. For
voiceover use the exact phrase "says in an off-screen voiceover", then state
that the character's lips remain closed. Use `<scenetrans>` where a line
crosses a cut and `<cutoff>` where speech is truncated by the ending. Quote
visible on-screen text verbatim. Reference labels `<Picture n>`, `<Video n>`,
and `<Audio n>` keep one meaning across every section.

## Generation context

Match the described duration to the requested four to fifteen seconds.
`overall_soundscape` is one to four sentences of ambience, action sounds, and
non-verbal sounds, never repeating dialogue or music. `non_diegetic_music`
covers instrumentation, tempo, and dynamics; `N/A` when no score is wanted.

## Examples

Input: a baker opens the shutters before sunrise and says one line

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening a street bakery's shutters before sunrise. The camera pushes in at slow speed as the baker with a calm, raspy voice (S1) says: <d>[English] First batch of the morning.</d>

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo.
```

## Pitfalls

Avoid plot summaries, unresolved labels, and timing that misses the duration.
Expansion produces this grammar when given the H3 route. Check live runtime
availability before promising speech; mold's tokenizer does not yet register
the official dialogue tokens (issue #1430).

## CLI

Read exactly one direct task leaf from `SKILL.md`: base modes for FL2VA
identities, or Ref2VA for reference-conditioned identities.

```bash
# Feed a written Context-IR prompt from a file
mold run minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p "$(cat h3-prompt.txt)" --first-frame presenter.png --duration 5 --seed 83009
```

## Sources

- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md
- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt
- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt
- https://huggingface.co/MiniMaxAI/MiniMax-H3
