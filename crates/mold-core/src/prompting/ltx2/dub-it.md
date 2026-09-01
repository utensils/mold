# LTX-2 Dub-It

## Prompt style

The reference video owns timing, visible speakers, cuts, and mouth motion.
Write replacement speech that fits the observed speaking windows, identify
each speaker consistently, and describe voice, accent, emotion, and delivery.
Do not change the visual action. Preserve the existing ambience unless asked
to replace it.

## Generation context

Duration is not yours to choose: the reference supplies frames and fps, and
both axes must be multiples of 64. Fit the new line inside the same speaking
window, because a longer line desynchronizes the mouth. The reference must
carry an audio track, since its speech is what the dub imitates.

## Examples

Input: make the presenter say the new tagline in a calmer tone

Output: The same speaker says in a calm, lower voice, "We render it locally
now." The room tone underneath is unchanged.

## CLI

Inspect `mold --help` and the selected host's capabilities for the current
Dub-It invocation; do not infer support from the family name alone. Verify the
returned clip before reporting success.

```bash
# Re-voice a clip of someone speaking; frames and fps come from the reference
mold run ltx-2.3-22b-distilled:fp8 "she says: the harbour freezes every winter" --ic-lora-control lipdub --video speaker.mp4 --width 704 --height 448
```

## Sources

- https://github.com/Lightricks/LTX-2 (`DubItPipeline`, `packages/ltx-pipelines/src/ltx_pipelines/dubit.py`)
