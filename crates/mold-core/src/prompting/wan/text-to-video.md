# Wan text-to-video

## Prompt style

Use for T2V identities. Open with the shot size, then the subject and setting,
then the action in two or three beats, then background motion and lighting.
Close with the one camera move. Keep to about {{word_limit}} words.

## Generation context

One continuous shot per clip, no cuts. A T2V checkpoint cannot carry a chain
seam, so clips in a longer sequence are independent shots. Ask for an
image-conditioned checkpoint when motion must continue across clips.

## Examples

Input: a fox in the snow, tracking shot

Output: Medium wide shot at sunrise, side lighting. A red fox trots through
fresh snow in a pine forest; breath drifts back and powder lifts from each paw.
The camera tracks alongside in one continuous move.

## Pitfalls

Do not name two camera moves or a second unrelated action.

## CLI

```bash
mold run wan22-t2v-a14b:q4 \
  "Medium-wide wildlife shot at sunrise. A red fox trots steadily through fresh snow in a quiet pine forest; visible breath drifts back and powder lifts from each paw. The camera tracks alongside at eye level in one continuous move. Soft side light, realistic fur, stable background, no text." \
  --width 832 --height 480 --frames 49 --fps 16 --seed 83005
mold run wan21-t2v-1.3b "a red fox trotting through snow" --frames 81 --fps 16
```

## Sources

- https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py
- https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py
