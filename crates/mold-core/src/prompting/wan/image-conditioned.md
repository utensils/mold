# Wan image-conditioned video

## Prompt style

Use for I2V and TI2V identities and any source frame. Stay under {{word_limit}}
words and write dynamics only. Drop description already visible in the frame,
which owns appearance. Supplement a bare action with the visible subject, so
"dancing" becomes "the girl is dancing". Keep and emphasize camera phrases such
as "the camera pans up".

## Generation context

Name what must stay stable. For a first and last frame pair, describe a
physically continuous path between them and the change that happens: walking
into, appearing, turning into, camera left, camera right. Check the model row
for its supported source roles.

## Examples

Input: dancing

Output: The girl is dancing, turning slowly on the spot as her skirt lifts. The
camera pushes forward.

## Pitfalls

Do not redescribe or contradict the frame.

## CLI

```bash
mold run wan22-i2v-a14b:q5 "the balloon lifts off" --image balloon.png
mold run wan22-i2v-a14b:q5 "the sapling grows into an oak" --image sapling.png --last-image oak.png
mold run wan22-ti2v-5b:q8 "the paper boat drifts on down the gutter" --image boat.png --frames 100 --clip-frames 49
```

## Sources

- https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/system_prompt.py
- https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/prompt_extend.py
