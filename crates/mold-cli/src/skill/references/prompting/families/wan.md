# Wan prompting

Manifest family: `wan`.

Wan 2.1/2.2 A14B and TI2V-5B generate silent video. Do not request native
dialogue or claim synchronized audio. Wan S2V is a separate speech-to-video
model and is not interchangeable with these checkpoints.

Select exactly one direct task leaf from `SKILL.md`: text-to-video when the
model identity is T2V, or image-conditioned for I2V/TI2V and requests with a
source frame.

Official guidance: [Wan 2.2](https://github.com/Wan-Video/Wan2.2).
