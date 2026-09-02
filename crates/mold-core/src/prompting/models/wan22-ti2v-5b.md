# Wan 2.2 TI2V-5B

## Generation context

The family's 720p path: 1280x704 at 24 fps, 4k+1 frames, both dimensions a
multiple of 32. One checkpoint serves text-to-video and image-conditioned work,
so a chain seam continues on it and a long sequence keeps one motion.

## Pitfalls

At 24 fps a 121-frame clip runs five seconds, so pace the action faster than on
the 16 fps tiers. Keep a `:turbo` prompt to one simple idea.

## CLI

```bash
mold run wan22-ti2v-5b "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24
mold run wan22-ti2v-5b:q8 "a paper boat drifting down a rain gutter" --frames 100 --clip-frames 49
mold run wan22-ti2v-5b:turbo "waves on a black sand beach" --width 1280 --height 704
mold run wan22-ti2v-5b:dmd "waves on a black sand beach" --width 1280 --height 704 --frames 121 --fps 24
mold run wan22-ti2v-5b:q8 "the balloon lifts off" --image balloon.png --frames 49
```

## Sources

- https://github.com/Wan-Video/Wan2.2
- https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
