# LTX-2 prompting

Manifest family: `ltx2`.

For ordinary audio-video generation, write one flowing paragraph under 200
words with 4-8 concrete present-tense sentences: shot scale, subject,
chronological action, one camera move, coherent lighting, ambience, and sound.
For dialogue, keep one visible speaker in one continuous take, describe voice
and delivery, and quote the complete short line. Keep the mouth unobstructed.

```bash
mold run ltx-2.5-22b-distilled:q6 \
  "A cinematic close-up shows a woman in a red raincoat beneath a glass awning during rain at night. The locked camera holds as she says with delighted, clear delivery, 'This was made locally with Mold, including the sound.' Rain taps on glass beneath her clean voice; no music or on-screen text." \
  --width 768 --height 512 --frames 121 --clip-frames 121 --fps 24 --audio --seed 83007
```

Keep lip sync inside one clip. Respect the selected checkpoint's advertised
alignment and recipe; distilled LTX-2.5 guidance is fixed at 1.0.

For specialized audio workflows, also read the directly linked Dub-It or
text-to-audio task leaf in `SKILL.md`.
