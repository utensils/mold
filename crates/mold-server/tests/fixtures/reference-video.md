# Fresh-video normalization fixtures

Generated with FFmpeg from lavfi solid color/testsrc2 and synthetic silence,
without model weights, third-party content, network inputs, or generation jobs.
Base64 keeps small binary inputs visible to source packaging and review.

- `reference-video.mp4.b64`: 32×32, 48 H.264 frames, 2 seconds, no audio.
- `reference-hevc.mov.b64`: 32×48 HEVC MOV, 48 frames, 2 seconds.
- `reference-vp9.webm.b64`: 32×48 VP9 WebM, 48 frames, 2 seconds.
- `reference-av1.mkv.b64`: 32×48 AV1 Matroska, 48 frames, 2 seconds.
- `reference-prores.mov.b64`: 32×48 ProRes MOV, 48 frames, 2 seconds.
- `reference-vfr.mp4.b64`: 32×48 H.264 MP4; alternate frames removed in the
  first second, all frames retained in the second; original timestamps kept.
- `reference-audio-overhang.mp4.b64`: 2-second blue H.264 video and 3 seconds
  of stereo AAC silence. Audio must remain distinct from the visual duration.
- `reference-rotation.mp4.b64`: the overhang fixture's video stream, without
  audio, with a 90-degree display matrix; canonical pixels become 48×32.

HEVC: libx265 with `pools=none:frame-threads=1`; VP9: libvpx-vp9; AV1:
libaom-av1; ProRes: prores_ks profile 0; H.264: libx264, yuv420p. All inputs
use one encoder thread. The exact regeneration commands and original binary
copies are retained in the task's external UAT evidence while validation runs.

- `reference-rational-cfr.mp4.b64`: 32×32 blue H.264/yuv420p, 48 frames at
  24000/1001 fps. Generated with `ffmpeg -f lavfi -i
  color=c=blue:s=32x32:r=24000/1001 -frames:v 48 -an -c:v libx264 -pix_fmt
  yuv420p -threads 1 rational-cfr.mp4`; exact bytes/rational rate must survive.
