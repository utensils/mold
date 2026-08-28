- **Opt-in LTX-2 video-only rendering.** `video_only: true` /
  `mold run --video-only` / the Advanced "Video only" toggle on web and
  desktop skip the audio-video transformer's audio branch entirely, the way
  upstream's video-only configurator omits it. Output-changing and never a
  default; refused beside `enable_audio=true`, conditioning audio, the
  text-to-audio pipeline, and `extend_video`. The debug-only
  `MOLD_LTX_DEBUG_DISABLE_AUDIO_BRANCH` environment variable is removed —
  the request field is the one switch
  ([#1037](https://github.com/utensils/mold/issues/1037)).
