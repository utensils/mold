<script setup lang="ts">
import { useVideoPlaybackStore } from "../../stores/videoPlayback";

const playback = useVideoPlaybackStore();
</script>

<template>
  <button
    type="button"
    class="ms-toolbar-button shrink-0"
    data-test="video-sound-toggle"
    aria-label="Mute video playback"
    :aria-pressed="playback.muted"
    :title="playback.muted ? 'Unmute videos' : 'Mute videos'"
    @click.stop="playback.toggleSound()"
  >
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.7"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M11 5 6 9H3v6h3l5 4V5Z" />
      <path v-if="playback.muted" d="m16 9 6 6m0-6-6 6" />
      <path v-else d="M15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14" />
    </svg>
    {{ playback.muted ? "Sound off" : "Sound on" }}
  </button>
</template>
