import { defineStore } from "pinia";

export const VIDEO_MUTED_STORAGE_KEY = "mold.video-playback.muted.v1";

function savedMuted(): boolean {
  try {
    return localStorage.getItem(VIDEO_MUTED_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

/** One sound preference for Create and Library, independent of generation audio. */
export const useVideoPlaybackStore = defineStore("videoPlayback", {
  state: () => ({ muted: savedMuted(), volume: 1 }),
  actions: {
    setMuted(muted: boolean) {
      this.muted = muted;
      try {
        localStorage.setItem(VIDEO_MUTED_STORAGE_KEY, String(muted));
      } catch {
        // Playback still works when storage is unavailable.
      }
    },
    toggleSound() {
      if (this.muted && this.volume === 0) this.volume = 1;
      this.setMuted(!this.muted);
    },
    syncFromPlayer(event: Event) {
      const video = event.currentTarget as HTMLVideoElement;
      this.volume = video.volume;
      this.setMuted(video.muted || video.volume === 0);
    },
  },
});
