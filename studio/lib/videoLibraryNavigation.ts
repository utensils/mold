import type { InjectionKey } from "vue";
import type { VideoJobOwner } from "../api/videoJobs";
export const VIDEO_LIBRARY_NAVIGATION: InjectionKey<
  (value: { owner: VideoJobOwner; filename: string }) => void
> = Symbol("video-library-navigation");
