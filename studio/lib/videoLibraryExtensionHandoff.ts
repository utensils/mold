import { videoSettingsForReuse } from "./videoSettingsReuse";
import type { VideoDurationAuthoring } from "./videoDurationAuthoring";
import { shallowRef, ref, type InjectionKey } from "vue";
import type { VideoJobOwner } from "../api/videoJobs";
import type { VideoJobSource } from "./api/videoJobTypes";
import type { RetainedSourceMediaInventory } from "../api/gallerySourceMedia";
/** Private in-memory navigation only. Credentials never enter draft storage. */
export interface VideoLibraryExtensionSelection {
  owner: VideoJobOwner;
  source: VideoJobSource;
  title: string;
  metadata: Record<string, unknown>;
  inventory: RetainedSourceMediaInventory;
  originalReferences?: {
    descriptor: Record<string, unknown>;
    member?: { display_name: string; role: string };
  }[];
}
export const pendingVideoLibraryExtension =
  shallowRef<VideoLibraryExtensionSelection | null>(null);
export const VIDEO_LIBRARY_EXTENSION_NAVIGATION: InjectionKey<
  (value: VideoLibraryExtensionSelection) => void | Promise<void>
> = Symbol("video-library-extension");
export function takeVideoLibraryExtension() {
  const value = pendingVideoLibraryExtension.value;
  pendingVideoLibraryExtension.value = null;
  return value;
}

/** Reuse/new-draft is explicit permission to stop extending the prior movie. */
export const videoDurationResetPending = ref(false);
export const videoDurationReusePending =
  shallowRef<VideoDurationAuthoring | null>(null);
export function requestVideoDurationReset(metadata?: object) {
  videoDurationReusePending.value = metadata
    ? (videoSettingsForReuse(metadata)?.draft ?? null)
    : null;
  videoDurationResetPending.value = true;
}

/** Full retained clip inputs are intentionally memory-only. */
export const pendingLegacyVideoSettings = shallowRef<
  import("../api/legacyVideoSettings").LegacyVideoSettings | null
>(null);
