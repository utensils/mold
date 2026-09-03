<script setup lang="ts">
import MeshViewer from "@studio/components/MeshViewer.vue";
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import {
  galleryThumbnailScheduler,
  type ThumbnailHandle,
  type ThumbnailPriority,
} from "@studio/lib/thumbnailScheduler";
import {
  authedMediaUrl,
  fullSizeMediaUrl,
  isThumbnailPath,
  prepareNativeThumbnail,
} from "../../lib/gallery/media";
import type { ApiTarget } from "../../lib/api/client";

const props = withDefaults(
  defineProps<{
    path: string;
    video?: boolean;
    /** Render as a 3-D mesh viewer. Mutually exclusive with `video`/`audio`. */
    mesh?: boolean;
    /** Audio-only print: renders a transport instead of a raster element. */
    audio?: boolean;
    /** Poster shown while a mesh loads, and kept on failure. Callers that
     *  already hold a resolved URL (blob/http/native) pass it here; `mesh`
     *  callers that only know the print's path pass `posterPath` instead and
     *  this component resolves it the same way a tile resolves its
     *  thumbnail. When both are given, this explicit URL wins. */
    poster?: string;
    /** The print's thumbnail path (`galleryMediaPath(..., true)`), resolved
     *  internally into the mesh poster URL. Ignored unless `mesh` is set. */
    posterPath?: string;
    alt?: string;
    controls?: boolean;
    /** Explicit host to fetch from; defaults to the primary connection. */
    target?: ApiTarget | null;
    /** Blob-cache bucket, usually the origin host id. */
    cacheKey?: string | null;
    mediaVersion?: string | null;
    /** Scheduler priority for a thumbnail: on-screen tiles are `visible`,
     *  overscan rows `near`, prewarm `background`. Raising it promotes a
     *  queued request in place; the scheduler never demotes. */
    priority?: ThumbnailPriority;
  }>(),
  {
    video: false,
    mesh: false,
    audio: false,
    alt: "",
    controls: false,
    target: null,
    cacheKey: null,
    mediaVersion: null,
    priority: "visible",
  },
);

const src = ref<string | null>(null);
const failed = ref(false);
let loadEpoch = 0;
let thumbnailHandle: ThumbnailHandle<string> | null = null;

/** The mesh poster resolved from `posterPath`; `undefined` while pending or
 *  on a failed resolve, in which case the viewer just keeps its own
 *  "loading" state until `src` itself is ready. */
const resolvedPoster = ref<string | undefined>(undefined);
let posterHandle: ThumbnailHandle<string> | null = null;
let posterEpoch = 0;

/** Shared with `load()`: schedule a path through the same native-thumbnail
 *  cache the gallery tiles use, falling back to the blob route. Factored out
 *  so the mesh poster resolves through the identical path as a real
 *  thumbnail rather than carrying a second, divergent implementation. */
function scheduleThumbnail(path: string, priority: ThumbnailPriority): ThumbnailHandle<string> {
  const options = {
    ...(props.target ? { target: props.target } : {}),
    ...(props.cacheKey ? { cacheKey: props.cacheKey } : {}),
    ...(props.mediaVersion ? { mediaVersion: props.mediaVersion } : {}),
  };
  return galleryThumbnailScheduler.schedule({
    key: `${props.cacheKey ?? "primary"}|${path}|${props.mediaVersion ?? "legacy"}|${props.target?.baseUrl ?? "primary"}|${props.target?.apiKey ?? ""}`,
    hostKey: props.cacheKey ?? props.target?.baseUrl ?? "primary",
    priority,
    run: async (signal) =>
      (await prepareNativeThumbnail({
        path,
        target: props.target,
        cacheKey: props.cacheKey,
        mediaVersion: props.mediaVersion,
        signal,
      })) ?? authedMediaUrl(path, { ...options, signal }),
  });
}

const retryDelaysMs = [0, 250, 1_000] as const;

async function load() {
  thumbnailHandle?.cancel();
  thumbnailHandle = null;
  const epoch = ++loadEpoch;
  src.value = null;
  failed.value = false;
  const delays = isThumbnailPath(props.path) ? retryDelaysMs : ([0] as const);
  for (const delayMs of delays) {
    if (delayMs > 0) await new Promise((resolve) => setTimeout(resolve, delayMs));
    if (epoch !== loadEpoch) return;
    try {
      const options = {
        ...(props.target ? { target: props.target } : {}),
        ...(props.cacheKey ? { cacheKey: props.cacheKey } : {}),
        ...(props.mediaVersion ? { mediaVersion: props.mediaVersion } : {}),
      };
      // Thumbnails and full-size stills/audio go native-first in the desktop
      // app (#1132's rationale: a media element pointed straight at a host
      // shares WebKit's per-host connection pool with every held-open
      // generation/download stream to that host). Video keeps the ticketed
      // or direct streaming URL so it can seek without buffering; outside
      // Tauri, or when the native route refuses, stills fall back to it too.
      // A thumbnail goes to the persistent native cache first (a `mold-thumb://`
      // URL WebKit decodes itself); the blob route remains the fallback
      // outside Tauri or for a print with no content version.
      const thumbnail = isThumbnailPath(props.path);
      const url = thumbnail
        ? await (() => {
            const handle = scheduleThumbnail(props.path, props.priority);
            thumbnailHandle = handle;
            return handle.promise;
          })()
        : await fullSizeMediaUrl(props.path, {
            ...options,
            // A mesh is fetched whole by the viewer, exactly as audio is,
            // so it may not take the legacy blob path either.
            allowLegacyBlob: !props.video && !props.audio && !props.mesh,
            video: props.video,
          });
      if (epoch === loadEpoch) src.value = url;
      return;
    } catch {
      // Retry bounded transient native/network failures. The media cache
      // evicts rejected promises, so each attempt performs a fresh request.
    }
  }
  if (epoch === loadEpoch) failed.value = true;
}

/** Best-effort: a mesh poster is a nicety while the geometry streams in, not
 *  the primary content, so a failed resolve just leaves it undefined rather
 *  than flipping `failed`. */
async function loadPoster() {
  posterHandle?.cancel();
  posterHandle = null;
  const epoch = ++posterEpoch;
  if (!props.mesh || !props.posterPath) {
    resolvedPoster.value = undefined;
    return;
  }
  try {
    const handle = scheduleThumbnail(props.posterPath, "background");
    posterHandle = handle;
    const url = await handle.promise;
    if (epoch === posterEpoch) resolvedPoster.value = url;
  } catch {
    if (epoch === posterEpoch) resolvedPoster.value = undefined;
  }
}

const posterForViewer = computed(() => props.poster ?? resolvedPoster.value);

/** The next silent "3-D view couldn't start" needs the scheme that failed
 *  (blob/http/mold-local) to tell a CSP refusal from a parse/GL failure. */
function onMeshFail(message: string) {
  let scheme = "unknown";
  try {
    scheme = src.value ? new URL(src.value).protocol : "unknown";
  } catch {
    // src.value was not resolvable to an absolute URL; leave "unknown".
  }
  console.warn("[mesh] viewer failed", { scheme, message });
}

// Watch the route's VALUES, never a freshly built array: a getter returning
// `[...]` yields a new array on every parent re-render, and Vue compares the
// source by identity, so each re-render (selecting a tile, a poll landing)
// re-ran `load()`, nulled `src`, and swapped the <img> for the shimmer. Parents
// build `target` per render, so that swap happened between the two clicks of
// a double-click on every remote tile — the lightbox never opened for
// remote-only prints. The multi-source form compares each value on its own.
watch(
  [
    () => props.path,
    () => props.cacheKey,
    () => props.mediaVersion,
    () => props.target?.baseUrl,
    () => props.target?.apiKey,
  ],
  load,
);
// A tile scrolling from the overscan band into view promotes its queued
// request without restarting it.
watch(
  () => props.priority,
  (priority) => thumbnailHandle?.setPriority(priority),
);
watch(
  [
    () => props.mesh,
    () => props.posterPath,
    () => props.cacheKey,
    () => props.mediaVersion,
    () => props.target?.baseUrl,
    () => props.target?.apiKey,
  ],
  loadPoster,
);
onMounted(() => {
  load();
  loadPoster();
});
onUnmounted(() => {
  loadEpoch += 1;
  thumbnailHandle?.cancel();
  thumbnailHandle = null;
  posterEpoch += 1;
  posterHandle?.cancel();
  posterHandle = null;
});
</script>

<template>
  <video
    v-if="video && src"
    :src="src"
    class="h-full w-full object-contain"
    :controls="controls"
    loop
    playsinline
    disablepictureinpicture
  />
  <MeshViewer
    v-else-if="mesh && src"
    :src="src"
    v-bind="posterForViewer ? { poster: posterForViewer } : {}"
    :alt="alt"
    class="h-full w-full"
    @fail="onMeshFail"
  />
  <audio v-else-if="audio && src" :src="src" class="w-full" controls />
  <img
    v-else-if="src"
    :src="src"
    :alt="alt"
    class="h-full w-full object-cover"
    decoding="async"
    draggable="false"
  />
  <div v-else-if="failed" class="flex h-full w-full items-center justify-center bg-bench">
    <span class="edge-code">UNREADABLE</span>
  </div>
  <div v-else class="media-placeholder grain-shimmer h-full w-full" aria-hidden="true">
    <span>Loading preview</span>
  </div>
</template>

<style scoped>
.media-placeholder {
  display: grid;
  place-items: center;
  background-color: color-mix(in srgb, var(--bench) 82%, var(--safelight) 18%);
  background-image:
    radial-gradient(
      circle at 28% 24%,
      color-mix(in srgb, var(--halide) 16%, transparent),
      transparent 34%
    ),
    radial-gradient(
      circle at 72% 76%,
      color-mix(in srgb, var(--safelight) 14%, transparent),
      transparent 38%
    );
}

.media-placeholder span {
  border: 1px solid color-mix(in srgb, var(--rebate) 20%, transparent);
  border-radius: 999px;
  padding: 5px 9px;
  background: color-mix(in srgb, var(--bench) 78%, transparent);
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
</style>
