<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from "vue";
import { imageUrl, thumbnailUrl } from "../api";
import type { GalleryImage } from "../types";
import { mediaKind } from "../types";
import {
  formatRelativeTime,
  formatResolution,
  shortModel,
} from "../util/format";

type Variant = "feed" | "grid";

const props = withDefaults(
  defineProps<{
    item: GalleryImage;
    variant?: Variant;
    // Global audio preference. Browser policy still requires the first
    // sound-on playback to follow a user gesture — once the user clicks
    // the header toggle, subsequent videos entering the viewport pick up
    // the preference automatically.
    muted?: boolean;
  }>(),
  { variant: "grid", muted: true },
);

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
}>();

/*
 * Lifecycle
 * ---------
 * Each card uses a single IntersectionObserver that drives three states:
 *   - not mounted        → lightweight pulse placeholder
 *   - near-visible       → start loading thumbnail
 *   - in viewport        → autoplay video
 * Videos pause when they scroll out of view so we don't have 50 decoders
 * running at once in a long gallery.
 */
const root = ref<HTMLElement | null>(null);
const videoEl = ref<HTMLVideoElement | null>(null);

const visible = ref(false);
const onScreen = ref(false);

/*
 * Source fallback chain: thumbnail → full image → broken tile.
 * Grid mode starts at `thumb` (cached /api/gallery/thumbnail/:name is fast
 * and we're rendering dozens of cards per viewport). Feed mode starts at
 * `full` so the big Tumblr-style card shows the image at its natural
 * resolution instead of a 256-px upscale — fewer cards are visible at a
 * time so the bandwidth cost is negligible. On `<img @error>` we step
 * through the chain and eventually land on a "can't render" tile rather
 * than the browser's default broken-icon.
 */
type SourceStage = "thumb" | "full" | "broken";
const stage = ref<SourceStage>(props.variant === "feed" ? "full" : "thumb");
const loaded = ref(false);

let observer: IntersectionObserver | null = null;

onMounted(() => {
  if (!root.value) return;
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          visible.value = true;
          onScreen.value = true;
        } else if (visible.value) {
          onScreen.value = false;
        }
      }
      const el = videoEl.value;
      if (el) {
        if (onScreen.value && el.paused) {
          el.play().catch(() => {
            /* autoplay may be blocked; ignore */
          });
        } else if (!onScreen.value && !el.paused) {
          el.pause();
        }
      }
    },
    { rootMargin: "600px 0px", threshold: 0.01 },
  );
  observer.observe(root.value);
});

onBeforeUnmount(() => {
  observer?.disconnect();
});

const kind = computed(() => mediaKind(props.item.format, props.item.filename));
const aspectStyle = computed(() => {
  const w = props.item.metadata.width;
  const h = props.item.metadata.height;
  if (w > 0 && h > 0) return { aspectRatio: `${w} / ${h}` };
  return kind.value === "video"
    ? { aspectRatio: "16 / 9" }
    : { aspectRatio: "1 / 1" };
});

// Image fallback chain: thumbnail → full file → broken.
const imageCurrentSrc = computed(() => {
  if (stage.value === "thumb") return thumbnailUrl(props.item.filename);
  if (stage.value === "full") return imageUrl(props.item.filename);
  return "";
});

// Videos are different: `src` must be the actual video bytes (a PNG
// thumbnail won't decode in a <video> element). We use the thumbnail URL
// as a static `poster` so the browser shows the first frame while the
// video loads, and fall back to the full file if even that fails.
const videoSrc = computed(() => imageUrl(props.item.filename));
const videoPoster = computed(() => thumbnailUrl(props.item.filename));

const relative = computed(() => formatRelativeTime(props.item.timestamp));
const modelLabel = computed(() => shortModel(props.item.metadata.model));
const resolution = computed(() => formatResolution(props.item.metadata));

function onImgError() {
  if (stage.value === "thumb") stage.value = "full";
  else stage.value = "broken";
}

function onVideoError() {
  stage.value = "broken";
}

function openDetail() {
  emit("open", props.item);
}
</script>

<template>
  <article
    ref="root"
    class="group relative block w-full cursor-zoom-in overflow-hidden bg-ink-900/80 shadow-[var(--shadow-card)] transition hover:ring-brand-400/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-400"
    :class="[
      variant === 'feed'
        ? 'ring-0 sm:rounded-3xl sm:ring-1 sm:ring-white/5'
        : 'rounded-2xl ring-1 ring-white/5',
    ]"
    role="button"
    tabindex="0"
    :aria-label="`Open ${item.filename}`"
    @click="openDetail"
    @keydown.enter.prevent="openDetail"
    @keydown.space.prevent="openDetail"
  >
    <!-- Media frame: aspect-ratio preserved, media absolutely positioned -->
    <div class="relative w-full overflow-hidden" :style="aspectStyle">
      <!-- Placeholder / broken state -->
      <div
        v-if="!visible || stage === 'broken'"
        class="absolute inset-0"
        :class="
          stage === 'broken'
            ? 'flex flex-col items-center justify-center gap-1.5 bg-rose-950/30 text-rose-200'
            : 'animate-pulse bg-gradient-to-br from-white/[0.03] to-white/[0.08]'
        "
        aria-hidden="true"
      >
        <template v-if="stage === 'broken'">
          <svg
            class="h-8 w-8 opacity-70"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.7"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <path d="m3 20 6-6 4 4" />
            <path d="m14 14 4-4 3 3" />
            <path d="m3 4 18 16" />
          </svg>
          <p class="px-3 text-center text-[12px] leading-tight opacity-80">
            can't render
          </p>
        </template>
      </div>

      <!-- Image: `object-contain` in feed so the whole image is visible at
           its natural aspect ratio (no edge cropping, Tumblr-style);
           `object-cover` in grid so tiles pack tightly. -->
      <img
        v-if="visible && stage !== 'broken' && kind !== 'video'"
        :src="imageCurrentSrc"
        :alt="item.metadata.prompt || item.filename"
        loading="lazy"
        decoding="async"
        class="absolute inset-0 h-full w-full transition-opacity duration-200"
        :class="variant === 'feed' ? 'object-contain' : 'object-cover'"
        :style="{ opacity: loaded ? 1 : 0 }"
        @load="loaded = true"
        @error="onImgError"
      />

      <!-- Video -->
      <video
        v-if="visible && stage !== 'broken' && kind === 'video'"
        ref="videoEl"
        :src="videoSrc"
        :poster="videoPoster"
        class="absolute inset-0 h-full w-full transition-opacity duration-200"
        :class="variant === 'feed' ? 'object-contain' : 'object-cover'"
        :style="{ opacity: loaded ? 1 : 0 }"
        :muted="muted"
        loop
        playsinline
        preload="metadata"
        :autoplay="onScreen"
        @loadeddata="loaded = true"
        @error="onVideoError"
      />

      <!-- Top-left type badge -->
      <div
        v-if="kind !== 'image'"
        class="absolute left-3 top-3 inline-flex items-center gap-1 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-white backdrop-blur"
      >
        <svg
          v-if="kind === 'video'"
          class="h-3 w-3"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M8 5v14l11-7z" />
        </svg>
        <svg
          v-else
          class="h-3 w-3"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          aria-hidden="true"
        >
          <path d="M3 12a9 9 0 1 1 18 0" />
        </svg>
        {{ kind === "video" ? "video" : "anim" }}
      </div>

      <!-- Format chip (top-right) -->
      <div
        v-if="item.format"
        class="absolute right-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-white/85 backdrop-blur"
      >
        {{ item.format }}
      </div>

      <!-- Grid variant: bottom-anchored hover overlay (compact). -->
      <div
        v-if="variant === 'grid'"
        class="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent p-3.5 pt-10 text-white"
      >
        <div class="flex items-end gap-3">
          <div class="min-w-0 flex-1">
            <div
              class="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider"
            >
              <span class="h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
              <span class="truncate">{{ modelLabel }}</span>
            </div>
            <p
              v-if="item.metadata.prompt"
              class="mt-1 line-clamp-2 text-[12.5px] leading-snug text-white/90 opacity-0 transition-opacity duration-200 group-hover:opacity-100"
            >
              {{ item.metadata.prompt }}
            </p>
          </div>
          <time
            class="shrink-0 text-[11px] tabular-nums text-white/75"
            :title="new Date(item.timestamp * 1000).toLocaleString()"
          >
            {{ relative }}
          </time>
        </div>
      </div>
    </div>

    <!-- Feed variant: dedicated caption strip under the media, always visible. -->
    <div
      v-if="variant === 'feed'"
      class="flex flex-col gap-2.5 border-t border-white/5 px-5 py-4 sm:px-6 sm:py-5"
    >
      <div
        class="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-ink-300"
      >
        <span
          class="inline-flex items-center gap-1.5 font-semibold text-ink-50"
        >
          <span class="h-1.5 w-1.5 rounded-full bg-brand-400" />
          {{ modelLabel }}
        </span>
        <span v-if="resolution" class="tabular-nums">{{ resolution }}</span>
        <span v-if="item.metadata.seed" class="tabular-nums">
          seed {{ item.metadata.seed }}
        </span>
        <span v-if="item.metadata.steps" class="tabular-nums">
          {{ item.metadata.steps }} steps
        </span>
        <span v-if="item.metadata.frames" class="tabular-nums">
          {{ item.metadata.frames }}f @ {{ item.metadata.fps || "?" }}fps
        </span>
        <span
          v-if="item.metadata_synthetic"
          class="italic text-ink-500"
          title="No mold:parameters chunk was embedded in this file."
        >
          no metadata
        </span>
        <time
          class="ml-auto shrink-0 text-[12px] tabular-nums text-ink-400"
          :title="new Date(item.timestamp * 1000).toLocaleString()"
        >
          {{ relative }}
        </time>
      </div>
      <p
        v-if="item.metadata.prompt"
        class="line-clamp-3 text-[15px] leading-relaxed text-ink-100"
      >
        {{ item.metadata.prompt }}
      </p>
      <p v-else class="text-[13px] italic text-ink-400">
        No prompt recorded for this file.
      </p>
    </div>
  </article>
</template>
