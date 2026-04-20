<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref } from "vue";
import { imageUrl, thumbnailUrl } from "../api";
import type { GalleryImage } from "../types";
import { mediaKind } from "../types";
import { useTweaks } from "../composables/useTweaks";
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
    muted?: boolean;
    starred?: boolean;
  }>(),
  { variant: "grid", muted: true, starred: false },
);

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
  (e: "star", item: GalleryImage): void;
  (e: "rerun", item: GalleryImage): void;
}>();

const { tweaks } = useTweaks();
const showChips = computed(() => tweaks.value.showChips);

/*
 * Lifecycle
 * ---------
 * A single IntersectionObserver drives three states per card:
 *   - not mounted        → pulse placeholder
 *   - near-visible       → start loading thumbnail
 *   - in viewport        → autoplay video
 * Videos pause when scrolled out of view so we don't end up with 50
 * decoders running in a long gallery.
 */
const root = ref<HTMLElement | null>(null);
const videoEl = ref<HTMLVideoElement | null>(null);

const visible = ref(false);
const onScreen = ref(false);

type SourceStage = "thumb" | "full" | "broken";
const stage = ref<SourceStage>(props.variant === "feed" ? "full" : "thumb");

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
          el.play().catch(() => undefined);
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

const imageCurrentSrc = computed(() => {
  if (stage.value === "thumb") return thumbnailUrl(props.item.filename);
  if (stage.value === "full") return imageUrl(props.item.filename);
  return "";
});

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

function onStar(e: Event) {
  e.stopPropagation();
  emit("star", props.item);
}

function onRerun(e: Event) {
  e.stopPropagation();
  emit("rerun", props.item);
}

function copyPrompt(e: Event) {
  e.stopPropagation();
  const p = props.item.metadata.prompt;
  if (!p) return;
  navigator.clipboard?.writeText(p).catch(() => undefined);
}
</script>

<template>
  <article
    ref="root"
    class="card"
    :class="[
      variant === 'feed' ? 'card-feed' : 'card-grid',
      { 'is-starred': starred },
    ]"
    role="button"
    tabindex="0"
    :aria-label="`Open ${item.filename}`"
    @click="openDetail"
    @keydown.enter.prevent="openDetail"
    @keydown.space.prevent="openDetail"
  >
    <div class="card-media" :style="aspectStyle">
      <div v-if="!visible" class="card-placeholder" aria-hidden="true" />

      <template v-if="visible && stage !== 'broken'">
        <img
          v-if="kind !== 'video'"
          :src="imageCurrentSrc"
          :alt="item.metadata.prompt || item.filename"
          loading="lazy"
          decoding="async"
          class="card-img"
          :style="{ objectFit: variant === 'feed' ? 'contain' : 'cover' }"
          @error="onImgError"
        />
        <video
          v-else
          ref="videoEl"
          :src="videoSrc"
          :poster="videoPoster"
          class="card-img"
          :style="{ objectFit: variant === 'feed' ? 'contain' : 'cover' }"
          :muted="muted"
          loop
          playsinline
          preload="metadata"
          :autoplay="onScreen"
          @error="onVideoError"
        />
        <div v-if="kind === 'video'" class="card-video-marker">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      </template>

      <div v-if="stage === 'broken'" class="card-broken">
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.7"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <rect x="3" y="4" width="18" height="16" rx="2" />
          <path d="m3 20 6-6 4 4" />
          <path d="m14 14 4-4 3 3" />
          <path d="m3 4 18 16" />
        </svg>
        <span>can't render</span>
      </div>

      <div v-if="showChips" class="card-chips">
        <span v-if="kind !== 'image'" class="card-chip">
          <svg
            v-if="kind === 'video'"
            width="10"
            height="10"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
          {{ kind }}
        </span>
        <span v-if="item.format" class="card-chip">{{ item.format }}</span>
      </div>

      <div class="card-actions">
        <button
          class="card-iconbtn"
          :class="{ on: starred }"
          :aria-label="starred ? 'Unstar' : 'Star'"
          @click="onStar"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="m12 3 2.9 5.9 6.5.9-4.7 4.6 1.1 6.5L12 17.8l-5.8 3.1 1.1-6.5L2.6 9.8l6.5-.9z"
              :fill="starred ? 'currentColor' : 'none'"
            />
          </svg>
        </button>
        <button class="card-iconbtn" aria-label="Re-run" @click="onRerun">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M12 8v4l3 2" />
          </svg>
        </button>
      </div>

      <div v-if="variant === 'grid'" class="card-overlay">
        <div class="card-overlay-top">
          <span class="card-dot" />
          <span class="card-model">{{ modelLabel }}</span>
          <time class="card-time">{{ relative }}</time>
        </div>
        <p v-if="item.metadata.prompt" class="card-prompt-compact">
          {{ item.metadata.prompt }}
        </p>
      </div>
    </div>

    <div v-if="variant === 'feed'" class="card-caption">
      <div class="card-meta-row">
        <span class="card-model-pill">
          <span class="card-dot" />
          {{ modelLabel }}
        </span>
        <span v-if="resolution" class="card-meta-chip">{{ resolution }}</span>
        <span v-if="item.metadata.seed" class="card-meta-chip">
          seed {{ item.metadata.seed }}
        </span>
        <span v-if="item.metadata.steps" class="card-meta-chip">
          {{ item.metadata.steps }} steps
        </span>
        <span v-if="item.metadata.frames" class="card-meta-chip">
          {{ item.metadata.frames }}f @ {{ item.metadata.fps || "?" }}fps
        </span>
        <span
          v-if="item.metadata_synthetic"
          class="card-meta-chip"
          style="font-style: italic; color: var(--fg-4)"
          title="No mold:parameters chunk was embedded in this file."
        >
          no metadata
        </span>
        <time
          class="card-time card-meta-push"
          :title="new Date(item.timestamp * 1000).toLocaleString()"
        >
          {{ relative }}
        </time>
      </div>
      <p v-if="item.metadata.prompt" class="card-prompt">
        {{ item.metadata.prompt }}
      </p>
      <p
        v-else
        class="card-prompt"
        style="font-style: italic; color: var(--fg-3)"
      >
        No prompt recorded for this file.
      </p>
      <div class="card-caption-tail">
        <button class="card-tail-btn" @click="onRerun">
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M3 12a9 9 0 0 1 15.5-6.3L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M12 8v4l3 2" />
          </svg>
          Re-run with tweaks
        </button>
        <button
          v-if="item.metadata.prompt"
          class="card-tail-btn"
          @click="copyPrompt"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <rect x="9" y="9" width="13" height="13" rx="2" />
            <path d="M5 15V5a2 2 0 0 1 2-2h10" />
          </svg>
          Copy prompt
        </button>
      </div>
    </div>
  </article>
</template>
