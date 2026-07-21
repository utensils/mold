<script setup lang="ts">
/*
 * Discover model card (spec WEB MODELS Discover, prototype lines 1607-1611):
 * a kind badge with a "Details ›" affordance, the mono model name, a
 * family · size line, and a full-width Pull button. The card body opens the
 * detail drawer; Pull enqueues the download straight into the shell.
 */
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import type { CatalogEntryWire } from "../types";

const props = defineProps<{ entry: CatalogEntryWire }>();
const emit = defineEmits<{ open: []; pull: [] }>();

const CATALOG_THUMBNAIL_WIDTH = 512;
const CIVITAI_IMAGE_HOSTS = new Set([
  "image.civitai.com",
  "imagecache.civitai.com",
]);

/*
 * Civitai preview URLs carry an inline transform segment (`original=true` or
 * `width=NNN`). Rewriting every card to the same small derivative means the
 * grid, a re-search, and a scroll-back all hit one cached response instead of
 * pulling a fresh full-size render per card. Anything else (HF, unknown hosts)
 * is loaded straight from its public URL — the mold server has no image proxy,
 * and adding a round trip through it would only slow the grid down. Kept local
 * to the card because it is the only consumer.
 */
function thumbnailSrc(raw: string): string {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return raw;
  }
  if (!CIVITAI_IMAGE_HOSTS.has(url.hostname)) return raw;
  const segments = url.pathname.split("/");
  const transform = segments.findIndex(
    (s) => s.startsWith("original=") || s.startsWith("width="),
  );
  if (transform < 0) return raw;
  segments[transform] = `width=${CATALOG_THUMBNAIL_WIDTH}`;
  url.pathname = segments.join("/");
  return url.toString();
}

const thumbUrl = computed(() =>
  props.entry.thumbnail_url ? thumbnailSrc(props.entry.thumbnail_url) : null,
);
const thumbLoaded = ref(false);
// A thumbnail that 404s, or a Civitai preview that turns out to be a video,
// falls back to the family mark rather than a broken-image glyph.
const thumbFailed = ref(false);
const showThumb = computed(() => thumbUrl.value !== null && !thumbFailed.value);

/*
 * The no-image state is a family mark rather than a grey box, so a grid of
 * catalog results still reads as a set of models. Mirrors the desktop app's
 * ModelFamilyPlaceholder so both surfaces label a family the same way.
 */
const family = computed(() => {
  const f = props.entry.family.trim().toLowerCase();
  if (f === "flux2" || f.startsWith("flux.2"))
    return { mark: "F2", label: "FLUX.2", tone: "cool" };
  if (f.startsWith("flux")) return { mark: "F", label: "FLUX", tone: "cool" };
  if (f === "sdxl") return { mark: "XL", label: "SDXL", tone: "warm" };
  if (f === "sd15" || f.includes("1.5"))
    return { mark: "1.5", label: "SD 1.5", tone: "warm" };
  if (f.startsWith("sd3")) return { mark: "3", label: "SD 3", tone: "warm" };
  if (f.startsWith("qwen"))
    return { mark: "Q", label: "QWEN", tone: "neutral" };
  if (f.startsWith("zimage") || f.startsWith("z-image"))
    return { mark: "Z", label: "Z-IMAGE", tone: "cool" };
  if (f.startsWith("ltx"))
    return { mark: "LTX", label: "LTX VIDEO", tone: "neutral" };
  if (f.startsWith("wuerstchen"))
    return { mark: "W", label: "WUERSTCHEN", tone: "warm" };
  const label = f.replaceAll("-", " ").toUpperCase() || "MODEL";
  return { mark: label.slice(0, 3), label, tone: "neutral" };
});

function formatGB(bytes: number | null): string {
  if (!bytes) return "—";
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

// engine_phase > 5 marks a catalog package mold can't run yet.
const supported = computed(() => props.entry.engine_phase <= 5);
const pullLabel = computed(() => {
  if (!supported.value) return "Unsupported";
  return props.entry.installed ? "Repair" : "Pull";
});
</script>

<template>
  <article class="card" data-test="discover-card">
    <button
      type="button"
      class="card__media"
      :aria-label="`${props.entry.name} — details`"
      @click="emit('open')"
    >
      <template v-if="showThumb">
        <span
          v-if="!thumbLoaded"
          class="ms-shimmer card__shimmer"
          data-test="card-thumb-shimmer"
          aria-hidden="true"
        />
        <img
          :src="thumbUrl!"
          alt=""
          loading="lazy"
          decoding="async"
          fetchpriority="low"
          class="card__thumb"
          data-test="card-thumb"
          @load="thumbLoaded = true"
          @error="thumbFailed = true"
        />
      </template>
      <span
        v-else
        class="card__placeholder"
        :data-tone="family.tone"
        data-test="card-thumb-placeholder"
      >
        <span class="card__placeholder-mark">{{ family.mark }}</span>
        <span class="card__placeholder-label">{{ family.label }}</span>
      </span>
    </button>

    <div class="card__top">
      <span class="card__kind">{{ props.entry.kind }}</span>
      <span
        v-if="props.entry.installed"
        class="card__installed"
        title="Already on disk"
      >
        installed
      </span>
      <button
        type="button"
        class="card__details"
        data-test="details-btn"
        @click="emit('open')"
      >
        Details ›
      </button>
    </div>

    <button
      type="button"
      class="card__nameline"
      data-test="card-open"
      @click="emit('open')"
    >
      <span class="card__name">{{ props.entry.name }}</span>
      <span class="card__meta">
        {{ props.entry.family }} · {{ formatGB(props.entry.size_bytes) }}
      </span>
    </button>

    <button
      type="button"
      class="card__pull"
      data-test="pull-btn"
      :disabled="!supported"
      :title="supported ? undefined : 'Unsupported catalog package'"
      @click="emit('pull')"
    >
      <Icon v-if="supported" name="download" :size="13" />
      {{ pullLabel }}
    </button>
  </article>
</template>

<style scoped>
.card {
  display: flex;
  flex-direction: column;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 16px;
  transition: border-color var(--dur-quick) var(--ease);
}

.card:hover {
  border-color: var(--ce);
}

/* Full-bleed preview: pulled out past the card padding, square-cornered where
   it meets the body so the card still reads as one surface. The height is
   fixed so every tile in a grid row lines up. */
.card__media {
  position: relative;
  display: block;
  width: calc(100% + 32px);
  /* A 3:2 box rather than a short strip: catalog previews are near-always
     portrait, and a shallow landscape crop of one shows little more than a
     pair of eyes. Every tile in a grid row is the same width, so an aspect
     ratio keeps the row aligned exactly as a fixed height would. */
  aspect-ratio: 3 / 2;
  margin: -16px -16px 12px;
  padding: 0;
  border: 0;
  border-bottom: 1px solid var(--edge);
  border-radius: calc(var(--radius-card) - 1px) calc(var(--radius-card) - 1px) 0
    0;
  overflow: hidden;
  background: var(--print);
  cursor: pointer;
}

.card__shimmer {
  position: absolute;
  inset: 0;
}

.card__thumb {
  position: relative;
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  /* Bias the crop upward: model previews are overwhelmingly portrait
     subjects, and a centred crop of a portrait in a landscape box cuts the
     face off. */
  object-position: 50% 28%;
}

.card__placeholder {
  --family-primary: var(--halide);
  --family-secondary: var(--safelight);
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--family-primary) 9%, var(--bench));
}

.card__placeholder[data-tone="warm"] {
  --family-primary: var(--safelight);
  --family-secondary: var(--halide);
}

.card__placeholder[data-tone="neutral"] {
  --family-primary: color-mix(in srgb, var(--rebate) 72%, transparent);
  --family-secondary: var(--halide);
}

.card__placeholder-mark {
  font-family: var(--f-mono);
  font-size: 30px;
  font-weight: 600;
  letter-spacing: -0.04em;
  color: color-mix(in srgb, var(--family-primary) 88%, var(--rebate));
}

.card__placeholder-label {
  position: absolute;
  left: 12px;
  bottom: 10px;
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  color: color-mix(in srgb, var(--family-primary) 72%, var(--rebate));
}

.card__media:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: -2px;
}

.card__top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
}

.card__kind {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--halide);
  border: 1px solid color-mix(in srgb, var(--halide) 40%, transparent);
  padding: 1px 7px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}

.card__installed {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--success);
  border: 1px solid color-mix(in srgb, var(--success) 40%, transparent);
  padding: 1px 7px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}

.card__details {
  margin-left: auto;
  border: 0;
  background: transparent;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
  cursor: pointer;
  padding: 2px 0;
  transition: color var(--dur-quick) var(--ease);
}

.card__details:hover {
  color: var(--rebate);
}

.card__nameline {
  border: 0;
  background: transparent;
  text-align: left;
  padding: 0;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.card__name {
  font-family: var(--f-mono);
  font-size: 13.5px;
  font-weight: 600;
  color: var(--rebate);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.card__meta {
  font-size: 11px;
  color: var(--ink-3);
  margin-bottom: 14px;
}

.card__pull {
  width: 100%;
  margin-top: auto;
  box-sizing: border-box;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--rebate);
  padding: 9px;
  border-radius: var(--radius-control);
  font-family: var(--f-body);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.card__pull:hover:not(:disabled) {
  border-color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 8%, transparent);
}

.card__pull:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.card__pull:focus-visible,
.card__details:focus-visible,
.card__nameline:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
