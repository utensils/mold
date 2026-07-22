<script setup lang="ts">
/*
 * Lightbox — the canonical print viewer (spec §03
 * Tile/lightbox rules, prototype desktop LIGHTBOX + mobile-web GALLERY VIEWER).
 *
 *   ≥640px  contained two-pane lightbox: media on the --print bed with
 *           prev/next circle buttons + a 320px details panel (kicker, n/N, ✕,
 *           prompt, mono Model/Seed/Dimensions/Host rows, accent "Reuse these
 *           settings", Use as source / Download, and an overflow Upscale/Delete).
 *   <640px  full-screen viewer: top bar (✕, n/N, download), centered media with
 *           arrows, bottom sheet (prompt, chips, Reuse + Use as source / Save).
 *
 * The page owns the reuse/source/delete side effects; this component just emits
 * the intent. Keyboard: ←/→ navigate, Esc closes.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { imageUrl, thumbnailUrl } from "../../api";
import { getHost } from "../../lib/hostRegistry";
import {
  MediaUpgradeRequiredError,
  directMediaUrl,
  directThumbnailUrl,
  needsAuthedMedia,
  resolveStreamableSrc,
  resolveThumbnailSrc,
} from "../../lib/galleryMedia";
import type { GalleryImage } from "../../types";
import { mediaKind } from "../../types";
import { formatResolution, shortModel } from "../../util/format";

const props = defineProps<{
  item: GalleryImage | null;
  // Position in the filtered list — surfaced as "n / N".
  index: number;
  total: number;
  hasPrev: boolean;
  hasNext: boolean;
  // Global audio preference; native controls still let the user override.
  muted: boolean;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "prev"): void;
  (e: "next"): void;
  (e: "reuse", item: GalleryImage): void;
  (e: "use-source", item: GalleryImage): void;
  (e: "upscale", item: GalleryImage): void;
  (e: "delete", item: GalleryImage): void;
}>();

const menuOpen = ref(false);

const kind = computed(() =>
  props.item ? mediaKind(props.item.format, props.item.filename) : "image",
);
const isVideoFile = computed(() => kind.value === "video");

/*
 * Full-size media is addressed on the host that owns the print. Keyless hosts
 * (the serving origin included) resolve synchronously to a plain URL. A keyed
 * host exchanges its key for a short-lived `media-token` ticket so WebKit can
 * Range-stream video without the durable key ever entering a URL; images may
 * fall back to a bounded blob on a host too old to issue tickets, but a video
 * surfaces the "can't stream" state rather than buffering the whole file.
 */
const hostEntry = computed(() => {
  const id = (props.item as { hostId?: string } | null)?.hostId;
  return id ? getHost(id) : null;
});
const hostLabel = computed(() => {
  const item = props.item as { hostLabel?: string } | null;
  const hostId = (props.item as { hostId?: string } | null)?.hostId;
  return item?.hostLabel ?? hostEntry.value?.name ?? hostId ?? "local";
});

const mediaSrc = ref("");
const posterSrc = ref("");
const streamBlocked = ref(false);
const streamMessage = ref("");
let resolveGeneration = 0;

function resolveMedia() {
  const item = props.item;
  const generation = ++resolveGeneration;
  streamBlocked.value = false;
  if (!item) {
    mediaSrc.value = "";
    posterSrc.value = "";
    return;
  }
  const host = hostEntry.value;
  const hostId = (item as { hostId?: string }).hostId;
  if (hostId && !host) {
    mediaSrc.value = "";
    posterSrc.value = "";
    streamBlocked.value = true;
    streamMessage.value = `${hostLabel.value} isn't connected anymore.`;
    return;
  }
  if (!host) {
    mediaSrc.value = imageUrl(item.filename);
    posterSrc.value = thumbnailUrl(item.filename);
    return;
  }
  if (!needsAuthedMedia(host)) {
    mediaSrc.value = directMediaUrl(host, item.filename);
    posterSrc.value = directThumbnailUrl(host, item.filename);
    return;
  }
  mediaSrc.value = "";
  posterSrc.value = "";
  void resolveThumbnailSrc(host, item.filename)
    .then((url) => {
      if (generation === resolveGeneration) posterSrc.value = url;
    })
    .catch(() => {
      /* no poster is fine — the media itself still loads */
    });
  void resolveStreamableSrc(host, item.filename)
    .then((url) => {
      if (generation === resolveGeneration) mediaSrc.value = url;
    })
    .catch((err) => {
      if (generation !== resolveGeneration) return;
      streamBlocked.value = true;
      streamMessage.value =
        err instanceof MediaUpgradeRequiredError
          ? "Connect a newer Mold host to stream this clip."
          : "Couldn't reach the host that holds this print.";
    });
}

watch(() => props.item, resolveMedia, { immediate: true });

const blockedTitle = computed(() =>
  isVideoFile.value ? "Can't stream this clip" : "Can't load this print",
);

const prompt = computed(() => props.item?.metadata.prompt ?? "");
const modelLabel = computed(() =>
  props.item ? shortModel(props.item.metadata.model) : "",
);
const dimensions = computed(() =>
  props.item ? formatResolution(props.item.metadata) : "",
);
const seed = computed(() => props.item?.metadata.seed ?? null);
const steps = computed(() => props.item?.metadata.steps ?? null);
const guidance = computed(() => props.item?.metadata.guidance ?? null);
const scheduler = computed(() => {
  const value = props.item?.metadata.scheduler;
  if (!value) return "—";
  if (typeof value === "string") return value;
  return Object.keys(value)[0] ?? "—";
});
const negativePrompt = computed(
  () => props.item?.metadata.negative_prompt?.trim() ?? "",
);
const originalPrompt = computed(
  () => props.item?.metadata.original_prompt?.trim() ?? "",
);
const loraLabel = computed(() => {
  const metadata = props.item?.metadata;
  const stack = metadata?.loras?.length
    ? metadata.loras
    : metadata?.lora
      ? [{ path: metadata.lora, scale: metadata.lora_scale ?? 1 }]
      : [];
  return stack.map((lora) => `${lora.path} · ${lora.scale}`).join(", ");
});
const formatLabel = computed(() =>
  (
    props.item?.format ??
    props.item?.metadata.output_format ??
    "—"
  ).toUpperCase(),
);
const sizeLabel = computed(() => {
  const bytes = props.item?.size_bytes;
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
});
const timestampLabel = computed(() => {
  const timestamp = props.item?.timestamp;
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleString();
});

async function copyText(value: string) {
  if (!value || typeof navigator === "undefined" || !navigator.clipboard)
    return;
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    // Clipboard access can be denied outside a secure browser context.
  }
}

// ── Responsive layout ──────────────────────────────────────────────────────
// Resolve the breakpoint at setup so the correct layout renders on the first
// paint; a resize listener keeps it honest afterwards. Defaults to the desktop
// two-pane when there is no window (SSR / non-DOM environments).
const wide = ref(
  typeof window !== "undefined" ? window.innerWidth >= 640 : true,
);
function updateWide() {
  if (typeof window !== "undefined") wide.value = window.innerWidth >= 640;
}

// ── Keyboard ────────────────────────────────────────────────────────────────
function onKey(e: KeyboardEvent) {
  if (!props.item) return;
  if (e.key === "Escape") {
    if (menuOpen.value) menuOpen.value = false;
    else emit("close");
  } else if (e.key === "ArrowLeft" && props.hasPrev) {
    emit("prev");
  } else if (e.key === "ArrowRight" && props.hasNext) {
    emit("next");
  }
}

onMounted(() => {
  updateWide();
  window.addEventListener("resize", updateWide);
  window.addEventListener("keydown", onKey);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", updateWide);
  window.removeEventListener("keydown", onKey);
  if (typeof document !== "undefined") document.body.style.overflow = "";
});

// Lock body scroll while open; reset the overflow menu on item change.
watch(
  () => props.item,
  (open, prev) => {
    if (typeof document !== "undefined") {
      document.body.style.overflow = open ? "hidden" : "";
    }
    if (!prev || !open || prev.filename !== open.filename)
      menuOpen.value = false;
  },
);

function onReuse() {
  if (props.item) emit("reuse", props.item);
}
function onUseSource() {
  if (props.item) emit("use-source", props.item);
}
function onUpscale() {
  menuOpen.value = false;
  if (props.item) emit("upscale", props.item);
}
function onDelete() {
  menuOpen.value = false;
  if (props.item) emit("delete", props.item);
}
</script>

<template>
  <Transition name="fade">
    <div
      v-if="item"
      class="lb"
      :class="wide ? 'lb--wide' : 'lb--full'"
      role="dialog"
      aria-modal="true"
      aria-label="Print viewer"
      @click.self="emit('close')"
    >
      <!-- ============================ DESKTOP ============================ -->
      <div v-if="wide" class="lb__card" @click.stop>
        <div class="lb__stage">
          <p v-if="streamBlocked" class="lb__blocked" role="status">
            <span class="lb__blocked-title">{{ blockedTitle }}</span>
            <span class="lb__blocked-body">{{ streamMessage }}</span>
          </p>
          <video
            v-else-if="isVideoFile && mediaSrc"
            :src="mediaSrc"
            :poster="posterSrc"
            class="lb__media"
            controls
            autoplay
            loop
            playsinline
            :muted="muted"
          />
          <img
            v-else-if="mediaSrc"
            :src="mediaSrc"
            :alt="prompt || item.filename"
            class="lb__media"
          />
          <button
            class="lb__nav lb__nav--prev"
            :disabled="!hasPrev"
            aria-label="Previous print"
            @click="emit('prev')"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M15 6l-6 6 6 6" />
            </svg>
          </button>
          <button
            class="lb__nav lb__nav--next"
            :disabled="!hasNext"
            aria-label="Next print"
            @click="emit('next')"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M9 6l6 6-6 6" />
            </svg>
          </button>
        </div>

        <div class="lb__panel">
          <div class="lb__phead">
            <span class="lb__kicker">Print details</span>
            <span class="lb__flex"></span>
            <span class="lb__pos">{{ index + 1 }} / {{ total }}</span>
            <div class="lb__more">
              <button
                class="lb__morebtn"
                aria-haspopup="menu"
                :aria-expanded="menuOpen"
                aria-label="More actions"
                @click="menuOpen = !menuOpen"
              >
                <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <circle cx="5" cy="12" r="1.6" />
                  <circle cx="12" cy="12" r="1.6" />
                  <circle cx="19" cy="12" r="1.6" />
                </svg>
              </button>
              <template v-if="menuOpen">
                <button
                  class="lb__menu-scrim"
                  tabindex="-1"
                  aria-hidden="true"
                  @click="menuOpen = false"
                ></button>
                <div class="lb__menu" role="menu">
                  <button role="menuitem" @click="onUpscale">Upscale…</button>
                  <button
                    role="menuitem"
                    class="lb__menu-danger"
                    @click="onDelete"
                  >
                    Delete
                  </button>
                </div>
              </template>
            </div>
            <button class="lb__x" aria-label="Close" @click="emit('close')">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2.2"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          </div>

          <div v-if="prompt" class="lb__prompt-wrap">
            <p class="lb__prompt">{{ prompt }}</p>
            <button
              data-test="copy-prompt"
              class="lb__copy"
              @click="copyText(prompt)"
            >
              Copy prompt
            </button>
          </div>
          <p v-else class="lb__prompt lb__prompt--empty">No prompt recorded.</p>

          <div class="lb__rows">
            <div class="lb__row">
              <span class="lb__rowk">Model</span><span>{{ modelLabel }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Seed</span
              ><button
                v-if="seed != null"
                data-test="copy-seed"
                class="lb__rowcopy"
                @click="copyText(String(seed))"
              >
                {{ seed }}
              </button>
              <span v-else>—</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Dimensions</span
              ><span>{{ dimensions || "—" }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Host</span><span>{{ hostLabel }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Steps</span><span>{{ steps ?? "—" }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Guidance</span
              ><span>{{ guidance ?? "—" }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Scheduler</span
              ><span>{{ scheduler }}</span>
            </div>
            <div v-if="loraLabel" class="lb__row">
              <span class="lb__rowk">LoRA</span><span>{{ loraLabel }}</span>
            </div>
            <div v-if="negativePrompt" class="lb__row lb__row--long">
              <span class="lb__rowk">Negative</span
              ><span>{{ negativePrompt }}</span>
            </div>
            <div v-if="originalPrompt" class="lb__row lb__row--long">
              <span class="lb__rowk">Original</span
              ><span>{{ originalPrompt }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">File</span
              ><span>{{ formatLabel }} · {{ sizeLabel }}</span>
            </div>
            <div class="lb__row">
              <span class="lb__rowk">Created</span
              ><span>{{ timestampLabel }}</span>
            </div>
          </div>

          <span class="lb__flex"></span>

          <button class="lb__reuse" @click="onReuse">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path d="M3 12a9 9 0 019-9 9 9 0 016.7 3M21 3v6h-6" />
              <path d="M21 12a9 9 0 01-9 9 9 9 0 01-6.7-3M3 21v-6h6" />
            </svg>
            Reuse these settings
          </button>
          <div class="lb__pair">
            <button class="lb__quiet" @click="onUseSource">
              Use as source
            </button>
            <a
              v-if="mediaSrc"
              class="lb__quiet"
              :href="mediaSrc"
              :download="item.filename"
              role="button"
            >
              Download
            </a>
            <span v-else class="lb__quiet lb__quiet--off">Download</span>
          </div>
        </div>
      </div>

      <!-- ============================ MOBILE ============================ -->
      <div v-else class="lb__full" @click.stop>
        <div class="lb__topbar">
          <button class="lb__circle" aria-label="Close" @click="emit('close')">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.4"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
          <span class="lb__pos lb__pos--light"
            >{{ index + 1 }} / {{ total }}</span
          >
          <span class="lb__flex"></span>
          <a
            v-if="mediaSrc"
            class="lb__circle"
            :href="mediaSrc"
            :download="item.filename"
            aria-label="Save"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M4 12v7a1 1 0 001 1h14a1 1 0 001-1v-7" />
              <path d="M12 15V3M8 7l4-4 4 4" />
            </svg>
          </a>
          <div class="lb__more">
            <button
              class="lb__circle"
              aria-haspopup="menu"
              :aria-expanded="menuOpen"
              aria-label="More actions"
              @click="menuOpen = !menuOpen"
            >
              <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <circle cx="5" cy="12" r="1.6" />
                <circle cx="12" cy="12" r="1.6" />
                <circle cx="19" cy="12" r="1.6" />
              </svg>
            </button>
            <template v-if="menuOpen">
              <button
                class="lb__menu-scrim"
                tabindex="-1"
                aria-hidden="true"
                @click="menuOpen = false"
              ></button>
              <div class="lb__menu lb__menu--right" role="menu">
                <button role="menuitem" @click="onUpscale">Upscale…</button>
                <button
                  role="menuitem"
                  class="lb__menu-danger"
                  @click="onDelete"
                >
                  Delete
                </button>
              </div>
            </template>
          </div>
        </div>

        <div class="lb__stage lb__stage--full">
          <p v-if="streamBlocked" class="lb__blocked" role="status">
            <span class="lb__blocked-title">{{ blockedTitle }}</span>
            <span class="lb__blocked-body">{{ streamMessage }}</span>
          </p>
          <video
            v-else-if="isVideoFile && mediaSrc"
            :src="mediaSrc"
            :poster="posterSrc"
            class="lb__media"
            controls
            autoplay
            loop
            playsinline
            :muted="muted"
          />
          <img
            v-else-if="mediaSrc"
            :src="mediaSrc"
            :alt="prompt || item.filename"
            class="lb__media"
          />
          <button
            class="lb__nav lb__nav--prev"
            :disabled="!hasPrev"
            aria-label="Previous print"
            @click="emit('prev')"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.4"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M15 6l-6 6 6 6" />
            </svg>
          </button>
          <button
            class="lb__nav lb__nav--next"
            :disabled="!hasNext"
            aria-label="Next print"
            @click="emit('next')"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2.4"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M9 6l6 6-6 6" />
            </svg>
          </button>
        </div>

        <div class="lb__sheet">
          <div v-if="prompt" class="lb__prompt-wrap">
            <p class="lb__prompt">{{ prompt }}</p>
            <button
              data-test="copy-prompt"
              class="lb__copy"
              @click="copyText(prompt)"
            >
              Copy prompt
            </button>
          </div>
          <div class="lb__chips">
            <span class="lb__chip">{{ modelLabel }}</span>
            <span class="lb__chip">seed {{ seed ?? "—" }}</span>
            <button
              v-if="seed != null"
              data-test="copy-seed"
              class="lb__chip"
              @click="copyText(String(seed))"
            >
              Copy seed
            </button>
            <span v-if="dimensions" class="lb__chip">{{ dimensions }}</span>
            <span class="lb__chip">{{ hostLabel }}</span>
            <span class="lb__chip">{{ steps ?? "—" }} steps</span>
            <span class="lb__chip">CFG {{ guidance ?? "—" }}</span>
            <span class="lb__chip">{{ formatLabel }}</span>
          </div>
          <p v-if="originalPrompt" class="lb__mobile-meta">
            <b>Original:</b> {{ originalPrompt }}
          </p>
          <p v-if="negativePrompt" class="lb__mobile-meta">
            <b>Negative:</b> {{ negativePrompt }}
          </p>
          <p v-if="loraLabel" class="lb__mobile-meta">
            <b>LoRA:</b> {{ loraLabel }}
          </p>
          <button class="lb__reuse" @click="onReuse">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path d="M3 12a9 9 0 019-9 9 9 0 016.7 3M21 3v6h-6" />
              <path d="M21 12a9 9 0 01-9 9 9 9 0 01-6.7-3M3 21v-6h6" />
            </svg>
            Reuse these settings
          </button>
          <div class="lb__pair">
            <button class="lb__quiet" @click="onUseSource">
              Use as source
            </button>
            <a
              v-if="mediaSrc"
              class="lb__quiet"
              :href="mediaSrc"
              :download="item.filename"
              role="button"
            >
              Save
            </a>
            <span v-else class="lb__quiet lb__quiet--off">Save</span>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
/* Backdrop is contained to the app frame (spec §05); fixed so the viewer
 * stays centered regardless of how far the gallery is scrolled. */
.lb {
  position: fixed;
  inset: 0;
  z-index: 60;
  background: rgba(6, 5, 10, 0.82);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
}
.lb--wide {
  padding: 40px;
}

/* ── Desktop two-pane ─────────────────────────────────────────────────── */
.lb__card {
  display: flex;
  width: 100%;
  max-width: 1000px;
  max-height: 86vh;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  overflow: hidden;
  box-shadow: var(--shadow-float);
}

.lb__stage {
  flex: 1;
  min-width: 0;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--print);
  padding: 20px;
}
.lb__media {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}

/* Shown instead of the media when the owning host can't serve it — a video on
   a host too old to issue a streaming ticket, or an unreachable host. */
.lb__blocked {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-width: 320px;
  margin: 0;
  padding: 22px;
  text-align: center;
  color: var(--on-media);
}
.lb__blocked-title {
  font-family: var(--f-display);
  font-size: 15px;
  font-weight: 700;
}
.lb__blocked-body {
  font-size: 13px;
  line-height: 1.45;
  opacity: 0.78;
}

.lb__quiet--off {
  opacity: 0.45;
  cursor: default;
}

.lb__nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 0;
  background: rgba(0, 0, 0, 0.5);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}
.lb__nav svg {
  width: 22px;
  height: 22px;
}
.lb__nav:hover:not(:disabled) {
  background: rgba(0, 0, 0, 0.68);
}
.lb__nav:disabled {
  opacity: 0.25;
  cursor: default;
}
.lb__nav--prev {
  left: 14px;
}
.lb__nav--next {
  right: 14px;
}
.lb__nav:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.lb__panel {
  width: 320px;
  flex: 0 0 320px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow-y: auto;
}

.lb__phead {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 18px;
}
.lb__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.lb__flex {
  flex: 1;
}
.lb__pos {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
.lb__pos--light {
  color: var(--on-media);
}

.lb__x,
.lb__morebtn {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 0;
  background: color-mix(in srgb, var(--rebate) 9%, transparent);
  color: var(--ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}
.lb__x svg,
.lb__morebtn svg {
  width: 16px;
  height: 16px;
}
.lb__x:hover,
.lb__morebtn:hover {
  color: var(--rebate);
}
.lb__x:focus-visible,
.lb__morebtn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.lb__more {
  position: relative;
}
.lb__menu-scrim {
  position: fixed;
  inset: 0;
  z-index: 1;
  border: 0;
  background: transparent;
  cursor: default;
}
.lb__menu {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  z-index: 2;
  min-width: 150px;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-control);
  box-shadow: var(--shadow-raised);
  padding: 5px;
  display: flex;
  flex-direction: column;
}
.lb__menu--right {
  left: auto;
  right: 0;
}
.lb__menu button {
  text-align: left;
  border: 0;
  background: transparent;
  color: var(--rebate);
  padding: 9px 11px;
  border-radius: var(--radius-control-sm);
  font-size: 13px;
  cursor: pointer;
}
.lb__menu button:hover {
  background: var(--surface);
}
.lb__menu-danger {
  color: var(--stop);
}

.lb__prompt {
  font-size: 14px;
  line-height: 1.5;
  color: var(--rebate);
  margin: 0 0 20px;
  white-space: pre-wrap;
}
.lb__prompt-wrap {
  margin-bottom: 20px;
}
.lb__prompt-wrap .lb__prompt {
  margin-bottom: 7px;
}
.lb__copy,
.lb__rowcopy {
  border: 0;
  padding: 0;
  background: transparent;
  color: var(--safelight);
  font: inherit;
  cursor: pointer;
}
.lb__copy {
  font-family: var(--f-mono);
  font-size: 10.5px;
}
.lb__rowcopy {
  text-align: right;
}
.lb__copy:focus-visible,
.lb__rowcopy:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.lb__prompt--empty {
  color: var(--ink-3);
  font-style: italic;
}

.lb__rows {
  display: flex;
  flex-direction: column;
  gap: 11px;
  font-family: var(--f-mono);
  font-size: 11.5px;
}
.lb__row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}
.lb__row span:last-child {
  color: var(--rebate);
  text-align: right;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.lb__row--long span:last-child {
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
}
.lb__rowk {
  color: var(--ink-3);
  flex: 0 0 auto;
}

.lb__reuse {
  width: 100%;
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 13px;
  border-radius: var(--radius-control-lg);
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  cursor: pointer;
}
.lb__reuse svg {
  width: 15px;
  height: 15px;
}
.lb__reuse:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.lb__pair {
  display: flex;
  gap: 9px;
}
.lb__quiet {
  flex: 1;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control);
  font-size: 12.5px;
  font-weight: 600;
  text-align: center;
  text-decoration: none;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}
.lb__quiet:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
  color: var(--rebate);
}
.lb__quiet:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* ── Mobile full-screen ───────────────────────────────────────────────── */
.lb--full {
  padding: 0;
  background: var(--print);
  backdrop-filter: none;
  display: block;
}
.lb__full {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
}
.lb__topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: max(16px, env(safe-area-inset-top)) 16px 10px;
  color: var(--on-media);
}
.lb__circle {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 0;
  background: rgba(255, 255, 255, 0.16);
  color: var(--on-media);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  text-decoration: none;
}
.lb__circle svg {
  width: 18px;
  height: 18px;
}
.lb__circle:focus-visible {
  outline: 2px solid var(--on-media);
  outline-offset: 2px;
}

.lb__stage--full {
  flex: 1;
  min-height: 0;
  padding: 6px 10px;
  background: transparent;
}
.lb__stage--full .lb__nav {
  width: 36px;
  height: 36px;
  background: rgba(0, 0, 0, 0.45);
}
.lb__stage--full .lb__nav svg {
  width: 20px;
  height: 20px;
}
.lb__stage--full .lb__nav--prev {
  left: 8px;
}
.lb__stage--full .lb__nav--next {
  right: 8px;
}

.lb__sheet {
  background: var(--bench);
  border-top: 1px solid var(--edge);
  border-radius: 20px 20px 0 0;
  padding: 17px 18px max(24px, env(safe-area-inset-bottom));
  color: var(--rebate);
}
.lb__sheet .lb__prompt {
  font-size: 14px;
  margin-bottom: 13px;
}
.lb__sheet .lb__prompt-wrap {
  margin-bottom: 13px;
}
.lb__mobile-meta {
  margin: 0 0 8px;
  color: var(--ink-2);
  font-size: 12px;
  line-height: 1.4;
}
.lb__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 16px;
  font-family: var(--f-mono);
  font-size: 10.5px;
}
.lb__chip {
  border: 1px solid var(--edge);
  background: var(--bath);
  color: var(--ink-2);
  padding: 4px 9px;
  border-radius: var(--radius-pill);
}
.lb__sheet .lb__reuse {
  border-radius: var(--radius-card);
}
.lb__sheet .lb__quiet {
  border-radius: var(--radius-card);
}
</style>
