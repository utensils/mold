<script setup lang="ts">
/*
 * Model detail drawer (spec §03/§06, prototype MODEL DETAIL DRAWER lines
 * 605-643). One drawer serves both Models tabs off the shared `useCatalog`
 * detail union: a Discover (catalog) row pulls a variant; an Installed model
 * loads into memory, unloads, or deletes. 452px @ui DrawerPanel on the web
 * breakpoint, full-screen SheetPanel on phones — both render inside the
 * app frame, never a fixed layer.
 */
import { computed, markRaw, onBeforeUnmount, onMounted, ref, watch } from "vue";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import Icon from "@ui/components/Icon.vue";
import { useCatalog } from "../../composables/useCatalog";
import { requestConfirm, toast } from "../../lib/toasts";
import type { CatalogEntryWire, ModelInfoExtended } from "../../types";

const cat = useCatalog();

/**
 * One explicit state for the whole drawer, so the template's branches cover
 * every case by construction. `unrenderable` is the terminal branch: a detail
 * that is set but carries nothing we can paint (a wire shape we don't know, a
 * missing entry/model) used to fall through every `v-if` and leave an open,
 * completely empty panel — the bug users photographed twice.
 */
type DrawerState =
  "closed" | "loading" | "error" | "catalog" | "installed" | "unrenderable";

const state = computed<DrawerState>(() => {
  if (cat.detailLoadingId.value != null) return "loading";
  if (cat.detailError.value != null) return "error";
  const d = cat.detail.value;
  if (d == null) return "closed";
  if (d.kind === "catalog" && isObject(d.entry)) return "catalog";
  if (d.kind === "installed" && isObject(d.model)) return "installed";
  return "unrenderable";
});

function isObject(v: unknown): boolean {
  return typeof v === "object" && v !== null;
}

/** The catalog entry, only when it is actually renderable. */
const entry = computed<CatalogEntryWire | null>(() =>
  state.value === "catalog"
    ? (cat.detail.value as { entry: CatalogEntryWire }).entry
    : null,
);

// The drawer opens for content AND for the loading / error non-content states
// so a Discover row whose detail fetch is slow or failing never shows a blank
// panel (spec G4).
const open = computed(() => state.value !== "closed");

// Phone surfaces get the full SheetPanel; everything else the right drawer.
const isPhone = ref(false);
let mq: MediaQueryList | null = null;
function onMediaChange(e: MediaQueryListEvent) {
  isPhone.value = e.matches;
}
onMounted(() => {
  mq = window.matchMedia?.("(max-width: 639px)") ?? null;
  if (mq) {
    isPhone.value = mq.matches;
    mq.addEventListener?.("change", onMediaChange);
  }
});
onBeforeUnmount(() => mq?.removeEventListener?.("change", onMediaChange));

const panelComponent = computed(() =>
  markRaw(isPhone.value ? SheetPanel : DrawerPanel),
);
const panelProps = computed(() =>
  isPhone.value ? { variant: "full" as const } : { width: 452 },
);

function formatGB(bytes: number | null | undefined): string {
  if (!bytes) return "—";
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

/** Compact counts for metadata rows: 999 → "999", 12_300 → "12.3k". */
function formatCount(count: number): string {
  const compact = (value: number, suffix: string) =>
    `${value.toFixed(1).replace(/\.0$/, "")}${suffix}`;
  if (count >= 1_000_000) return compact(count / 1_000_000, "M");
  if (count >= 1_000) return compact(count / 1_000, "k");
  return String(count);
}

/** Catalog timestamps are unix seconds; unparseable values drop the row. */
function formatDate(unixSeconds: number): string {
  const d = new Date(unixSeconds * 1000);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

// ── Normalized view over the detail union ──────────────────────────────
const isInstalled = computed(() => state.value === "installed");
const installedModel = computed<ModelInfoExtended | null>(() =>
  state.value === "installed"
    ? (cat.detail.value as { model: ModelInfoExtended }).model
    : null,
);
const isLoaded = computed(() => installedModel.value?.is_loaded ?? false);

const mediaLabel = computed(() => {
  if (entry.value) return entry.value.modality;
  const m = installedModel.value;
  if (!m) return "";
  return /ltx/i.test(m.family ?? "") ? "video" : "image";
});

const name = computed(
  () => entry.value?.name ?? installedModel.value?.name ?? "",
);

const family = computed(
  () => entry.value?.family ?? installedModel.value?.family ?? "",
);

const description = computed(
  () => entry.value?.description ?? installedModel.value?.description ?? "",
);

// ── Hero preview ───────────────────────────────────────────────────────
// Catalog thumbnails are public CDN URLs (the server has no proxy route), so
// they load direct. Anything that fails — hotlink block, mixed content, a
// dead asset — hides the block instead of showing a broken-image glyph.
const VIDEO_URL_RE = /\.(mp4|webm|mov|m4v)(\?|#|$)/i;

const heroUrl = computed<string | null>(() => {
  const raw = entry.value?.thumbnail_url;
  return typeof raw === "string" && raw.trim() !== "" ? raw : null;
});
const heroIsVideo = computed(() =>
  heroUrl.value ? VIDEO_URL_RE.test(heroUrl.value) : false,
);
const heroFailed = ref(false);
watch(heroUrl, () => {
  heroFailed.value = false;
});

// Checkpoint weights = SIZE (model weights only, per catalog semantics).
const checkpointBytes = computed<number | null>(() => {
  const e = entry.value;
  if (e) return e.size_bytes ?? null;
  const m = installedModel.value;
  return m?.size_gb ? m.size_gb * 1_000_000_000 : null;
});

// Full footprint = FETCH (weights + every shared companion; >= SIZE).
const footprintBytes = computed<number | null>(() => {
  const e = entry.value;
  if (!e) {
    const m = installedModel.value;
    if (!m) return null;
    return m.disk_usage_bytes ?? (m.size_gb ? m.size_gb * 1_000_000_000 : null);
  }
  // Defensive optional chaining: a malformed entry (null download_recipe /
  // files) must never throw here — a render exception is exactly what blanks
  // the whole drawer.
  const primary = (e.download_recipe?.files ?? []).reduce(
    (sum, f) => sum + (f?.size_bytes ?? 0),
    0,
  );
  const companions = (e.companion_details ?? []).reduce(
    (sum, c) => sum + (c?.size_bytes ?? 0),
    0,
  );
  const total = primary + companions;
  return total > 0 ? total : (e.size_bytes ?? null);
});

const components = computed<string[]>(() => {
  const e = entry.value;
  if (e) return (e.companion_details ?? []).map((c) => c.name);
  const d = cat.detail.value;
  if (state.value !== "installed") return [];
  return ((d as { components?: { name: string }[] }).components ?? []).map(
    (c) => c.name,
  );
});

const variants = computed(() =>
  state.value === "catalog"
    ? ((cat.detail.value as { variants?: { id: string; label: string }[] })
        .variants ?? [])
    : [],
);

/** Trigger phrases — Civitai LoRAs are unusable without them. */
const trainedWords = computed<string[]>(() =>
  (entry.value?.trained_words ?? []).filter(
    (w) => typeof w === "string" && w.trim() !== "",
  ),
);

const pageUrl = computed<string | null>(() => {
  const raw = entry.value?.page_url;
  return typeof raw === "string" && /^https?:\/\//i.test(raw) ? raw : null;
});

const source = computed(() => {
  const e = entry.value;
  if (e) return e.source === "civitai" ? "Civitai" : "Hugging Face";
  return installedModel.value?.hf_repo || "local";
});

/** Key/value rows. A field the server didn't send is omitted outright — a
 *  column of em dashes reads as a broken panel, which is how this drawer
 *  got reported in the first place. */
interface MetaRow {
  key: string;
  val: string;
}

const metaRows = computed<MetaRow[]>(() => {
  const rows: MetaRow[] = [];
  const e = entry.value;
  if (e) {
    if (e.author) rows.push({ key: "Author", val: e.author });
    if (e.kind) rows.push({ key: "Kind", val: e.kind });
    if (e.file_format) rows.push({ key: "Format", val: e.file_format });
    if (e.download_count)
      rows.push({ key: "Downloads", val: formatCount(e.download_count) });
    if (e.likes) rows.push({ key: "Likes", val: formatCount(e.likes) });
    if (typeof e.rating === "number")
      rows.push({ key: "Rating", val: `${e.rating.toFixed(1)} / 5` });
    if (e.updated_at) {
      const val = formatDate(e.updated_at);
      if (val) rows.push({ key: "Updated", val });
    }
    rows.push({ key: "Source", val: source.value });
    if (e.license) rows.push({ key: "License", val: e.license });
    return rows;
  }
  if (installedModel.value) rows.push({ key: "Source", val: source.value });
  return rows;
});

// ── Discover (catalog) pull ────────────────────────────────────────────
const selectedVariantId = ref<string | null>(null);
watch(
  () => cat.detail.value,
  () => {
    const e = entry.value;
    selectedVariantId.value = e ? (variants.value[0]?.id ?? e.id) : null;
  },
  { immediate: true },
);

const canPull = computed(() => {
  const e = entry.value;
  return e ? cat.canDownload(e) : false;
});

const isRepair = computed(() => entry.value?.installed === true);

async function handlePull() {
  const e = entry.value;
  if (!e) return;
  const target = isRepair.value ? e.id : (selectedVariantId.value ?? e.id);
  await cat.startDownload(target);
  cat.closeDetail();
}

// ── Installed model actions ────────────────────────────────────────────
const busy = ref(false);

async function handleLoad() {
  const m = installedModel.value;
  if (!m || busy.value) return;
  busy.value = true;
  try {
    await cat.loadInstalled(m.name);
    toast("success", "loaded into memory");
  } catch (e: unknown) {
    toast("error", e instanceof Error ? e.message : "load failed");
  } finally {
    busy.value = false;
  }
}

async function handleUnload() {
  const m = installedModel.value;
  if (!m || busy.value) return;
  busy.value = true;
  try {
    await cat.unloadInstalled(m.name);
    toast("success", "unloaded");
  } catch (e: unknown) {
    toast("error", e instanceof Error ? e.message : "unload failed");
  } finally {
    busy.value = false;
  }
}

async function handleDelete() {
  const m = installedModel.value;
  if (!m || busy.value) return;
  const modelName = m.name;
  const ok = await requestConfirm({
    title: "Delete model?",
    body: `Remove ${modelName} and its files from disk. Shared components used by other models are kept.`,
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  busy.value = true;
  try {
    const result = await cat.deleteInstalled(modelName);
    toast("success", `deleted — freed ${formatGB(result.freed_bytes)}`);
  } catch (e: unknown) {
    toast("error", e instanceof Error ? e.message : "delete failed");
  } finally {
    busy.value = false;
  }
}

function onClose() {
  cat.closeDetail();
}

function onRetry() {
  const id = cat.detailError.value?.id;
  if (id) cat.retryDetail(id);
}
</script>

<template>
  <!-- Viewport-fixed host: DrawerPanel/SheetPanel are `position: absolute;
       inset: 0` so they fill their nearest positioned ancestor by design. The
       Models page is a long scrolling column, so mounting the panel inline
       drew it at the TOP OF THE PAGE — scroll down to a card, open it, and the
       drawer renders off-screen above the viewport while only its backdrop
       shows. Pinning the host to the viewport puts the panel where the user is
       looking, at every scroll offset. Same fix the Create Advanced sheet
       needed. -->
  <div v-if="open" class="fixed inset-0 z-40" data-test="detail-drawer-host">
    <component
      :is="panelComponent"
      v-bind="panelProps"
      :open="open"
      title="Model details"
      @close="onClose"
    >
      <div
        v-if="state === 'loading'"
        class="md md--state"
        data-test="detail-loading"
      >
        <div class="md__spinner" aria-hidden="true" />
        <p class="md__state-msg">loading model details…</p>
      </div>
      <div
        v-else-if="state === 'error'"
        class="md md--state"
        data-test="detail-error"
      >
        <div class="md__name">Model details</div>
        <p class="md__state-msg">{{ cat.detailError.value?.message }}</p>
        <button
          type="button"
          class="md__ghost"
          data-test="detail-retry"
          @click="onRetry"
        >
          <Icon name="refresh" :size="14" />
          Try again
        </button>
      </div>
      <div
        v-else-if="state === 'catalog' || state === 'installed'"
        class="md"
        data-test="detail-content"
      >
        <!-- Hero preview: public CDN thumbnail, hidden outright when absent or
           unloadable so a broken-image glyph never leads the panel. -->
        <div
          v-if="heroUrl && !heroFailed"
          class="md__hero"
          data-test="detail-hero"
        >
          <video
            v-if="heroIsVideo"
            :src="heroUrl"
            class="md__hero-media"
            autoplay
            loop
            muted
            playsinline
            preload="metadata"
            @error="heroFailed = true"
          />
          <img
            v-else
            :src="heroUrl"
            alt=""
            loading="lazy"
            decoding="async"
            class="md__hero-media"
            @error="heroFailed = true"
          />
        </div>

        <span class="md__media">{{ mediaLabel }}</span>
        <div class="md__name">{{ name || "Untitled model" }}</div>
        <div v-if="family" class="md__fam">{{ family }}</div>
        <p v-if="description" class="md__desc">{{ description }}</p>

        <div class="md__tiles">
          <div class="md__tile">
            <div class="md__tile-key">Checkpoint weights</div>
            <div class="md__tile-val" data-test="tile-checkpoint">
              {{ formatGB(checkpointBytes) }}
            </div>
          </div>
          <div class="md__tile">
            <div class="md__tile-key">Full footprint</div>
            <div class="md__tile-val" data-test="tile-footprint">
              {{ formatGB(footprintBytes) }}
            </div>
          </div>
        </div>

        <template v-if="components.length">
          <div class="md__kicker">Components</div>
          <div class="md__chips" data-test="components">
            <span v-for="c in components" :key="c" class="md__chip">{{
              c
            }}</span>
          </div>
        </template>

        <template v-if="variants.length">
          <div class="md__kicker">Variants</div>
          <div class="md__chips" data-test="variants">
            <button
              v-for="v in variants"
              :key="v.id"
              type="button"
              class="md__variant"
              :data-on="selectedVariantId === v.id ? 'true' : undefined"
              @click="selectedVariantId = v.id"
            >
              {{ v.label }}
            </button>
          </div>
        </template>

        <template v-if="trainedWords.length">
          <div class="md__kicker">Trigger words</div>
          <div class="md__chips" data-test="trained-words">
            <span v-for="w in trainedWords" :key="w" class="md__chip">
              {{ w }}
            </span>
          </div>
        </template>

        <div v-if="metaRows.length" class="md__rows" data-test="meta-rows">
          <div v-for="row in metaRows" :key="row.key" class="md__row">
            <span class="md__row-key">{{ row.key }}</span>
            <span class="md__row-val">{{ row.val }}</span>
          </div>
        </div>

        <a
          v-if="pageUrl"
          class="md__link"
          :href="pageUrl"
          target="_blank"
          rel="noopener noreferrer"
          data-test="page-link"
        >
          View on {{ source }}
          <svg
            viewBox="0 0 12 12"
            width="11"
            height="11"
            fill="none"
            stroke="currentColor"
            stroke-width="1.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="M8.5 6.75v2.75a1 1 0 0 1-1 1H2.5a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1h2.75"
            />
            <path d="M7 1.5h3.5V5" />
            <path d="M10.5 1.5 5.75 6.25" />
          </svg>
        </a>

        <!-- Installed model: load / unload / delete -->
        <template v-if="isInstalled">
          <button
            type="button"
            class="md__action"
            data-test="load-btn"
            :disabled="isLoaded || busy"
            @click="handleLoad"
          >
            <Icon name="download" :size="15" />
            {{ isLoaded ? "loaded into memory" : "Load into memory" }}
          </button>
          <div class="md__secondary">
            <button
              type="button"
              class="md__ghost"
              data-test="unload-btn"
              :disabled="!isLoaded || busy"
              @click="handleUnload"
            >
              Unload
            </button>
            <button
              type="button"
              class="md__ghost md__ghost--danger"
              data-test="delete-btn"
              :disabled="isLoaded || busy"
              :title="
                isLoaded
                  ? 'Unload before deleting'
                  : 'Delete this model from disk'
              "
              @click="handleDelete"
            >
              Delete
            </button>
          </div>
        </template>

        <!-- Discover (catalog) row: pull / repair -->
        <button
          v-else
          type="button"
          class="md__action"
          :data-test="isRepair ? 'repair-btn' : 'pull-btn'"
          :disabled="!canPull"
          :title="canPull ? undefined : 'Unsupported catalog package'"
          @click="handlePull"
        >
          <Icon v-if="!isRepair" name="download" :size="15" />
          {{ isRepair ? "Repair" : "Pull" }}
        </button>
      </div>
      <!-- Terminal branch. Reached only when the drawer is open with a detail we
         can't render — the panel says so and offers a way out instead of
         painting nothing. -->
      <div v-else class="md md--state" data-test="detail-unrenderable">
        <div class="md__name">Model details</div>
        <p class="md__state-msg">
          This model's details came back in a shape we can't read. Try opening
          it again, or check the server version.
        </p>
        <button
          type="button"
          class="md__ghost"
          data-test="detail-close"
          @click="onClose"
        >
          Close
        </button>
      </div>
    </component>
  </div>
</template>

<style scoped>
.md {
  color: var(--rebate);
}

.md__hero {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 10;
  overflow: hidden;
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  background: var(--bath);
  margin-bottom: 16px;
}

.md__hero-media {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.md__media {
  display: inline-block;
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--halide);
  border: 1px solid color-mix(in srgb, var(--halide) 40%, transparent);
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  margin-bottom: 12px;
}

.md__name {
  font-family: var(--f-display);
  font-size: 23px;
  font-weight: 700;
  letter-spacing: -0.01em;
  word-break: break-word;
}

.md__fam {
  font-family: var(--f-mono);
  font-size: 12px;
  color: var(--ink-3);
  margin-top: 4px;
}

.md__desc {
  font-size: 14px;
  color: var(--ink-2);
  line-height: 1.55;
  margin: 16px 0 20px;
}

.md__tiles {
  display: flex;
  gap: 10px;
  margin: 20px 0 22px;
}

.md__tile {
  flex: 1;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  padding: 13px;
}

.md__tile-key {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.md__tile-val {
  font-family: var(--f-mono);
  font-size: 17px;
  margin-top: 5px;
}

.md__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  margin-bottom: 10px;
}

.md__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-bottom: 22px;
}

.md__chip {
  border: 1px solid var(--edge);
  background: var(--bath);
  color: var(--ink-2);
  padding: 6px 11px;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-family: var(--f-mono);
}

.md__variant {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 6px 13px;
  border-radius: var(--radius-pill);
  font-size: 12px;
  font-family: var(--f-mono);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease),
    box-shadow var(--dur-quick) var(--ease);
}

.md__variant[data-on="true"] {
  border-color: var(--sel-border);
  color: var(--sel-ink);
  background: var(--sel-bg);
  box-shadow: var(--sel-ring);
}

.md__variant:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.md__rows {
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  padding: 4px 14px;
  margin-bottom: 22px;
}

.md__row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 11px 0;
  border-bottom: 1px solid var(--edge);
  font-size: 12.5px;
}

.md__row:last-child {
  border-bottom: 0;
}

.md__link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 22px;
  font-size: 12.5px;
  color: var(--safelight);
  text-decoration: none;
}

.md__link:hover {
  text-decoration: underline;
}

.md__link:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.md__row-key {
  color: var(--ink-3);
}

.md__row-val {
  font-family: var(--f-mono);
  font-size: 11px;
  text-align: right;
  word-break: break-all;
}

.md__action {
  width: 100%;
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 13px;
  border-radius: var(--radius-control-lg);
  font-family: var(--f-body);
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: filter var(--dur-quick) var(--ease);
}

.md__action:hover:not(:disabled) {
  filter: brightness(1.05);
}

.md__action:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.md__action:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.md__secondary {
  display: flex;
  gap: 9px;
  margin-top: 9px;
}

.md__ghost {
  flex: 1;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.md__ghost:hover:not(:disabled) {
  border-color: var(--ink-3);
  color: var(--rebate);
}

.md__ghost--danger:hover:not(:disabled) {
  border-color: var(--stop);
  color: var(--stop);
}

.md__ghost:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.md__ghost:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* ── Non-content states (loading / error) — never a blank panel ────────── */
.md--state {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 12px;
  padding-top: 8px;
}
.md--state .md__ghost {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.md__state-msg {
  margin: 0;
  font-size: 13px;
  color: var(--ink-3);
  line-height: 1.5;
}
.md__spinner {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 2px solid var(--ce);
  border-top-color: var(--safelight);
  animation: md-spin 0.7s linear infinite;
}
@keyframes md-spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .md__spinner {
    animation: none;
  }
}
</style>
