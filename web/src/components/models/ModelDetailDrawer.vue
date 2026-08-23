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
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import { modelKindValue, modelWeightsLabel } from "@studio/lib/modelMetadata";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import SheetPanel from "@ui/components/SheetPanel.vue";
import Icon from "@ui/components/Icon.vue";
import { useCatalog } from "../../composables/useCatalog";
import { useModelInstallTargets } from "../../composables/useModelInstallTargets";
import { useOverlayFocus } from "../../composables/useOverlayFocus";
import { modelDisplayName } from "@studio/lib/modelDisplay";
import {
  modelRuntimeNotice,
  modelRuntimeNoticeForId,
} from "@studio/lib/modelRuntimeAvailability";
import { requestConfirm, toast } from "../../lib/toasts";
import type {
  CatalogEntryWire,
  ModelComponentStatus,
  ModelInfoExtended,
} from "../../types";
import { formatGB } from "../../util/format";

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
const installedCatalogEntry = computed<CatalogEntryWire | null>(() =>
  state.value === "installed"
    ? ((cat.detail.value as { catalogEntry?: CatalogEntryWire | null })
        .catalogEntry ?? null)
    : null,
);
/** Catalog metadata used for presentation. Installed actions and component
 * status remain keyed to the `/api/models` row. */
const metadataEntry = computed(
  () => entry.value ?? installedCatalogEntry.value,
);

// The drawer opens for content AND for the loading / error non-content states
// so a Discover row whose detail fetch is slow or failing never shows a blank
// panel (spec G4).
const open = computed(() => state.value !== "closed");
const host = ref<HTMLElement | null>(null);
const { onKeydown } = useOverlayFocus(open, host, () => onClose());

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
/** `runtime_available: false` (download-only rows such as the NVFP4 H3
 * partitions) means the server rejects every load/generate attempt with a
 * 501 — never offer the load action, and say why instead of a toast.
 *
 * A Discover row is answered too, from the host's own `/api/models` listing
 * (#1276): these checkpoints are 21-42 GB, so "this machine cannot run it"
 * has to arrive before the Pull button, not after the download. The pull
 * itself stays available — the whole point is that the model is downloadable
 * and not runnable. */
const runtimeNotice = computed(
  () =>
    modelRuntimeNotice(installedModel.value) ??
    modelRuntimeNoticeForId(entry.value?.id, cat.availableManifests.value),
);
const runtimeUnavailable = computed(() => runtimeNotice.value !== null);

const mediaLabel = computed(() => {
  if (metadataEntry.value) return metadataEntry.value.modality;
  const m = installedModel.value;
  if (!m) return "";
  if (m.modality?.trim()) return m.modality;
  return /ltx/i.test(m.family ?? "") ? "video" : "image";
});

const name = computed(
  () =>
    entry.value?.name ??
    (installedModel.value ? modelDisplayName(installedModel.value) : ""),
);

const family = computed(
  () => metadataEntry.value?.family ?? installedModel.value?.family ?? "",
);

const kind = computed(() =>
  modelKindValue({
    kind: metadataEntry.value?.kind ?? installedModel.value?.kind,
    family: family.value,
  }),
);
const nsfw = computed(
  () => metadataEntry.value?.nsfw ?? installedModel.value?.nsfw ?? false,
);
const weightsLabel = computed(() => modelWeightsLabel(kind.value));
const description = computed(() => {
  const raw = (
    metadataEntry.value?.description ??
    installedModel.value?.description ??
    ""
  ).trim();
  if (!raw || metadataEntry.value) return raw;

  // Older servers synthesized installed catalog descriptions from the display
  // title (`Title` / `Title by Author`). Repeating that directly under the
  // title looks like broken metadata; preserve only genuinely descriptive
  // copy while catalog entries continue to render their upstream description.
  const title = name.value.trim().toLocaleLowerCase();
  const normalized = raw.toLocaleLowerCase();
  const bylinePrefix = `${title} by `;
  if (
    normalized === title ||
    (normalized.startsWith(bylinePrefix) &&
      normalized.length > bylinePrefix.length)
  ) {
    return "";
  }
  return raw;
});

// ── Hero preview ───────────────────────────────────────────────────────
// Catalog thumbnails are public CDN URLs (the server has no proxy route), so
// they load direct. Anything that fails — hotlink block, mixed content, a
// dead asset — hides the block instead of showing a broken-image glyph.
const VIDEO_URL_RE = /\.(mp4|webm|mov|m4v)(\?|#|$)/i;

const heroUrl = computed<string | null>(() => {
  const raw = metadataEntry.value?.thumbnail_url;
  return typeof raw === "string" && raw.trim() !== "" ? raw : null;
});
const heroIsVideo = computed(() =>
  heroUrl.value ? VIDEO_URL_RE.test(heroUrl.value) : false,
);
const heroFailed = ref(false);
watch(heroUrl, () => {
  heroFailed.value = false;
});

// Model weights = SIZE (primary weights only, per catalog semantics).
const modelWeightsBytes = computed<number | null>(() => {
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

const components = computed<ModelComponentStatus[]>(() => {
  const e = entry.value;
  if (e)
    return (e.companion_details ?? []).map((component) => ({
      kind: component.kind,
      name: component.name,
      present: e.installed,
    }));
  const d = cat.detail.value;
  if (state.value !== "installed") return [];
  return (d as { components?: ModelComponentStatus[] }).components ?? [];
});
const componentStatusAvailable = computed(() => state.value === "installed");

const variants = computed(() =>
  state.value === "catalog"
    ? ((cat.detail.value as { variants?: { id: string; label: string }[] })
        .variants ?? [])
    : [],
);

/** Trigger phrases — Civitai LoRAs are unusable without them. */
const trainedWords = computed<string[]>(() =>
  (metadataEntry.value?.trained_words ?? []).filter(
    (w) => typeof w === "string" && w.trim() !== "",
  ),
);

/** Search tags are useful discovery context, but keep the panel scannable. */
const tags = computed<string[]>(() => {
  const unique = new Set<string>();
  for (const raw of metadataEntry.value?.tags ?? []) {
    if (typeof raw !== "string") continue;
    const tag = raw.trim();
    if (tag) unique.add(tag);
  }
  return [...unique];
});
const visibleTags = computed(() => tags.value.slice(0, 8));
const hiddenTagCount = computed(() =>
  Math.max(0, tags.value.length - visibleTags.value.length),
);

const pageUrl = computed<string | null>(() => {
  const raw = metadataEntry.value?.page_url;
  return typeof raw === "string" && /^https?:\/\//i.test(raw) ? raw : null;
});

const source = computed(() => {
  const e = metadataEntry.value;
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
  const e = metadataEntry.value;
  if (e) {
    if (e.author) rows.push({ key: "Author", val: e.author });
    rows.push({
      key: "Role",
      val: e.family_role === "foundation" ? "Foundation" : "Fine-tune",
    });
    if (e.sub_family) rows.push({ key: "Variant", val: e.sub_family });
    rows.push({
      key: "Package",
      val: e.bundling === "single-file" ? "Single file" : "Separated files",
    });
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

/*
 * Install targeting. `installed` is one machine's answer, so the drawer asks
 * the shared plan instead: the action stays a Pull while any reachable machine
 * lacks the model, and an installed model this server has is still installable
 * on the machine next to it.
 */
const installTargets = useModelInstallTargets();
const installPlan = computed(() => {
  const e = entry.value;
  if (e) return installTargets.planFor(e.id, e.installed);
  const m = installedModel.value;
  return installTargets.planFor(m?.name ?? "", true);
});
const isRepair = computed(() => installPlan.value.label === "Repair");
/** The Installed segment's cross-machine install; hidden when nobody lacks it. */
const canInstallElsewhere = computed(
  () => isInstalled.value && installPlan.value.canInstall,
);

/** Resolve a machine, send the download there, and report what happened. */
async function runInstall(modelId: string, displayName: string) {
  const choice = await installTargets.chooseInstallTarget({
    modelId,
    displayName,
    ownedByOrigin: entry.value ? entry.value.installed : true,
  });
  if (choice.kind === "cancelled") return false;
  try {
    await installTargets.startDownloadOn(choice.target, modelId);
    toast(
      "success",
      installTargets.queuedMessage(
        choice.target,
        isRepair.value ? "repair" : "install",
      ),
    );
    return true;
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
    return false;
  }
}

async function handlePull() {
  const e = entry.value;
  if (!e) return;
  const modelId = isRepair.value ? e.id : (selectedVariantId.value ?? e.id);
  if (await runInstall(modelId, e.name)) cat.closeDetail();
}

async function handleInstallElsewhere() {
  const m = installedModel.value;
  if (!m || busy.value) return;
  await runInstall(m.name, modelDisplayName(m));
}

async function handleComponentRepair(component: ModelComponentStatus) {
  if (!component.repair_model) return;
  try {
    await cat.startDownload(component.repair_model);
    toast("success", `repairing ${component.name}`);
  } catch (e: unknown) {
    toast("error", e instanceof Error ? e.message : "repair failed");
  }
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
    body: `Remove ${modelDisplayName(m)} and its files from disk. Shared components used by other models are kept.`,
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
  <div
    v-if="open"
    ref="host"
    class="fixed inset-0 z-40"
    data-test="detail-drawer-host"
    @keydown="onKeydown"
  >
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

        <div class="md__classification" data-test="model-classification">
          <ModelMetadataBadges
            :kind="kind"
            :family="family"
            :modality="mediaLabel"
            :nsfw="nsfw"
          />
        </div>
        <div class="md__name">{{ name || "Untitled model" }}</div>
        <div v-if="family" class="md__fam">{{ family }}</div>
        <p v-if="description" class="md__desc">{{ description }}</p>

        <div class="md__tiles">
          <div class="md__tile">
            <div class="md__tile-key" data-test="tile-model-weights-label">
              {{ weightsLabel }}
            </div>
            <div class="md__tile-val" data-test="tile-model-weights">
              {{ formatGB(modelWeightsBytes) }}
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
            <span
              v-for="component in components"
              :key="`${component.kind}:${component.name}`"
              class="md__chip"
              :data-test="
                !componentStatusAvailable
                  ? 'component-included'
                  : component.present
                    ? 'component-ready'
                    : 'component-missing'
              "
            >
              {{ component.name }}
              <template v-if="componentStatusAvailable">
                · {{ component.present ? "ready" : "missing" }}
              </template>
              <button
                v-if="
                  componentStatusAvailable &&
                  !component.present &&
                  component.repair_model
                "
                type="button"
                class="md__chip-action"
                data-test="component-repair"
                @click="handleComponentRepair(component)"
              >
                Repair
              </button>
            </span>
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

        <template v-if="visibleTags.length">
          <div class="md__kicker">Tags</div>
          <div class="md__chips" data-test="model-tags">
            <span v-for="tag in visibleTags" :key="tag" class="md__chip">
              {{ tag }}
            </span>
            <span v-if="hiddenTagCount" class="md__chip">
              +{{ hiddenTagCount }} more
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
            v-if="!runtimeUnavailable"
            type="button"
            class="md__action"
            data-test="load-btn"
            :disabled="isLoaded || busy"
            @click="handleLoad"
          >
            <Icon name="download" :size="15" />
            {{ isLoaded ? "loaded into memory" : "Load into memory" }}
          </button>
          <p
            v-else-if="runtimeNotice"
            class="md__state-msg"
            data-test="runtime-unavailable-note"
          >
            {{ runtimeNotice.message }}
          </p>
          <div class="md__secondary">
            <button
              v-if="!runtimeUnavailable"
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
          <!-- Installed here, missing there: the install action has to stay
               reachable until every connected machine owns it. -->
          <button
            v-if="canInstallElsewhere"
            type="button"
            class="md__ghost md__ghost--wide"
            data-test="install-elsewhere-btn"
            :disabled="busy"
            @click="handleInstallElsewhere"
          >
            <Icon name="download" :size="14" />
            Install on another machine
          </button>
        </template>

        <!-- Discover (catalog) row: pull / repair -->
        <template v-else>
          <!-- Downloadable, not runnable. Said before the Pull, never as a
               toast after it, and it never disables the Pull. -->
          <p
            v-if="runtimeNotice"
            class="md__state-msg"
            data-test="runtime-unavailable-note"
          >
            {{ runtimeNotice.message }}
          </p>
          <button
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
        </template>
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

.md__classification {
  display: flex;
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

.md__chip-action {
  margin-left: 7px;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font: inherit;
  cursor: pointer;
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

.md__ghost--wide {
  width: 100%;
  margin-top: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
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
