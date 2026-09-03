<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import { modelKindValue, modelWeightsLabel } from "@studio/lib/modelMetadata";
import SourceGlyph from "../generate/SourceGlyph.vue";
import ModelFamilyPlaceholder from "./ModelFamilyPlaceholder.vue";
import { fetchCatalogDetail, startCatalogDownload } from "../../lib/api/catalog";
import { fetchModelComponents } from "../../lib/api/models";
import {
  buildDownloadContents,
  canDownloadEntry,
  catalogActionLabel,
  downloadContentsTotalBytes,
  mergeCatalogSummaryDetail,
} from "../../lib/catalogDetail";
import { catalogPageUrl, catalogSizeInfo } from "../../lib/catalog";
import { isVideoFamily } from "../../lib/capabilities";
import { catalogThumbnailUrl } from "../../lib/catalogThumbnails";
import { formatCount, formatGB } from "../../lib/format";
import { openExternal } from "../../lib/openExternal";
import { useToastStore } from "../../stores/toasts";
import { ApiError, type ApiTarget } from "../../lib/api/client";
import type { ModelSource } from "../../lib/modelSource";
import type { ModelRuntimeNotice } from "@studio/lib/modelRuntimeAvailability";
import type { CatalogEntry, ModelComponentStatus } from "../../lib/api/types";

/** A selectable pull variant (e.g. quantization); selecting one sets the exact
 *  id a Pull targets, honoring the manifest-variant precedence the list built. */
export interface DrawerVariant {
  id: string;
  label: string;
  sizeBytes?: number | null;
}

/**
 * In-app catalog detail: description, license, tags, modality/format, and
 * the itemized download contents (primary weights + shared companions with
 * per-file sizes and a computed total) — so a pull is an informed decision
 * without leaving for huggingface.co / civitai.com. Search-summary rows
 * arrive without the descriptive fields, so the drawer enriches its entry
 * via `GET /api/catalog/:id` on the same host the catalog list came from,
 * falling back to the summary when the detail fetch fails (older servers,
 * live rows, upstream hiccups).
 */
const props = defineProps<{
  entry: CatalogEntry;
  pulling: boolean;
  /** Host the catalog view is browsing; undefined = current primary. */
  target?: ApiTarget | undefined;
  forwardCredentials?: boolean | undefined;
  /** Selectable pull variants; the chosen chip is the exact pull target. */
  variants?: DrawerVariant[] | undefined;
  /**
   * Pull or Repair, decided by the owner of the host list: a model installed
   * on one machine is still a Pull for every machine that lacks it. Omitted
   * (single-machine callers) falls back to this entry's own install flag.
   */
  action?: "Pull" | "Repair" | undefined;
  /** This machine's own runtime answer for the model, resolved by the parent
   *  through `@studio/lib/modelRuntimeAvailability`. Rendered as an inline
   *  note above the action — before the pull, never as a toast after it —
   *  and it never disables the action: the model is downloadable, it just
   *  cannot generate here (#1276). */
  runtimeNotice?: ModelRuntimeNotice | null | undefined;
}>();
const emit = defineEmits<{
  (e: "close"): void;
  (e: "pull", entry: CatalogEntry): void;
  (e: "select-variant", id: string): void;
}>();

const detail = ref<CatalogEntry | null>(null);
const loading = ref(false);

/** Summary fields render immediately; the detail overlays as it arrives.
 *  `installed` never downgrades: the opener knows the model is on its host
 *  even when the detail endpoint (possibly another host) says otherwise.
 *  `thumbnail_url` never swaps: Civitai publishes several previews per model
 *  version and the detail endpoint need not pick the one the search listing
 *  picked, so letting it overlay would show a different image than the card
 *  the user just clicked. The detail's preview only fills a listing gap. */
const merged = computed<CatalogEntry>(() =>
  detail.value ? mergeCatalogSummaryDetail(props.entry, detail.value) : props.entry,
);

/** Same staleness guard for the descriptive detail fetch. */
let detailEpoch = 0;

async function loadDetail(): Promise<void> {
  const epoch = ++detailEpoch;
  detail.value = null;
  loading.value = true;
  try {
    const res = await fetchCatalogDetail(
      props.entry.id,
      props.forwardCredentials ?? false,
      props.target,
    );
    if (epoch === detailEpoch) detail.value = res;
  } catch {
    // Older server / live row not fetchable — the summary keeps the drawer
    // useful and the Pull action intact.
    if (epoch === detailEpoch) detail.value = null;
  } finally {
    if (epoch === detailEpoch) loading.value = false;
  }
}

watch(() => props.entry.id, loadDetail, { immediate: true });

// ── On-disk component state (installed models only) ───────────────────────

const components = ref<ModelComponentStatus[] | "loading" | "error" | null>(null);

/** Invalidates in-flight component fetches when a newer entry supersedes
 *  them — a slow host's late response must not show (or Repair) the
 *  previous model's components. */
let componentsEpoch = 0;

/** Per-component presence from the owning host; quietly absent elsewhere. */
async function loadComponents(): Promise<void> {
  const epoch = ++componentsEpoch;
  if (!merged.value.installed) {
    components.value = null;
    return;
  }
  components.value = "loading";
  try {
    const res = await fetchModelComponents(props.entry.id, props.target);
    if (epoch === componentsEpoch) components.value = res.components;
  } catch {
    // Not an installed manifest/catalog model on this host, or an older
    // server — the section simply hides.
    if (epoch === componentsEpoch) components.value = "error";
  }
}

watch([() => props.entry.id, () => merged.value.installed], loadComponents, { immediate: true });

const componentList = computed<ModelComponentStatus[]>(() =>
  Array.isArray(components.value) ? components.value : [],
);
const componentsPresent = computed(() => componentList.value.filter((c) => c.present).length);

/** Component rows currently mid-repair, keyed by component name. */
const repairing = ref<Set<string>>(new Set());

/**
 * Re-fetch one missing component without deleting the model: the download
 * queue is keyed on the server-provided `repair_model` and skips files
 * already on disk. Targets the same host the component listing came from.
 */
async function repairComponent(c: ModelComponentStatus): Promise<void> {
  if (!c.repair_model || repairing.value.has(c.name)) return;
  const toasts = useToastStore();
  repairing.value.add(c.name);
  try {
    await startCatalogDownload(c.repair_model, props.target, props.forwardCredentials ?? false);
    toasts.push(`Repairing ${merged.value.name} — re-fetching ${c.name}`);
    await loadComponents();
  } catch (err) {
    toasts.push(
      err instanceof ApiError && err.status === 409
        ? `${c.repair_model} is already queued.`
        : String(err),
      "error",
    );
  } finally {
    repairing.value.delete(c.name);
  }
}

const glyphSource = computed<ModelSource>(() => {
  if (merged.value.source === "civitai") return "civitai";
  // Installed local-file models open this drawer too — keep their disk mark.
  if (merged.value.source === "local") return "local";
  return "hf";
});
const thumbnailUrl = computed(() =>
  merged.value.thumbnail_url ? catalogThumbnailUrl(merged.value.thumbnail_url) : null,
);
const thumbFailed = ref(false);
const showHero = computed(() => thumbnailUrl.value !== null && !thumbFailed.value);

const downloadItems = computed(() => buildDownloadContents(merged.value));
const downloadTotal = computed(() => downloadContentsTotalBytes(downloadItems.value));
const actionLabel = computed(() => props.action ?? catalogActionLabel(merged.value));
const downloadable = computed(() => canDownloadEntry(merged.value));
const unsupported = computed(() => !downloadable.value);

/**
 * The itemized total, prefixed with "≥" when some items lack a size — the sum
 * is then a lower bound, not the true download size. "—" when nothing reports.
 */
const downloadTotalLabel = computed(() => {
  const { bytes, complete } = downloadTotal.value;
  if (bytes == null) return "—";
  return `${complete ? "" : "≥ "}${formatGB(bytes)}`;
});

const pullLabel = computed(() => {
  if (props.pulling) return "Pulling…";
  return downloadTotal.value.bytes != null ? `Get it · ${downloadTotalLabel.value}` : "Get it";
});

/** Media badge (image/video) — from the detail's modality, else the family. */
const mediaBadge = computed(
  () => merged.value.modality ?? (isVideoFamily(merged.value.family) ? "video" : "image"),
);
const kindValue = computed(() => modelKindValue(merged.value));
const weightsHeading = computed(() => modelWeightsLabel(kindValue.value));

/** SIZE = checkpoint weights; FETCH = full footprint (weights + shared
 *  components). FETCH ≥ SIZE always (project SIZE/FETCH semantics). */
const sizeInfo = computed(() => catalogSizeInfo(merged.value));
const checkpointLabel = computed(() =>
  sizeInfo.value.weightsBytes != null ? formatGB(sizeInfo.value.weightsBytes) : "—",
);
const footprintLabel = computed(() => {
  const fetchBytes = sizeInfo.value.fetchBytes ?? sizeInfo.value.weightsBytes;
  return fetchBytes != null ? formatGB(fetchBytes) : "—";
});

const SOURCE_LABEL: Record<string, string> = {
  hf: "Hugging Face",
  civitai: "Civitai",
  local: "Local file",
};
const sourceLabel = computed(() => SOURCE_LABEL[merged.value.source] ?? merged.value.source);
const detailPageUrl = computed(() => catalogPageUrl(merged.value));

const catalogDate = computed(() => {
  const value = merged.value.updated_at ?? merged.value.created_at ?? merged.value.added_at;
  if (value == null || !Number.isFinite(value)) return null;
  const milliseconds = Math.abs(value) < 100_000_000_000 ? value * 1_000 : value;
  const date = new Date(milliseconds);
  if (Number.isNaN(date.getTime())) return null;
  return {
    label:
      merged.value.updated_at != null
        ? "Updated"
        : merged.value.created_at != null
          ? "Created"
          : "Added",
    value: new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    }).format(date),
  };
});

function openModelPage(): void {
  if (detailPageUrl.value) void openExternal(detailPageUrl.value);
}

/**
 * Which variant a Pull targets. Defaults to the entry's own id, so a drawer
 * opened without explicit variants pulls exactly the row it came from
 * (preserving the list's manifest-variant precedence). Selecting a chip
 * repoints the pull without re-opening the drawer.
 */
const selectedVariantId = ref<string>(props.entry.id);
watch(
  () => props.entry.id,
  (id) => {
    selectedVariantId.value = id;
  },
);
watch(
  () => props.variants,
  (variants) => {
    if (variants?.length && !variants.some((variant) => variant.id === selectedVariantId.value)) {
      selectedVariantId.value = variants[0]!.id;
    }
  },
  { immediate: true },
);

/** The exact entry a Pull/Repair submits — `merged`, repointed to the chosen
 *  variant id so the selection is the download target. */
const pullEntry = computed<CatalogEntry>(() => ({ ...merged.value, id: selectedVariantId.value }));

/** Select immediately for responsive feedback, then let the catalog owner
 * replace the drawer entry so every title, size, component, and action field
 * follows the chosen runnable model instead of changing only the Pull id. */
function selectVariant(id: string): void {
  selectedVariantId.value = id;
  emit("select-variant", id);
}

function formatSize(bytes: number | null): string {
  return bytes != null ? formatGB(bytes) : "—";
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") emit("close");
}
onMounted(() => window.addEventListener("keydown", onKeydown));
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <aside
    class="border-border fixed inset-y-0 right-0 z-40 flex w-96 max-w-full flex-col border-l bg-bg shadow-md"
    role="dialog"
    aria-modal="false"
    :aria-label="merged.display_name ?? merged.name"
    data-test="catalog-detail-drawer"
  >
    <!-- Header -->
    <div class="border-border flex items-center gap-2 border-b px-4 py-3">
      <SourceGlyph :source="glyphSource" :size="16" class="shrink-0 text-fg-dim" />
      <h2
        class="min-w-0 flex-1 truncate text-sm font-semibold text-fg"
        :title="merged.display_name ?? merged.name"
      >
        {{ merged.display_name ?? merged.name }}
      </h2>
      <button
        type="button"
        class="h-7 shrink-0 rounded-control px-2 text-fg-2 hover:bg-bg-deep hover:text-fg"
        aria-label="Close model detail"
        data-test="drawer-close"
        @click="emit('close')"
      >
        ✕
      </button>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <!-- Preview -->
      <!-- A 4:3 hero keeps more of a portrait preview than 16:9 did; the
           placeholder branch stays 16:9 because its mark is a fixed height. -->
      <div
        class="border-border relative w-full overflow-hidden border-b"
        :class="showHero ? 'aspect-[4/3]' : 'aspect-video'"
      >
        <img
          v-if="showHero"
          :src="thumbnailUrl!"
          alt=""
          loading="lazy"
          decoding="async"
          data-test="drawer-hero"
          class="catalog-hero h-full w-full object-cover"
          @error="thumbFailed = true"
        />
        <ModelFamilyPlaceholder v-else :family="merged.family" layout="grid" />
      </div>

      <div class="flex flex-col gap-3 p-4">
        <!-- Classification stays together: what it is, what it creates, and
             whether it contains mature material. -->
        <ModelMetadataBadges
          :kind="kindValue"
          :family="merged.family"
          :modality="mediaBadge"
          :nsfw="merged.nsfw"
          data-test="drawer-classification"
        />

        <!-- State chips -->
        <div v-if="merged.installed || unsupported" class="flex flex-wrap gap-1.5">
          <span
            v-if="merged.installed"
            class="border-border font-mono rounded-control border px-2 py-0.5 text-micro text-sapphire"
            title="Files are present under this host's models directory"
          >
            ● installed
          </span>
          <span
            v-if="unsupported"
            class="rounded-control border border-error px-2 py-0.5 text-micro text-error"
          >
            Unsupported catalog package
          </span>
        </div>

        <!-- Description belongs with identity/classification rather than
             being buried after the technical metadata grid. -->
        <p v-if="merged.description" class="text-micro leading-relaxed text-fg-2">
          {{ merged.description }}
        </p>
        <p v-else-if="loading" class="text-micro text-fg-dim">Loading details…</p>

        <!-- Meta grid -->
        <dl class="grid grid-cols-2 gap-x-3 gap-y-2">
          <div v-if="merged.author" class="min-w-0">
            <dt class="text-micro text-fg-dim">Author</dt>
            <dd class="truncate text-sm text-fg-2">{{ merged.author }}</dd>
          </div>
          <div>
            <dt class="text-micro text-fg-dim">Family</dt>
            <dd class="font-mono text-sm text-fg-2">{{ merged.family }}</dd>
          </div>
          <div>
            <dt class="text-micro text-fg-dim">Source</dt>
            <dd class="text-sm text-fg-2" data-test="drawer-source">{{ sourceLabel }}</dd>
          </div>
          <div v-if="merged.file_format">
            <dt class="text-micro text-fg-dim">Format</dt>
            <dd class="font-mono text-sm text-fg-2">{{ merged.file_format }}</dd>
          </div>
          <div v-if="merged.size_bytes != null">
            <dt class="text-micro text-fg-dim">Weights</dt>
            <dd class="font-mono text-sm text-fg-2">{{ formatGB(merged.size_bytes) }}</dd>
          </div>
          <div v-if="merged.download_count">
            <dt class="text-micro text-fg-dim">Downloads</dt>
            <dd class="font-mono text-sm text-fg-2">
              {{ formatCount(merged.download_count) }}
            </dd>
          </div>
          <div v-if="merged.likes" data-test="drawer-likes">
            <dt class="text-micro text-fg-dim">Likes</dt>
            <dd class="font-mono text-sm text-fg-2">♥ {{ formatCount(merged.likes) }}</dd>
          </div>
          <div v-if="merged.rating != null">
            <dt class="text-micro text-fg-dim">Rating</dt>
            <dd class="font-mono text-sm text-fg-2">★ {{ merged.rating.toFixed(1) }}</dd>
          </div>
          <div v-if="merged.license" class="col-span-2 min-w-0">
            <dt class="text-micro text-fg-dim">License</dt>
            <dd class="truncate text-sm text-fg-2" :title="merged.license">
              {{ merged.license }}
            </dd>
          </div>
          <div v-if="catalogDate" data-test="drawer-updated">
            <dt class="text-micro text-fg-dim">{{ catalogDate.label }}</dt>
            <dd class="text-sm text-fg-2">{{ catalogDate.value }}</dd>
          </div>
          <div v-if="detailPageUrl" class="col-span-2 min-w-0">
            <dt class="text-micro text-fg-dim">Model page</dt>
            <dd>
              <button
                type="button"
                class="text-sm text-accent hover:underline"
                data-test="drawer-page-link"
                @click="openModelPage"
              >
                View on {{ sourceLabel }}
              </button>
            </dd>
          </div>
        </dl>

        <!-- Footprint tiles: model weights (SIZE) vs full footprint (FETCH,
             which includes shared components and is always ≥ SIZE). -->
        <div
          v-if="sizeInfo.weightsBytes != null"
          class="flex gap-2.5"
          data-test="drawer-stat-tiles"
        >
          <div class="border-border flex-1 rounded-card border bg-bg-deep p-3">
            <div class="font-mono text-micro text-fg-dim whitespace-nowrap uppercase">
              {{ weightsHeading }}
            </div>
            <div class="font-mono mt-1 text-base text-fg" data-test="stat-checkpoint">
              {{ checkpointLabel }}
            </div>
          </div>
          <div class="border-border flex-1 rounded-card border bg-bg-deep p-3">
            <div class="font-mono text-micro text-fg-dim whitespace-nowrap uppercase">
              Full footprint
            </div>
            <div class="font-mono mt-1 text-base text-fg" data-test="stat-footprint">
              {{ footprintLabel }}
            </div>
          </div>
        </div>

        <!-- Variants: the selected chip is the exact pull target. -->
        <section v-if="variants?.length" data-test="drawer-variants">
          <div class="font-mono text-micro text-fg-dim whitespace-nowrap mb-1.5 uppercase">
            Variants
          </div>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="variant in variants"
              :key="variant.id"
              type="button"
              data-test="variant-chip"
              class="font-mono rounded-control border px-2.5 py-1 text-micro transition-colors duration-100"
              :class="
                variant.id === selectedVariantId
                  ? 'border-accent text-accent'
                  : 'border-border text-fg-2 hover:text-fg'
              "
              :aria-pressed="variant.id === selectedVariantId"
              @click="selectVariant(variant.id)"
            >
              {{ variant.label }}
              <span v-if="variant.sizeBytes != null" class="text-fg-dim">
                · {{ formatSize(variant.sizeBytes) }}
              </span>
            </button>
          </div>
        </section>

        <!-- Download contents -->
        <section
          v-if="downloadItems.length"
          class="border-border border-t pt-3"
          data-test="download-contents"
        >
          <div class="mb-1.5 flex items-baseline justify-between">
            <span class="font-mono text-micro text-fg-dim whitespace-nowrap"
              >DOWNLOAD CONTENTS</span
            >
            <span class="font-mono text-micro text-fg">{{ downloadTotalLabel }}</span>
          </div>
          <ul class="flex flex-col gap-1">
            <li
              v-for="item in downloadItems"
              :key="item.key"
              class="grid grid-cols-[1fr_auto] items-baseline gap-2"
            >
              <span class="min-w-0 truncate text-micro text-fg-2" :title="item.label">
                {{ item.label }}
              </span>
              <span class="font-mono text-micro text-fg-dim">
                {{ item.kind }} · {{ formatSize(item.sizeBytes) }}
              </span>
            </li>
          </ul>
        </section>

        <!-- On-disk components (installed models) -->
        <section
          v-if="componentList.length"
          class="border-border border-t pt-3"
          data-test="component-list"
        >
          <div class="mb-1.5 flex items-baseline justify-between">
            <span class="font-mono text-micro text-fg-dim whitespace-nowrap">ON THIS HOST</span>
            <span class="font-mono text-micro text-fg">
              {{ componentsPresent }}/{{ componentList.length }} present
            </span>
          </div>
          <ul class="flex flex-col gap-1">
            <li
              v-for="c in componentList"
              :key="c.name"
              class="flex items-center gap-2"
              data-test="component-row"
            >
              <span
                class="h-1.5 w-1.5 shrink-0 rounded-full"
                :class="c.present ? 'bg-sapphire' : 'bg-error'"
                role="img"
                :title="c.present ? 'Present' : 'Missing'"
                :aria-label="c.present ? 'Present' : 'Missing'"
              />
              <span class="min-w-0 truncate text-micro text-fg-2">{{ c.name }}</span>
              <span class="font-mono text-micro text-fg-dim whitespace-nowrap ml-auto shrink-0">{{
                c.kind
              }}</span>
              <button
                v-if="!c.present && c.repair_model"
                type="button"
                data-test="component-repair"
                class="border-border h-6 shrink-0 rounded-control border px-2 text-micro text-accent transition-colors duration-150 hover:border-accent active:translate-y-px disabled:opacity-40"
                :disabled="repairing.has(c.name)"
                :title="`Re-download the missing ${c.name}`"
                @click="repairComponent(c)"
              >
                {{ repairing.has(c.name) ? "Repairing…" : "Repair" }}
              </button>
            </li>
          </ul>
        </section>

        <!-- Civitai trigger phrases are high-value authoring metadata. -->
        <section
          v-if="merged.trained_words?.length"
          class="border-border border-t pt-3"
          data-test="drawer-trained-words"
        >
          <div class="font-mono text-micro text-fg-dim whitespace-nowrap mb-1.5 uppercase">
            Trigger words
          </div>
          <div class="flex flex-wrap gap-1">
            <span
              v-for="word in merged.trained_words"
              :key="word"
              class="border-border font-mono rounded-control border px-2 py-0.5 text-micro text-fg-2"
            >
              {{ word }}
            </span>
          </div>
        </section>

        <!-- Tags -->
        <div v-if="merged.tags?.length" class="flex flex-wrap gap-1">
          <span
            v-for="tag in merged.tags"
            :key="tag"
            class="border-border rounded-control border px-1.5 py-0.5 text-micro text-fg-dim"
          >
            {{ tag }}
          </span>
        </div>
      </div>
    </div>

    <!-- Action -->
    <div class="border-border border-t p-4">
      <p
        v-if="props.runtimeNotice"
        data-test="runtime-unavailable-note"
        class="text-micro text-fg-2"
      >
        {{ props.runtimeNotice.message }}
      </p>
      <button
        v-if="actionLabel === 'Repair'"
        type="button"
        data-test="drawer-repair"
        class="border-border h-8 w-full rounded-control border text-sm text-fg-2 transition-colors duration-150 hover:border-accent hover:text-fg active:translate-y-px disabled:opacity-50"
        :disabled="pulling || !downloadable"
        title="Re-fetch any missing or incomplete files for this model"
        @click="emit('pull', pullEntry)"
      >
        {{ pulling ? "Repairing…" : "Repair" }}
      </button>
      <button
        v-else
        type="button"
        data-test="drawer-pull"
        class="border-border h-8 w-full rounded-control border text-sm text-accent transition-colors duration-150 hover:border-accent active:translate-y-px disabled:opacity-50"
        :disabled="pulling || !downloadable"
        :title="downloadable ? 'Download this model' : 'Unsupported catalog package'"
        @click="emit('pull', pullEntry)"
      >
        {{ pullLabel }}
      </button>
    </div>
  </aside>
</template>

<style scoped>
/* Same upward crop bias as the grid card: portrait previews otherwise lose
   the subject's head to a centred cover crop. */
.catalog-hero {
  object-position: 50% 25%;
}
</style>
