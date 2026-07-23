<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import SourceGlyph from "../generate/SourceGlyph.vue";
import ModelFamilyPlaceholder from "./ModelFamilyPlaceholder.vue";
import { fetchCatalogDetail, startCatalogDownload } from "../../lib/api/catalog";
import { fetchModelComponents } from "../../lib/api/models";
import {
  buildDownloadContents,
  canDownloadEntry,
  catalogActionLabel,
  downloadContentsTotalBytes,
} from "../../lib/catalogDetail";
import { catalogSizeInfo } from "../../lib/catalog";
import { isVideoFamily } from "../../lib/capabilities";
import { catalogThumbnailUrl } from "../../lib/catalogThumbnails";
import { formatCount, formatGB } from "../../lib/format";
import { useToastStore } from "../../stores/toasts";
import { ApiError, type ApiTarget } from "../../lib/api/client";
import type { ModelSource } from "../../lib/modelSource";
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
}>();
const emit = defineEmits<{
  (e: "close"): void;
  (e: "pull", entry: CatalogEntry): void;
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
  detail.value
    ? {
        ...props.entry,
        ...detail.value,
        installed: props.entry.installed || detail.value.installed,
        thumbnail_url: props.entry.thumbnail_url || detail.value.thumbnail_url || null,
      }
    : props.entry,
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
const actionLabel = computed(() => catalogActionLabel(merged.value));
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
  return downloadTotal.value.bytes != null ? `Pull · ${downloadTotalLabel.value}` : "Pull";
});

/** Media badge (image/video) — from the detail's modality, else the family. */
const mediaBadge = computed(
  () => merged.value.modality ?? (isVideoFamily(merged.value.family) ? "video" : "image"),
);

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
    class="border-edge fixed inset-y-0 right-0 z-40 flex w-96 max-w-full flex-col border-l bg-bench shadow-raised"
    role="dialog"
    aria-modal="false"
    :aria-label="merged.display_name ?? merged.name"
    data-test="catalog-detail-drawer"
  >
    <!-- Header -->
    <div class="border-edge flex items-center gap-2 border-b px-4 py-3">
      <SourceGlyph :source="glyphSource" :size="16" class="shrink-0 text-ink-3" />
      <h2
        class="min-w-0 flex-1 truncate text-body font-semibold text-ink"
        :title="merged.display_name ?? merged.name"
      >
        {{ merged.display_name ?? merged.name }}
      </h2>
      <button
        type="button"
        class="h-7 shrink-0 rounded-control px-2 text-ink-2 hover:bg-bath hover:text-ink"
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
        class="border-edge relative w-full overflow-hidden border-b"
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
        <!-- Media kind -->
        <div>
          <span
            class="data-mono inline-block rounded-full border border-halide/40 px-2 py-0.5 text-caption uppercase text-halide"
            data-test="drawer-media-badge"
          >
            {{ mediaBadge }}
          </span>
        </div>

        <!-- State chips -->
        <div v-if="merged.installed || unsupported" class="flex flex-wrap gap-1.5">
          <span
            v-if="merged.installed"
            class="border-edge data-mono rounded-full border px-2 py-0.5 text-caption text-halide"
            title="Files are present under this host's models directory"
          >
            ● installed
          </span>
          <span
            v-if="unsupported"
            class="rounded-full border border-stop px-2 py-0.5 text-caption text-stop"
          >
            Unsupported catalog package
          </span>
        </div>

        <!-- Meta grid -->
        <dl class="grid grid-cols-2 gap-x-3 gap-y-2">
          <div v-if="merged.author" class="min-w-0">
            <dt class="text-caption text-ink-3">Author</dt>
            <dd class="truncate text-body text-ink-2">{{ merged.author }}</dd>
          </div>
          <div>
            <dt class="text-caption text-ink-3">Family</dt>
            <dd class="data-mono text-body text-ink-2">{{ merged.family }}</dd>
          </div>
          <div>
            <dt class="text-caption text-ink-3">Source</dt>
            <dd class="text-body text-ink-2" data-test="drawer-source">{{ sourceLabel }}</dd>
          </div>
          <div v-if="merged.modality">
            <dt class="text-caption text-ink-3">Modality</dt>
            <dd class="text-body text-ink-2 capitalize">{{ merged.modality }}</dd>
          </div>
          <div v-if="merged.file_format">
            <dt class="text-caption text-ink-3">Format</dt>
            <dd class="data-mono text-body text-ink-2">{{ merged.file_format }}</dd>
          </div>
          <div v-if="merged.size_bytes != null">
            <dt class="text-caption text-ink-3">Weights</dt>
            <dd class="data-mono text-body text-ink-2">{{ formatGB(merged.size_bytes) }}</dd>
          </div>
          <div v-if="merged.download_count">
            <dt class="text-caption text-ink-3">Downloads</dt>
            <dd class="data-mono text-body text-ink-2">{{ formatCount(merged.download_count) }}</dd>
          </div>
          <div v-if="merged.rating != null">
            <dt class="text-caption text-ink-3">Rating</dt>
            <dd class="data-mono text-body text-ink-2">★ {{ merged.rating.toFixed(1) }}</dd>
          </div>
          <div v-if="merged.license" class="col-span-2 min-w-0">
            <dt class="text-caption text-ink-3">License</dt>
            <dd class="truncate text-body text-ink-2" :title="merged.license">
              {{ merged.license }}
            </dd>
          </div>
        </dl>

        <!-- Description -->
        <p v-if="merged.description" class="text-caption leading-relaxed text-ink-2">
          {{ merged.description }}
        </p>
        <p v-else-if="loading" class="text-caption text-ink-3">Loading details…</p>

        <!-- Footprint tiles: checkpoint weights (SIZE) vs full footprint (FETCH,
             which includes shared components and is always ≥ SIZE). -->
        <div
          v-if="sizeInfo.weightsBytes != null"
          class="flex gap-2.5"
          data-test="drawer-stat-tiles"
        >
          <div class="border-edge flex-1 rounded-card border bg-bath p-3">
            <div class="edge-code uppercase">Checkpoint weights</div>
            <div class="data-mono mt-1 text-body-lg text-ink" data-test="stat-checkpoint">
              {{ checkpointLabel }}
            </div>
          </div>
          <div class="border-edge flex-1 rounded-card border bg-bath p-3">
            <div class="edge-code uppercase">Full footprint</div>
            <div class="data-mono mt-1 text-body-lg text-ink" data-test="stat-footprint">
              {{ footprintLabel }}
            </div>
          </div>
        </div>

        <!-- Variants: the selected chip is the exact pull target. -->
        <section v-if="variants?.length" data-test="drawer-variants">
          <div class="edge-code mb-1.5 uppercase">Variants</div>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="variant in variants"
              :key="variant.id"
              type="button"
              data-test="variant-chip"
              class="data-mono rounded-full border px-2.5 py-1 text-caption transition-colors duration-100"
              :class="
                variant.id === selectedVariantId
                  ? 'border-safelight text-safelight'
                  : 'border-edge text-ink-2 hover:text-ink'
              "
              :aria-pressed="variant.id === selectedVariantId"
              @click="selectedVariantId = variant.id"
            >
              {{ variant.label }}
              <span v-if="variant.sizeBytes != null" class="text-ink-3">
                · {{ formatSize(variant.sizeBytes) }}
              </span>
            </button>
          </div>
        </section>

        <!-- Download contents -->
        <section
          v-if="downloadItems.length"
          class="border-edge border-t pt-3"
          data-test="download-contents"
        >
          <div class="mb-1.5 flex items-baseline justify-between">
            <span class="edge-code">DOWNLOAD CONTENTS</span>
            <span class="data-mono text-caption text-ink">{{ downloadTotalLabel }}</span>
          </div>
          <ul class="flex flex-col gap-1">
            <li
              v-for="item in downloadItems"
              :key="item.key"
              class="grid grid-cols-[1fr_auto] items-baseline gap-2"
            >
              <span class="min-w-0 truncate text-caption text-ink-2" :title="item.label">
                {{ item.label }}
              </span>
              <span class="data-mono text-caption text-ink-3">
                {{ item.kind }} · {{ formatSize(item.sizeBytes) }}
              </span>
            </li>
          </ul>
        </section>

        <!-- On-disk components (installed models) -->
        <section
          v-if="componentList.length"
          class="border-edge border-t pt-3"
          data-test="component-list"
        >
          <div class="mb-1.5 flex items-baseline justify-between">
            <span class="edge-code">ON THIS HOST</span>
            <span class="data-mono text-caption text-ink">
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
                :class="c.present ? 'bg-halide' : 'bg-stop'"
                role="img"
                :title="c.present ? 'Present' : 'Missing'"
                :aria-label="c.present ? 'Present' : 'Missing'"
              />
              <span class="min-w-0 truncate text-caption text-ink-2">{{ c.name }}</span>
              <span class="edge-code ml-auto shrink-0">{{ c.kind }}</span>
              <button
                v-if="!c.present && c.repair_model"
                type="button"
                data-test="component-repair"
                class="border-edge h-6 shrink-0 rounded-control border px-2 text-caption text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-40"
                :disabled="repairing.has(c.name)"
                :title="`Re-download the missing ${c.name}`"
                @click="repairComponent(c)"
              >
                {{ repairing.has(c.name) ? "Repairing…" : "Repair" }}
              </button>
            </li>
          </ul>
        </section>

        <!-- Tags -->
        <div v-if="merged.tags?.length" class="flex flex-wrap gap-1">
          <span
            v-for="tag in merged.tags"
            :key="tag"
            class="border-edge rounded-full border px-1.5 py-0.5 text-caption text-ink-3"
          >
            {{ tag }}
          </span>
        </div>
      </div>
    </div>

    <!-- Action -->
    <div class="border-edge border-t p-4">
      <button
        v-if="actionLabel === 'Repair'"
        type="button"
        data-test="drawer-repair"
        class="border-edge h-8 w-full rounded-control border text-body text-ink-2 transition-colors duration-150 hover:border-safelight hover:text-ink active:translate-y-px disabled:opacity-50"
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
        class="border-edge h-8 w-full rounded-control border text-body text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-50"
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
