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
import { catalogThumbnailUrl } from "../../lib/catalogThumbnails";
import { formatCount, formatGB } from "../../lib/format";
import { useToastStore } from "../../stores/toasts";
import { ApiError, type ApiTarget } from "../../lib/api/client";
import type { ModelSource } from "../../lib/modelSource";
import type { CatalogEntry, ModelComponentStatus } from "../../lib/api/types";

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
}>();
const emit = defineEmits<{
  (e: "close"): void;
  (e: "pull", entry: CatalogEntry): void;
}>();

const detail = ref<CatalogEntry | null>(null);
const loading = ref(false);

/** Summary fields render immediately; the detail overlays as it arrives.
 *  `installed` never downgrades: the opener knows the model is on its host
 *  even when the detail endpoint (possibly another host) says otherwise. */
const merged = computed<CatalogEntry>(() =>
  detail.value
    ? {
        ...props.entry,
        ...detail.value,
        installed: props.entry.installed || detail.value.installed,
      }
    : props.entry,
);

async function loadDetail(): Promise<void> {
  detail.value = null;
  loading.value = true;
  try {
    detail.value = await fetchCatalogDetail(
      props.entry.id,
      props.forwardCredentials ?? false,
      props.target,
    );
  } catch {
    // Older server / live row not fetchable — the summary keeps the drawer
    // useful and the Pull action intact.
    detail.value = null;
  } finally {
    loading.value = false;
  }
}

watch(() => props.entry.id, loadDetail, { immediate: true });

// ── On-disk component state (installed models only) ───────────────────────

const components = ref<ModelComponentStatus[] | "loading" | "error" | null>(null);

/** Per-component presence from the owning host; quietly absent elsewhere. */
async function loadComponents(): Promise<void> {
  if (!merged.value.installed) {
    components.value = null;
    return;
  }
  components.value = "loading";
  try {
    components.value = (await fetchModelComponents(props.entry.id, props.target)).components;
  } catch {
    // Not an installed manifest/catalog model on this host, or an older
    // server — the section simply hides.
    components.value = "error";
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

const glyphSource = computed<ModelSource>(() =>
  merged.value.source === "civitai" ? "civitai" : "hf",
);
const thumbnailUrl = computed(() =>
  merged.value.thumbnail_url ? catalogThumbnailUrl(merged.value.thumbnail_url) : null,
);
const thumbFailed = ref(false);

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
    :aria-label="merged.name"
    data-test="catalog-detail-drawer"
  >
    <!-- Header -->
    <div class="border-edge flex items-center gap-2 border-b px-4 py-3">
      <SourceGlyph :source="glyphSource" :size="16" class="shrink-0 text-ink-3" />
      <h2 class="min-w-0 flex-1 truncate text-body font-semibold text-ink" :title="merged.name">
        {{ merged.name }}
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
      <div class="border-edge relative aspect-video w-full overflow-hidden border-b">
        <img
          v-if="thumbnailUrl && !thumbFailed"
          :src="thumbnailUrl"
          alt=""
          loading="lazy"
          decoding="async"
          class="h-full w-full object-cover"
          @error="thumbFailed = true"
        />
        <ModelFamilyPlaceholder v-else :family="merged.family" layout="grid" />
      </div>

      <div class="flex flex-col gap-3 p-4">
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
        @click="emit('pull', merged)"
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
        @click="emit('pull', merged)"
      >
        {{ pullLabel }}
      </button>
    </div>
  </aside>
</template>
