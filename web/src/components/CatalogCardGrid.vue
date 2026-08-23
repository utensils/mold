<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  isAlreadyQueuedError,
  planBatchInstallTargets,
} from "@studio/lib/modelBatchInstall";
import { useCatalog } from "../composables/useCatalog";
import { useModelInstallTargets } from "../composables/useModelInstallTargets";
import { toast } from "../lib/toasts";
import type { CatalogEntryWire } from "../types";
import CatalogCard from "./CatalogCard.vue";
import { modelRuntimeNoticeForId } from "@studio/lib/modelRuntimeAvailability";

const cat = useCatalog();
const installTargets = useModelInstallTargets();
const sentinel = ref<HTMLElement | null>(null);
const selected = ref(new Map<string, CatalogEntryWire>());
const selectedTargetId = ref("");
const batchStarting = ref(false);
let observer: IntersectionObserver | null = null;

/** Whether this host can run the model behind a Discover row, from its own
 *  `/api/models` listing — knowable before the pull, which for the affected
 *  checkpoints is 21-42 GB (#1276). A live catalog id nobody lists is simply
 *  unknown and gets no badge. */
function runtimeNoticeFor(id: string) {
  return modelRuntimeNoticeForId(id, cat.availableManifests.value);
}

function detach() {
  observer?.disconnect();
  observer = null;
}

// rootMargin lets us start loading the next page ~one screenful before the
// sentinel actually scrolls into view, so the user rarely sees an empty
// stretch waiting for the fetch to complete.
function attach() {
  detach();
  if (!sentinel.value) return;
  if (typeof IntersectionObserver === "undefined") return;
  observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        void cat.loadMore();
      }
    },
    { rootMargin: "400px" },
  );
  observer.observe(sentinel.value);
}

onMounted(attach);
onBeforeUnmount(detach);
// Re-attach when the sentinel ref changes (e.g. transitions from
// hasMore=false → true after a filter change loads a fresh result set).
watch(sentinel, attach);

function openCard(id: string) {
  void cat.openDetail(id);
}

// A model can be installed here and missing on the next machine, so the pull
// resolves a target first: one candidate acts straight away, several ask.
async function pullCard(entry: CatalogEntryWire) {
  const choice = await installTargets.chooseInstallTarget({
    modelId: entry.id,
    displayName: entry.name,
    ownedByOrigin: entry.installed,
  });
  if (choice.kind === "cancelled") return;
  try {
    await installTargets.startDownloadOn(choice.target, entry.id);
    toast("success", installTargets.queuedMessage(choice.target));
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

function selectionTargets(entry: CatalogEntryWire) {
  return installTargets.planFor(entry.id, entry.installed).targets;
}

function selectable(entry: CatalogEntryWire): boolean {
  return entry.supported && selectionTargets(entry).length > 0;
}

function toggleSelection(entry: CatalogEntryWire, checked: boolean): void {
  const next = new Map(selected.value);
  if (checked) next.set(entry.id, entry);
  else next.delete(entry.id);
  selected.value = next;
}

const batchTargets = computed(() =>
  planBatchInstallTargets(
    [...selected.value.values()].map((entry) => ({
      modelId: entry.id,
      targets: selectionTargets(entry),
    })),
  ),
);

watch(
  batchTargets,
  (targets) => {
    if (targets.some(({ host }) => host.id === selectedTargetId.value)) return;
    selectedTargetId.value = targets.length === 1 ? targets[0]!.host.id : "";
  },
  { immediate: true },
);

const selectedBatchTarget = computed(() =>
  batchTargets.value.find(({ host }) => host.id === selectedTargetId.value),
);

function targetSummary(installCount: number, repairCount: number): string {
  const parts = [];
  if (installCount) parts.push(`${installCount} new`);
  if (repairCount) parts.push(`${repairCount} repair`);
  return parts.join(", ");
}

async function startBatch(): Promise<void> {
  const target = selectedBatchTarget.value;
  if (!target || batchStarting.value) return;
  batchStarting.value = true;
  const entriesById = new Map(selected.value);
  const results = await Promise.allSettled(
    target.items.map(async (item) => {
      try {
        await installTargets.startDownloadOn(
          { host: target.host, action: item.action },
          item.modelId,
        );
      } catch (error) {
        if (!isAlreadyQueuedError(error)) throw error;
      }
      return item.modelId;
    }),
  );
  const next = new Map(selected.value);
  let succeeded = 0;
  const failures: string[] = [];
  results.forEach((result, index) => {
    if (result.status === "fulfilled") {
      next.delete(result.value);
      succeeded += 1;
    } else {
      const item = target.items[index]!;
      const entry = entriesById.get(item.modelId);
      failures.push(
        `${entry?.name ?? "Model"}: ${result.reason instanceof Error ? result.reason.message : String(result.reason)}`,
      );
    }
  });
  selected.value = next;
  if (succeeded)
    toast(
      "success",
      `${succeeded} ${succeeded === 1 ? "download" : "downloads"} queued on ${target.host.label}`,
    );
  if (failures.length) toast("error", failures.join(" · "));
  batchStarting.value = false;
}
</script>

<template>
  <div class="min-w-0 flex-1">
    <!-- Loading state -->
    <div
      v-if="cat.loading.value && cat.visibleEntries.value.length === 0"
      class="flex items-center justify-center py-12 text-sm text-ink-3"
    >
      Loading…
    </div>

    <!-- Error state -->
    <div
      v-if="cat.errorMsg.value || cat.providerErrors.value.length"
      data-test="catalog-provider-warning"
      class="bg-bench border border-edge flex items-start gap-3 rounded-2xl px-4 py-3 text-sm text-rose-200"
      role="alert"
    >
      <span class="mt-0.5">⚠</span>
      <div class="min-w-0 flex-1">
        <p class="font-medium text-rose-100">
          {{
            cat.errorMsg.value
              ? "Couldn't refresh the catalog."
              : "Part of the catalog is unavailable."
          }}
        </p>
        <p class="text-rose-200/80">
          {{
            cat.errorMsg.value ??
            cat.providerErrors.value.map((item) => item.message).join(" ")
          }}
          <span v-if="cat.providerErrors.value.length">
            Showing available models.</span
          >
        </p>
      </div>
      <button
        type="button"
        data-test="catalog-retry"
        class="border border-rose-200/40 shrink-0 rounded-lg px-3 py-1.5 font-medium text-rose-100 hover:bg-rose-100/10"
        :disabled="cat.loading.value"
        @click="cat.refresh()"
      >
        {{ cat.loading.value ? "Retrying…" : "Retry" }}
      </button>
    </div>

    <!-- Empty state -->
    <div
      v-if="!cat.loading.value && cat.visibleEntries.value.length === 0"
      class="flex flex-col items-center justify-center gap-2 py-16 text-ink-3"
    >
      <p class="text-sm">No models found.</p>
      <p class="text-xs">
        No catalog entry matches every filter you've set — try widening one.
      </p>
    </div>

    <!-- Grid -->
    <template v-else-if="cat.visibleEntries.value.length > 0">
      <p
        data-testid="catalog-result-count"
        class="mb-3 text-[12px] text-ink-3"
        aria-live="polite"
      >
        {{ cat.resultCount.value.toLocaleString() }}
        {{ cat.resultCount.value === 1 ? "result" : "results" }}
      </p>

      <div
        v-if="selected.size > 0"
        class="batch-bar"
        data-test="catalog-batch-bar"
        aria-live="polite"
      >
        <strong>{{ selected.size }} selected</strong>
        <template v-if="batchTargets.length">
          <label class="batch-bar__target">
            <span>Target machine</span>
            <select
              v-model="selectedTargetId"
              data-test="catalog-batch-target"
              :disabled="batchStarting"
            >
              <option value="" disabled>Choose a machine…</option>
              <option
                v-for="target in batchTargets"
                :key="target.host.id"
                :value="target.host.id"
              >
                {{ target.host.label }} ·
                {{ targetSummary(target.installCount, target.repairCount) }}
              </option>
            </select>
          </label>
          <button
            type="button"
            class="batch-bar__download"
            data-test="catalog-batch-download"
            :disabled="!selectedBatchTarget || batchStarting"
            @click="startBatch"
          >
            {{ batchStarting ? "Starting…" : `Download ${selected.size}` }}
          </button>
        </template>
        <span v-else class="batch-bar__warning"
          >No machine can receive every selected model.</span
        >
        <button
          type="button"
          class="batch-bar__clear"
          :disabled="batchStarting"
          @click="selected = new Map()"
        >
          Clear
        </button>
      </div>

      <div
        data-test="catalog-results"
        :data-layout="cat.layout.value"
        :class="
          cat.layout.value === 'grid'
            ? 'grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3'
            : 'flex flex-col gap-2'
        "
      >
        <CatalogCard
          v-for="entry in cat.visibleEntries.value"
          :key="entry.id"
          :entry="entry"
          :layout="cat.layout.value"
          :selectable="!batchStarting && selectable(entry)"
          :checked="selected.has(entry.id)"
          :runtime-notice="runtimeNoticeFor(entry.id)"
          @open="openCard(entry.id)"
          @pull="pullCard(entry)"
          @toggle-select="toggleSelection(entry, $event)"
        />
      </div>

      <div
        v-if="cat.loadingMore.value"
        class="py-6 text-center text-sm text-ink-3"
      >
        Loading more…
      </div>

      <!-- The sentinel triggers loadMore() when scrolled into view.
           We only render it while there's actually more to fetch so a
           stale observer callback can't double-fetch the final page. -->
      <div
        v-if="cat.hasMore.value"
        ref="sentinel"
        data-testid="catalog-load-more-sentinel"
        class="h-1 w-full"
        aria-hidden="true"
      />
    </template>
  </div>
</template>

<style scoped>
.batch-bar {
  position: sticky;
  top: 8px;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  padding: 10px 12px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  background: color-mix(in srgb, var(--bench) 94%, transparent);
  box-shadow: var(--shadow-raised);
  backdrop-filter: blur(12px);
  color: var(--rebate);
  font-size: 12px;
}

.batch-bar__target {
  display: flex;
  flex: 1;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  color: var(--ink-2);
}

.batch-bar select,
.batch-bar button {
  min-height: 34px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  padding: 0 10px;
}

.batch-bar__download {
  border-color: var(--safelight) !important;
  background: var(--safelight) !important;
  color: var(--on-accent) !important;
  font-weight: 700;
}

.batch-bar button:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.batch-bar__clear {
  background: transparent !important;
}

.batch-bar__warning {
  flex: 1;
  color: var(--stop);
}

@media (max-width: 700px) {
  .batch-bar {
    align-items: stretch;
    flex-wrap: wrap;
  }

  .batch-bar__target {
    width: 100%;
    flex-basis: 100%;
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
