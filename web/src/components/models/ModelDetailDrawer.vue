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

const cat = useCatalog();

// The drawer opens for content AND for the loading / error non-content states
// so a Discover row whose detail fetch is slow or failing never shows a blank
// panel (spec G4).
const open = computed(
  () =>
    cat.detail.value != null ||
    cat.detailLoadingId.value != null ||
    cat.detailError.value != null,
);

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

// ── Normalized view over the detail union ──────────────────────────────
const isInstalled = computed(() => cat.detail.value?.kind === "installed");
const installedModel = computed(() =>
  cat.detail.value?.kind === "installed" ? cat.detail.value.model : null,
);
const isLoaded = computed(() => installedModel.value?.is_loaded ?? false);

const mediaLabel = computed(() => {
  const d = cat.detail.value;
  if (!d) return "";
  if (d.kind === "catalog") return d.entry.modality;
  return /ltx/i.test(d.model.family) ? "video" : "image";
});

const name = computed(() => {
  const d = cat.detail.value;
  if (!d) return "";
  return d.kind === "catalog" ? d.entry.name : d.model.name;
});

const family = computed(() => {
  const d = cat.detail.value;
  if (!d) return "";
  return d.kind === "catalog" ? d.entry.family : d.model.family;
});

const description = computed(() => {
  const d = cat.detail.value;
  if (!d) return "";
  return (
    (d.kind === "catalog" ? d.entry.description : d.model.description) ?? ""
  );
});

// Checkpoint weights = SIZE (model weights only, per catalog semantics).
const checkpointBytes = computed<number | null>(() => {
  const d = cat.detail.value;
  if (!d) return null;
  if (d.kind === "catalog") return d.entry.size_bytes;
  return d.model.size_gb ? d.model.size_gb * 1_000_000_000 : null;
});

// Full footprint = FETCH (weights + every shared companion; >= SIZE).
const footprintBytes = computed<number | null>(() => {
  const d = cat.detail.value;
  if (!d) return null;
  if (d.kind === "installed") {
    return (
      d.model.disk_usage_bytes ??
      (d.model.size_gb ? d.model.size_gb * 1_000_000_000 : null)
    );
  }
  // Defensive optional chaining: a malformed entry (null download_recipe /
  // files) must never throw here — a render exception is exactly what blanks
  // the whole drawer.
  const primary = (d.entry.download_recipe?.files ?? []).reduce(
    (sum, f) => sum + (f.size_bytes ?? 0),
    0,
  );
  const companions = (d.entry.companion_details ?? []).reduce(
    (sum, c) => sum + (c.size_bytes ?? 0),
    0,
  );
  const total = primary + companions;
  return total > 0 ? total : d.entry.size_bytes;
});

const components = computed<string[]>(() => {
  const d = cat.detail.value;
  if (!d) return [];
  if (d.kind === "installed") return d.components.map((c) => c.name);
  return (d.entry.companion_details ?? []).map((c) => c.name);
});

const variants = computed(() =>
  cat.detail.value?.kind === "catalog" ? cat.detail.value.variants : [],
);

const source = computed(() => {
  const d = cat.detail.value;
  if (!d) return "—";
  if (d.kind === "installed") return d.model.hf_repo || "local";
  return d.entry.source === "civitai" ? "Civitai" : "Hugging Face";
});

const license = computed(() => {
  const d = cat.detail.value;
  if (d?.kind === "catalog") return d.entry.license ?? "—";
  return "—";
});

// ── Discover (catalog) pull ────────────────────────────────────────────
const selectedVariantId = ref<string | null>(null);
watch(
  () => cat.detail.value,
  (d) => {
    selectedVariantId.value =
      d?.kind === "catalog" ? (d.variants[0]?.id ?? d.entry.id) : null;
  },
  { immediate: true },
);

const canPull = computed(() => {
  const d = cat.detail.value;
  return d?.kind === "catalog" ? cat.canDownload(d.entry) : false;
});

const isRepair = computed(
  () =>
    cat.detail.value?.kind === "catalog" && cat.detail.value.entry.installed,
);

async function handlePull() {
  const d = cat.detail.value;
  if (d?.kind !== "catalog") return;
  const target = isRepair.value
    ? d.entry.id
    : (selectedVariantId.value ?? d.entry.id);
  await cat.startDownload(target);
  cat.closeDetail();
}

// ── Installed model actions ────────────────────────────────────────────
const busy = ref(false);

async function handleLoad() {
  const d = cat.detail.value;
  if (d?.kind !== "installed" || busy.value) return;
  busy.value = true;
  try {
    await cat.loadInstalled(d.model.name);
    toast("success", "loaded into memory");
  } catch (e: unknown) {
    toast("error", e instanceof Error ? e.message : "load failed");
  } finally {
    busy.value = false;
  }
}

async function handleUnload() {
  const d = cat.detail.value;
  if (d?.kind !== "installed" || busy.value) return;
  busy.value = true;
  try {
    await cat.unloadInstalled(d.model.name);
    toast("success", "unloaded");
  } catch (e: unknown) {
    toast("error", e instanceof Error ? e.message : "unload failed");
  } finally {
    busy.value = false;
  }
}

async function handleDelete() {
  const d = cat.detail.value;
  if (d?.kind !== "installed" || busy.value) return;
  const modelName = d.model.name;
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
</script>

<template>
  <component
    :is="panelComponent"
    v-bind="panelProps"
    :open="open"
    title="Model details"
    @close="onClose"
  >
    <div
      v-if="cat.detailLoadingId.value"
      class="md md--state"
      data-test="detail-loading"
    >
      <div class="md__spinner" aria-hidden="true" />
      <p class="md__state-msg">loading model details…</p>
    </div>
    <div
      v-else-if="cat.detailError.value"
      class="md md--state"
      data-test="detail-error"
    >
      <div class="md__name">Model details</div>
      <p class="md__state-msg">{{ cat.detailError.value.message }}</p>
      <button
        type="button"
        class="md__ghost"
        data-test="detail-retry"
        @click="cat.retryDetail(cat.detailError.value.id)"
      >
        <Icon name="refresh" :size="14" />
        Try again
      </button>
    </div>
    <div v-else-if="cat.detail.value" class="md">
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
          <span v-for="c in components" :key="c" class="md__chip">{{ c }}</span>
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

      <div class="md__rows">
        <div class="md__row">
          <span class="md__row-key">Source</span>
          <span class="md__row-val">{{ source }}</span>
        </div>
        <div class="md__row md__row--last">
          <span class="md__row-key">License</span>
          <span class="md__row-val">{{ license }}</span>
        </div>
      </div>

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
  </component>
</template>

<style scoped>
.md {
  color: var(--rebate);
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

.md__row--last {
  border-bottom: 0;
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
