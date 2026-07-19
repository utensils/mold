<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import {
  filterModelsForTarget,
  findInstalledModel,
  mergeInstalledModels,
  preferredInstalledModel,
  shouldShowStarterCards,
} from "../lib/generateModels";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import StarterCards from "../components/generate/StarterCards.vue";
import ParamPanel from "../components/generate/ParamPanel.vue";
import LoraStack from "../components/generate/LoraStack.vue";
import TemplatesPanel from "../components/generate/TemplatesPanel.vue";
import SourceImageWell from "../components/generate/SourceImageWell.vue";
import EstimateBadge from "../components/generate/EstimateBadge.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import HostSelector from "../components/generate/HostSelector.vue";
import SourceGlyph from "../components/generate/SourceGlyph.vue";
import PanelResizeHandle from "../components/shell/PanelResizeHandle.vue";
import { modelSource } from "../lib/modelSource";
import { modelAvailabilityTag } from "../lib/hosts";
import { dragWidth } from "../lib/panelResize";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useConnectionStore } from "../stores/connection";
import {
  useGenerationStore,
  jobPhase,
  jobProgress,
  needsHostRoute,
  type Job,
} from "../stores/generation";
import { useGenerateFormStore } from "../stores/generateForm";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { copyBase64ImageToClipboard } from "../lib/clipboard";
import { useUiStore } from "../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { decideChainRouting } from "../lib/chainRouting";
import { applyPrefillToForm, buildRequest } from "../lib/generateForm";
import { applySourceFitPreprocess } from "../lib/sourceFitPreprocess";
import { domCanvasOps } from "../lib/sourceFitCanvas";
import { upscaleImage } from "../lib/api/upscale";
import type { HostRoute } from "../stores/hosts";
import type { GenerationTemplate } from "../lib/generationTemplates";
import { autoGrowRows } from "../lib/autogrow";
import { PromptCycler, caretOnFirstLine, caretOnLastLine } from "../lib/promptCycler";
import { fetchHistory } from "../lib/api/history";
import { formatGB } from "../lib/format";
import { randomSeed } from "../stores/generation";
import type { ModelEntry } from "../lib/api/types";
import { fitAspectRatio } from "../lib/fitAspectRatio";
import { primaryModifierPressed, shortcutLabel } from "../lib/platform";

const router = useRouter();
const conn = useConnectionStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const models = useModelStore();
const composer = useComposerStore();
const toasts = useToastStore();
const ui = useUiStore();
const contextMenu = useContextMenuStore();

// Store-backed so the model, prompt, and params survive navigating away and
// back — this view unmounts on every route change.
const formStore = useGenerateFormStore();
const form = formStore.form;
const promptEl = ref<HTMLTextAreaElement | null>(null);
const previewRegion = ref<HTMLDivElement | null>(null);
const previewFrameSize = ref({ width: 0, height: 0 });
const expandControl = ref<InstanceType<typeof ExpandControl> | null>(null);
const pickerEl = ref<HTMLDivElement | null>(null);
const pickerOpen = ref(false);

function onDocumentPointerDown(event: PointerEvent) {
  if (!pickerOpen.value || !pickerEl.value) return;
  if (!event.composedPath().includes(pickerEl.value)) pickerOpen.value = false;
}

// Live inspector width while dragging its left-edge handle; null follows the
// persisted preference (appPrefs.generateParamsWidth). Persist only on commit.
const draftAsideWidth = ref<number | null>(null);
const asideWidth = computed(() => draftAsideWidth.value ?? appPrefs.generateParamsWidth);

function onAsideResize(dx: number) {
  draftAsideWidth.value = dragWidth("generateParams", appPrefs.generateParamsWidth, dx, "left");
}

async function onAsideCommit() {
  const width = draftAsideWidth.value;
  if (width === null) return;
  if (width !== appPrefs.generateParamsWidth) await appPrefs.update({ generateParamsWidth: width });
  draftAsideWidth.value = null;
}

function onAsideReset() {
  draftAsideWidth.value = null;
  void appPrefs.update({ generateParamsWidth: null });
}

const job = computed(() => generation.active);
const siblings = computed(() => generation.siblings);
const caps = computed(() => generationCapabilitiesForFamily(form.family));
/** Over-budget video frames on a non-chainable model would fail server-side —
 *  ParamPanel shows the reason under Frames; this blocks the submit. */
const chainReject = computed(() => {
  if (!caps.value.supportsVideo) return false;
  return decideChainRouting(form.frames, form.family, form.model).kind === "reject";
});
const installedModels = computed(() =>
  mergeInstalledModels(models.installed, hostModels.unionInstalled),
);
// Falls back to the union entry so a model that only exists on an extra host
// still populates params/defaults after being picked.
const selectedModel = computed<ModelEntry | null>(() =>
  findInstalledModel(installedModels.value, form.model),
);

const showStarterCards = computed(() =>
  shouldShowStarterCards({
    connectionReady: conn.ready,
    primaryLoading: models.loading,
    hostsInitialized: hosts.initialized,
    hostModelsLoading: hostModels.loading,
    allReadyHostsFetched: hostModels.allReadyHostsFetched,
    installed: installedModels.value,
  }),
);

/** The request the estimate badge previews — null until a model is chosen. */
const estimateRequest = computed(() => (form.model ? buildRequest(form) : null));

/** True when submits (and the estimate preflight) must resolve a route:
 *  multiple hosts, or a dead primary while another host can serve. */
const routeRequired = computed(() =>
  needsHostRoute({
    multiHost: hosts.multiHost,
    primaryReady: hosts.primaryHost?.status === "ready",
    anyHostReady: hosts.all.some((h) => h.status === "ready"),
  }),
);

/** Preflight against the host the batch will actually route to. */
const estimateTarget = computed(() =>
  routeRequired.value
    ? (hosts.resolveRoute(appPrefs.settings?.generateTargetHost ?? null, form.model || null)
        ?.target ?? null)
    : null,
);

/**
 * What the picker renders: the all-host union under Auto / Most capable,
 * narrowed to the sticky host's installed set when one is picked. The current
 * `form.model` is left alone — `stickyHostMissingModel` already warns that a
 * generate there will auto-pull the weights.
 */
const pickerModels = computed<ModelEntry[]>(() => {
  const target = appPrefs.settings?.generateTargetHost ?? null;
  const fetched = target && target !== "capable" && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
  return filterModelsForTarget(
    installedModels.value,
    target,
    fetched ? new Set(hostModels.installedOn(target).map((m) => m.name)) : null,
  );
});

/**
 * The picker's list: the primary's installed models merged with every model
 * installed on an extra host, grouped by family (primary entries win the
 * dedup so their defaults are used).
 */
const pickerFamilies = computed<Map<string, ModelEntry[]>>(() => {
  const byName = new Map<string, ModelEntry>();
  for (const m of pickerModels.value) byName.set(m.name, m);
  const groups = new Map<string, ModelEntry[]>();
  for (const m of byName.values()) {
    const list = groups.get(m.family) ?? [];
    list.push(m);
    groups.set(m.family, list);
  }
  return groups;
});

/** Subtle per-row tag for models that live only on non-primary hosts. */
function availabilityTag(m: ModelEntry): string | null {
  if (!hosts.multiHost) return null;
  // With a sticky host every rendered row is on that host — tags are noise.
  const target = appPrefs.settings?.generateTargetHost ?? null;
  if (target && target !== "capable") return null;
  return modelAvailabilityTag(hostModels.hostsFor(m.name), hosts.all);
}

/**
 * The sticky target host's label when it lacks the selected model (per the
 * last availability snapshot) — the job will auto-pull the weights there.
 */
const stickyHostMissingModel = computed<string | null>(() => {
  const sel = appPrefs.settings?.generateTargetHost ?? null;
  if (!sel || sel === "capable" || !form.model) return null;
  const host = hosts.all.find((h) => h.id === sel);
  if (!host) return null;
  const ids = hostModels.hostsFor(form.model);
  if (ids.length === 0 || ids.includes(sel)) return null;
  return host.label;
});

// Availability data is demand-driven: fetch on mount / when the set of ready
// hosts changes, and force-fresh whenever the picker opens (a model pulled on
// an extra host by another client shows up the moment the user looks). No
// global timers.
watch(pickerOpen, (open) => {
  if (open) void hostModels.refresh(true);
});
watch(
  () =>
    hosts.all
      .filter((h) => h.status === "ready")
      .map((h) => h.id)
      .join("\n"),
  () => void hostModels.refresh(),
  // immediate: routing must be model-aware on the FIRST Generate click, not
  // only after the picker has been opened once (peer review on #390).
  { immediate: true },
);

const buttonLabel = computed(() =>
  generation.pending.length > 0 ? `Generate (+${generation.pending.length} queued)` : "Generate",
);

const previewWidth = computed(() => job.value?.width ?? form.width);
const previewHeight = computed(() => job.value?.height ?? form.height);
const previewFrameStyle = computed(() => ({
  aspectRatio: `${previewWidth.value} / ${previewHeight.value}`,
  width: `${previewFrameSize.value.width}px`,
  height: `${previewFrameSize.value.height}px`,
}));

let previewResizeObserver: ResizeObserver | null = null;

function resizePreview(width?: number, height?: number) {
  const rect = previewRegion.value?.getBoundingClientRect();
  previewFrameSize.value = fitAspectRatio(
    width ?? rect?.width ?? 0,
    height ?? rect?.height ?? 0,
    previewWidth.value,
    previewHeight.value,
  );
}

watch([previewWidth, previewHeight], () => resizePreview());

const edgeCode = computed(() => {
  const j = job.value;
  if (!j) return "";
  const name = j.model.toUpperCase().replace(":", "·");
  const s = j.result ? `S ${j.result.seed_used}` : `S ${j.visualSeed.slice(0, 12)}`;
  const stepPart = `${j.status === "complete" ? j.total : j.step}/${j.total}`;
  const size = j.result ? `${j.result.width}×${j.result.height}` : `${j.width}×${j.height}`;
  const time = j.result ? `${(j.result.generation_time_ms / 1000).toFixed(1)}s` : "";
  return [name, s, stepPart, size, time].filter(Boolean).join("  ");
});

function pickModel(m: ModelEntry) {
  formStore.applyModel(m);
  pickerOpen.value = false;
}

function loadTemplate(template: GenerationTemplate) {
  // Base64 media was stripped on save; buildRequest's pruneRequestForFamily
  // still guards anything the (possibly different) family can't use.
  Object.assign(form, template.form);
  if (form.model && !findInstalledModel(installedModels.value, form.model)) {
    toasts.push(`Model "${form.model}" isn't installed — settings applied anyway.`);
  }
  if (template.mediaReferences.length > 0) {
    toasts.push(`Re-add media: ${template.mediaReferences.join(", ")}.`);
  }
}

function siblingDot(s: Job): string {
  if (s.status === "complete") return "text-ink"; // ◉ developed
  if (s.status === "error") return "text-stop";
  return "text-ink-3"; // ◎ pending
}

function canvasMenu(): MenuEntry[] {
  const j = job.value;
  if (!j) return [];
  const live = j.status !== "complete" && j.status !== "error";
  return [
    {
      label: "Cancel",
      danger: true,
      disabled: !live,
      action: () => void generation.cancel(j.clientId).then(() => toasts.push("Cancelled")),
    },
    { separator: true },
    {
      label: "Copy prompt",
      action: () => void navigator.clipboard.writeText(j.prompt),
    },
    {
      label: "Copy seed",
      disabled: !j.result,
      action: () => void navigator.clipboard.writeText(String(j.result?.seed_used ?? "")),
    },
    {
      label: "Copy image",
      disabled: !j.result || !!j.result.video_frames,
      action: () => {
        if (!j.result) return;
        const mime = j.result.format === "jpeg" ? "image/jpeg" : `image/${j.result.format}`;
        void copyBase64ImageToClipboard(j.result.image, mime)
          .then(() => toasts.push("Image copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
      },
    },
    { separator: true },
    {
      label: "Show in Gallery",
      disabled: j.status !== "complete",
      action: () => void router.push("/gallery"),
    },
  ];
}

function onExpandApply(payload: { expanded: string; original: string }) {
  form.prompt = payload.expanded;
  form.originalPrompt = payload.original;
}
function onExpandRestore(original: string) {
  form.prompt = original;
  form.originalPrompt = null;
}
function appendPromptWord(word: string) {
  const trimmed = word.trim();
  if (!trimmed) return;
  form.prompt = form.prompt.trim() ? `${form.prompt.trimEnd()}, ${trimmed}` : trimmed;
}

/** Status line while the source is upscaled/refit ahead of the submit. */
const preprocessingStatus = ref<string | null>(null);

/**
 * Apply the source-fit policy to the attached source (and mask) before the
 * request is built: canvas-fit a mismatched source, generate the pad mask
 * for pad-repaint, and for upscale-then-fit run the source through
 * `POST /api/upscale/stream` first. `route` is the ALREADY-RESOLVED
 * generation host so the upscaler model auto-downloads on the same machine
 * the job will run on. Returns false when the submit must abort.
 */
async function preprocessSourceFit(route: HostRoute | null): Promise<boolean> {
  if (!caps.value.supportsImg2img || caps.value.sourceImageMode !== "single") return true;
  if (!form.sourceImage) return true;
  try {
    const result = await applySourceFitPreprocess(
      {
        source: form.sourceImage,
        mask: form.maskImage,
        policy: form.sourceFit,
        target: { width: form.width, height: form.height },
      },
      {
        ops: domCanvasOps,
        upscale: (image, model) =>
          upscaleImage({
            model,
            image,
            ...(route ? { target: route.target } : {}),
            onProgress: (message) => (preprocessingStatus.value = message),
          }),
        onStatus: (message) => (preprocessingStatus.value = message),
      },
    );
    form.sourceImage = result.source;
    form.maskImage = result.mask;
    return true;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    toasts.push(`Source preprocessing failed: ${message}`, "error");
    return false;
  } finally {
    preprocessingStatus.value = null;
  }
}

async function generate() {
  if (!form.prompt.trim() || !form.model || chainReject.value) return;
  const batch = caps.value.forcesBatchSizeOne ? 1 : form.batchSize;
  // With multiple live hosts — or a dead primary while another host can
  // serve — route the batch (sticky pick, Auto = least busy, or Most
  // capable) — model-aware, so hosts that already have the weights win.
  // A pinned host that went away is an error, not a reroute. Resolved
  // BEFORE source preprocessing so upscale-then-fit hits the same host.
  let route = null;
  if (routeRequired.value) {
    route = hosts.resolveRoute(appPrefs.settings?.generateTargetHost ?? null, form.model || null);
    if (!route) {
      toasts.push("The selected host isn't reachable — pick another host.", "error");
      return;
    }
  }
  if (!(await preprocessSourceFit(route))) return;
  const request = buildRequest(form);
  // Submitting while another print develops queues server-side; each job
  // snapshots its own model + params, so tweaking the form afterwards is safe.
  const { settled } = generation.submitBatch(request, batch, route);
  void loadPromptHistory();
  const done = await settled;
  const ok = done.filter((s) => s.status === "complete").length;
  const failed = done.find((s) => s.status === "error");
  if (ok > 0) {
    toasts.push(
      ok === 1 ? "Generated — saved to Gallery" : `Generated ${ok} prints — saved to Gallery`,
    );
    // Gallery refresh is handled by the generation store's complete hook
    // (per-origin bucket) plus the SSE / fallback-poll paths.
  } else if (failed?.error && failed.error !== "Cancelled") {
    toasts.push(failed.error, "error");
  }
}

// ↑/↓ cycle recent prompts (shell-history style) when the caret is on the
// composer's first/last line, so multi-line editing keeps native arrows.
const cycler = new PromptCycler();

/** The composer grows with its content (capped) instead of scrolling at 2 rows. */
function growPrompt() {
  if (promptEl.value) autoGrowRows(promptEl.value);
}
// Programmatic prompt changes (history cycling, expand, templates) resize too.
watch(
  () => form.prompt,
  () => void nextTick(growPrompt),
  { flush: "post" },
);
onMounted(() => growPrompt());

async function loadPromptHistory() {
  try {
    cycler.setEntries((await fetchHistory("", 100)).map((e) => e.prompt));
  } catch {
    // No history API (older engine / DB off) — arrows just move the caret.
  }
}

function cycleHistory(direction: "prev" | "next") {
  const replacement = direction === "prev" ? cycler.prev(form.prompt) : cycler.next();
  if (replacement === null) return;
  form.prompt = replacement;
  void nextTick(() => {
    const el = promptEl.value;
    el?.setSelectionRange(el.value.length, el.value.length);
  });
}

function onComposerKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && primaryModifierPressed(e)) {
    e.preventDefault();
    void generate();
  } else if ((e.key === "e" || e.key === "E") && primaryModifierPressed(e)) {
    e.preventDefault();
    expandControl.value?.expand();
  } else if (e.key === "ArrowUp" && promptEl.value && caretOnFirstLine(promptEl.value)) {
    e.preventDefault();
    cycleHistory("prev");
  } else if (e.key === "ArrowDown" && promptEl.value && caretOnLastLine(promptEl.value)) {
    e.preventDefault();
    cycleHistory("next");
  }
}

// Auto-select a default model only when none is set. With the form persisted
// in a store, the `!form.model` guard now means "the first ever visit" — a
// remount after the user chose a model leaves their choice untouched.
watch(
  () => installedModels.value,
  (installed) => {
    if (!form.model && installed.length > 0) {
      const preferred = preferredInstalledModel(installed);
      if (preferred) formStore.applyModel(preferred);
    }
  },
  { immediate: true },
);

// Refetch on every visit — a pull may have finished since the last look.
watch(
  () => conn.ready,
  (ready) => {
    if (ready) {
      void models.fetch();
      void loadPromptHistory();
    }
  },
  { immediate: true },
);

function applyPrefill() {
  const prefill = composer.take();
  if (!prefill) return;
  // Gallery reuse ships full metadata (full-fidelity restore); palette /
  // history / jobs keep the legacy scalar copy.
  applyPrefillToForm(form, prefill, installedModels.value);
  void nextTick(() => promptEl.value?.focus());
}

// Apply a prefill whenever one arrives (Reuse settings, history, "Generate
// with <model>"), including one already queued before this view mounted.
watch(() => composer.prefill, applyPrefill, { immediate: true });

// ⌘N — clear the composer for a fresh generation, keeping the model.
watch(
  () => ui.newGenerationTick,
  () => {
    formStore.clearComposer();
    void nextTick(() => promptEl.value?.focus());
  },
);

// ⌘R — randomize the seed.
watch(
  () => ui.randomizeSeedTick,
  () => {
    form.seed = String(randomSeed());
  },
);

// Menu ▸ Generate / Expand Prompt reuse the composer actions.
watch(
  () => ui.generateTick,
  () => void generate(),
);
watch(
  () => ui.expandTick,
  () => expandControl.value?.expand(),
);

onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointerDown);
  promptEl.value?.focus();
  if (previewRegion.value && typeof ResizeObserver !== "undefined") {
    previewResizeObserver = new ResizeObserver(([entry]) => {
      if (entry) resizePreview(entry.contentRect.width, entry.contentRect.height);
    });
    previewResizeObserver.observe(previewRegion.value);
  }
  resizePreview();
});

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onDocumentPointerDown);
  previewResizeObserver?.disconnect();
});
</script>

<template>
  <StarterCards v-if="showStarterCards" @browse="router.push('/models')" />

  <div
    v-else
    data-test="generate-layout"
    class="relative grid h-full min-h-0 overflow-hidden"
    :style="{ gridTemplateColumns: `1fr ${asideWidth}px` }"
  >
    <!-- Canvas + composer -->
    <div data-test="generate-workbench" class="flex min-h-0 min-w-0 flex-col overflow-hidden p-6">
      <div class="flex min-h-0 flex-1 flex-col">
        <div
          ref="previewRegion"
          data-test="preview-region"
          class="flex min-h-0 flex-1 items-center justify-center overflow-hidden"
        >
          <div
            class="relative w-full overflow-hidden rounded-media border border-control-edge"
            :class="job ? 'bg-print-surface' : 'bg-empty-surface'"
            data-test="preview-frame"
            :style="previewFrameStyle"
            @contextmenu="job && contextMenu.open($event, canvasMenu())"
          >
            <video
              v-if="job?.resultUrl && job.result?.video_frames"
              :src="job.resultUrl"
              class="absolute inset-0 h-full w-full object-contain"
              autoplay
              loop
              controls
            />
            <img
              v-else-if="job?.resultUrl"
              :src="job.resultUrl"
              alt=""
              class="absolute inset-0 h-full w-full object-contain transition-opacity duration-500"
            />
            <!-- Live latent preview: a tiny PNG upscaled by CSS; the blur
                 tightens as denoising progresses and the grain resolves
                 over it, so the print literally develops on the canvas. -->
            <img
              v-if="job && job.status !== 'complete' && job.previewUrl"
              :src="job.previewUrl"
              alt=""
              class="absolute inset-0 h-full w-full object-cover"
              :style="{ filter: `blur(${Math.max(2, 14 - 12 * jobProgress(job))}px)` }"
            />
            <!-- The grain canvas paints edge-to-edge (temperature wash), so
                 once previews exist it thins out with progress to reveal
                 the forming print underneath. -->
            <DevelopCanvas
              v-if="job && job.status !== 'complete'"
              :seed="job.visualSeed"
              :progress="jobProgress(job)"
              :phase="jobPhase(job)"
              class="absolute inset-0"
              :style="{
                opacity: job.previewUrl ? String(Math.max(0.18, 1 - jobProgress(job) * 0.9)) : '1',
              }"
            />
            <div
              v-if="!job"
              data-test="empty-canvas"
              class="absolute inset-0 flex items-center justify-center p-6 text-center"
            >
              <div class="flex max-w-64 flex-col items-center">
                <div
                  class="border-halide/40 mb-4 flex h-20 w-24 items-center justify-center rounded-media border bg-[color-mix(in_srgb,var(--halide)_7%,transparent)]"
                  aria-hidden="true"
                >
                  <svg
                    viewBox="0 0 48 40"
                    class="h-10 w-12 text-halide/70"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.5"
                  >
                    <rect x="4" y="4" width="40" height="32" rx="1" />
                    <circle cx="33" cy="13" r="3" />
                    <path d="m9 31 10-11 7 7 5-5 8 9" />
                  </svg>
                </div>
                <div class="font-display text-display-sm font-semibold text-ink">No print yet</div>
                <p class="mt-1 text-caption text-ink-2">
                  Choose a model, describe your print, then generate.
                </p>
              </div>
            </div>
            <div
              v-if="job && (job.status === 'denoising' || job.status === 'finishing')"
              class="edge-code absolute bottom-2 left-3"
            >
              <template v-if="job.status === 'denoising'">{{ job.step }}/{{ job.total }}</template>
              <template v-else>Fixing — {{ job.stage ?? "finishing" }}…</template>
            </div>
          </div>
        </div>

        <div v-if="job" class="edge-code mt-2 truncate" :title="edgeCode">{{ edgeCode }}</div>

        <!-- Batch dots -->
        <div v-if="siblings.length > 1" class="mt-2 flex items-center gap-1.5">
          <span
            v-for="(s, i) in siblings"
            :key="i"
            class="data-mono text-body"
            :class="siblingDot(s)"
            :title="`${i + 1} of ${siblings.length}`"
          >
            {{ s.status === "complete" ? "◉" : s.status === "error" ? "◉" : "◎" }}
          </span>
          <span class="edge-code ml-1">
            {{ siblings.filter((s) => s.status === "complete").length }} of {{ siblings.length }}
          </span>
        </div>

        <p v-if="job?.status === 'error'" class="mt-2 text-caption text-stop">{{ job.error }}</p>
      </div>

      <!-- Composer -->
      <div
        data-test="generate-composer"
        class="mt-4 shrink-0 rounded-chrome border border-control-edge bg-bench p-3 transition-colors duration-100 focus-within:border-safelight"
      >
        <textarea
          ref="promptEl"
          v-model="form.prompt"
          data-selectable
          rows="2"
          aria-label="Prompt"
          placeholder="Describe the print — a lighthouse at dusk, kodak portra…"
          class="w-full resize-none overflow-x-hidden bg-transparent text-body-lg text-ink outline-none placeholder:text-ink-3"
          @keydown="onComposerKeydown"
          @input="
            cycler.reset();
            growPrompt();
          "
        />
        <div class="mt-2 flex min-w-0 flex-wrap items-center justify-between gap-2">
          <ExpandControl
            ref="expandControl"
            :prompt="form.prompt"
            :family="form.family"
            @apply="onExpandApply"
            @restore="onExpandRestore"
          />
          <div class="flex min-w-0 items-center gap-3">
            <span
              v-if="preprocessingStatus"
              class="truncate text-caption text-ink-3"
              data-test="preprocessing-status"
            >
              {{ preprocessingStatus }}
            </span>
            <EstimateBadge :request="estimateRequest" :target="estimateTarget" />
            <button
              type="button"
              class="h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-60"
              :disabled="!form.prompt.trim() || !form.model || chainReject"
              @click="generate"
            >
              {{ buttonLabel }}
              <kbd class="kbd-hint ml-1.5 opacity-80">{{ shortcutLabel("↩") }}</kbd>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Inspector. overflow-x-hidden is load-bearing: with only overflow-y
         set, any child momentarily wider than the 320px column during a
         window resize computes overflow-x to auto and grows a horizontal
         scrollbar under the prompt area. -->
    <aside class="border-edge overflow-x-hidden overflow-y-auto border-l bg-bench p-4">
      <HostSelector />
      <div class="mb-2 flex items-center gap-2">
        <span class="edge-code">Model</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>
      <div ref="pickerEl" class="relative">
        <button
          type="button"
          :aria-expanded="pickerOpen"
          class="border-edge flex min-h-9 w-full items-center justify-between gap-2 rounded-control border bg-bath px-2 py-1.5 text-body text-ink"
          @click="pickerOpen = !pickerOpen"
        >
          <span data-test="selected-model-name" class="min-w-0 break-all text-left">{{
            selectedModel?.name ?? "Choose a model"
          }}</span>
          <span v-if="selectedModel?.disk_usage_bytes" class="data-mono shrink-0 text-ink-3">
            {{ formatGB(selectedModel.disk_usage_bytes) }}
          </span>
        </button>
        <div
          v-if="pickerOpen"
          data-test="model-picker-menu"
          class="border-edge absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-chrome border bg-bench shadow-raised"
        >
          <template v-for="[family, list] in pickerFamilies" :key="family">
            <div class="edge-code px-2 pt-2 pb-1">{{ family.toUpperCase() }}</div>
            <button
              v-for="m in list"
              :key="m.name"
              type="button"
              class="flex w-full items-start gap-2 px-2 py-1.5 text-left text-body text-ink-2 hover:bg-bath hover:text-ink"
              @click="pickModel(m)"
            >
              <SourceGlyph :source="modelSource(m)" class="mt-0.5 shrink-0 text-ink-3" />
              <span class="min-w-0 flex-1">
                <span
                  data-test="model-option-name"
                  class="block break-all text-ink"
                  :title="m.name"
                >
                  {{ m.name }}
                </span>
                <span
                  v-if="availabilityTag(m)"
                  data-test="model-availability"
                  class="edge-code mt-0.5 block break-all whitespace-normal"
                >
                  {{ availabilityTag(m) }}
                </span>
              </span>
              <span
                class="mt-2 h-1.5 w-1.5 shrink-0 rounded-full"
                :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
                :title="m.is_loaded ? 'On GPU' : ''"
              />
            </button>
          </template>
          <button
            type="button"
            data-test="browse-catalog"
            class="border-edge flex w-full items-center border-t px-2 py-2 text-left text-body text-halide hover:bg-bath"
            @click="
              pickerOpen = false;
              void router.push(caps.supportsVideo ? '/models?type=video' : '/models');
            "
          >
            Browse all models →
          </button>
        </div>
      </div>
      <p v-if="stickyHostMissingModel" class="mt-1.5 text-caption text-ink-3">
        Not on {{ stickyHostMissingModel }} — will download there.
      </p>

      <ParamPanel
        :form="form"
        :last-seed="generation.lastSeedUsed"
        :upscalers="models.upscalers"
        class="mt-5"
      />
      <SourceImageWell :form="form" />
      <LoraStack
        v-if="caps.supportsLora"
        :form="form"
        :model="form.model"
        @append-word="appendPromptWord"
      />
      <TemplatesPanel :form="form" @load="loadTemplate" />
    </aside>

    <!-- The aside scrolls, so its resize handle lives on the (relative) grid
         container, pinned to the column boundary and straddling the border. -->
    <PanelResizeHandle
      class="absolute inset-y-0 z-10 translate-x-1/2"
      :style="{ right: `${asideWidth}px` }"
      label="Resize inspector"
      @resize="onAsideResize"
      @commit="onAsideCommit"
      @reset="onAsideReset"
    />
  </div>
</template>
