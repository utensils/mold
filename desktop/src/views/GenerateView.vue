<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import StarterCards from "../components/generate/StarterCards.vue";
import ParamPanel from "../components/generate/ParamPanel.vue";
import LoraStack from "../components/generate/LoraStack.vue";
import TemplatesPanel from "../components/generate/TemplatesPanel.vue";
import SourceImageWell from "../components/generate/SourceImageWell.vue";
import EstimateBadge from "../components/generate/EstimateBadge.vue";
import ExpandControl from "../components/generate/ExpandControl.vue";
import HostSelector from "../components/generate/HostSelector.vue";
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostsStore } from "../stores/hosts";
import { useConnectionStore } from "../stores/connection";
import { useGenerationStore, jobPhase, jobProgress, type Job } from "../stores/generation";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGalleryStore } from "../stores/gallery";
import { useModelStore } from "../stores/models";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { copyBase64ImageToClipboard } from "../lib/clipboard";
import { useUiStore } from "../stores/ui";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { buildRequest } from "../lib/generateForm";
import type { GenerationTemplate } from "../lib/generationTemplates";
import { PromptCycler, caretOnFirstLine, caretOnLastLine } from "../lib/promptCycler";
import { fetchHistory } from "../lib/api/history";
import { formatGB } from "../lib/format";
import { randomSeed } from "../stores/generation";
import type { ModelEntry } from "../lib/api/types";
import { fitAspectRatio } from "../lib/fitAspectRatio";

const router = useRouter();
const conn = useConnectionStore();
const hosts = useHostsStore();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const gallery = useGalleryStore();
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
const pickerOpen = ref(false);

const job = computed(() => generation.active);
const siblings = computed(() => generation.siblings);
const caps = computed(() => generationCapabilitiesForFamily(form.family));
const selectedModel = computed<ModelEntry | null>(
  () => models.installed.find((m) => m.name === form.model) ?? null,
);

/** The request the estimate badge previews — null until a model is chosen. */
const estimateRequest = computed(() => (form.model ? buildRequest(form) : null));

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
  if (form.model && !models.installed.some((m) => m.name === form.model)) {
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

async function generate() {
  if (!form.prompt.trim() || !form.model) return;
  const request = buildRequest(form);
  const batch = caps.value.forcesBatchSizeOne ? 1 : form.batchSize;
  // With multiple live hosts, route the batch (sticky pick or Auto = least
  // busy). A pinned host that went away is an error, not a silent reroute.
  let route = null;
  if (hosts.multiHost) {
    route = hosts.resolveRoute(appPrefs.settings?.generateTargetHost ?? null);
    if (!route) {
      toasts.push("The selected host isn't reachable — pick another host.", "error");
      return;
    }
  }
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
    void gallery.fetch();
  } else if (failed?.error && failed.error !== "Cancelled") {
    toasts.push(failed.error, "error");
  }
}

// ↑/↓ cycle recent prompts (shell-history style) when the caret is on the
// composer's first/last line, so multi-line editing keeps native arrows.
const cycler = new PromptCycler();

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
  if (e.key === "Enter" && e.metaKey) {
    e.preventDefault();
    void generate();
  } else if ((e.key === "e" || e.key === "E") && e.metaKey) {
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
  () => models.installed,
  (installed) => {
    if (!form.model && installed.length > 0) {
      const preferred = installed.find((m) => m.family === "flux") ?? installed[0]!;
      formStore.applyModel(preferred);
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
  form.prompt = prefill.prompt;
  form.model = prefill.model;
  form.seed = prefill.seed !== null ? String(prefill.seed) : "";
  form.width = prefill.width;
  form.height = prefill.height;
  form.steps = prefill.steps;
  form.guidance = prefill.guidance;
  const m = models.installed.find((x) => x.name === prefill.model);
  if (m) form.family = m.family;
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
  promptEl.value?.focus();
  if (previewRegion.value && typeof ResizeObserver !== "undefined") {
    previewResizeObserver = new ResizeObserver(([entry]) => {
      if (entry) resizePreview(entry.contentRect.width, entry.contentRect.height);
    });
    previewResizeObserver.observe(previewRegion.value);
  }
  resizePreview();
});

onBeforeUnmount(() => previewResizeObserver?.disconnect());
</script>

<template>
  <StarterCards
    v-if="conn.ready && !models.loading && models.installed.length === 0"
    @browse="router.push('/models')"
  />

  <div
    v-else
    data-test="generate-layout"
    class="grid h-full min-h-0 grid-cols-[1fr_320px] overflow-hidden"
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
            class="relative w-full overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
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
              class="absolute inset-0 flex items-center justify-center text-caption text-ink-3"
            >
              The print develops here
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
        class="mt-4 shrink-0 rounded-chrome border border-edge bg-bench p-3 transition-colors duration-100 focus-within:border-safelight"
      >
        <textarea
          ref="promptEl"
          v-model="form.prompt"
          data-selectable
          rows="2"
          aria-label="Prompt"
          placeholder="Describe the print — a lighthouse at dusk, kodak portra…"
          class="w-full resize-none bg-transparent text-body-lg text-ink outline-none placeholder:text-ink-3"
          @keydown="onComposerKeydown"
          @input="cycler.reset()"
        />
        <div class="mt-2 flex items-center justify-between gap-2">
          <ExpandControl
            ref="expandControl"
            :prompt="form.prompt"
            :family="form.family"
            @apply="onExpandApply"
            @restore="onExpandRestore"
          />
          <div class="flex items-center gap-3">
            <EstimateBadge :request="estimateRequest" />
            <button
              type="button"
              class="h-9 rounded-chrome bg-safelight px-4 text-body font-semibold text-[#141110] transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-60"
              :disabled="!form.prompt.trim() || !form.model"
              @click="generate"
            >
              {{ buttonLabel }}
              <kbd class="kbd-hint ml-1.5 opacity-70">⌘↩</kbd>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Inspector -->
    <aside class="border-edge overflow-y-auto border-l bg-bench p-4">
      <HostSelector />
      <div class="mb-2 flex items-center gap-2">
        <span class="edge-code">Model</span>
        <div class="border-edge h-px flex-1 border-t" />
      </div>
      <div class="relative">
        <button
          type="button"
          class="border-edge flex h-9 w-full items-center justify-between rounded-control border bg-bath px-2 text-body text-ink"
          @click="pickerOpen = !pickerOpen"
        >
          <span class="truncate">{{ selectedModel?.name ?? "Choose a model" }}</span>
          <span v-if="selectedModel?.disk_usage_bytes" class="data-mono ml-2 text-ink-3">
            {{ formatGB(selectedModel.disk_usage_bytes) }}
          </span>
        </button>
        <div
          v-if="pickerOpen"
          class="border-edge absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-chrome border bg-bench shadow-raised"
        >
          <template v-for="[family, list] in models.byFamily" :key="family">
            <div class="edge-code px-2 pt-2 pb-1">{{ family.toUpperCase() }}</div>
            <button
              v-for="m in list"
              :key="m.name"
              type="button"
              class="flex w-full items-center justify-between px-2 py-1.5 text-left text-body text-ink-2 hover:bg-bath hover:text-ink"
              @click="pickModel(m)"
            >
              <span class="truncate">{{ m.name }}</span>
              <span
                class="ml-2 h-1.5 w-1.5 shrink-0 rounded-full"
                :class="m.is_loaded ? 'bg-safelight' : 'bg-transparent'"
                :title="m.is_loaded ? 'On GPU' : ''"
              />
            </button>
          </template>
        </div>
      </div>

      <ParamPanel :form="form" class="mt-5" />
      <SourceImageWell :form="form" />
      <LoraStack
        v-if="caps.supportsLora"
        :form="form"
        :model="form.model"
        @append-word="appendPromptWord"
      />
      <TemplatesPanel :form="form" @load="loadTemplate" />
    </aside>
  </div>
</template>
