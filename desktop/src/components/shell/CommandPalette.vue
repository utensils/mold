<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { useUiStore } from "../../stores/ui";
import { useModelStore } from "../../stores/models";
import { useComposerStore } from "../../stores/composer";
import { useGenerationStore } from "../../stores/generation";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";
import { matchCommands, type Matchable } from "../../lib/palette";
import { fetchHistory, type HistoryEntry } from "../../lib/api/history";
import { newGenerateForm, applyModelDefaults } from "../../lib/generateForm";

interface Command extends Matchable {
  id: string;
  title: string;
  subtitle?: string;
  run: () => void;
}

const ui = useUiStore();
const router = useRouter();
const models = useModelStore();
const composer = useComposerStore();
const generation = useGenerationStore();
const conn = useConnectionStore();
const toasts = useToastStore();

const query = ref("");
const selected = ref(0);
const inputEl = ref<HTMLInputElement | null>(null);
const history = ref<HistoryEntry[]>([]);
let debounce: ReturnType<typeof setTimeout> | null = null;

function close() {
  ui.closePalette();
}
function go(route: string) {
  void router.push(route);
  close();
}

/** Prefill the composer with a model's defaults (and optional prompt). */
function prefillModel(model: string, prompt = "") {
  const installed = models.installed.find((m) => m.name === model);
  const form = newGenerateForm();
  if (installed) applyModelDefaults(form, installed);
  composer.set({
    prompt,
    model: installed ? model : form.model,
    seed: null,
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: form.guidance,
  });
  go("/generate");
}

const staticCommands = computed<Command[]>(() => {
  const cmds: Command[] = [
    { id: "nav-generate", title: "Go to Generate", run: () => go("/generate") },
    { id: "nav-gallery", title: "Go to Gallery", run: () => go("/gallery") },
    { id: "nav-chains", title: "Go to Chains", run: () => go("/chains") },
    { id: "nav-models", title: "Go to Models", run: () => go("/models") },
    { id: "nav-history", title: "Go to History", run: () => go("/history") },
    { id: "nav-settings", title: "Go to Settings", run: () => go("/settings") },
    {
      id: "act-new",
      title: "New generation",
      keywords: ["clear", "compose"],
      run: () => {
        ui.newGeneration();
        go("/generate");
      },
    },
    {
      id: "act-seed",
      title: "Randomize seed",
      run: () => {
        ui.randomizeSeed();
        go("/generate");
      },
    },
    {
      id: "act-engine-local",
      title: "Switch to built-in engine",
      keywords: ["local", "engine"],
      run: () => {
        void conn.useLocal();
        close();
      },
    },
    {
      id: "act-engine-remote",
      title: "Set remote host…",
      keywords: ["engine", "server"],
      run: () => go("/settings"),
    },
  ];
  if (
    generation.active &&
    generation.active.status !== "complete" &&
    generation.active.status !== "error"
  ) {
    cmds.push({
      id: "act-cancel",
      title: "Cancel job",
      run: () => {
        void generation.cancel().then(() => toasts.push("Cancelled"));
        close();
      },
    });
  }
  for (const m of models.installed) {
    cmds.push({
      id: `model-${m.name}`,
      title: `Generate with ${m.name}`,
      subtitle: m.family,
      keywords: [m.family, "model"],
      run: () => prefillModel(m.name),
    });
  }
  return cmds;
});

const historyCommands = computed<Command[]>(() =>
  history.value.map((h, i) => ({
    id: `history-${i}`,
    title: h.prompt,
    subtitle: `${h.model} · history`,
    run: () => prefillModel(h.model, h.prompt),
  })),
);

const results = computed<Command[]>(() => [
  ...matchCommands(query.value, staticCommands.value),
  ...historyCommands.value,
]);

watch(results, () => {
  selected.value = 0;
});

// Debounced history search whenever the palette is open.
watch(query, (q) => {
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(() => {
    void fetchHistory(q, 6)
      .then((entries) => (history.value = entries))
      .catch(() => (history.value = []));
  }, 200);
});

watch(
  () => ui.paletteOpen,
  (open) => {
    if (open) {
      query.value = "";
      selected.value = 0;
      history.value = [];
      void fetchHistory("", 6)
        .then((entries) => (history.value = entries))
        .catch(() => {});
      void nextTick(() => inputEl.value?.focus());
    }
  },
);

function move(delta: number) {
  const n = results.value.length;
  if (n === 0) return;
  selected.value = (selected.value + delta + n) % n;
}
function runSelected() {
  results.value[selected.value]?.run();
}
function onKeydown(e: KeyboardEvent) {
  if (e.key === "ArrowDown") {
    e.preventDefault();
    move(1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    move(-1);
  } else if (e.key === "Enter") {
    e.preventDefault();
    runSelected();
  } else if (e.key === "Escape") {
    e.preventDefault();
    close();
  }
}
</script>

<template>
  <div
    v-if="ui.paletteOpen"
    class="fixed inset-0 z-50 flex justify-center pt-24"
    @click.self="close"
  >
    <div
      class="border-edge h-fit w-[36rem] max-w-[90vw] overflow-hidden rounded-chrome border bg-bench shadow-raised"
    >
      <input
        ref="inputEl"
        v-model="query"
        data-selectable
        type="text"
        placeholder="Search or run a command…"
        class="border-edge w-full border-b bg-transparent px-4 py-3 text-body-lg text-ink outline-none placeholder:text-ink-3"
        @keydown="onKeydown"
      />
      <div class="max-h-80 overflow-y-auto py-1">
        <p v-if="results.length === 0" class="px-4 py-3 text-caption text-ink-3">No matches.</p>
        <button
          v-for="(cmd, i) in results"
          :key="cmd.id"
          type="button"
          class="flex w-full items-center gap-3 px-4 py-2 text-left"
          :class="i === selected ? 'bg-bath' : 'hover:bg-bath'"
          @mouseenter="selected = i"
          @click="cmd.run()"
        >
          <span
            class="min-w-0 flex-1 truncate text-body"
            :class="i === selected ? 'text-ink' : 'text-ink-2'"
          >
            {{ cmd.title }}
          </span>
          <span v-if="cmd.subtitle" class="edge-code shrink-0">{{ cmd.subtitle }}</span>
        </button>
      </div>
    </div>
  </div>
</template>
