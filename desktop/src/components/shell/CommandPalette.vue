<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import { useUiStore } from "../../stores/ui";
import { useGalleryStore } from "../../stores/gallery";
import { useModelStore } from "../../stores/models";
import { useComposerStore } from "../../stores/composer";
import { useGenerationStore } from "../../stores/generation";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { matchCommands, type Matchable } from "../../lib/palette";
import { fetchHistory, type HistoryEntry } from "../../lib/api/history";
import { loadModel, unloadModel } from "../../lib/api/models";
import { newGenerateForm, applyModelDefaults } from "../../lib/generateForm";

interface Command extends Matchable {
  id: string;
  title: string;
  subtitle?: string;
  run: () => void;
}

const ui = useUiStore();
const router = useRouter();
const gallery = useGalleryStore();
const models = useModelStore();
const composer = useComposerStore();
const generation = useGenerationStore();
const conn = useConnectionStore();
const toasts = useToastStore();
const appPrefs = useAppPrefsStore();

/** Mono section label for a result row, derived from its id prefix. */
function sectionLabel(id: string): string {
  if (id.startsWith("nav-")) return "go";
  if (id.startsWith("theme-") || id.startsWith("appear-")) return "theme";
  if (id.startsWith("model-") || id.startsWith("load-") || id.startsWith("unload-")) return "model";
  if (id.startsWith("history-")) return "recent";
  if (id.startsWith("print-")) return "print";
  return "run";
}

const query = ref("");
const selected = ref(0);
const inputEl = ref<HTMLInputElement | null>(null);
const history = ref<HistoryEntry[]>([]);
let debounce: ReturnType<typeof setTimeout> | null = null;
// The element focused before the palette opened, so we can hand focus back on
// close (keyboard users don't get dumped at the top of the document).
let restoreFocusEl: HTMLElement | null = null;

const activeOptionId = computed(() =>
  results.value.length > 0 ? `cmd-palette-option-${selected.value}` : undefined,
);

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
  go("/create");
}

const staticCommands = computed<Command[]>(() => {
  const cmds: Command[] = [
    {
      id: "nav-create",
      title: "Go to Create",
      // "generate" keeps the pre-rename muscle memory working.
      keywords: ["generate", "compose"],
      run: () => go("/create"),
    },
    {
      id: "nav-library",
      title: "Go to Library",
      keywords: ["gallery", "prints"],
      run: () => go("/library"),
    },
    {
      id: "nav-models",
      title: "Go to Models",
      // "catalog" keeps the pre-rename muscle memory working.
      keywords: ["models", "catalog"],
      run: () => go("/models"),
    },
    {
      id: "nav-machines",
      title: "Go to Machines",
      keywords: ["hosts", "jobs", "gpu"],
      run: () => go("/machines"),
    },
    {
      id: "nav-sequence",
      title: "Create sequence",
      keywords: ["sequence", "clips", "video"],
      run: () => go("/create?output=sequence"),
    },
    {
      id: "nav-history",
      title: "Open history",
      keywords: ["runs", "prompts", "recent"],
      run: () => go("/library?panel=history"),
    },
    {
      id: "nav-runpod",
      title: "Manage RunPod GPUs",
      keywords: ["cloud", "gpu", "instance"],
      run: () => go("/machines/runpod"),
    },
    { id: "nav-settings", title: "Go to Settings", run: () => go("/settings") },
    {
      id: "act-new",
      title: "New generation",
      keywords: ["clear", "compose"],
      run: () => {
        ui.newGeneration();
        go("/create");
      },
    },
    {
      id: "act-seed",
      title: "Randomize seed",
      run: () => {
        ui.randomizeSeed();
        go("/create");
      },
    },
    // No "Switch to built-in engine" command: the built-in engine is always
    // the primary now — recovery is covered by "Restart engine" below.
    {
      id: "act-add-host",
      title: "Add host…",
      keywords: ["engine", "server", "host", "remote"],
      run: () => go("/settings?section=hosts"),
    },
    {
      id: "act-engine-restart",
      title: "Restart engine",
      keywords: ["engine", "reload", "reboot"],
      run: () => {
        void conn
          .stopEngine()
          .then(() => conn.useLocal())
          .then(() => toasts.push("Engine restarted"));
        close();
      },
    },
    {
      id: "act-copy-seed",
      title: "Copy last seed",
      keywords: ["seed", "clipboard"],
      run: () => {
        ui.copySeed();
        close();
      },
    },
    {
      id: "theme-mold",
      title: "Theme: Mold",
      subtitle: "cyan & magenta",
      keywords: ["theme", "palette", "appearance", "family", "color"],
      run: () => {
        void appPrefs.update({ themeFamily: "mold" });
        close();
      },
    },
    {
      id: "theme-safelight",
      title: "Theme: Safelight",
      subtitle: "warm darkroom",
      keywords: ["theme", "palette", "appearance", "family", "color"],
      run: () => {
        void appPrefs.update({ themeFamily: "safelight" });
        close();
      },
    },
    {
      id: "appear-dark",
      title: "Appearance: Dark",
      keywords: ["theme", "appearance", "dark", "lights off"],
      run: () => {
        void appPrefs.update({ theme: "dark" });
        close();
      },
    },
    {
      id: "appear-light",
      title: "Appearance: Light",
      keywords: ["theme", "appearance", "light", "lights on"],
      run: () => {
        void appPrefs.update({ theme: "light" });
        close();
      },
    },
    {
      id: "appear-system",
      title: "Appearance: System",
      keywords: ["theme", "appearance", "system", "auto"],
      run: () => {
        void appPrefs.update({ theme: "system" });
        close();
      },
    },
  ];
  if (generation.pending.length > 0) {
    cmds.push({
      id: "act-cancel",
      title: "Cancel job",
      run: () => {
        void generation.cancel().then(() => toasts.push("Cancelled"));
        close();
      },
    });
  }
  if (generation.pending.length > 1) {
    cmds.push({
      id: "act-cancel-all",
      title: `Cancel all ${generation.pending.length} jobs`,
      keywords: ["queue", "stop"],
      run: () => {
        const ids = generation.pending.map((j) => j.clientId);
        void Promise.all(ids.map((id) => generation.cancel(id))).then(() =>
          toasts.push("Cancelled all jobs"),
        );
        close();
      },
    });
  }
  if (generation.jobs.some((j) => j.status === "complete" || j.status === "error")) {
    cmds.push({
      id: "act-clear-finished",
      title: "Clear finished jobs",
      keywords: ["jobs", "queue"],
      run: () => {
        generation.prune(0);
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
    if (m.is_loaded) {
      cmds.push({
        id: `unload-${m.name}`,
        title: `Unload ${m.name}`,
        subtitle: "free VRAM",
        keywords: ["unload", "model", "vram", m.family],
        run: () => {
          void unloadModel(m.name)
            .then(() => models.fetch())
            .then(() => toasts.push(`Unloaded ${m.name}`))
            .catch(() => toasts.push(`Couldn't unload ${m.name}.`, "error"));
          close();
        },
      });
    } else {
      cmds.push({
        id: `load-${m.name}`,
        title: `Load ${m.name}`,
        subtitle: "warm up on GPU",
        keywords: ["load", "model", "warm", m.family],
        run: () => {
          toasts.push(`Loading ${m.name}…`);
          void loadModel(m.name)
            .then(() => models.fetch())
            .then(() => toasts.push(`Loaded ${m.name}`))
            .catch(() => toasts.push(`Couldn't load ${m.name}.`, "error"));
          close();
        },
      });
    }
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

/** Prints matching the query — the same fields the Gallery search scans.
 *  Selecting one deep-links to the gallery with that print's lightbox open. */
const galleryCommands = computed<Command[]>(() => {
  const q = query.value.trim().toLowerCase();
  if (!q) return [];
  return gallery.merged
    .filter(
      (e) =>
        e.item.filename.toLowerCase().includes(q) ||
        e.item.metadata.model.toLowerCase().includes(q) ||
        e.item.metadata.prompt.toLowerCase().includes(q),
    )
    .slice(0, 6)
    .map((e) => ({
      id: `print-${e.sourceKey}-${e.item.filename}`,
      title: e.item.metadata.prompt || e.item.filename,
      subtitle: `${e.item.metadata.model} · library`,
      run: () => go(`/library?print=${encodeURIComponent(e.item.filename)}`),
    }));
});

const results = computed<Command[]>(() => [
  ...matchCommands(query.value, staticCommands.value),
  ...galleryCommands.value,
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
      restoreFocusEl = document.activeElement as HTMLElement | null;
      query.value = "";
      selected.value = 0;
      history.value = [];
      void fetchHistory("", 6)
        .then((entries) => (history.value = entries))
        .catch(() => {});
      // Warm the gallery buckets so print results can match, but never
      // refetch on every open — a loaded store is reused as-is.
      if (!gallery.loaded) void gallery.fetchAll();
      void nextTick(() => inputEl.value?.focus());
    } else {
      // Return focus to whatever launched the palette.
      restoreFocusEl?.focus?.();
      restoreFocusEl = null;
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
    class="cmdk-scrim absolute inset-0 z-[120] flex justify-center pt-[84px]"
    @click.self="close"
  >
    <div
      class="ms-fade-up flex h-fit max-h-[440px] w-[560px] max-w-[86%] flex-col overflow-hidden rounded-card-lg border border-ce bg-bench shadow-raised"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div class="flex items-center gap-2.5 border-b border-edge px-4 py-3.5">
        <Icon name="search" :size="17" class="shrink-0 text-ink-3" />
        <input
          ref="inputEl"
          v-model="query"
          data-selectable
          type="text"
          placeholder="Search actions, models, settings…"
          class="min-w-0 flex-1 bg-transparent text-body-lg text-ink outline-none placeholder:text-ink-3"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded="true"
          aria-controls="cmd-palette-listbox"
          :aria-activedescendant="activeOptionId"
          aria-label="Search or run a command"
          @keydown="onKeydown"
        />
        <Keycap>esc</Keycap>
      </div>
      <div id="cmd-palette-listbox" class="min-h-0 flex-1 overflow-y-auto p-2" role="listbox">
        <p v-if="results.length === 0" class="px-3 py-6 text-center text-body text-ink-3">
          No matches.
        </p>
        <button
          v-for="(cmd, i) in results"
          :id="`cmd-palette-option-${i}`"
          :key="cmd.id"
          type="button"
          role="option"
          :aria-selected="i === selected"
          class="flex w-full items-center gap-3 rounded-control px-3 py-2.5 text-left transition-colors duration-100"
          :class="i === selected ? 'bg-surface' : ''"
          @mouseenter="selected = i"
          @click="cmd.run()"
        >
          <span
            class="w-12 shrink-0 font-utility text-[9px] tracking-[0.06em] text-ink-3 uppercase"
          >
            {{ sectionLabel(cmd.id) }}
          </span>
          <span
            class="min-w-0 flex-1 truncate text-[13.5px]"
            :class="i === selected ? 'text-ink' : 'text-ink-2'"
          >
            {{ cmd.title }}
          </span>
          <span v-if="cmd.subtitle" class="edge-code shrink-0">{{ cmd.subtitle }}</span>
          <Icon name="arrow-right" :size="14" class="shrink-0 text-ce" />
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cmdk-scrim {
  background: rgba(6, 5, 10, 0.55);
  backdrop-filter: blur(4px);
}
</style>
