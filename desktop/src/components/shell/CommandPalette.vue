<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import { useUiStore } from "../../stores/ui";
import { useGalleryStore } from "../../stores/gallery";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useDownloadsStore } from "../../stores/downloads";
import { useComposerStore } from "../../stores/composer";
import { useGenerationStore } from "../../stores/generation";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { THEME_META } from "../../lib/theme";
import { matchCommands, type Matchable } from "../../lib/palette";
import { fetchHistory, type HistoryEntry } from "../../lib/api/history";
import { loadModel, unloadModel } from "../../lib/api/models";
import { searchCatalog, startCatalogDownload } from "../../lib/api/catalog";
import { useInventoryKnown } from "../../lib/modelInventory";
import { planModelInstall } from "@studio/lib/modelInstallTargets";
import {
  buildInstallModelCommands,
  buildUseModelCommands,
  shouldSearchCatalog,
  CATALOG_SEARCH_DEBOUNCE_MS,
  type SearchableCatalogEntry,
} from "@studio/lib/modelSearch";
import { newGenerateForm, applyModelDefaults } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";

interface Command extends Matchable {
  id: string;
  title: string;
  subtitle?: string;
  run: () => void;
}

/** Target sentinels meaning "let the router pick a machine"; null is Auto. */
const AUTO_TARGET_IDS = ["capable"] as const;

const ui = useUiStore();
const router = useRouter();
const gallery = useGalleryStore();
const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const downloads = useDownloadsStore();
const composer = useComposerStore();
const generation = useGenerationStore();
const conn = useConnectionStore();
const toasts = useToastStore();
const appPrefs = useAppPrefsStore();
const inventoryKnown = useInventoryKnown();

/** Mono section label for a result row, derived from its id prefix. */
function sectionLabel(id: string): string {
  if (id.startsWith("nav-")) return "go";
  if (id.startsWith("theme-") || id.startsWith("appear-")) return "theme";
  if (id.startsWith("model-") || id.startsWith("load-") || id.startsWith("unload-")) return "model";
  if (id.startsWith("install-")) return "install";
  if (id.startsWith("history-")) return "recent";
  if (id.startsWith("print-")) return "print";
  return "run";
}

const query = ref("");
const selected = ref(0);
const inputEl = ref<HTMLInputElement | null>(null);
const history = ref<HistoryEntry[]>([]);
const catalogEntries = ref<SearchableCatalogEntry[]>([]);
let debounce: ReturnType<typeof setTimeout> | null = null;
let catalogDebounce: ReturnType<typeof setTimeout> | null = null;
// Monotonic epoch so a slow response for an abandoned query can never
// overwrite the results of a newer one.
let catalogEpoch = 0;
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

/**
 * Every installed generation model in the fleet with the machines that hold
 * it. The primary's canonical list lives in its own store, so it is merged in
 * rather than waiting on the per-host fan-out — otherwise a palette opened
 * before `hostModels.refresh()` lands would list no models at all.
 */
const fleetModels = computed(() => {
  // `hostModels` keeps a machine's last inventory after it drops offline, so
  // an errored host's cache must not become a palette row: the model would be
  // unusable, and repinning generation to that owner would leave Create
  // unable to submit at all. A model every remaining owner has lost is gone.
  const unreachable = new Set(hosts.all.filter((h) => h.status === "error").map((h) => h.id));
  const byName = new Map<string, { entry: ModelEntry; owners: string[] }>();
  if (!unreachable.has("local")) {
    for (const m of models.installed) byName.set(m.name, { entry: m, owners: ["local"] });
  }
  for (const m of hostModels.unionInstalled) {
    const reachable = m.hostIds.filter((id) => !unreachable.has(id));
    if (reachable.length === 0) continue;
    const existing = byName.get(m.name);
    byName.set(m.name, {
      entry: existing?.entry ?? m,
      owners: [...new Set([...(existing?.owners ?? []), ...reachable])],
    });
  }
  return byName;
});

/** The pinned generation target; null is Auto, "capable" is Most capable. */
const generateTargetHost = computed(() => appPrefs.settings?.generateTargetHost ?? null);

function hostLabel(id: string): string {
  return hosts.all.find((h) => h.id === id)?.label ?? id;
}

/** Prefill the composer with a model's defaults (and optional prompt). */
function prefillModel(model: string, prompt = "") {
  // Resolve across the fleet, not just the primary: a model that only exists
  // on another machine still has to bring its own defaults with it.
  const installed = fleetModels.value.get(model)?.entry;
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

/** Use a model, repinning the generation target when it lives elsewhere. */
async function useModel(name: string, switchToHostId: string | null) {
  if (switchToHostId) {
    // Await the write before navigating. `appPrefs.update` re-reads the whole
    // settings file, merges, and writes it back; navigating to Create kicks
    // off its own last-route persistence, and the two interleaved would
    // restore the stale `generateTargetHost` — pinning Create to the machine
    // that does not have the model the user just picked.
    await appPrefs.update({ generateTargetHost: switchToHostId });
    toasts.push(`Generating on ${hostLabel(switchToHostId)}`);
  }
  prefillModel(name);
}

/**
 * Queue a pull for a model nobody has yet.
 *
 * Unlike the Models workspace this never opens the machine picker — that
 * dialog belongs to Models, and a palette invoked from any other route would
 * pop it somewhere the user cannot see. The plan's first target (installs
 * before repairs, primary first) is taken and named in the toast instead.
 */
async function installModel(modelId: string, displayName: string) {
  const readyHosts = hosts.all.filter((h) => h.status === "ready" && h.baseUrl);
  const host = planModelInstall(readyHosts, [], { inventoryKnown }).targets[0]?.host ?? null;
  close();
  try {
    const target = host?.baseUrl ? { baseUrl: host.baseUrl, apiKey: host.apiKey } : undefined;
    // Attach the snapshot-first stream before enqueueing so a cached,
    // near-instant pull still produces a visible terminal event.
    await downloads.subscribe(host ?? undefined);
    await startCatalogDownload(modelId, target, host ? host.kind === "remote" : false);
    toasts.push(`Pulling ${displayName}${host && hosts.all.length > 1 ? ` on ${host.label}` : ""}`);
  } catch (err) {
    toasts.push(`Couldn't queue ${displayName}: ${String(err)}`, "error");
  }
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
        void conn.restartEngine().then((result) => {
          if (result === "restarted") toasts.push("Engine restarted");
          else if (result === "failed" && conn.error) toasts.push(conn.error, "error");
        });
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
    ...THEME_META.map((meta) => ({
      id: `theme-${meta.id}`,
      title: `Theme: ${meta.label}`,
      subtitle: meta.blurb,
      keywords: ["theme", "look", "appearance", "color", meta.tone, meta.label.toLowerCase()],
      run: () => {
        void appPrefs.update({ theme: meta.id });
        close();
      },
    })),
    {
      id: "appear-match-system",
      title: appPrefs.matchSystem
        ? "Stop matching the system appearance"
        : "Match the system appearance",
      subtitle: "Swap to the paired light or dark theme when macOS does",
      keywords: ["theme", "appearance", "system", "auto", "light", "dark"],
      run: () => {
        void appPrefs.update({ matchSystem: !appPrefs.matchSystem });
        close();
      },
    },
  ];
  if (generation.pending.length > 0) {
    cmds.push({
      id: "act-cancel",
      title: "Cancel job",
      run: () => {
        void generation
          .cancel()
          .then((cancelled) => {
            if (cancelled) toasts.push("Cancelled");
          })
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          );
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
        void Promise.all(ids.map((id) => generation.cancel(id)))
          .then((outcomes) => {
            const cancelled = outcomes.filter(Boolean).length;
            if (cancelled === outcomes.length) toasts.push("Cancelled all jobs");
            else if (cancelled > 0)
              toasts.push(
                `Cancelled ${cancelled} ${cancelled === 1 ? "job" : "jobs"}; remaining jobs already settled`,
              );
          })
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
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
  // Fleet-wide "Use <model>" rows, ranked with everything else by the shared
  // matcher so a model name typed in full outranks a nav row that merely
  // contains it. Load/unload stay scoped to the primary below — they act on
  // this machine's GPU, not on whichever machine happens to hold the weights.
  const fleet = fleetModels.value;
  for (const row of buildUseModelCommands(
    [...fleet.values()].map((f) => f.entry),
    {
      ownersFor: (name) => fleet.get(name)?.owners ?? [],
      targetHostId: generateTargetHost.value,
      autoTargetIds: AUTO_TARGET_IDS,
      hostLabel,
    },
  )) {
    cmds.push({
      id: row.id,
      title: row.label,
      ...(row.hint ? { subtitle: row.hint } : {}),
      keywords: row.keywords,
      run: () => {
        void useModel(row.model, row.switchToHostId);
      },
    });
  }
  for (const m of models.installed) {
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

/** Live catalog hits nobody has yet. Kept in the server's relevance order and
 *  appended, so a slow round trip can never reorder rows under a keystroke. */
const installCommands = computed<Command[]>(() =>
  buildInstallModelCommands(catalogEntries.value, {
    installedNames: [...fleetModels.value.keys()],
  }).map((row) => ({
    id: row.id,
    title: row.label,
    subtitle: row.hint,
    keywords: row.keywords,
    run: () => {
      void installModel(row.model, row.displayName);
    },
  })),
);

const results = computed<Command[]>(() => [
  ...matchCommands(query.value, staticCommands.value),
  ...installCommands.value,
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

/** Debounced live catalog search, so a model you don't have is still findable. */
watch(query, (q) => {
  if (catalogDebounce) clearTimeout(catalogDebounce);
  const epoch = ++catalogEpoch;
  // Drop the previous query's rows immediately. Install rows are appended
  // unmatched, so leaving them up through the debounce and request would let
  // Enter queue a model the user is no longer looking at — a multi-gigabyte
  // download they never asked for.
  catalogEntries.value = [];
  if (!shouldSearchCatalog(q)) return;
  catalogDebounce = setTimeout(() => {
    // Checkpoints only: the palette's promise is "switch to this model", and
    // a LoRA or VAE is not something you can switch to.
    void searchCatalog({ q: q.trim(), kind: "checkpoint", page_size: 12 })
      .then((response) => {
        if (epoch !== catalogEpoch) return;
        catalogEntries.value = response.entries
          // An unsupported entry has no runnable pipeline — offering to pull
          // it would end in a model the user cannot switch to.
          .filter((entry) => entry.supported !== false)
          .map((entry) => ({
            id: entry.id,
            name: entry.display_name ?? entry.name,
            family: entry.family ?? null,
            source: entry.source ?? null,
            installed: entry.installed ?? null,
          }));
      })
      .catch(() => {
        // The palette still works offline; it just stops offering installs.
        if (epoch === catalogEpoch) catalogEntries.value = [];
      });
  }, CATALOG_SEARCH_DEBOUNCE_MS);
});

watch(
  () => ui.paletteOpen,
  (open) => {
    if (open) {
      restoreFocusEl = document.activeElement as HTMLElement | null;
      query.value = "";
      selected.value = 0;
      history.value = [];
      catalogEntries.value = [];
      void fetchHistory("", 6)
        .then((entries) => (history.value = entries))
        .catch(() => {});
      // Warm the gallery buckets so print results can match, but never
      // refetch on every open — a loaded store is reused as-is.
      if (!gallery.loaded) void gallery.fetchAll();
      // Fleet inventory so a model on another machine is findable from here.
      // The store's own staleness window keeps this from being a fan-out on
      // every ⌘K.
      void hostModels.refresh();
      void nextTick(() => inputEl.value?.focus());
    } else {
      if (catalogDebounce) clearTimeout(catalogDebounce);
      catalogDebounce = null;
      catalogEpoch++;
      // Return focus to whatever launched the palette.
      restoreFocusEl?.focus?.();
      restoreFocusEl = null;
    }
  },
);

// A palette torn down mid-keystroke must not leave a search armed: both
// debounces would otherwise fire against a dead component, issuing a request
// whose response nothing can consume.
onBeforeUnmount(() => {
  if (debounce) clearTimeout(debounce);
  if (catalogDebounce) clearTimeout(catalogDebounce);
  debounce = null;
  catalogDebounce = null;
  catalogEpoch++;
});

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
