<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRouter } from "vue-router";
import EmptyState from "../components/shell/EmptyState.vue";
import AuthedMedia from "../components/gallery/AuthedMedia.vue";
import HostFilterChips from "../components/shell/HostFilterChips.vue";
import {
  clearHistoryOn,
  clearScope,
  fetchHistoryAll,
  groupByDay,
  type HistoryEntry,
  type HistoryHostTarget,
  type HostHistoryEntry,
} from "../lib/api/history";
import { galleryMediaPath } from "../lib/gallery/media";
import { useConnectionStore } from "../stores/connection";
import { useComposerStore } from "../stores/composer";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useToastStore } from "../stores/toasts";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { applyModelDefaults, newGenerateForm } from "../lib/generateForm";
import type { MergedPrint } from "../stores/gallery";
import type { GalleryImage } from "../lib/api/types";

const router = useRouter();
const conn = useConnectionStore();
const composer = useComposerStore();
const gallery = useGalleryStore();
const hosts = useHostsStore();
const models = useModelStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();

/**
 * Two lenses on the past: Runs (every finished generation with its print,
 * settings, and seed — the gallery DB is the source of truth) and Prompts
 * (the prompt log, including prompts whose outputs are gone).
 */
const tab = ref<"runs" | "prompts">("runs");
const query = ref("");

/** The Gallery's chip filter narrows both tabs; row chips only in All view. */
const showChips = computed(() => gallery.chipCounts.length > 1);
const showBadges = computed(() => gallery.filter === "all" && gallery.chipCounts.length > 1);

// ── Runs (gallery-backed, merged across every connected host) ──────────────

const runs = computed<MergedPrint[]>(() => {
  const q = query.value.trim().toLowerCase();
  // Same filtered set the Gallery renders, so the two views can't drift.
  const entries = gallery.filtered;
  if (!q) return entries;
  return entries.filter(
    (e) =>
      e.item.metadata.prompt.toLowerCase().includes(q) ||
      e.item.metadata.model.toLowerCase().includes(q),
  );
});

/** Day buckets, reusing the prompt-log grouping via the shared shape. The
 *  grouping key carries the origin so same-named prints on two hosts stay
 *  distinct rows. */
const runKey = (e: MergedPrint) => `${e.sourceKey}\u0000${e.item.filename}`;
const runGroups = computed(() => {
  const pseudo = runs.value.map((e) => ({
    prompt: runKey(e),
    model: e.item.metadata.model,
    used_at: e.item.timestamp * 1000,
  }));
  const groups = groupByDay(pseudo);
  const byKey = new Map(runs.value.map((e) => [runKey(e), e]));
  return groups.map((g) => ({
    label: g.label,
    runs: g.entries.map((e) => byKey.get(e.prompt)!).filter(Boolean),
  }));
});

function runTime(img: GalleryImage): string {
  return new Date(img.timestamp * 1000).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function useRun(img: GalleryImage) {
  const m = img.metadata;
  const installed = models.installed.find((entry) => entry.name === m.model);
  composer.set({
    prompt: m.prompt,
    model: installed ? m.model : newGenerateForm().model,
    seed: m.seed,
    width: m.width,
    height: m.height,
    steps: m.steps,
    guidance: m.guidance,
  });
  void router.push("/generate");
}

function runMenu(img: GalleryImage): MenuEntry[] {
  return [
    { label: "Reuse settings", action: () => useRun(img) },
    {
      label: "Copy prompt",
      action: () => {
        void navigator.clipboard.writeText(img.metadata.prompt).then(() => toasts.push("Copied"));
      },
    },
    {
      label: "Copy seed",
      action: () => {
        void navigator.clipboard
          .writeText(String(img.metadata.seed))
          .then(() => toasts.push("Copied seed"));
      },
    },
    { label: "Show in Gallery", action: () => void router.push("/gallery") },
  ];
}

// (Re)fetch whenever the gallery's source set changes — mount, the
// connection coming up, hosts joining mid-session. History must be complete
// even if the Gallery view was never opened. (A source joining refetches
// every bucket — fetchAll doesn't skip settled ones; cheap and rare.)
watch(
  () => gallery.sources.map((s) => s.key).join("|"),
  () => {
    // Guard on a non-empty source set: during connection init / an engine
    // switch, sources is briefly empty and fetchAll's syncBuckets would wipe
    // already-fetched buckets and evict their cached media for nothing.
    if (gallery.sources.length > 0 && !gallery.loaded) void gallery.fetchAll();
  },
  { immediate: true },
);

// ── Prompts (the prompt log, fanned out over every ready host) ─────────────

/** Hosts the prompt log reads from: every ready host, primary included. */
const historyHosts = computed<HistoryHostTarget[]>(() =>
  hosts.all.flatMap((h) =>
    h.status === "ready" && h.baseUrl
      ? [{ hostId: h.id, label: h.label, target: { baseUrl: h.baseUrl, apiKey: h.apiKey } }]
      : [],
  ),
);

const promptEntries = ref<HostHistoryEntry[]>([]);
/** Hosts whose GET /api/history succeeded on the last load. */
const supportedHostIds = ref<string[]>([]);
const loaded = ref(false);
const unavailable = ref(false);
/** All hosts failed, but at least one looked like a network blip — show
 *  "couldn't reach" instead of wrongly blaming old servers. */
const unreachable = ref(false);
const confirmingClear = ref(false);

/** The chip filter applies here too ("local" = the built-in engine's id;
 *  a remote primary's This-Mac IPC bucket has no prompt log — empty). */
const visiblePrompts = computed<HostHistoryEntry[]>(() =>
  gallery.filter === "all"
    ? promptEntries.value
    : promptEntries.value.filter((e) => e.hostId === gallery.filter),
);

const groups = computed(() => groupByDay(visiblePrompts.value));

/** The chip filter points at the This-Mac IPC bucket, which is a gallery
 *  source but not a history source (only exists with a remote primary). */
const filterIsLocalOnlyBucket = computed(
  () => gallery.filter !== "all" && !historyHosts.value.some((h) => h.hostId === gallery.filter),
);

async function load() {
  const targets = historyHosts.value;
  const listing = await fetchHistoryAll(targets, query.value);
  promptEntries.value = listing.entries;
  supportedHostIds.value = listing.supportedHostIds;
  // 404 = a server that predates the history API; 503 = DB off. Only when
  // EVERY host is like that is history truly unavailable; a fan-out where
  // something merely didn't answer is a reachability problem instead.
  const allFailed = targets.length > 0 && listing.supportedHostIds.length === 0;
  unreachable.value = allFailed && listing.unreachableHostIds.length > 0;
  unavailable.value = allFailed && !unreachable.value;
  loaded.value = true;
}

let debounce: ReturnType<typeof setTimeout> | null = null;
watch(query, () => {
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(() => {
    if (tab.value === "prompts") void load();
  }, 250);
});

watch(
  [() => conn.ready, tab, () => historyHosts.value.map((h) => h.hostId).join("|")],
  ([ready, current]) => {
    if (ready && current === "prompts") void load();
  },
  { immediate: true },
);

function usePrompt(entry: HistoryEntry) {
  const installed = models.installed.find((m) => m.name === entry.model);
  const form = newGenerateForm();
  if (installed) applyModelDefaults(form, installed);
  composer.set({
    prompt: entry.prompt,
    model: installed ? entry.model : form.model,
    seed: null,
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: form.guidance,
  });
  void router.push("/generate");
}

function promptMenu(entry: HistoryEntry): MenuEntry[] {
  return [
    { label: "Use prompt", action: () => usePrompt(entry) },
    {
      label: "Copy prompt",
      action: () => {
        void navigator.clipboard.writeText(entry.prompt).then(() => toasts.push("Copied"));
      },
    },
  ];
}

/** Clear respects the chip filter: one host, or every history-capable host. */
const clearTargets = computed<HistoryHostTarget[]>(() =>
  clearScope(
    gallery.filter,
    historyHosts.value.filter((h) => supportedHostIds.value.includes(h.hostId)),
  ),
);

const clearLabel = computed(() => {
  if (!confirmingClear.value) return "Clear…";
  const scope = clearTargets.value;
  const suffix = scope.length > 1 ? ` on ${scope.map((h) => h.label).join(", ")}` : "";
  return `Clear ${visiblePrompts.value.length} prompts${suffix}?`;
});

async function clearAll() {
  if (!confirmingClear.value) {
    confirmingClear.value = true;
    return;
  }
  confirmingClear.value = false;
  const targets = clearTargets.value;
  // The This-Mac chip (or an unpopulated support set) has nothing to clear —
  // a success toast there would be a lie.
  if (targets.length === 0) {
    toasts.push("Nothing to clear here");
    return;
  }
  await Promise.allSettled(targets.map((h) => clearHistoryOn(h.target)));
  toasts.push("Cleared history");
  await load();
}

const timeOf = (e: HistoryEntry) =>
  new Date(e.used_at).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
</script>

<template>
  <div class="flex h-full flex-col">
    <header class="border-edge flex h-11 items-center gap-3 border-b px-4">
      <span class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        History
      </span>
      <div
        class="border-edge flex rounded-control border bg-bath p-0.5"
        role="group"
        aria-label="History view"
      >
        <button
          type="button"
          data-test="tab-runs"
          :aria-pressed="tab === 'runs'"
          class="rounded-control px-2.5 py-1 text-body transition-colors"
          :class="tab === 'runs' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'"
          @click="tab = 'runs'"
        >
          Runs
        </button>
        <button
          type="button"
          data-test="tab-prompts"
          :aria-pressed="tab === 'prompts'"
          class="rounded-control px-2.5 py-1 text-body transition-colors"
          :class="tab === 'prompts' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'"
          @click="tab = 'prompts'"
        >
          Prompts
        </button>
      </div>
      <input
        v-model="query"
        data-selectable
        type="search"
        :placeholder="tab === 'runs' ? 'Search runs…' : 'Search prompts…'"
        class="border-edge ml-auto h-7 w-64 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
      />
      <button
        v-if="tab === 'prompts'"
        type="button"
        data-test="clear-history"
        class="border-edge h-7 rounded-control border px-2.5 text-body transition-colors duration-100"
        :class="
          confirmingClear
            ? 'border-stop bg-stop font-semibold text-[#141110]'
            : 'text-ink-2 hover:text-stop'
        "
        @blur="confirmingClear = false"
        @click="clearAll"
      >
        {{ clearLabel }}
      </button>
    </header>

    <!-- One origin filter for both tabs — the same chips the Gallery uses -->
    <div v-if="showChips" class="border-edge flex items-center border-b px-4 py-2">
      <HostFilterChips v-model="gallery.filter" :chips="gallery.chipCounts" />
    </div>

    <!-- Runs: every finished generation with print + settings -->
    <div v-if="tab === 'runs'" class="min-h-0 flex-1 overflow-y-auto px-4 py-3">
      <EmptyState
        v-if="gallery.loaded && runs.length === 0 && !query"
        headline="No runs yet"
        detail="Every print you develop shows up here with its settings and seed."
      />
      <p v-else-if="query && runs.length === 0" class="mt-6 text-center text-body text-ink-2">
        No runs match “{{ query }}”.
      </p>
      <template v-for="group in runGroups" :key="group.label">
        <div class="mt-3 mb-1 flex items-center gap-2 first:mt-0">
          <span class="edge-code">{{ group.label.toUpperCase() }}</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <button
          v-for="entry in group.runs"
          :key="runKey(entry)"
          type="button"
          data-test="run-row"
          class="group flex w-full items-center gap-3 rounded-control px-2 py-1.5 text-left hover:bg-bench"
          @click="useRun(entry.item)"
          @contextmenu="contextMenu.open($event, runMenu(entry.item))"
        >
          <div
            class="h-12 w-12 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
          >
            <AuthedMedia
              :path="
                galleryMediaPath(entry.item.filename, gallery.mediaSourceOf(entry.sourceKey), true)
              "
              :target="gallery.targetOf(entry.sourceKey)"
              :cache-key="entry.sourceKey"
              :alt="entry.item.metadata.prompt"
            />
          </div>
          <div class="min-w-0 flex-1">
            <div class="truncate text-body text-ink" :title="entry.item.metadata.prompt">
              {{ entry.item.metadata.prompt }}
            </div>
            <div class="data-mono mt-0.5 truncate text-caption text-ink-3">
              {{ entry.item.metadata.model }} · {{ entry.item.metadata.width }}×{{
                entry.item.metadata.height
              }}
              · S {{ entry.item.metadata.seed }} · {{ entry.item.metadata.steps }} steps
            </div>
          </div>
          <span
            v-if="showBadges"
            data-test="host-badge"
            class="edge-code max-w-32 shrink-0 truncate"
          >
            {{ entry.hostLabel }}
          </span>
          <span class="data-mono shrink-0 text-caption text-ink-3">{{ runTime(entry.item) }}</span>
          <span
            class="shrink-0 text-caption text-safelight opacity-0 transition-opacity duration-100 group-hover:opacity-100"
          >
            ↩ Reuse
          </span>
        </button>
      </template>
    </div>

    <!-- Prompts: the raw prompt log, merged across every ready host -->
    <div v-else class="min-h-0 flex-1 overflow-y-auto px-4 py-3">
      <EmptyState
        v-if="loaded && unreachable"
        headline="Couldn't reach any host"
        detail="The prompt log lives on each engine — reconnect and it reloads on its own."
      />
      <EmptyState
        v-else-if="loaded && unavailable"
        headline="Prompt history isn't available"
        :detail="
          historyHosts.length > 1
            ? 'None of the connected engines expose prompt history — they may predate the history API or run without their database.'
            : 'This engine doesn\'t expose prompt history — it may predate the history API or run without its database.'
        "
      />
      <EmptyState
        v-else-if="loaded && filterIsLocalOnlyBucket"
        headline="This Mac keeps prints, not prompts"
        detail="Prompts are tracked by each engine — pick a host chip (or All) to see them."
      />
      <EmptyState
        v-else-if="loaded && visiblePrompts.length === 0 && !query"
        headline="No prompts yet"
        detail="Every prompt you develop is kept here to reuse."
      />
      <template v-for="group in groups" :key="group.label">
        <div class="mt-3 mb-1 flex items-center gap-2 first:mt-0">
          <span class="edge-code">{{ group.label.toUpperCase() }}</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <button
          v-for="(entry, i) in group.entries"
          :key="`${group.label}-${i}`"
          type="button"
          data-test="prompt-row"
          class="group flex w-full items-center gap-3 rounded-control px-2 py-1.5 text-left hover:bg-bench"
          @click="usePrompt(entry)"
          @contextmenu="contextMenu.open($event, promptMenu(entry))"
        >
          <span class="min-w-0 flex-1 truncate text-body text-ink" :title="entry.prompt">
            {{ entry.prompt }}
          </span>
          <span
            v-if="showBadges"
            data-test="host-badge"
            class="edge-code max-w-32 shrink-0 truncate"
          >
            {{ entry.hostLabel }}
          </span>
          <span class="data-mono shrink-0 text-caption text-ink-3">{{ entry.model }}</span>
          <span class="data-mono shrink-0 text-caption text-ink-3">{{ timeOf(entry) }}</span>
          <span
            class="shrink-0 text-caption text-safelight opacity-0 transition-opacity duration-100 group-hover:opacity-100"
          >
            ↩ Use
          </span>
        </button>
      </template>
      <p
        v-if="query && visiblePrompts.length === 0 && !unavailable"
        class="mt-6 text-center text-body text-ink-2"
      >
        No prompts match “{{ query }}”.
      </p>
    </div>
  </div>
</template>
