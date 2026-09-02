<script setup lang="ts">
/*
 * History drawer (spec §06) — the Library's Runs + Prompts + Sequences log on a
 * right-side @ui DrawerPanel. Runs are gallery-backed (every finished
 * generation with its print, settings, and seed); Prompts is the raw prompt log
 * fanned out over every ready host; Sequences is the one place the durable
 * server-side sequence jobs are enumerated, acted on, and cleaned up — the
 * Create strip is present tense and no longer carries settled work or its
 * maintenance buttons. All the reuse/clear actions from the old History screen
 * are preserved. Opens via ?panel=history with ?tab= selecting the lens (the
 * retired /history route, the activity digest chip, and the command palette all
 * deep-link here); Library owns the open state and closing.
 */
import { computed, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import SequenceJobRow from "@ui/components/SequenceJobRow.vue";
import {
  HISTORY_JOBS_RENDER_CAP,
  mergeActivity,
  sequenceToVM,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import { isPrintOfChainJob } from "@studio/lib/sequenceReuse";
import EmptyState from "../shell/EmptyState.vue";
import AuthedMedia from "../gallery/AuthedMedia.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";
import HostFilterChips from "../shell/HostFilterChips.vue";
import {
  clearHistoryOn,
  clearScope,
  fetchHistoryAll,
  groupByDay,
  type HistoryEntry,
  type HistoryHostTarget,
  type HostHistoryEntry,
} from "../../lib/api/history";
import { galleryMediaPath } from "../../lib/gallery/media";
import { readGalleryMediaBlob } from "../../lib/gallery/sourceMedia";
import {
  applyGalleryEntryAsSource,
  canUseGalleryEntryAsSource,
} from "../../lib/gallery/useAsSource";
import { modelDisplayNameForId } from "../../lib/models";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useConnectionStore } from "../../stores/connection";
import { useComposerStore } from "../../stores/composer";
import { useGalleryStore } from "../../stores/gallery";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { copyLocalOutputPath } from "../../lib/localOutputPath";
import { applyModelDefaults, newGenerateForm } from "../../lib/generateForm";
import type { MergedPrint } from "../../stores/gallery";
import type { GalleryImage } from "../../lib/api/types";
import { dragWidth } from "../../lib/panelResize";
import { useAppPrefsStore } from "../../stores/appPrefs";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();

const route = useRoute();
const router = useRouter();
const chains = useChainJobsStore();
const conn = useConnectionStore();
const composer = useComposerStore();
const gallery = useGalleryStore();
const generateForm = useGenerateFormStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const models = useModelStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();
const appPrefs = useAppPrefsStore();

const draftDrawerWidth = ref<number | null>(null);
const drawerWidth = computed(() => draftDrawerWidth.value ?? appPrefs.historyDrawerWidth);

function onDrawerResize(dx: number) {
  draftDrawerWidth.value = dragWidth("historyDrawer", appPrefs.historyDrawerWidth, dx, "left");
}

async function onDrawerCommit() {
  const width = draftDrawerWidth.value;
  if (width === null) return;
  if (width !== appPrefs.historyDrawerWidth) {
    await appPrefs.update({ historyDrawerWidth: width });
  }
  draftDrawerWidth.value = null;
}

function onDrawerReset() {
  draftDrawerWidth.value = null;
  void appPrefs.update({ historyDrawerWidth: null });
}

/**
 * Three lenses on the past: Runs (every finished generation with its print,
 * settings, and seed — the gallery DB is the source of truth), Prompts (the
 * prompt log, including prompts whose outputs are gone), and Sequences (the
 * durable chain jobs, which are the past *with actions attached*).
 *
 * The tab lives in the URL so the Create activity digest can deep-link
 * straight to Sequences.
 */
type HistoryTab = "runs" | "prompts" | "sequences";
const TABS: HistoryTab[] = ["runs", "prompts", "sequences"];

const tab = computed<HistoryTab>(() => {
  const q = route.query.tab;
  return typeof q === "string" && (TABS as string[]).includes(q) ? (q as HistoryTab) : "runs";
});

function selectTab(next: HistoryTab) {
  if (next === tab.value) return;
  void router.replace({ path: route.path, query: { ...route.query, tab: next } });
}

const query = ref("");

/** The Library's chip filter narrows both tabs; row chips only in All view. */
const showChips = computed(() => gallery.chipCounts.length > 1);
const showBadges = computed(() => gallery.filter === "all" && gallery.chipCounts.length > 1);
const availabilityLabel = (entry: MergedPrint) =>
  entry.availableOn.map((source) => source.label).join(" · ");

// ── Runs (gallery-backed, merged across every connected host) ──────────────

const runs = computed<MergedPrint[]>(() => {
  const q = query.value.trim().toLowerCase();
  // Same host-chip set the Library renders. Deliberately NOT `filtered`:
  // the Library view's own search box and Images/Video chips must not
  // invisibly narrow History runs (this tab has its own search).
  const entries = gallery.hostFiltered;
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
const runKey = (e: MergedPrint) => `${e.sourceKey}\0${e.item.filename}`;
/** The Runs list is not windowed, and every row mounts a thumbnail, so it is
 *  capped exactly like Sequences: a 1 000-print library must not mount 1 000
 *  media requests the moment the drawer opens. */
const visibleRuns = computed(() => runs.value.slice(0, HISTORY_JOBS_RENDER_CAP));
const runsCapNote = computed(() =>
  runs.value.length > HISTORY_JOBS_RENDER_CAP
    ? `showing ${HISTORY_JOBS_RENDER_CAP} of ${runs.value.length} — search to narrow`
    : null,
);
const runGroups = computed(() => {
  const pseudo = visibleRuns.value.map((e) => ({
    prompt: runKey(e),
    model: e.item.metadata.model,
    used_at: e.item.timestamp * 1000,
  }));
  const groups = groupByDay(pseudo);
  const byKey = new Map(visibleRuns.value.map((e) => [runKey(e), e]));
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
  composer.set({ metadata: img.metadata });
  emit("close");
  void router.push("/create");
}

/**
 * "Use as source" on a past run — the same rule the Library tile menu and the
 * Lightbox use, reading this row's bytes from its own origin. Deliberately
 * not `composer.set`: that is a settings prefill and would clobber the source
 * this just attached.
 */
async function useRunAsSource(entry: MergedPrint) {
  const outcome = await applyGalleryEntryAsSource(entry, generateForm.form, (target) =>
    readGalleryMediaBlob(target, gallery),
  );
  if (!outcome.ok) {
    toasts.push(outcome.error, "error");
    return;
  }
  toasts.push(outcome.message);
  emit("close");
  void router.push("/create");
}

function runMenu(entry: MergedPrint): MenuEntry[] {
  const img = entry.item;
  return [
    { label: "Reuse settings", action: () => useRun(img) },
    {
      label: "Use as source",
      disabled: !canUseGalleryEntryAsSource(img),
      action: () => void useRunAsSource(entry),
    },
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
    {
      label: "Copy file path",
      action: () =>
        void copyLocalOutputPath(img.filename)
          .then(() => toasts.push("File path copied"))
          .catch((error) =>
            toasts.push(error instanceof Error ? error.message : String(error), "error"),
          ),
    },
    { label: "Show in library", action: () => emit("close") },
  ];
}

// (Re)fetch whenever the gallery's source set changes while the drawer is open.
// The Library view usually has these buckets loaded already; this guards the
// deep-link case (opening straight onto ?panel=history).
watch(
  [() => props.open, () => gallery.sources.map((s) => s.key).join("|")],
  ([open]) => {
    if (open && gallery.sources.length > 0 && !gallery.loaded) void gallery.fetchAll();
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
    if (props.open && tab.value === "prompts") void load();
  }, 250);
});

watch(
  [
    () => props.open,
    () => conn.ready,
    tab,
    () => historyHosts.value.map((h) => h.hostId).join("|"),
  ],
  ([open, ready, current]) => {
    if (open && ready && current === "prompts") void load();
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
  emit("close");
  void router.push("/create");
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

// ── Sequences (durable chain jobs, every host) ─────────────────────────────

type SequenceVM = ActivityJobVM & { kind: "sequence" };

/** `chains.syncPolling()` only runs while something is active, so a settled
 *  list goes stale the moment the last job finishes. Refetch on open and on
 *  every switch back to this tab. */
watch(
  [() => props.open, tab],
  ([open, current]) => {
    if (open && current === "sequences") void chains.fetchAll();
  },
  { immediate: true },
);

const sequenceModelLabel = (name: string) => modelDisplayNameForId(name, hostModels.unionInstalled);

/** Every host's jobs through the shared merge (active first, then recency),
 *  narrowed by the same host chips the other two tabs use. */
const sequenceRows = computed<SequenceVM[]>(() => {
  const vms = chains.allJobs
    .filter(({ hostId }) => gallery.filter === "all" || hostId === gallery.filter)
    .map(({ hostId, job }) =>
      sequenceToVM(job, {
        hostId,
        hostLabel: hosts.all.find((h) => h.id === hostId)?.label ?? hostId,
      }),
    );
  return mergeActivity([], vms).filter((vm): vm is SequenceVM => vm.kind === "sequence");
});

const visibleSequences = computed(() => sequenceRows.value.slice(0, HISTORY_JOBS_RENDER_CAP));
const sequenceCapNote = computed(() =>
  sequenceRows.value.length > HISTORY_JOBS_RENDER_CAP
    ? `showing ${HISTORY_JOBS_RENDER_CAP} of ${sequenceRows.value.length}`
    : null,
);

const sequenceTime = (vm: SequenceVM) =>
  new Date(vm.settledAtMs ?? vm.createdAtMs).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

/** The print a job produced, on that job's OWN host only — a merged print may
 *  live on three hosts, but the other two hold auto-saved copies. */
function printOf(vm: SequenceVM): { sourceKey: string; filename: string } | null {
  const hit = gallery.merged.find(
    (entry) => entry.sourceKey === vm.hostId && isPrintOfChainJob(entry.item.metadata, vm.jobId),
  );
  return hit ? { sourceKey: hit.sourceKey, filename: hit.item.filename } : null;
}

function showPrint(vm: SequenceVM) {
  const print = printOf(vm);
  if (!print) return;
  emit("close");
  void router.push({ path: "/library", query: { print: print.filename } });
}

const confirmDeleteSequence = ref<SequenceVM | null>(null);

function onSequenceAction(action: ActivityAction, vm: SequenceVM) {
  switch (action) {
    case "watch":
      // Re-enter the job: Create's canvas holds whatever it finds, running or
      // finished. Same verb the strip uses, one route away.
      composer.setSequence({ kind: "inspect", hostId: vm.hostId, jobId: vm.jobId });
      emit("close");
      void router.push("/create");
      return;
    case "edit":
      composer.setSequence({ kind: "edit", hostId: vm.hostId, jobId: vm.jobId });
      emit("close");
      void router.push("/create");
      return;
    case "cancel":
      void chains.cancel(vm.hostId, vm.jobId).catch((err) => toasts.push(String(err), "error"));
      return;
    case "resume":
      void chains.resume(vm.hostId, vm.jobId).catch((err) => toasts.push(String(err), "error"));
      return;
    case "delete":
      confirmDeleteSequence.value = vm;
      return;
    default:
      return;
  }
}

function selectSequence(vm: SequenceVM) {
  composer.setSequence({ kind: "inspect", hostId: vm.hostId, jobId: vm.jobId });
  emit("close");
  void router.push("/create");
}

function deleteSequenceConfirmed() {
  const vm = confirmDeleteSequence.value;
  confirmDeleteSequence.value = null;
  if (!vm) return;
  void chains.remove(vm.hostId, vm.jobId).catch((err) => toasts.push(String(err), "error"));
}

// ── Maintenance (moved out of the Create composer) ─────────────────────────

/** `Clear inactive` is a server DELETE, not the strip's client-side dismiss —
 *  the confirm names the host and the count so the two never blur. */
const confirmingClearSequences = ref(false);
const clearingSequences = ref(false);
const confirmCleanUpDisk = ref(false);

const inactiveSequenceCount = computed(
  () => sequenceRows.value.filter((vm) => vm.settledAtMs !== null).length,
);
const sequenceHostScope = computed(() => (gallery.filter === "all" ? null : gallery.filter));
const sequenceHostLabel = computed(() =>
  sequenceHostScope.value
    ? (hosts.all.find((h) => h.id === sequenceHostScope.value)?.label ?? sequenceHostScope.value)
    : "every machine",
);

const clearSequencesLabel = computed(() => {
  if (!confirmingClearSequences.value) return "Clear inactive";
  const n = inactiveSequenceCount.value;
  return `Delete ${n} inactive sequence${n === 1 ? "" : "s"} on ${sequenceHostLabel.value}?`;
});

async function clearInactiveSequences() {
  if (!confirmingClearSequences.value) {
    confirmingClearSequences.value = true;
    return;
  }
  confirmingClearSequences.value = false;
  clearingSequences.value = true;
  try {
    const scope = sequenceHostScope.value;
    const { cleared, failed } = scope
      ? await chains.clearInactive(scope)
      : await chains.clearInactive();
    if (failed > 0) toasts.push(`Cleared ${cleared}, ${failed} could not be deleted`, "error");
    else toasts.push(`Cleared ${cleared} sequence${cleared === 1 ? "" : "s"}`);
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    clearingSequences.value = false;
  }
}

function cleanUpDisk() {
  confirmCleanUpDisk.value = true;
}

async function cleanUpDiskConfirmed() {
  confirmCleanUpDisk.value = false;
  try {
    const hostIds = sequenceHostScope.value
      ? [sequenceHostScope.value]
      : Object.keys(chains.byHost);
    let swept = 0;
    let pruned = 0;
    for (const hostId of hostIds) {
      const outcome = await chains.gc(hostId);
      swept += outcome.swept_ephemeral_jobs;
      pruned += outcome.pruned_artifact_dirs;
    }
    toasts.push(`Swept ${swept} jobs, pruned ${pruned} dirs`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}
</script>

<template>
  <DrawerPanel :open="open" :width="drawerWidth" title="History" @close="emit('close')">
    <template #leading>
      <PanelResizeHandle
        class="absolute inset-y-0 -left-0.5 z-10"
        label="Resize history"
        @resize="onDrawerResize"
        @commit="onDrawerCommit"
        @reset="onDrawerReset"
      />
    </template>
    <template #header>
      <span class="ms-drawer-kicker">History</span>
      <div
        class="ml-3 flex rounded-control border border-control-edge bg-bath p-0.5"
        role="group"
        aria-label="History view"
      >
        <button
          type="button"
          data-test="tab-runs"
          :aria-pressed="tab === 'runs'"
          class="rounded-control px-2.5 py-1 text-body transition-colors"
          :class="tab === 'runs' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'"
          @click="selectTab('runs')"
        >
          Runs
        </button>
        <button
          type="button"
          data-test="tab-prompts"
          :aria-pressed="tab === 'prompts'"
          class="rounded-control px-2.5 py-1 text-body transition-colors"
          :class="tab === 'prompts' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'"
          @click="selectTab('prompts')"
        >
          Prompts
        </button>
        <button
          type="button"
          data-test="tab-sequences"
          :aria-pressed="tab === 'sequences'"
          class="rounded-control px-2.5 py-1 text-body transition-colors"
          :class="tab === 'sequences' ? 'bg-bench text-ink shadow-sm' : 'text-ink-2 hover:text-ink'"
          @click="selectTab('sequences')"
        >
          Sequences
        </button>
      </div>
    </template>

    <div class="flex flex-col gap-3">
      <div v-if="tab !== 'sequences'" class="flex items-center gap-2">
        <input
          v-model="query"
          data-selectable
          type="search"
          :placeholder="tab === 'runs' ? 'Search runs…' : 'Search prompts…'"
          class="border-edge h-7 flex-1 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
        />
        <button
          v-if="tab === 'prompts'"
          type="button"
          data-test="clear-history"
          class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body transition-colors duration-100"
          :class="
            confirmingClear
              ? 'border-stop bg-stop font-semibold text-on-accent'
              : 'text-ink-2 hover:text-stop'
          "
          @blur="confirmingClear = false"
          @click="clearAll"
        >
          {{ clearLabel }}
        </button>
      </div>

      <!-- One origin filter for both tabs — the same chips the Library uses -->
      <HostFilterChips
        v-if="showChips"
        v-model="gallery.filter"
        :chips="gallery.chipCounts"
        :all-count="gallery.merged.length"
      />

      <!-- Runs: every finished generation with print + settings -->
      <div v-if="tab === 'runs'">
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
            class="group flex w-full items-center gap-3 rounded-control px-2 py-1.5 text-left hover:bg-bath"
            @click="useRun(entry.item)"
            @contextmenu="contextMenu.open($event, runMenu(entry))"
          >
            <div
              class="h-12 w-12 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
            >
              <AuthedMedia
                :path="
                  galleryMediaPath(
                    entry.item.filename,
                    gallery.mediaSourceOf(entry.sourceKey),
                    true,
                  )
                "
                :target="gallery.targetOf(entry.sourceKey)"
                :cache-key="entry.sourceKey"
                :media-version="entry.item.media_version ?? null"
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
              class="edge-code max-w-24 shrink-0 truncate"
            >
              {{ availabilityLabel(entry) }}
            </span>
            <span class="data-mono shrink-0 text-caption text-ink-3">{{
              runTime(entry.item)
            }}</span>
          </button>
        </template>
        <p v-if="runsCapNote" data-test="runs-cap-note" class="edge-code mt-1">
          {{ runsCapNote }}
        </p>
      </div>

      <!-- Sequences: the durable chain jobs, with their maintenance -->
      <div v-else-if="tab === 'sequences'" class="flex flex-col gap-2">
        <EmptyState
          v-if="sequenceRows.length === 0"
          data-test="sequences-empty"
          headline="No sequences yet"
          detail="A sequence you develop keeps its job here — watch it, edit its clips, or clear it out."
          action="Go to Create"
          @action="
            emit('close');
            router.push('/create');
          "
        />
        <template v-else>
          <div
            v-for="vm in visibleSequences"
            :key="vm.key"
            data-test="sequence-row"
            class="rounded-control px-2 py-1.5 hover:bg-bath"
          >
            <SequenceJobRow
              class="min-w-0"
              dense
              :vm="vm"
              :model-label="sequenceModelLabel(vm.model)"
              :time-label="sequenceTime(vm)"
              @action="onSequenceAction"
              @select="selectSequence"
            >
              <template #actions>
                <button
                  v-if="printOf(vm)"
                  type="button"
                  data-test="seq-show-print"
                  class="border-ce shrink-0 rounded-control border px-2 py-0.5 text-caption text-ink-2 hover:text-rebate"
                  @click.stop="showPrint(vm)"
                >
                  Show print
                </button>
              </template>
            </SequenceJobRow>
          </div>
          <p v-if="sequenceCapNote" data-test="sequence-cap-note" class="edge-code mt-1">
            {{ sequenceCapNote }}
          </p>
        </template>

        <!-- Maintenance lives here now: destructive, host-scoped, rarely used. -->
        <div class="border-edge mt-2 flex items-center gap-2 border-t pt-2">
          <button
            type="button"
            data-test="seq-clear-inactive"
            class="border-edge h-7 rounded-control border px-2.5 text-body transition-colors duration-100"
            :class="
              confirmingClearSequences
                ? 'border-stop bg-stop font-semibold text-on-accent'
                : 'text-ink-2 hover:text-stop'
            "
            :disabled="clearingSequences"
            title="Delete every sequence job that isn't running or queued"
            @blur="confirmingClearSequences = false"
            @click="clearInactiveSequences"
          >
            {{ clearSequencesLabel }}
          </button>
          <button
            type="button"
            data-test="seq-cleanup-disk"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 transition-colors hover:text-ink"
            title="Sweep ephemeral jobs and prune artifact directories"
            @click="cleanUpDisk"
          >
            Clean up disk
          </button>
        </div>
      </div>

      <!-- Prompts: the raw prompt log, merged across every ready host -->
      <div v-else>
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
            class="group flex w-full items-center gap-3 rounded-control px-2 py-1.5 text-left hover:bg-bath"
            @click="usePrompt(entry)"
            @contextmenu="contextMenu.open($event, promptMenu(entry))"
          >
            <span class="min-w-0 flex-1 truncate text-body text-ink" :title="entry.prompt">
              {{ entry.prompt }}
            </span>
            <span
              v-if="showBadges"
              data-test="host-badge"
              class="edge-code max-w-24 shrink-0 truncate"
            >
              {{ entry.hostLabel }}
            </span>
            <span class="data-mono shrink-0 text-caption text-ink-3">{{ entry.model }}</span>
            <span class="data-mono shrink-0 text-caption text-ink-3">{{ timeOf(entry) }}</span>
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

    <ConfirmDialog
      :open="confirmDeleteSequence !== null"
      title="Delete this sequence?"
      :message="`Removes the job and its cached clips${confirmDeleteSequence ? ` from ${confirmDeleteSequence.hostLabel}` : ''}. The finished video in the Library is kept.`"
      confirm-label="Delete"
      danger
      @confirm="deleteSequenceConfirmed"
      @cancel="confirmDeleteSequence = null"
    />
    <ConfirmDialog
      :open="confirmCleanUpDisk"
      title="Clean up sequence cache?"
      message="This discards cached scene media used for scene playback and sequence editing. Final videos in the Library remain."
      confirm-label="Clean up"
      danger
      @confirm="cleanUpDiskConfirmed"
      @cancel="confirmCleanUpDisk = false"
    />
  </DrawerPanel>
</template>

<style scoped>
.ms-drawer-kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
</style>
