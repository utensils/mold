<script setup lang="ts">
/*
 * History — the Library's Runs + Prompts log as an INLINE column beside the
 * grid, never a modal drawer: no scrim, no `aria-modal`, and the tiles reflow
 * next to it and stay clickable. Runs are gallery-backed (every finished
 * generation with its print, settings, and seed); Prompts is the raw prompt log
 * fanned out over every ready host. All the reuse/clear actions from the old
 * History screen are preserved. Opens via ?panel=history with ?tab= selecting
 * the lens (the retired /history route and the command palette both deep-link
 * here); Library owns the open state and closing.
 */
import { computed, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import Icon from "@ui/components/Icon.vue";
import { HISTORY_JOBS_RENDER_CAP } from "@studio/lib/activity";
import EmptyState from "../shell/EmptyState.vue";
import AuthedMedia from "../gallery/AuthedMedia.vue";
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
import { useConnectionStore } from "../../stores/connection";
import { useComposerStore } from "../../stores/composer";
import { useGalleryStore } from "../../stores/gallery";
import { useGenerateFormStore } from "../../stores/generateForm";
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
import { useReuseStillPrint } from "../../composables/useReuseStillPrint";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();

const route = useRoute();
const router = useRouter();
const conn = useConnectionStore();
const composer = useComposerStore();
const gallery = useGalleryStore();
const generateForm = useGenerateFormStore();
const hosts = useHostsStore();
const models = useModelStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();
const appPrefs = useAppPrefsStore();
const reuseStillPrint = useReuseStillPrint();

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
 * Two lenses on the past: Runs (every finished generation with its print,
 * settings, and seed — the gallery DB is the source of truth) and Prompts (the
 * prompt log, including prompts whose outputs are gone).
 *
 * The tab lives in the URL so the shell can deep-link a lens. Anything else —
 * a bookmarked `?tab=sequences` from the retired sequence log included —
 * normalizes to Runs rather than rendering nothing.
 */
type HistoryTab = "runs" | "prompts";
const TABS: HistoryTab[] = ["runs", "prompts"];

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
 *  capped: a 1 000-print library must not mount 1 000 media requests the
 *  moment the drawer opens. */
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

/** The mock's one meta line: `flux-dev:q4 · 1024² · seed 4821 · 12:41`. A
 *  square canvas is written once with a superscript two, as the mock does. */
function runMeta(img: GalleryImage): string {
  const { model, width, height, seed } = img.metadata;
  const size = width === height ? `${width}²` : `${width}×${height}`;
  return `${model} · ${size} · seed ${seed} · ${runTime(img)}`;
}

/**
 * "Use these settings" on a past run. It takes the SAME road the Lightbox and
 * the Create view's Recent tab take — `composer.set` restored the numbers and
 * dropped the photo the print was made from, because it invalidates
 * retained-source authority.
 */
function useRun(entry: MergedPrint) {
  reuseStillPrint(entry);
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
    { label: "Use these settings", action: () => useRun(entry) },
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
    { label: "Show in My images", action: () => emit("close") },
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
</script>

<template>
  <aside
    v-if="open"
    data-test="history-panel"
    aria-label="History"
    class="border-border relative flex shrink-0 flex-col border-l bg-chrome"
    :style="{ width: `${drawerWidth}px` }"
  >
    <PanelResizeHandle
      class="absolute inset-y-0 -left-0.5 z-10"
      label="Resize history"
      @resize="onDrawerResize"
      @commit="onDrawerCommit"
      @reset="onDrawerReset"
    />
    <header
      class="border-border flex h-[var(--mold-shell-viewbar-h)] shrink-0 items-center gap-2 border-b bg-bg px-3.5"
    >
      <span class="text-sm font-semibold text-fg">History</span>
      <span class="flex-1" />
      <button
        type="button"
        class="flex h-6 w-6 items-center justify-center rounded-control text-fg-dim hover:text-fg"
        aria-label="Close history"
        @click="emit('close')"
      >
        <Icon name="close" :size="13" />
      </button>
    </header>

    <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
      <div
        class="flex rounded-control border border-border-control bg-bg-deep p-0.5"
        role="group"
        aria-label="History view"
      >
        <button
          type="button"
          data-test="tab-runs"
          :aria-pressed="tab === 'runs'"
          class="rounded-control px-2.5 py-1 text-sm transition-colors"
          :class="tab === 'runs' ? 'bg-bg text-fg shadow-sm' : 'text-fg-2 hover:text-fg'"
          @click="selectTab('runs')"
        >
          Runs
        </button>
        <button
          type="button"
          data-test="tab-prompts"
          :aria-pressed="tab === 'prompts'"
          class="rounded-control px-2.5 py-1 text-sm transition-colors"
          :class="tab === 'prompts' ? 'bg-bg text-fg shadow-sm' : 'text-fg-2 hover:text-fg'"
          @click="selectTab('prompts')"
        >
          Prompts
        </button>
      </div>

      <div class="flex items-center gap-2">
        <input
          v-model="query"
          data-selectable
          type="search"
          :placeholder="tab === 'runs' ? 'Search runs…' : 'Search prompts…'"
          class="border-border h-7 flex-1 rounded-control border bg-bg-deep px-2 text-sm text-fg placeholder:text-fg-dim"
        />
        <button
          v-if="tab === 'prompts'"
          type="button"
          data-test="clear-history"
          class="border-border h-7 shrink-0 rounded-control border px-2.5 text-sm transition-colors duration-100"
          :class="
            confirmingClear
              ? 'border-error bg-error font-semibold text-on-accent'
              : 'text-fg-2 hover:text-error'
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
          detail="Every picture you make shows up here with its settings and seed."
        />
        <p v-else-if="query && runs.length === 0" class="mt-6 text-center text-sm text-fg-2">
          No runs match “{{ query }}”.
        </p>
        <template v-for="group in runGroups" :key="group.label">
          <div class="mt-3 mb-1 flex items-center gap-2 first:mt-0">
            <span class="font-mono text-micro text-fg-dim whitespace-nowrap">{{
              group.label.toUpperCase()
            }}</span>
            <div class="border-border h-px flex-1 border-t" />
          </div>
          <button
            v-for="entry in group.runs"
            :key="runKey(entry)"
            type="button"
            data-test="run-row"
            class="border-border flex w-full items-center gap-2.5 rounded-control border p-2.5 text-left hover:bg-surface"
            @click="useRun(entry)"
            @contextmenu="contextMenu.open($event, runMenu(entry))"
          >
            <div
              class="h-12 w-12 shrink-0 overflow-hidden rounded-inner border border-border bg-media-bed"
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
            <div class="flex min-w-0 flex-1 flex-col gap-1">
              <span class="truncate text-xs text-fg" :title="entry.item.metadata.prompt">
                {{ entry.item.metadata.prompt }}
              </span>
              <span class="font-mono truncate text-micro text-fg-dim" data-test="run-meta">
                {{ runMeta(entry.item) }}
              </span>
              <span
                v-if="showBadges"
                data-test="host-badge"
                class="font-mono truncate text-micro text-fg-dim"
              >
                {{ availabilityLabel(entry) }}
              </span>
              <span class="text-micro font-semibold text-accent">Use these settings</span>
            </div>
          </button>
        </template>
        <p
          v-if="runsCapNote"
          data-test="runs-cap-note"
          class="font-mono text-micro text-fg-dim whitespace-nowrap mt-1"
        >
          {{ runsCapNote }}
        </p>
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
            <span class="font-mono text-micro text-fg-dim whitespace-nowrap">{{
              group.label.toUpperCase()
            }}</span>
            <div class="border-border h-px flex-1 border-t" />
          </div>
          <button
            v-for="(entry, i) in group.entries"
            :key="`${group.label}-${i}`"
            type="button"
            data-test="prompt-row"
            class="group flex w-full items-center gap-3 rounded-control px-2 py-1.5 text-left hover:bg-bg-deep"
            @click="usePrompt(entry)"
            @contextmenu="contextMenu.open($event, promptMenu(entry))"
          >
            <span class="min-w-0 flex-1 truncate text-sm text-fg" :title="entry.prompt">
              {{ entry.prompt }}
            </span>
            <span
              v-if="showBadges"
              data-test="host-badge"
              class="font-mono text-micro text-fg-dim whitespace-nowrap max-w-24 shrink-0 truncate"
            >
              {{ entry.hostLabel }}
            </span>
            <span class="font-mono shrink-0 text-micro text-fg-dim">{{ entry.model }}</span>
            <span class="font-mono shrink-0 text-micro text-fg-dim">{{ timeOf(entry) }}</span>
          </button>
        </template>
        <p
          v-if="query && visiblePrompts.length === 0 && !unavailable"
          class="mt-6 text-center text-sm text-fg-2"
        >
          No prompts match “{{ query }}”.
        </p>
      </div>
    </div>
  </aside>
</template>
