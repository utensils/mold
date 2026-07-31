<script setup lang="ts">
/*
 * ⌘K command palette (spec §06). Presentational PalettePanel + the command
 * registry: navigation, quick actions, theme, and model search.
 *
 * Model search covers the whole fleet, not just the machine the next job would
 * land on. One query answers all three states a model can be in: installed
 * here (use it), installed on another machine (repin the target, then use it),
 * or nowhere yet (pull it). The last case comes from a debounced live catalog
 * search, appended after the ranked commands so a slow round trip can never
 * reorder rows out from under a keystroke.
 *
 * Installs from here deliberately do NOT open the machine picker: that dialog
 * is mounted by the Models page, so it would be invisible from any other route.
 * The palette takes the plan's first target (installs before repairs, origin
 * first) and names the machine in the toast; Models remains the surface for
 * choosing explicitly.
 */
import { computed, onBeforeUnmount, ref, toRef, watch } from "vue";
import { useRouter } from "vue-router";
import PalettePanel from "@ui/components/PalettePanel.vue";
import {
  CATALOG_SEARCH_DEBOUNCE_MS,
  shouldSearchCatalog,
  type SearchableCatalogEntry,
} from "@studio/lib/modelSearch";
import {
  baseCommands,
  catalogCommands,
  filterCommands,
  modelCommands,
  type Command,
  type CommandContext,
} from "../../lib/commands";
import { fetchCatalogSearch } from "../../api";
import { isStandaloneGenerationModel } from "../../lib/modelFilters";
import { toast } from "../../lib/toasts";
import { useGenerateForm } from "../../composables/useGenerateForm";
import { useHostRouting } from "../../composables/useHostRouting";
import { useModelInstallTargets } from "../../composables/useModelInstallTargets";
import { useOverlayFocus } from "../../composables/useOverlayFocus";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();

const router = useRouter();
const form = useGenerateForm();
const routing = useHostRouting();
const installTargets = useModelInstallTargets();

const query = ref("");
const catalogEntries = ref<SearchableCatalogEntry[]>([]);
const catalogBusy = ref(false);
const host = ref<HTMLElement | { $el?: unknown } | null>(null);
const { onKeydown } = useOverlayFocus(toRef(props, "open"), host, () =>
  emit("close"),
);

let debounce: ReturnType<typeof setTimeout> | null = null;
// Monotonic epoch so a slow response for an abandoned query can never
// overwrite the results of a newer one.
let searchEpoch = 0;

/** Installed generation models across every ready machine. */
const installed = computed(() =>
  routing.installedModels.value.filter(isStandaloneGenerationModel),
);

const installedNames = computed(() => installed.value.map((m) => m.name));

function hostLabel(hostId: string): string {
  return routing.hosts.value.find((h) => h.id === hostId)?.label ?? hostId;
}

const ctx: CommandContext = {
  go: (path) => {
    void router.push(path);
  },
  runModel: (name, switchToHostId) => {
    // Repin the generation target first: the user picked a model, and honouring
    // a pin that cannot run it would silently contradict the choice they made.
    if (switchToHostId) {
      routing.setTarget(switchToHostId);
      toast("info", `Generating on ${hostLabel(switchToHostId)}`);
    }
    // Apply the model's full defaults (family, dimensions, family-gated fields,
    // LoRA reset) exactly like the Create page's model rail. Setting only
    // `model` would leave stale family state behind — e.g. a Qwen-edit → FLUX
    // swap could still emit edit_images from the old family.
    const model = installed.value.find((m) => m.name === name);
    if (model) form.applyModelDefaults(model);
    else form.state.value.model = name;
    void router.push("/create");
  },
  installModel: (modelId, displayName) => {
    const target = installTargets.planFor(modelId).targets[0] ?? null;
    void installTargets
      .startDownloadOn(target, modelId)
      .then(() => {
        toast("success", installTargets.queuedMessage(target));
        window.dispatchEvent(new CustomEvent("mold:open-downloads"));
      })
      .catch((error: unknown) => {
        const detail = error instanceof Error ? error.message : String(error);
        toast("error", `Couldn't queue ${displayName}: ${detail}`);
      });
  },
  openDownloads: () => {
    window.dispatchEvent(new CustomEvent("mold:open-downloads"));
  },
  newPrint: () => {
    void router.push("/create");
    window.dispatchEvent(new CustomEvent("mold:new-print"));
  },
};

/** Ranked commands: navigation, actions, theme, and installed models. */
const commands = computed<Command[]>(() => [
  ...baseCommands(ctx),
  ...modelCommands(installed.value, ctx, {
    ownersFor: (name) => routing.modelOwnerIds(name),
    targetHostId: routing.targetId.value,
    hostLabel,
  }),
]);

/** Catalog rows keep the server's own relevance order and stay a tail. */
const installRows = computed<Command[]>(() =>
  catalogCommands(catalogEntries.value, ctx, {
    installedNames: installedNames.value,
  }),
);

const filtered = computed(() => [
  ...filterCommands(commands.value, query.value),
  ...installRows.value,
]);

const items = computed(() =>
  filtered.value.map((c) => ({
    id: c.id,
    section: c.section,
    label: c.label,
    ...(c.hint ? { hint: c.hint } : {}),
  })),
);

function run(id: string) {
  const command = filtered.value.find((c) => c.id === id);
  if (command) command.run();
  emit("close");
}

function clearDebounce() {
  if (debounce !== null) {
    clearTimeout(debounce);
    debounce = null;
  }
}

async function searchCatalog(q: string, epoch: number) {
  try {
    // Checkpoints only: the palette's promise is "switch to this model", and a
    // LoRA or VAE is not something you can switch to.
    const response = await fetchCatalogSearch({
      q,
      kind: "checkpoint",
      page_size: 12,
    });
    if (epoch !== searchEpoch) return;
    catalogEntries.value = response.entries
      // An unsupported entry has no runnable pipeline — offering to pull it
      // would end in a model the user cannot switch to.
      .filter((entry) => entry.supported !== false)
      .map((entry) => ({
        id: entry.id,
        name: entry.name,
        family: entry.family ?? null,
        source: entry.source ?? null,
        installed: entry.installed ?? null,
      }));
  } catch {
    if (epoch !== searchEpoch) return;
    // The palette still works offline; it just stops offering installs.
    catalogEntries.value = [];
  } finally {
    if (epoch === searchEpoch) catalogBusy.value = false;
  }
}

watch(query, (q) => {
  clearDebounce();
  const epoch = ++searchEpoch;
  if (!shouldSearchCatalog(q)) {
    catalogEntries.value = [];
    catalogBusy.value = false;
    return;
  }
  catalogBusy.value = true;
  debounce = setTimeout(() => {
    void searchCatalog(q.trim(), epoch);
  }, CATALOG_SEARCH_DEBOUNCE_MS);
});

watch(
  () => props.open,
  (open) => {
    if (!open) {
      clearDebounce();
      searchEpoch++;
      catalogBusy.value = false;
      return;
    }
    query.value = "";
    catalogEntries.value = [];
    // One poll so a palette opened from Library still knows the fleet's
    // inventory; the routing composable dedupes concurrent refreshes.
    void routing.refresh().catch(() => {
      // Navigation, action, and theme commands stay usable offline.
    });
  },
);

onBeforeUnmount(clearDebounce);
</script>

<template>
  <PalettePanel
    ref="host"
    :open="open"
    :query="query"
    :items="items"
    :busy="catalogBusy"
    empty-text="No matches"
    @close="emit('close')"
    @update:query="query = $event"
    @run="run"
    @keydown="onKeydown"
  />
</template>
