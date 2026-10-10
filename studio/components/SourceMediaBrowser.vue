<script setup lang="ts">
import { computed, ref, shallowRef, watch, onBeforeUnmount } from "vue";
import SourceMediaCard from "./SourceMediaCard.vue";
import { sourceMediaInventories } from "../api/sourceMediaBrowser";
import type { RetainedSourceMediaInventory } from "../api/gallerySourceMedia";
import {
  sourceOwnerIdentity,
  filterSourceMedia,
  sourceHostIdentity,
  groupSourceMedia,
  type SourceMediaOwner,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
import {
  observeSourceMediaEvents,
  sourceEventMatches,
  cachedSourceInventory,
  cacheSourceInventory,
  invalidateSourceOwner,
} from "../lib/sourceMediaResources";
const props = withDefaults(
  defineProps<{ owners: SourceMediaOwner[]; showTitle?: boolean }>(),
  { showTitle: true },
);
const emit = defineEmits<{
  use: [selection: SourceMediaSelection];
  openOwner: [owner: SourceMediaOwner];
}>();
const inventories = shallowRef<Record<string, RetainedSourceMediaInventory>>(
  {},
);
const failures = shallowRef<Record<string, string>>({});
const search = ref("");
const kind = ref("");
const machine = ref("");
const machines = computed(() => [
  ...new Map(
    props.owners.map((owner) => [sourceHostIdentity(owner), owner]),
  ).entries(),
]);
const machineChoice = computed({
  get: () =>
    machine.value
      ? String(
          machines.value.findIndex(([identity]) => identity === machine.value),
        )
      : "",
  set: (value: string) => {
    machine.value =
      value === "" ? "" : (machines.value[Number(value)]?.[0] ?? "");
  },
});

const shown = ref(48);
const active = ref(0);
const loading = computed(() => active.value > 0);
const checked = new Set<string>();
const requests = new Map<AbortController, Set<string>>();
let known = new Set<string>();
let disposed = false;
const queued: SourceMediaOwner[][] = [];
let workers = 0;
const currentOwners = computed(
  () =>
    new Map(props.owners.map((owner) => [sourceOwnerIdentity(owner), owner])),
);
const scanned = computed(
  () =>
    Object.keys(inventories.value).length + Object.keys(failures.value).length,
);
const groups = computed(() =>
  groupSourceMedia(
    props.owners.flatMap((owner) => {
      const inventory = inventories.value[sourceOwnerIdentity(owner)];
      return inventory?.availability === "available"
        ? inventory.members.map((member) => ({ owner, member }))
        : [];
    }),
  ),
);
const rows = computed(() =>
  filterSourceMedia(
    groups.value.flatMap((group) => group.usages),
    { search: search.value, kind: kind.value, host: machine.value },
  ),
);

watch(
  () => rows.value.length,
  (count) => {
    shown.value = Math.min(
      shown.value,
      Math.max(48, Math.ceil(count / 48) * 48),
    );
  },
  { flush: "sync" },
);
watch([search, kind, machine], () => {
  shown.value = 48;
});
async function load(owners: SourceMediaOwner[]) {
  const missing = [
    ...new Map(
      owners.map((owner) => [sourceOwnerIdentity(owner), owner]),
    ).values(),
  ].filter((owner) => !checked.has(sourceOwnerIdentity(owner)));
  if (!missing.length || disposed) return;
  const cached = { ...inventories.value };
  const hosts = new Map<string, SourceMediaOwner[]>();
  for (const owner of missing) {
    const key = sourceOwnerIdentity(owner);
    checked.add(key);
    const value = cachedSourceInventory(key);
    if (value) {
      cached[key] = value;
      continue;
    }
    const host = JSON.stringify([
      owner.target.baseUrl,
      owner.target.apiKey,
      owner.instanceId,
    ]);
    let batch = hosts.get(host);
    if (!batch) hosts.set(host, (batch = []));
    batch.push(owner);
  }
  inventories.value = cached;
  const hostOwners = [...hosts.values()];
  const batches = [
    ...hostOwners.map((owners) => owners.slice(0, 25)),
    ...hostOwners.flatMap((owners) =>
      Array.from(
        { length: Math.ceil(Math.max(0, owners.length - 25) / 100) },
        (_, n) => owners.slice(25 + n * 100, 125 + n * 100),
      ),
    ),
  ];
  // Paint a small first page, then discover the full library in bounded background batches.
  queued.push(...batches);
  async function worker() {
    if (workers >= 2) return;
    workers++;
    try {
      while (queued.length && !disposed) {
        const owners = queued
          .shift()!
          .filter((owner) =>
            currentOwners.value.has(sourceOwnerIdentity(owner)),
          );
        if (!owners.length) continue;
        const controller = new AbortController();
        requests.set(controller, new Set(owners.map(sourceOwnerIdentity)));
        active.value++;
        try {
          const first = owners[0]!;
          const result = await sourceMediaInventories(
            [...new Set(owners.map((o) => o.filename))],
            first.target,
            controller.signal,
            first.instanceId,
          );
          if (controller.signal.aborted || disposed) continue;
          const values = { ...inventories.value },
            errors = { ...failures.value };
          for (const owner of owners) {
            const key = sourceOwnerIdentity(owner);
            if (!currentOwners.value.has(key)) continue;
            const value = result.inventories[owner.filename];
            if (
              value &&
              ["available", "unavailable_legacy"].includes(value.availability)
            ) {
              values[key] = value;
              delete errors[key];
              cacheSourceInventory(key, value);
            } else {
              delete values[key];
              invalidateSourceOwner(key);
              errors[key] =
                `${owner.title}: ${value?.availability.replaceAll("_", " ") ?? result.errors?.[owner.filename] ?? "Source inventory unavailable"}`;
            }
          }
          inventories.value = values;
          failures.value = errors;
        } catch (error) {
          if (!controller.signal.aborted && !disposed) {
            const errors = { ...failures.value },
              values = { ...inventories.value };
            for (const owner of owners) {
              const key = sourceOwnerIdentity(owner);
              if (currentOwners.value.has(key)) {
                delete values[key];
                invalidateSourceOwner(key);
                errors[key] =
                  `${owner.hostLabel}: ${error instanceof Error ? error.message : String(error)}`;
              }
            }
            inventories.value = values;
            failures.value = errors;
          }
        } finally {
          requests.delete(controller);
          active.value--;
        }
      }
    } finally {
      workers--;
    }
  }
  await Promise.all([worker(), worker()]);
}
function refresh() {
  queued.length = 0;
  for (const controller of requests.keys()) controller.abort();
  const owners = props.owners.filter((owner) =>
    checked.has(sourceOwnerIdentity(owner)),
  );
  for (const owner of owners) {
    const key = sourceOwnerIdentity(owner);
    checked.delete(key);
    invalidateSourceOwner(key, true);
  }
  inventories.value = {};
  failures.value = {};
  void load(owners.length ? owners : props.owners.slice(0, 25));
}
watch(
  () => props.owners.map(sourceOwnerIdentity).join("\n"),
  () => {
    const keys = new Set(currentOwners.value.keys());
    const values = { ...inventories.value },
      errors = { ...failures.value };
    for (const key of known)
      if (!keys.has(key)) {
        delete values[key];
        delete errors[key];
        checked.delete(key);
        invalidateSourceOwner(key);
      }
    const interrupted = new Set<string>();
    for (const [controller, held] of requests)
      if ([...held].some((key) => !keys.has(key))) {
        controller.abort();
        for (const key of held)
          if (!values[key]) {
            checked.delete(key);
            if (keys.has(key)) interrupted.add(key);
          }
      }
    inventories.value = values;
    failures.value = errors;
    const added = props.owners.filter(
      (owner) =>
        !known.has(sourceOwnerIdentity(owner)) ||
        interrupted.has(sourceOwnerIdentity(owner)),
    );
    known = keys;
    void load([
      ...added,
      ...props.owners.filter((owner) =>
        cachedSourceInventory(sourceOwnerIdentity(owner)),
      ),
    ]);
  },
  { immediate: true, flush: "sync" },
);
const stopEvents = observeSourceMediaEvents((event) => {
  const matches = props.owners.filter((owner) =>
    sourceEventMatches(owner, event),
  );
  const keys = new Set(matches.map(sourceOwnerIdentity));
  for (const [controller, batch] of requests) {
    if ([...batch].some((key) => keys.has(key))) {
      controller.abort();
      for (const key of batch) keys.add(key);
    }
  }
  const next = { ...inventories.value };
  for (const key of keys) {
    checked.delete(key);
    delete next[key];
  }
  inventories.value = next;
  void load(
    props.owners.filter((owner) => keys.has(sourceOwnerIdentity(owner))),
  );
});
const timer = setInterval(() => {
  const stale: SourceMediaOwner[] = [];
  const next = { ...inventories.value };
  for (const owner of props.owners) {
    const key = sourceOwnerIdentity(owner);
    if (!next[key]) continue;
    const cached = cachedSourceInventory(key);
    if (cached) next[key] = cached;
    else {
      checked.delete(key);
      stale.push(owner);
    }
  }
  inventories.value = next;
  void load(stale);
}, 60_000);
onBeforeUnmount(() => {
  disposed = true;
  stopEvents();
  clearInterval(timer);
  for (const controller of requests.keys()) controller.abort();
});
</script>
<template>
  <section
    class="source-browser"
    aria-label="Source Media"
    data-test="source-media-browser"
  >
    <header>
      <h2 v-if="showTitle">Source Media</h2>
      <p>
        Original inputs retained by your library. Reuse them in a new creation;
        their lifetime follows the prints that keep them.
      </p>
    </header>
    <div class="source-browser__filters">
      <label class="source-browser__search"
        >Search loaded sources<input
          v-model="search"
          type="search"
          placeholder="Name, role, print or machine"
      /></label>
      <label
        >Media type<select v-model="kind" aria-label="Source media type">
          <option value="">All media</option>
          <option value="image">Images</option>
          <option value="video">Video</option>
          <option value="audio">Audio</option>
          <option value="unknown">Other files</option>
        </select></label
      >
      <label
        >Machine<select v-model="machineChoice" aria-label="Source machine">
          <option value="">All machines</option>
          <option
            v-for="([key, owner], index) in machines"
            :key="key"
            :value="String(index)"
          >
            {{ owner.hostLabel }} · {{ owner.target.baseUrl }}
          </option>
        </select></label
      >
    </div>
    <p class="source-browser__status" role="status">
      {{ scanned }} of {{ owners.length }} library items checked ·
      {{ groups.length }} unique sources loaded<span v-if="loading">
        · Loading…</span
      >
    </p>
    <button type="button" :disabled="loading" @click="refresh">
      Refresh sources
    </button>
    <div
      v-if="Object.keys(failures).length"
      class="source-browser__error"
      role="status"
    >
      <p>Some sources could not be read.</p>
      <p
        v-for="message in [...new Set(Object.values(failures))].slice(0, 4)"
        :key="message"
      >
        {{ message }}
      </p>
      <button type="button" @click="refresh">Retry</button>
    </div>
    <div class="source-browser__grid">
      <SourceMediaCard
        v-for="group in rows.slice(shown - 48, shown)"
        :key="group.key"
        :selection="group.usages[0]!"
        :usages="group.usages"
        :show-owner="true"
        @use="emit('use', $event)"
        @open-owner="emit('openOwner', $event)"
      />
    </div>
    <p v-if="!loading && !rows.length">
      {{
        search || kind || machine
          ? "No loaded sources match these filters."
          : "No retained sources found in the checked library items."
      }}
    </p>
    <button v-if="shown > 48" type="button" @click="shown -= 48">
      Previous sources
    </button>
    <button v-if="rows.length > shown" type="button" @click="shown += 48">
      Next sources ({{ Math.min(shown, rows.length) }} of {{ rows.length }})
    </button>
  </section>
</template>
<style scoped>
.source-browser {
  padding: 20px;
  overflow: auto;
  min-height: 0;
  flex: 1;
  color: var(--mold-text, inherit);
}
.source-browser h2 {
  font-size: var(--mold-fs-lg);
  font-weight: 600;
  margin: 0 0 8px;
}
.source-browser header p,
.source-browser__status,
.source-browser__machine {
  color: var(--mold-text-dim, #888);
  font-size: var(--mold-fs-sm);
}
.source-browser__filters {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-block: 20px;
  align-items: end;
}
.source-browser__filters label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  flex: 1 1 160px;
}
.source-browser__search {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-block: 0;
  font-size: var(--mold-fs-sm);
}
.source-browser input,
.source-browser select {
  width: 100%;
  min-width: 0;
  min-height: 44px;
  padding: 10px 12px;
  background: var(--mold-surface, #8881);
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-3);
  color: inherit;
  font-size: max(1rem, var(--mold-fs-md));
}
.source-browser__owner {
  margin-block: 24px;
}
.source-browser__owner-link {
  font-weight: 600;
  text-align: left;
  text-decoration: underline;
  text-underline-offset: 3px;
}
.source-browser__machine {
  margin-left: 10px;
}
.source-browser__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
  gap: 12px;
  margin-top: 10px;
}
.source-browser button {
  min-height: 44px;
  color: inherit;
  cursor: pointer;
}
.source-browser > button,
.source-browser header button,
.source-browser__more,
.source-browser__error button {
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-3);
  padding: 8px 16px;
}
.source-browser__error {
  padding: 12px;
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-3);
}
@media (max-width: 600px) {
  .source-browser {
    padding: 16px 0;
  }
  .source-browser__grid {
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 148px), 1fr));
    align-items: start;
    gap: 10px;
  }
}
</style>
