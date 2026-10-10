<script setup lang="ts">
import { computed, ref, watch } from "vue";
import SourceMediaCard from "./SourceMediaCard.vue";
import { sourceMediaInventories } from "../api/sourceMediaBrowser";
import type { RetainedSourceMediaInventory } from "../api/gallerySourceMedia";
import {
  sourceOwnerIdentity,
  sourceMemberLabel,
  type SourceMediaOwner,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const props = withDefaults(
  defineProps<{ owners: SourceMediaOwner[]; showTitle?: boolean }>(),
  { showTitle: true },
);
const emit = defineEmits<{
  use: [selection: SourceMediaSelection];
  openOwner: [owner: SourceMediaOwner];
}>();
const inventories = ref<Record<string, RetainedSourceMediaInventory>>({});
const failures = ref<Record<string, string>>({});
const scanned = ref(0);
const loading = ref(false);
const search = ref("");
let controller: AbortController | undefined;
const rows = computed(() =>
  props.owners.flatMap((owner) => {
    const inventory = inventories.value[sourceOwnerIdentity(owner)];
    if (inventory?.availability !== "available") return [];
    const members = (inventory.members ?? []).filter((member) =>
      `${owner.title} ${owner.hostLabel} ${sourceMemberLabel(member)} ${member.display_name}`
        .toLocaleLowerCase()
        .includes(search.value.toLocaleLowerCase()),
    );
    return members.length ? [{ owner, members }] : [];
  }),
);
async function more() {
  if (loading.value) return;
  const request = controller;
  if (!request || request.signal.aborted) return;
  const page = props.owners.slice(scanned.value, scanned.value + 100);
  loading.value = true;
  const groups = new Map<string, SourceMediaOwner[]>();
  for (const owner of page) {
    const key = JSON.stringify([
      owner.target.baseUrl,
      owner.target.apiKey,
      owner.instanceId,
    ]);
    groups.set(key, [...(groups.get(key) ?? []), owner]);
  }
  for (const owners of groups.values()) {
    if (request.signal.aborted) return;
    const first = owners[0]!;
    try {
      const result = await sourceMediaInventories(
        [...new Set(owners.map((o) => o.filename))],
        first.target,
        request.signal,
        first.instanceId,
      );
      if (request.signal.aborted) return;
      for (const owner of owners) {
        const key = sourceOwnerIdentity(owner),
          value = result.inventories[owner.filename];
        if (value) {
          inventories.value[key] = value;
          if (!["available", "unavailable_legacy"].includes(value.availability))
            failures.value[key] =
              `${owner.title}: ${value.availability.replaceAll("_", " ")}`;
        } else
          failures.value[key] =
            `${owner.title}: ${result.errors?.[owner.filename] ?? "Source inventory unavailable"}`;
      }
    } catch (error) {
      if (request.signal.aborted) return;
      for (const owner of owners)
        failures.value[sourceOwnerIdentity(owner)] =
          `${owner.hostLabel}: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
  if (!request.signal.aborted) {
    scanned.value += page.length;
    loading.value = false;
  }
}
function reset() {
  controller?.abort();
  controller = new AbortController();
  inventories.value = {};
  failures.value = {};
  scanned.value = 0;
  loading.value = false;
  void more();
}
watch(
  () => props.owners.map(sourceOwnerIdentity).join("\n"),
  (_, __, cleanup) => {
    reset();
    cleanup(() => controller?.abort());
  },
  { immediate: true },
);
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
    <label class="source-browser__search"
      >Search loaded sources
      <input
        v-model="search"
        type="search"
        placeholder="Name, role, print or machine"
    /></label>
    <p class="source-browser__status" role="status">
      {{ scanned }} of {{ owners.length }} library items checked<span
        v-if="loading"
      >
        · Loading…</span
      >
    </p>
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
      <button type="button" @click="reset">Retry</button>
    </div>
    <div
      v-for="row in rows"
      :key="sourceOwnerIdentity(row.owner)"
      class="source-browser__owner"
    >
      <button
        class="source-browser__owner-link"
        type="button"
        @click="emit('openOwner', row.owner)"
      >
        {{ row.owner.title }}</button
      ><span class="source-browser__machine">{{ row.owner.hostLabel }}</span>
      <div class="source-browser__grid">
        <SourceMediaCard
          v-for="member in row.members"
          :key="member.member_id"
          :selection="{ owner: row.owner, member }"
          @use="emit('use', $event)"
        />
      </div>
    </div>
    <p v-if="!loading && !rows.length">
      {{
        search
          ? "No loaded sources match your search."
          : "No retained sources found in the checked library items."
      }}
    </p>
    <button
      v-if="scanned < owners.length"
      class="source-browser__more"
      type="button"
      :disabled="loading"
      @click="more"
    >
      {{ loading ? "Loading…" : "Check more library items" }}
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
.source-browser__search {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-block: 20px;
  font-size: var(--mold-fs-sm);
}
.source-browser input {
  max-width: 480px;
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
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 148px), 1fr));
  gap: 12px;
  margin-top: 10px;
}
.source-browser button {
  min-height: 44px;
  color: inherit;
  cursor: pointer;
}
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
    gap: 10px;
  }
}
</style>
