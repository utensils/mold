<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRouter } from "vue-router";
import EmptyState from "../components/shell/EmptyState.vue";
import { clearHistory, fetchHistory, groupByDay, type HistoryEntry } from "../lib/api/history";
import { useConnectionStore } from "../stores/connection";
import { useComposerStore } from "../stores/composer";
import { useModelStore } from "../stores/models";
import { useToastStore } from "../stores/toasts";
import { applyModelDefaults, newGenerateForm } from "../lib/generateForm";

const router = useRouter();
const conn = useConnectionStore();
const composer = useComposerStore();
const models = useModelStore();
const toasts = useToastStore();

const entries = ref<HistoryEntry[]>([]);
const query = ref("");
const loaded = ref(false);
const unavailable = ref(false);
const confirmingClear = ref(false);

const groups = computed(() => groupByDay(entries.value));

async function load() {
  try {
    entries.value = await fetchHistory(query.value);
    loaded.value = true;
    unavailable.value = false;
  } catch (err) {
    // 404 = talking to a server that predates the history API; 503 = DB off.
    unavailable.value = true;
    loaded.value = true;
    entries.value = [];
    void err;
  }
}

let debounce: ReturnType<typeof setTimeout> | null = null;
watch(query, () => {
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(load, 250);
});

watch(
  () => conn.ready,
  (ready) => {
    if (ready) void load();
  },
  { immediate: true },
);

function use(entry: HistoryEntry) {
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

async function clearAll() {
  if (!confirmingClear.value) {
    confirmingClear.value = true;
    return;
  }
  confirmingClear.value = false;
  await clearHistory();
  toasts.push("Cleared history");
  await load();
}

const timeOf = (e: HistoryEntry) =>
  new Date(e.used_at * 1000).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
</script>

<template>
  <EmptyState
    v-if="loaded && unavailable"
    headline="History isn't available"
    detail="This engine doesn't expose prompt history — it may predate the history API or run without its database."
  />
  <EmptyState
    v-else-if="loaded && entries.length === 0 && !query"
    headline="No prompts yet"
    detail="Every prompt you develop is kept here to reuse."
  />

  <div v-else class="flex h-full flex-col">
    <header class="border-edge flex h-11 items-center gap-3 border-b px-4">
      <span class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        History
      </span>
      <input
        v-model="query"
        data-selectable
        type="search"
        placeholder="Search prompts…"
        class="border-edge ml-auto h-7 w-64 rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
      />
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-body transition-colors duration-100"
        :class="
          confirmingClear
            ? 'border-stop bg-stop font-semibold text-[#141110]'
            : 'text-ink-2 hover:text-stop'
        "
        @blur="confirmingClear = false"
        @click="clearAll"
      >
        {{ confirmingClear ? `Clear ${entries.length} prompts?` : "Clear…" }}
      </button>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-4 py-3">
      <template v-for="group in groups" :key="group.label">
        <div class="mt-3 mb-1 flex items-center gap-2 first:mt-0">
          <span class="edge-code">{{ group.label.toUpperCase() }}</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <button
          v-for="(entry, i) in group.entries"
          :key="`${group.label}-${i}`"
          type="button"
          class="group flex w-full items-center gap-3 rounded-control px-2 py-1.5 text-left hover:bg-bench"
          @click="use(entry)"
        >
          <span class="min-w-0 flex-1 truncate text-body text-ink" :title="entry.prompt">
            {{ entry.prompt }}
          </span>
          <span class="data-mono shrink-0 text-caption text-ink-3">{{ entry.model }}</span>
          <span class="data-mono shrink-0 text-caption text-ink-3">{{ timeOf(entry) }}</span>
          <span
            class="shrink-0 text-caption text-safelight opacity-0 transition-opacity duration-100 group-hover:opacity-100"
          >
            ↩ Use
          </span>
        </button>
        <p v-if="query && entries.length === 0" class="mt-6 text-center text-body text-ink-2">
          No prompts match “{{ query }}”.
        </p>
      </template>
    </div>
  </div>
</template>
