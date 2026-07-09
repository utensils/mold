<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import ConfigRowItem from "../components/settings/ConfigRowItem.vue";
import { useConnectionStore } from "../stores/connection";
import { useToastStore } from "../stores/toasts";
import { ipc, type HostTest } from "../lib/ipc";
import { ApiError } from "../lib/api/client";
import { fetchConfig, fetchProfiles, resetConfig, setConfig, setProfile } from "../lib/api/config";
import { groupConfigRows } from "../lib/config";
import type { ConfigRow } from "../lib/api/types";

const conn = useConnectionStore();
const toasts = useToastStore();

type Tab = "general" | "engine" | "generation" | "advanced";
const tab = ref<Tab>("general");
const TABS: { id: Tab; label: string }[] = [
  { id: "general", label: "General" },
  { id: "engine", label: "Engine" },
  { id: "generation", label: "Generation" },
  { id: "advanced", label: "Advanced" },
];

// ── Engine (IPC-backed, not the config API) ────────────────────────────────
const remoteUrl = ref("");
const remoteKey = ref("");
const testResult = ref<HostTest | null>(null);
const testing = ref(false);
const switching = ref(false);
const switchError = ref<string | null>(null);

async function testConnection() {
  testing.value = true;
  testResult.value = null;
  try {
    testResult.value = await ipc.testRemoteHost(remoteUrl.value, remoteKey.value || null);
  } finally {
    testing.value = false;
  }
}
async function useRemote() {
  switching.value = true;
  switchError.value = null;
  try {
    await conn.useRemote(remoteUrl.value, remoteKey.value || null);
  } catch (err) {
    switchError.value = String(err);
  } finally {
    switching.value = false;
  }
}
async function useLocal() {
  switching.value = true;
  switchError.value = null;
  await conn.useLocal();
  if (conn.status === "error") switchError.value = conn.error;
  switching.value = false;
}

// ── Config surface (may 404 on older engines) ──────────────────────────────
const configAvailable = ref<boolean | null>(null);
const rows = ref<ConfigRow[]>([]);
const profiles = ref<string[]>([]);
const activeProfile = ref("default");

const grouped = computed(() => groupConfigRows(rows.value));

async function loadConfig() {
  try {
    rows.value = await fetchConfig();
    configAvailable.value = true;
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      configAvailable.value = false;
      return;
    }
    configAvailable.value = false;
  }
}

async function loadProfiles() {
  try {
    const p = await fetchProfiles();
    profiles.value = p.profiles;
    activeProfile.value = p.active;
  } catch {
    /* profiles are optional; General still renders */
  }
}

async function saveRow(row: ConfigRow, value: ConfigRow["value"]) {
  try {
    await setConfig(row.key, value);
    toasts.push(`Saved ${row.key}`);
    await loadConfig();
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

async function resetRow(row: ConfigRow) {
  try {
    await resetConfig(row.key);
    toasts.push(`Reset ${row.key}`);
    await loadConfig();
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

async function switchProfile(name: string) {
  try {
    await setProfile(name);
    activeProfile.value = name;
    toasts.push(`Switched to ${name}`);
    await loadConfig();
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

onMounted(async () => {
  const settings = await ipc.appSettingsGet();
  remoteUrl.value = settings.remoteUrl ?? "";
  remoteKey.value = settings.remoteApiKey ?? "";
  void loadConfig();
  void loadProfiles();
});
</script>

<template>
  <div class="flex h-full flex-col">
    <header class="border-edge flex items-center gap-3 border-b bg-bench px-4 py-2">
      <span class="font-display text-display-sm font-bold text-ink" style="font-stretch: 90%">
        Settings
      </span>
      <div class="ml-4 flex items-center gap-1">
        <button
          v-for="t in TABS"
          :key="t.id"
          type="button"
          class="h-7 rounded-control px-2.5 text-body"
          :class="tab === t.id ? 'bg-bath text-ink' : 'text-ink-2 hover:text-ink'"
          @click="tab = t.id"
        >
          {{ t.label }}
        </button>
      </div>
    </header>

    <div class="min-h-0 flex-1 overflow-y-auto px-6 py-5">
      <!-- General: profiles -->
      <section v-if="tab === 'general'" class="max-w-2xl">
        <div class="mb-2 flex items-center gap-2">
          <span class="edge-code">Profiles</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <div v-if="profiles.length" class="flex items-center gap-2">
          <select
            :value="activeProfile"
            class="border-edge h-8 rounded-control border bg-bath px-2 text-body text-ink"
            @change="switchProfile(($event.target as HTMLSelectElement).value)"
          >
            <option v-for="p in profiles" :key="p" :value="p">{{ p }}</option>
          </select>
          <span class="text-caption text-ink-3">Settings marked ⌂ are per-profile.</span>
        </div>
        <p v-else class="text-caption text-ink-3">No profiles reported by this engine.</p>
      </section>

      <!-- Engine: IPC panel + engine config -->
      <section v-else-if="tab === 'engine'" class="max-w-2xl">
        <div class="mb-2 flex items-center gap-2">
          <span class="edge-code">Engine</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <div class="border-edge rounded-chrome border bg-bench p-4">
          <div class="flex items-center gap-6">
            <label class="flex items-center gap-2 text-body">
              <input
                type="radio"
                :checked="conn.mode === 'local' || conn.mode === 'external'"
                class="accent-[var(--safelight)]"
                @change="useLocal"
              />
              Built-in (this Mac)
            </label>
            <label class="flex items-center gap-2 text-body">
              <input
                type="radio"
                :checked="conn.mode === 'remote'"
                class="accent-[var(--safelight)]"
                @change="useRemote"
              />
              Remote server
            </label>
            <span class="data-mono ml-auto text-caption text-ink-3">{{ conn.baseUrl ?? "—" }}</span>
          </div>

          <div class="border-edge mt-4 border-t pt-4">
            <label class="block text-caption text-ink-2" for="remote-host">Remote host</label>
            <div class="mt-1 flex gap-2">
              <input
                id="remote-host"
                v-model="remoteUrl"
                type="text"
                placeholder="http://studio.local:7680"
                class="border-edge data-mono h-8 flex-1 rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3"
              />
              <button
                type="button"
                class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink disabled:opacity-50"
                :disabled="testing || !remoteUrl"
                @click="testConnection"
              >
                {{ testing ? "Testing…" : "Test connection" }}
              </button>
            </div>

            <label class="mt-3 block text-caption text-ink-2" for="remote-key">API key</label>
            <input
              id="remote-key"
              v-model="remoteKey"
              type="password"
              autocomplete="off"
              placeholder="Optional"
              class="border-edge data-mono mt-1 h-8 w-full rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3"
            />

            <p
              v-if="testResult"
              class="mt-2 text-caption"
              :class="testResult.ok ? 'text-halide' : 'text-stop'"
            >
              <template v-if="testResult.ok">
                Connected{{ testResult.version ? ` — mold ${testResult.version}` : "" }}.
              </template>
              <template v-else>{{ testResult.error }}</template>
            </p>

            <div class="mt-3 flex items-center gap-3">
              <button
                type="button"
                class="h-8 rounded-control bg-safelight px-3 text-body font-semibold text-[#141110] hover:brightness-105 active:translate-y-px disabled:opacity-50"
                :disabled="switching || !remoteUrl"
                @click="useRemote"
              >
                Use this host
              </button>
              <span v-if="switchError" class="text-caption text-stop">{{ switchError }}</span>
            </div>
          </div>
        </div>

        <div v-if="configAvailable && grouped.engine.length" class="mt-5">
          <ConfigRowItem
            v-for="row in grouped.engine"
            :key="row.key"
            :row="row"
            @save="(v) => saveRow(row, v)"
            @reset="resetRow(row)"
          />
        </div>
      </section>

      <!-- Generation / Advanced: config rows -->
      <section v-else class="max-w-2xl">
        <div class="mb-2 flex items-center gap-2">
          <span class="edge-code">{{ tab }}</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <p v-if="configAvailable === false" class="text-caption text-ink-3">
          This engine doesn't expose configuration.
        </p>
        <p v-else-if="configAvailable === null" class="text-caption text-ink-3">Loading…</p>
        <template v-else>
          <p
            v-if="grouped[tab === 'generation' ? 'generation' : 'advanced'].length === 0"
            class="text-caption text-ink-3"
          >
            Nothing to configure here.
          </p>
          <ConfigRowItem
            v-for="row in grouped[tab === 'generation' ? 'generation' : 'advanced']"
            :key="row.key"
            :row="row"
            @save="(v) => saveRow(row, v)"
            @reset="resetRow(row)"
          />
        </template>
      </section>
    </div>
  </div>
</template>
