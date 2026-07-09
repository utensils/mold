<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useConnectionStore } from "../stores/connection";
import { ipc, type HostTest } from "../lib/ipc";

const conn = useConnectionStore();

const remoteUrl = ref("");
const remoteKey = ref("");
const testResult = ref<HostTest | null>(null);
const testing = ref(false);
const switching = ref(false);
const switchError = ref<string | null>(null);

onMounted(async () => {
  const settings = await ipc.appSettingsGet();
  remoteUrl.value = settings.remoteUrl ?? "";
  remoteKey.value = settings.remoteApiKey ?? "";
});

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
</script>

<template>
  <div class="h-full overflow-y-auto px-8 py-6">
    <h1 class="font-display text-display font-bold text-ink" style="font-stretch: 90%">Settings</h1>

    <section class="mt-6 max-w-xl">
      <div class="mb-2 flex items-center gap-2">
        <span class="edge-code">ENGINE</span>
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
              class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 transition-colors duration-100 hover:text-ink disabled:opacity-50"
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
              class="h-8 rounded-control bg-safelight px-3 text-body font-semibold text-[#141110] transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-50"
              :disabled="switching || !remoteUrl"
              @click="useRemote"
            >
              Use this host
            </button>
            <span v-if="switchError" class="text-caption text-stop">{{ switchError }}</span>
          </div>
        </div>
      </div>
      <p class="mt-2 text-caption text-ink-3">
        Generation defaults, profiles, and the full configuration surface arrive in a later
        milestone.
      </p>
    </section>
  </div>
</template>
