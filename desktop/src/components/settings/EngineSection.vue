<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import { ipc, type DiscoveredHost, type HostTest } from "../../lib/ipc";
import { addressLabel, prepareHosts, versionLabel } from "../../lib/discovery";
import { useConnectionStore } from "../../stores/connection";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const conn = useConnectionStore();
const config = useSettingsConfigStore();
const toasts = useToastStore();

const remoteUrl = ref("");
const remoteKey = ref("");
const remoteKeyInput = ref<HTMLInputElement | null>(null);
const testResult = ref<HostTest | null>(null);
const testing = ref(false);
const switching = ref(false);
const switchError = ref<string | null>(null);
const restarting = computed(() => conn.status === "starting");

// LAN discovery ("On your network").
const discovered = ref<DiscoveredHost[]>([]);
const scanning = ref(false);
const scanError = ref<string | null>(null);
// Set when a scan was auto-triggered by a failed remote connection, so the
// list can explain why it appeared.
const afterFailure = ref(false);

async function scan() {
  scanning.value = true;
  scanError.value = null;
  try {
    discovered.value = prepareHosts(await ipc.discoverServers());
  } catch (err) {
    scanError.value = String(err);
    discovered.value = [];
  } finally {
    scanning.value = false;
  }
}

onMounted(async () => {
  const settings = await ipc.appSettingsGet();
  remoteUrl.value = settings.remoteUrl ?? "";
  // Keychain first; legacy settings.json field for files written before S4.
  remoteKey.value = (await ipc.secretGet("remote-api-key")) ?? settings.remoteApiKey ?? "";
  void scan();
});

// When a remote connection drops (server moved, laptop roamed), re-scan so the
// user can re-select the same server at its new address without hunting.
watch(
  () => conn.status,
  (status) => {
    if (status === "error" && conn.mode === "remote") {
      afterFailure.value = true;
      void scan();
    }
  },
);

/** Pick a discovered host: fill the form, prompt for a key if needed, test. */
async function useDiscovered(host: DiscoveredHost) {
  remoteUrl.value = host.url;
  if (host.authRequired && !remoteKey.value) {
    await nextTick();
    remoteKeyInput.value?.focus();
    return;
  }
  await testConnection();
}

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
    void config.load();
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
  else void config.load();
  switching.value = false;
}

async function restartEngine() {
  await conn.stopEngine();
  await conn.useLocal();
  if (conn.ready) toasts.push("Engine restarted");
  else if (conn.error) toasts.push(conn.error, "error");
}
</script>

<template>
  <div class="max-w-2xl">
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
        <button
          v-if="conn.mode === 'local'"
          type="button"
          class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink disabled:opacity-50"
          :disabled="restarting"
          @click="restartEngine"
        >
          {{ restarting ? "Restarting…" : "Restart" }}
        </button>
      </div>

      <div class="border-edge mt-4 border-t pt-4">
        <label class="block text-caption text-ink-2" for="remote-host">Remote host</label>
        <div class="mt-1 flex gap-2">
          <input
            id="remote-host"
            v-model="remoteUrl"
            data-selectable
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

        <label class="mt-3 block text-caption text-ink-2" for="remote-key">
          API key
          <span class="edge-code ml-1">KEYCHAIN</span>
        </label>
        <input
          id="remote-key"
          ref="remoteKeyInput"
          v-model="remoteKey"
          data-selectable
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

        <div class="border-edge mt-4 border-t pt-4">
          <div class="flex items-center gap-2">
            <span class="text-caption text-ink-2">On your network</span>
            <button
              type="button"
              class="border-edge ml-auto h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink disabled:opacity-50"
              :disabled="scanning"
              @click="scan"
            >
              {{ scanning ? "Scanning…" : "Scan again" }}
            </button>
          </div>

          <p v-if="afterFailure && conn.status === 'error'" class="mt-2 text-caption text-ink-3">
            Your server may have moved — found these on the network.
          </p>

          <p v-if="scanError" class="mt-2 text-caption text-stop">{{ scanError }}</p>

          <ul v-if="discovered.length" class="mt-2 space-y-1.5">
            <li
              v-for="host in discovered"
              :key="host.url"
              class="border-edge flex items-center gap-3 rounded-control border bg-bath px-3 py-2"
            >
              <div class="min-w-0 flex-1">
                <div class="flex items-center gap-2">
                  <span class="truncate text-body text-ink">{{ host.name }}</span>
                  <span v-if="host.isThisMachine" class="edge-code">THIS MAC</span>
                  <span v-if="host.authRequired" class="edge-code">KEY</span>
                </div>
                <div class="data-mono mt-0.5 truncate text-caption text-ink-3">
                  {{ addressLabel(host) }} · {{ versionLabel(host) }}
                </div>
              </div>
              <button
                type="button"
                class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
                @click="useDiscovered(host)"
              >
                Use
              </button>
            </li>
          </ul>

          <p v-else-if="!scanning && !scanError" class="mt-2 text-caption text-ink-3">
            No mold servers found on your network.
          </p>
        </div>
      </div>
    </div>

    <div v-if="config.available" class="mt-5">
      <ConfigSettingRow schema-key="models_dir" />
      <ConfigSettingRow schema-key="output_dir" />
    </div>
  </div>
</template>
