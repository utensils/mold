<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import { ipc, type DiscoveredHost, type HostTest, type SavedHost } from "../../lib/ipc";
import { addressLabel, prepareHosts, versionLabel } from "../../lib/discovery";
import { timeAgo } from "../../lib/format";
import { hostIdFromUrl } from "../../lib/hosts";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore, type HostView } from "../../stores/hosts";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const conn = useConnectionStore();
const hosts = useHostsStore();
const config = useSettingsConfigStore();
const toasts = useToastStore();

// This-device card.
const restarting = computed(() => conn.status === "starting");
const localKeyVisible = ref(false);
const localApiKey = computed(() => conn.localInfo?.apiKey ?? "");

async function copyLocalApiKey() {
  if (!localApiKey.value) return;
  await navigator.clipboard.writeText(localApiKey.value);
  toasts.push("This device API key copied");
}

async function restartEngine() {
  await conn.stopEngine();
  await conn.useLocal();
  if (conn.ready) toasts.push("Engine restarted");
  else if (conn.error) toasts.push(conn.error, "error");
}

// Add-host form.
const addUrl = ref("");
const addKey = ref("");
const addUrlInput = ref<HTMLInputElement | null>(null);
const addKeyInput = ref<HTMLInputElement | null>(null);
const testResult = ref<HostTest | null>(null);
const testing = ref(false);
const adding = ref(false);
const addError = ref<string | null>(null);

async function testConnection() {
  testing.value = true;
  testResult.value = null;
  try {
    testResult.value = await ipc.testRemoteHost(addUrl.value, addKey.value || null);
  } catch (err) {
    // Malformed input rejects before the probe (normalize_host_url) — surface
    // it in the same slot a failed probe uses instead of failing silently.
    testResult.value = {
      ok: false,
      version: null,
      error: String(err),
      instanceId: null,
      hostname: null,
    };
  } finally {
    testing.value = false;
  }
}

/** mDNS instance name for a URL in the discovered list, to label the host. */
function discoveredName(url: string): string | null {
  return discovered.value.find((h) => h.url === url)?.name ?? null;
}

async function addHost() {
  if (!addUrl.value.trim()) {
    await nextTick();
    addUrlInput.value?.focus();
    return;
  }
  adding.value = true;
  addError.value = null;
  try {
    const host = await hosts.connect(
      addUrl.value,
      addKey.value || null,
      discoveredName(addUrl.value),
    );
    toasts.push(`Connected to ${host.label}`);
    addUrl.value = "";
    addKey.value = "";
    testResult.value = null;
    savedHosts.value = (await ipc.appSettingsGet()).savedHosts ?? [];
  } catch (err) {
    addError.value = String(err);
  } finally {
    adding.value = false;
  }
}

// Connected remotes (the local primary lives in the This-device card above).
const connectedRemotes = computed<HostView[]>(() => hosts.all.filter((h) => !h.primary));

/** Connected-remote ids and their instance ids, to hide already-connected
 *  boxes from the discovery list. The local primary is deliberately excluded —
 *  this machine still shows up on the network (badged), just never as an
 *  "Add"-able remote. */
const connectedRemoteIds = computed(() => new Set(connectedRemotes.value.map((h) => h.id)));
const connectedRemoteInstanceIds = computed(
  () => new Set(connectedRemotes.value.map((h) => h.instanceId).filter((id): id is string => !!id)),
);

const forgetPendingId = ref<string | null>(null);

/** Drop the host and its saved entry + stored key (two-step confirm). */
async function forgetHost(id: string) {
  if (forgetPendingId.value !== id) {
    forgetPendingId.value = id;
    return;
  }
  forgetPendingId.value = null;
  try {
    await hosts.disconnect(id);
    savedHosts.value = await ipc.forgetRemoteHost(id);
    toasts.push("Host forgotten");
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

// Remembered hosts (saved but not currently connected).
const savedHosts = ref<SavedHost[]>([]);
const rememberedHosts = computed(() =>
  savedHosts.value.filter((h) => !connectedRemoteIds.value.has(h.id)),
);

function savedHostLabel(host: SavedHost): string {
  return host.name ?? host.url.replace(/^https?:\/\//, "");
}

/** Reconnect a remembered host with its own stored key. */
async function connectSaved(host: SavedHost) {
  adding.value = true;
  addError.value = null;
  try {
    const key = await ipc.secretGet(`remote-api-key.${host.id}`);
    await hosts.connect(host.url, key, host.name);
    savedHosts.value = (await ipc.appSettingsGet()).savedHosts ?? [];
  } catch (err) {
    addError.value = String(err);
  } finally {
    adding.value = false;
  }
}

// LAN discovery ("On your network").
const discovered = ref<DiscoveredHost[]>([]);
const scanning = ref(false);
const scanError = ref<string | null>(null);

/** Discovered servers to show: hide boxes already connected as remotes (by
 *  slug or instance id); this machine stays visible with a THIS DEVICE badge. */
const undiscovered = computed(() =>
  discovered.value.filter(
    (d) =>
      !connectedRemoteIds.value.has(hostIdFromUrl(d.url)) &&
      !(d.instanceId && connectedRemoteInstanceIds.value.has(d.instanceId)),
  ),
);

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

/** A stored key for a discovered host: its URL slug's secret, or — when the
 *  advertisement carries an instance id matching a remembered host — that
 *  saved slug's secret (a box remembered by hostname is often advertised by
 *  IP under a different slug). */
async function storedKeyFor(host: DiscoveredHost): Promise<string | null> {
  const bySlug = await ipc.secretGet(`remote-api-key.${hostIdFromUrl(host.url)}`);
  if (bySlug) return bySlug;
  if (!host.instanceId) return null;
  const saved = savedHosts.value.find((s) => s.instanceId === host.instanceId);
  return saved ? await ipc.secretGet(`remote-api-key.${saved.id}`) : null;
}

/** One-click add of a discovered host; a key the app already holds is reused,
 *  and the key field is only focused when no key exists anywhere. */
async function addDiscovered(host: DiscoveredHost) {
  const storedKey = addKey.value ? null : await storedKeyFor(host);
  if (host.authRequired && !addKey.value && !storedKey) {
    addUrl.value = host.url;
    await nextTick();
    addKeyInput.value?.focus();
    return;
  }
  adding.value = true;
  addError.value = null;
  try {
    const view = await hosts.connect(host.url, addKey.value || storedKey, host.name);
    toasts.push(`Connected to ${view.label}`);
    addKey.value = "";
    savedHosts.value = (await ipc.appSettingsGet()).savedHosts ?? [];
  } catch (err) {
    addError.value = String(err);
  } finally {
    adding.value = false;
  }
}

function isThisMachine(host: DiscoveredHost): boolean {
  return (
    host.isThisMachine || (!!host.instanceId && host.instanceId === hosts.primaryHost?.instanceId)
  );
}

function hostDot(status: HostView["status"]): string {
  switch (status) {
    case "ready":
      return "bg-safelight";
    case "connecting":
      return "bg-halide animate-pulse";
    default:
      return "bg-stop";
  }
}

onMounted(async () => {
  savedHosts.value = (await ipc.appSettingsGet()).savedHosts ?? [];
  void scan();
});
</script>

<template>
  <div class="max-w-2xl">
    <!-- This device -->
    <div class="border-edge rounded-chrome border bg-bench p-4">
      <div class="flex items-center gap-3">
        <span
          class="h-1.5 w-1.5 shrink-0 rounded-full"
          :class="
            hostDot(conn.ready ? 'ready' : conn.status === 'starting' ? 'connecting' : 'error')
          "
        />
        <span class="text-body font-medium text-ink">This device</span>
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
        <span class="text-caption text-ink-2">This device API key</span>
        <p class="mt-1 text-caption text-ink-3">
          Use this key when another Mold app connects to this device over the network.
        </p>
        <div class="mt-2 flex items-center gap-2">
          <code
            class="border-edge data-mono rounded-control border bg-bath px-2 py-1 text-caption text-ink-2"
          >
            {{
              localApiKey
                ? localKeyVisible
                  ? localApiKey
                  : "••••••••••••••••"
                : "Local server unavailable"
            }}
          </code>
          <button
            v-if="localApiKey"
            type="button"
            data-test="reveal-local-api-key"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="localKeyVisible = !localKeyVisible"
          >
            {{ localKeyVisible ? "Hide" : "Reveal" }}
          </button>
          <button
            v-if="localApiKey"
            type="button"
            data-test="copy-local-api-key"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="copyLocalApiKey"
          >
            Copy
          </button>
        </div>
        <p v-if="conn.localError" class="mt-2 text-caption text-stop">{{ conn.localError }}</p>
      </div>
    </div>

    <!-- Add a host -->
    <div class="border-edge mt-4 rounded-chrome border bg-bench p-4">
      <span class="text-caption text-ink-2">Add a host</span>
      <div class="mt-2 flex gap-2">
        <input
          id="add-host-url"
          ref="addUrlInput"
          v-model="addUrl"
          data-selectable
          data-test="add-host-url"
          type="text"
          placeholder="hal9000"
          class="border-edge data-mono h-8 flex-1 rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3"
          @keydown.enter="addHost"
        />
        <button
          type="button"
          class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink disabled:opacity-50"
          :disabled="testing || !addUrl"
          @click="testConnection"
        >
          {{ testing ? "Testing…" : "Test connection" }}
        </button>
      </div>
      <input
        id="add-host-key"
        ref="addKeyInput"
        v-model="addKey"
        data-selectable
        data-test="add-host-key"
        type="password"
        autocomplete="off"
        placeholder="API key — optional, stored only on this device"
        class="border-edge data-mono mt-2 h-8 w-full rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3"
        @keydown.enter="addHost"
      />
      <p
        v-if="testResult"
        class="mt-2 text-caption"
        :class="testResult.ok ? 'text-halide' : 'text-stop'"
      >
        <template v-if="testResult.ok">
          Reached{{ testResult.hostname ? ` ${testResult.hostname}` : ""
          }}{{ testResult.version ? ` — mold ${testResult.version}` : "" }}.
        </template>
        <template v-else>{{ testResult.error }}</template>
      </p>
      <div class="mt-3 flex items-center gap-3">
        <button
          type="button"
          data-test="add-host"
          class="h-8 rounded-control bg-safelight px-3 text-body font-semibold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
          :disabled="adding || !addUrl"
          @click="addHost"
        >
          Add host
        </button>
        <span v-if="addError" class="text-caption text-stop">{{ addError }}</span>
      </div>
    </div>

    <!-- Connected hosts -->
    <div v-if="connectedRemotes.length" class="border-edge mt-4 rounded-chrome border bg-bench p-4">
      <span class="text-caption text-ink-2">Connected</span>
      <ul class="mt-2 space-y-1.5">
        <li
          v-for="host in connectedRemotes"
          :key="host.id"
          data-test="connected-host"
          class="border-edge flex items-center gap-3 rounded-control border bg-bath px-3 py-2"
        >
          <span class="h-1.5 w-1.5 shrink-0 rounded-full" :class="hostDot(host.status)" />
          <div class="min-w-0 flex-1">
            <div class="flex items-center gap-2">
              <span class="truncate text-body text-ink">{{ host.label }}</span>
              <span v-if="host.version" class="edge-code">mold {{ host.version }}</span>
            </div>
            <div class="data-mono mt-0.5 truncate text-caption text-ink-3">{{ host.baseUrl }}</div>
          </div>
          <button
            type="button"
            data-test="host-disconnect"
            class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="hosts.disconnect(host.id)"
          >
            Disconnect
          </button>
          <button
            type="button"
            data-test="saved-host-forget"
            class="h-7 shrink-0 rounded-control px-2.5 text-body"
            :class="forgetPendingId === host.id ? 'text-stop' : 'text-ink-3 hover:text-ink-2'"
            @click="forgetHost(host.id)"
            @blur="forgetPendingId = null"
          >
            {{ forgetPendingId === host.id ? "Forget?" : "Forget" }}
          </button>
        </li>
      </ul>
    </div>

    <!-- Remembered hosts -->
    <div v-if="rememberedHosts.length" class="border-edge mt-4 rounded-chrome border bg-bench p-4">
      <span class="text-caption text-ink-2">Remembered</span>
      <ul class="mt-2 space-y-1.5">
        <li
          v-for="saved in rememberedHosts"
          :key="saved.id"
          class="border-edge flex items-center gap-3 rounded-control border bg-bath px-3 py-2"
        >
          <div class="min-w-0 flex-1">
            <span class="truncate text-body text-ink">{{ savedHostLabel(saved) }}</span>
            <div class="data-mono mt-0.5 truncate text-caption text-ink-3">
              {{ saved.url
              }}<template v-if="saved.lastUsedMs"> · {{ timeAgo(saved.lastUsedMs) }}</template>
            </div>
          </div>
          <button
            type="button"
            data-test="saved-host-connect"
            class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink disabled:opacity-50"
            :disabled="adding"
            @click="connectSaved(saved)"
          >
            Connect
          </button>
          <button
            type="button"
            data-test="saved-host-forget"
            class="h-7 shrink-0 rounded-control px-2.5 text-body"
            :class="forgetPendingId === saved.id ? 'text-stop' : 'text-ink-3 hover:text-ink-2'"
            @click="forgetHost(saved.id)"
            @blur="forgetPendingId = null"
          >
            {{ forgetPendingId === saved.id ? "Forget?" : "Forget" }}
          </button>
        </li>
      </ul>
    </div>

    <!-- On your network -->
    <div class="border-edge mt-4 rounded-chrome border bg-bench p-4">
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

      <p v-if="scanError" class="mt-2 text-caption text-stop">{{ scanError }}</p>

      <ul v-if="undiscovered.length" class="mt-2 space-y-1.5">
        <li
          v-for="host in undiscovered"
          :key="host.url"
          data-test="discovered-host"
          class="border-edge flex items-center gap-3 rounded-control border bg-bath px-3 py-2"
        >
          <div class="min-w-0 flex-1">
            <div class="flex items-center gap-2">
              <span class="truncate text-body text-ink">{{ host.name }}</span>
              <span v-if="isThisMachine(host)" class="edge-code">THIS DEVICE</span>
              <span v-if="host.authRequired" class="edge-code">KEY</span>
            </div>
            <div class="data-mono mt-0.5 truncate text-caption text-ink-3">
              {{ addressLabel(host) }} · {{ versionLabel(host) }}
            </div>
          </div>
          <button
            v-if="!isThisMachine(host)"
            type="button"
            data-test="discovered-host-add"
            class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="addDiscovered(host)"
          >
            Add
          </button>
        </li>
      </ul>

      <p v-else-if="!scanning && !scanError" class="mt-2 text-caption text-ink-3">
        No other mold servers found on your network.
      </p>
    </div>

    <!-- Storage (engine config) -->
    <div v-if="config.available" class="mt-5">
      <ConfigSettingRow schema-key="models_dir" />
      <ConfigSettingRow schema-key="output_dir" />
    </div>
  </div>
</template>
