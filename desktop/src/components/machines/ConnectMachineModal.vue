<script setup lang="ts">
/*
 * Connect a machine (README §04 dialog): one screen — the machines found on
 * the network, or an address typed by hand, an API key when the machine asks
 * for one, and "Make images here from now on". Every connect goes through
 * the hosts store's connect() — the same path Machines and boot reconnect
 * use — so instance-UUID dedupe and per-host key storage are shared.
 */
import { computed, nextTick, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import ToggleControl from "../settings/ToggleControl.vue";
import { ipc, type DiscoveredHost } from "../../lib/ipc";
import { addressLabel, prepareHosts, versionLabel } from "../../lib/discovery";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useHostsStore } from "../../stores/hosts";

const props = defineProps<{ open: boolean; initialHost?: DiscoveredHost | null }>();
const emit = defineEmits<{ close: []; connected: [] }>();

const appPrefs = useAppPrefsStore();
const hosts = useHostsStore();

const address = ref("");
const apiKey = ref("");
const makeTarget = ref(false);
const error = ref<string | null>(null);
const connecting = ref(false);

const discovered = ref<DiscoveredHost[]>([]);
const selected = ref<DiscoveredHost | null>(null);
const scanning = ref(false);

const apiKeyInput = ref<HTMLInputElement | null>(null);

function reset() {
  address.value = "";
  apiKey.value = "";
  makeTarget.value = false;
  error.value = null;
  connecting.value = false;
  discovered.value = [];
  selected.value = null;
}

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    reset();
    if (props.initialHost) {
      // Opened for one machine that asked for a key: it is the list.
      discovered.value = [props.initialHost];
      selected.value = props.initialHost;
      await nextTick();
      apiKeyInput.value?.focus();
      return;
    }
    void scan();
  },
  { immediate: true },
);

// A stale error is misleading once the address it judged has changed, and a
// typed address means the discovered pick is no longer what connects.
watch(address, (value) => {
  error.value = null;
  if (value.trim()) selected.value = null;
});

async function scan() {
  scanning.value = true;
  error.value = null;
  try {
    discovered.value = prepareHosts(await ipc.discoverServers());
  } catch (err) {
    error.value = String(err);
    discovered.value = [];
  } finally {
    scanning.value = false;
  }
}

function isThisMachine(host: DiscoveredHost): boolean {
  return (
    host.isThisMachine || (!!host.instanceId && host.instanceId === hosts.primaryHost?.instanceId)
  );
}

async function pick(host: DiscoveredHost) {
  if (isThisMachine(host)) return;
  selected.value = host;
  address.value = "";
  error.value = null;
  if (host.authRequired) {
    await nextTick();
    apiKeyInput.value?.focus();
  }
}

function clearSelection() {
  selected.value = null;
  apiKey.value = "";
  error.value = null;
}

/** The selected machine asks for a key this dialog does not have yet. */
const keyRequired = computed(() => selected.value?.authRequired === true && !apiKey.value.trim());
const canConnect = computed(
  () =>
    !connecting.value && !keyRequired.value && (selected.value !== null || !!address.value.trim()),
);

/** All connects go through the store so dedupe + key storage are shared. */
async function connect() {
  const url = selected.value?.url ?? address.value.trim();
  if (!url) {
    error.value = "Enter a machine's address or pick one from the list.";
    return;
  }
  connecting.value = true;
  error.value = null;
  try {
    const host = await hosts.connect(url, apiKey.value || null, selected.value?.name ?? null);
    if (makeTarget.value) await appPrefs.update({ generateTargetHost: host.id });
    emit("connected");
    emit("close");
  } catch (err) {
    // Blunt inline error; the entered values stay put so a retry is one edit.
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    connecting.value = false;
  }
}
</script>

<template>
  <ModalPanel :open="open" :width="480" title="Connect a machine" @close="emit('close')">
    <template #description>
      Point mold at another computer running <code class="font-mono">mold serve</code> to borrow its
      GPU.
    </template>
    <div class="flex flex-col gap-3.5">
      <!-- Found on your network -->
      <div class="flex flex-col gap-1.5">
        <span class="flex items-center text-xs text-fg-dim">
          Found on your network
          <span class="flex-1" />
          <button
            v-if="!initialHost"
            type="button"
            class="text-micro text-fg-dim hover:text-fg disabled:text-fg-faint"
            :disabled="scanning"
            @click="scan"
          >
            {{ scanning ? "Scanning…" : "Scan again" }}
          </button>
        </span>
        <button
          v-for="host in discovered"
          :key="host.url"
          type="button"
          :data-test="
            selected?.url === host.url ? 'connect-discovered-selected' : 'connect-discovered'
          "
          class="flex items-center gap-2.5 rounded-control border bg-bg-deep p-2.5 text-left transition-colors duration-100"
          :class="
            selected?.url === host.url
              ? 'border-accent'
              : 'border-border hover:border-border-focus disabled:opacity-60'
          "
          :disabled="isThisMachine(host)"
          :aria-pressed="selected?.url === host.url"
          @click="pick(host)"
        >
          <span
            class="h-[7px] w-[7px] shrink-0 rounded-full"
            :class="isThisMachine(host) ? 'bg-fg-dim' : 'bg-success'"
          />
          <span class="flex min-w-0 flex-1 flex-col gap-0.5">
            <span class="truncate font-mono text-xs text-fg">
              {{ host.name }}
              <span v-if="isThisMachine(host)" class="text-fg-dim"> · this device</span>
            </span>
            <span class="truncate font-mono text-micro text-fg-dim">
              {{ addressLabel(host) }} · {{ versionLabel(host) }}
              <template v-if="host.authRequired"> · asks for a key</template>
            </span>
          </span>
          <span v-if="selected?.url === host.url" class="font-mono text-micro text-accent">
            selected
          </span>
        </button>
        <button
          v-if="selected && !initialHost"
          type="button"
          data-test="connect-discovered-clear"
          class="self-start text-micro text-fg-dim hover:text-fg"
          @click="clearSelection"
        >
          Choose another
        </button>
        <p v-if="!discovered.length && !scanning" class="text-micro text-fg-dim">
          No other mold servers found on your network.
        </p>
      </div>

      <!-- Or type an address -->
      <label v-if="!initialHost" class="flex flex-col gap-1.5">
        <span class="text-xs text-fg-dim">Or type an address</span>
        <input
          v-model="address"
          data-selectable
          data-test="connect-address"
          type="text"
          placeholder="http://192.168.1.31:7680"
          class="h-8 rounded-control border border-border bg-bg px-2.5 font-mono text-xs text-fg outline-none placeholder:text-fg-faint focus:border-border-focus"
          @keydown.enter="canConnect && connect()"
        />
      </label>

      <!-- API key -->
      <label class="flex flex-col gap-1.5">
        <span class="text-xs text-fg-dim">
          {{
            selected?.authRequired
              ? "API key — this machine asks for one"
              : "API key, only if the machine asks for one"
          }}
        </span>
        <input
          ref="apiKeyInput"
          v-model="apiKey"
          data-selectable
          data-test="connect-key"
          type="password"
          autocomplete="off"
          placeholder="Stored only on this device"
          class="h-8 rounded-control border border-border bg-bg px-2.5 font-mono text-xs text-fg outline-none placeholder:text-fg-faint focus:border-border-focus"
          @keydown.enter="canConnect && connect()"
        />
      </label>

      <div class="flex items-center justify-between gap-4">
        <span class="text-xs text-fg">Make images here from now on</span>
        <ToggleControl
          :model-value="makeTarget"
          aria-label="Make images here from now on"
          data-test="connect-make-target"
          @commit="(v) => (makeTarget = v)"
        />
      </div>

      <p v-if="error" data-test="connect-error" class="text-micro text-error">{{ error }}</p>
    </div>

    <template #footer>
      <button
        type="button"
        data-test="connect-cancel"
        class="ms-toolbar-button h-[30px]"
        @click="emit('close')"
      >
        Cancel
      </button>
      <button
        type="button"
        data-test="connect-continue"
        class="ms-toolbar-button ms-toolbar-button--on h-[30px] font-semibold disabled:opacity-50"
        :disabled="!canConnect"
        @click="connect"
      >
        {{ connecting ? "Connecting…" : "Connect" }}
      </button>
    </template>
  </ModalPanel>
</template>
