<script setup lang="ts">
/*
 * Connect-a-machine flow (spec §06) — a 3-step task modal on the shared
 * ModalPanel. Step 1 picks the machine type; step 2 collects connection
 * details (or, for Local network, offers the live mDNS list); step 3 confirms.
 * RunPod hands straight off to the provisioning view. Every connect goes
 * through the hosts store's connect() — the same path Machines and boot
 * reconnect use — so instance-UUID dedupe and per-host key storage are shared.
 */
import { computed, nextTick, ref, watch } from "vue";
import { useRouter } from "vue-router";
import ModalPanel from "@ui/components/ModalPanel.vue";
import Icon from "@ui/components/Icon.vue";
import type { IconName } from "@ui/icons";
import { ipc, type DiscoveredHost } from "../../lib/ipc";
import { addressLabel, prepareHosts, versionLabel } from "../../lib/discovery";
import { useHostsStore } from "../../stores/hosts";

const props = defineProps<{ open: boolean; initialHost?: DiscoveredHost | null }>();
const emit = defineEmits<{ close: []; connected: [] }>();

const router = useRouter();
const hosts = useHostsStore();

type MachineType = "remote" | "runpod" | "lan";

const step = ref(1);
const type = ref<MachineType>("remote");
const address = ref("");
const displayName = ref("");
const apiKey = ref("");
const error = ref<string | null>(null);
const connecting = ref(false);
const connectedName = ref("");

const discovered = ref<DiscoveredHost[]>([]);
const selectedDiscovered = ref<DiscoveredHost | null>(null);
const scanning = ref(false);

const addressInput = ref<HTMLInputElement | null>(null);
const apiKeyInput = ref<HTMLInputElement | null>(null);

async function focusApiKey() {
  await nextTick();
  window.requestAnimationFrame(() => apiKeyInput.value?.focus());
}

interface TypeOption {
  value: MachineType;
  icon: IconName;
  label: string;
  hint: string;
}
const TYPES: TypeOption[] = [
  {
    value: "remote",
    icon: "machines",
    label: "Remote server",
    hint: "A machine running mold serve",
  },
  { value: "runpod", icon: "cloud", label: "RunPod cloud", hint: "Rent a GPU by the minute" },
  { value: "lan", icon: "wifi", label: "Local network", hint: "Auto-discover mold hosts nearby" },
];

const typeLabel = computed(() => TYPES.find((t) => t.value === type.value)?.label ?? "");
const canGoBack = computed(() => step.value === 2);

function reset() {
  step.value = 1;
  type.value = "remote";
  address.value = "";
  displayName.value = "";
  apiKey.value = "";
  error.value = null;
  connecting.value = false;
  connectedName.value = "";
  discovered.value = [];
  selectedDiscovered.value = null;
}

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    reset();
    if (props.initialHost) {
      type.value = "lan";
      step.value = 2;
      discovered.value = [props.initialHost];
      selectedDiscovered.value = props.initialHost;
      await focusApiKey();
    }
  },
);

// A stale error is misleading once the address it judged has changed.
watch(address, () => {
  error.value = null;
});

function pickType(next: MachineType) {
  type.value = next;
  error.value = null;
}

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

/** All connects go through the store so dedupe + key storage are shared. */
async function connect(url: string, key: string | null, name: string | null) {
  if (!url.trim()) {
    error.value = "Enter a host address.";
    return;
  }
  connecting.value = true;
  error.value = null;
  try {
    const host = await hosts.connect(url, key, name);
    connectedName.value = host.label;
    emit("connected");
    step.value = 3;
  } catch (err) {
    // Blunt inline error; the entered values stay put so a retry is one edit.
    error.value = err instanceof Error ? err.message : String(err);
  } finally {
    connecting.value = false;
  }
}

function isThisMachine(host: DiscoveredHost): boolean {
  return (
    host.isThisMachine || (!!host.instanceId && host.instanceId === hosts.primaryHost?.instanceId)
  );
}

async function pickDiscovered(host: DiscoveredHost) {
  if (host.authRequired && !apiKey.value.trim()) {
    selectedDiscovered.value = host;
    error.value = null;
    await focusApiKey();
    return;
  }
  await connect(host.url, apiKey.value || null, host.name);
}

function clearDiscoveredSelection() {
  selectedDiscovered.value = null;
  apiKey.value = "";
  error.value = null;
}

async function onContinue() {
  error.value = null;
  if (step.value === 1) {
    if (type.value === "runpod") {
      emit("close");
      void router.push("/machines/runpod");
      return;
    }
    step.value = 2;
    if (type.value === "lan") {
      void scan();
    } else {
      await nextTick();
      addressInput.value?.focus();
    }
    return;
  }
  if (step.value === 2) {
    if (type.value === "remote") {
      await connect(address.value, apiKey.value || null, displayName.value || null);
    } else if (selectedDiscovered.value) {
      await pickDiscovered(selectedDiscovered.value);
    }
    return;
  }
  emit("close");
}

function onBack() {
  error.value = null;
  selectedDiscovered.value = null;
  apiKey.value = "";
  step.value = 1;
}

const continueLabel = computed(() =>
  step.value === 1 ? "Continue" : step.value === 2 ? "Connect" : "Done",
);
const continueDisabled = computed(
  () =>
    connecting.value ||
    (step.value === 2 && type.value === "remote" && !address.value.trim()) ||
    (step.value === 2 &&
      type.value === "lan" &&
      selectedDiscovered.value?.authRequired === true &&
      !apiKey.value.trim()),
);
</script>

<template>
  <ModalPanel
    :open="open"
    :width="480"
    :steps="3"
    :step="step"
    label="Connect a machine"
    @close="emit('close')"
  >
    <!-- Step 1 — machine type -->
    <template v-if="step === 1">
      <div class="font-sans font-semibold text-md font-bold text-fg" style="font-stretch: 92%">
        Add a machine
      </div>
      <p class="mb-4 mt-1 text-micro text-fg-dim">
        Generate on another GPU. Everything still runs on hardware you control.
      </p>
      <div class="flex flex-col gap-2">
        <button
          v-for="option in TYPES"
          :key="option.value"
          type="button"
          :data-test="`connect-type-${option.value}`"
          class="flex items-center gap-3 rounded-window border p-3.5 text-left transition-colors"
          :class="
            type === option.value
              ? 'border-accent bg-accent-tint text-fg'
              : 'border-border-control text-fg hover:border-fg-dim'
          "
          :aria-pressed="type === option.value"
          @click="pickType(option.value)"
        >
          <span
            class="flex h-9 w-9 shrink-0 items-center justify-center rounded-control text-sapphire"
            style="background: color-mix(in srgb, var(--mold-sapphire) 16%, transparent)"
          >
            <Icon :name="option.icon" :size="18" :stroke-width="1.8" />
          </span>
          <span class="min-w-0">
            <span class="block text-sm font-semibold text-fg">{{ option.label }}</span>
            <span class="block text-micro text-fg-dim">{{ option.hint }}</span>
          </span>
        </button>
      </div>
    </template>

    <!-- Step 2 — details -->
    <template v-else-if="step === 2">
      <div class="font-sans font-semibold text-md font-bold text-fg" style="font-stretch: 92%">
        Connection details
      </div>
      <p class="mb-4 mt-1 text-micro text-fg-dim">{{ typeLabel }}</p>

      <template v-if="type === 'remote'">
        <label class="text-micro font-semibold text-fg-2" for="connect-address">Address</label>
        <div
          class="mt-1.5 flex items-center gap-2 rounded-window border border-border-control bg-bg-deep px-3 py-2.5"
        >
          <Icon name="lock" :size="14" :stroke-width="2" class="shrink-0 text-fg-dim" />
          <input
            id="connect-address"
            ref="addressInput"
            v-model="address"
            data-selectable
            data-test="connect-address"
            type="text"
            placeholder="192.168.1.42:7680"
            class="font-mono text-xs min-w-0 flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-fg-dim"
            @keydown.enter="onContinue"
          />
        </div>

        <label class="mt-4 block text-micro font-semibold text-fg-2" for="connect-name">
          Display name
        </label>
        <input
          id="connect-name"
          v-model="displayName"
          data-selectable
          data-test="connect-name"
          type="text"
          placeholder="Optional"
          class="mt-1.5 w-full rounded-window border border-border-control bg-bg-deep px-3 py-2.5 text-sm text-fg outline-none placeholder:text-fg-dim"
          @keydown.enter="onContinue"
        />

        <label class="mt-4 block text-micro font-semibold text-fg-2" for="connect-key">
          API key
        </label>
        <input
          id="connect-key"
          v-model="apiKey"
          data-selectable
          data-test="connect-key"
          type="password"
          autocomplete="off"
          placeholder="Optional — stored only on this device"
          class="font-mono text-xs mt-1.5 w-full rounded-window border border-border-control bg-bg-deep px-3 py-2.5 text-sm text-fg outline-none placeholder:text-fg-dim"
          @keydown.enter="onContinue"
        />

        <p class="mt-4 flex items-center gap-2 text-micro text-fg-dim">
          <span class="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden="true" />
          Connection is direct — no data leaves your network.
        </p>
      </template>

      <template v-else>
        <div class="flex items-center gap-2">
          <span class="text-micro text-fg-2">Nearby hosts</span>
          <div class="flex-1" />
          <button
            type="button"
            class="border-border h-7 rounded-control border px-2.5 text-micro text-fg-2 hover:text-fg disabled:opacity-50"
            :disabled="scanning"
            @click="scan"
          >
            {{ scanning ? "Scanning…" : "Scan again" }}
          </button>
        </div>
        <ul v-if="discovered.length && !selectedDiscovered" class="mt-2 flex flex-col gap-1.5">
          <li
            v-for="host in discovered"
            :key="host.url"
            data-test="connect-discovered"
            class="border-border flex items-center gap-3 rounded-control border bg-bg-deep px-3 py-2"
          >
            <div class="min-w-0 flex-1">
              <div class="flex items-center gap-2">
                <span class="truncate text-sm text-fg">{{ host.name }}</span>
                <span
                  v-if="isThisMachine(host)"
                  class="font-mono text-micro text-fg-dim whitespace-nowrap"
                  >THIS DEVICE</span
                >
                <span
                  v-if="host.authRequired"
                  class="font-mono text-micro text-fg-dim whitespace-nowrap"
                  >KEY</span
                >
              </div>
              <div class="font-mono text-xs mt-0.5 truncate text-micro text-fg-dim">
                {{ addressLabel(host) }} · {{ versionLabel(host) }}
              </div>
            </div>
            <button
              v-if="!isThisMachine(host)"
              type="button"
              data-test="connect-discovered-add"
              class="border-border h-7 shrink-0 rounded-control border px-2.5 text-micro text-fg-2 hover:text-fg disabled:opacity-50"
              :disabled="connecting"
              @click="pickDiscovered(host)"
            >
              Connect
            </button>
          </li>
        </ul>
        <p v-else-if="!scanning && !selectedDiscovered" class="mt-2 text-micro text-fg-dim">
          No other mold servers found on your network.
        </p>
        <template v-if="selectedDiscovered">
          <div
            data-test="connect-discovered-selected"
            class="border-border mt-2 flex items-center gap-3 rounded-control border bg-bg-deep px-3 py-2"
          >
            <div class="min-w-0 flex-1">
              <div class="truncate text-sm text-fg">{{ selectedDiscovered.name }}</div>
              <div class="font-mono text-xs mt-0.5 truncate text-micro text-fg-dim">
                {{ addressLabel(selectedDiscovered) }} · {{ versionLabel(selectedDiscovered) }}
              </div>
            </div>
            <button
              v-if="!initialHost"
              type="button"
              class="text-micro text-fg-dim hover:text-fg"
              @click="clearDiscoveredSelection"
            >
              Choose another
            </button>
          </div>
          <label class="mt-4 block text-micro font-semibold text-fg-2" for="connect-lan-key">
            API key
          </label>
          <input
            id="connect-lan-key"
            ref="apiKeyInput"
            v-model="apiKey"
            data-selectable
            data-test="connect-key"
            type="password"
            autocomplete="off"
            placeholder="Required by this machine"
            class="font-mono text-xs mt-1.5 w-full rounded-window border border-border-control bg-bg-deep px-3 py-2.5 text-sm text-fg outline-none placeholder:text-fg-dim"
            @keydown.enter="onContinue"
          />
          <p class="mt-2 text-micro text-fg-dim">
            This machine requires its own API key. The key is stored only on this device.
          </p>
        </template>
      </template>

      <p v-if="error" data-test="connect-error" class="mt-3 text-micro text-error">{{ error }}</p>
    </template>

    <!-- Step 3 — confirmation -->
    <template v-else>
      <div data-test="connect-confirm" class="py-2 text-center">
        <div
          class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-control text-accent"
          style="background: color-mix(in srgb, var(--mold-blue) 18%, transparent)"
        >
          <Icon name="check" :size="30" :stroke-width="2.4" />
        </div>
        <div class="font-sans font-semibold text-md font-bold text-fg" style="font-stretch: 92%">
          Machine connected
        </div>
        <p class="mt-1.5 text-micro leading-relaxed text-fg-dim">
          {{ connectedName }} is online and ready. Pick it as a generation target from Machines.
        </p>
      </div>
    </template>

    <template #footer>
      <button
        v-if="step === 1"
        type="button"
        data-test="connect-cancel"
        class="border-border rounded-window border px-4 py-2.5 text-sm font-semibold text-fg-2 hover:text-fg"
        @click="emit('close')"
      >
        Cancel
      </button>
      <button
        v-if="canGoBack"
        type="button"
        data-test="connect-back"
        class="border-border rounded-window border px-4 py-2.5 text-sm font-semibold text-fg-2 hover:text-fg"
        @click="onBack"
      >
        Back
      </button>
      <div class="flex-1" />
      <button
        v-if="step !== 2 || type === 'remote' || selectedDiscovered"
        type="button"
        data-test="connect-continue"
        class="rounded-window bg-accent px-6 py-2.5 text-sm font-bold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
        :disabled="continueDisabled"
        @click="onContinue"
      >
        {{ connecting ? "Connecting…" : continueLabel }}
      </button>
    </template>
  </ModalPanel>
</template>
