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

const props = defineProps<{ open: boolean }>();
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
const scanning = ref(false);

const addressInput = ref<HTMLInputElement | null>(null);

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
}

watch(
  () => props.open,
  (open) => {
    if (open) reset();
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
  await connect(host.url, apiKey.value || null, host.name);
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
    }
    // Local network advances via pickDiscovered, not the primary button.
    return;
  }
  emit("close");
}

function onBack() {
  error.value = null;
  step.value = 1;
}

const continueLabel = computed(() =>
  step.value === 1 ? "Continue" : step.value === 2 ? "Connect" : "Done",
);
const continueDisabled = computed(
  () => connecting.value || (step.value === 2 && type.value === "remote" && !address.value.trim()),
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
      <div class="font-display text-display-sm font-bold text-ink" style="font-stretch: 92%">
        Add a machine
      </div>
      <p class="mb-4 mt-1 text-caption text-ink-3">
        Generate on another GPU. Everything still runs on hardware you control.
      </p>
      <div class="flex flex-col gap-2">
        <button
          v-for="option in TYPES"
          :key="option.value"
          type="button"
          :data-test="`connect-type-${option.value}`"
          class="flex items-center gap-3 rounded-chrome border p-3.5 text-left transition-colors"
          :class="
            type === option.value
              ? 'border-[var(--sel-border)] bg-[var(--sel-bg)] text-ink'
              : 'border-ce text-ink hover:border-ink-3'
          "
          :aria-pressed="type === option.value"
          @click="pickType(option.value)"
        >
          <span
            class="flex h-9 w-9 shrink-0 items-center justify-center rounded-control text-halide"
            style="background: color-mix(in srgb, var(--halide) 16%, transparent)"
          >
            <Icon :name="option.icon" :size="18" :stroke-width="1.8" />
          </span>
          <span class="min-w-0">
            <span class="block text-body font-semibold text-ink">{{ option.label }}</span>
            <span class="block text-caption text-ink-3">{{ option.hint }}</span>
          </span>
        </button>
      </div>
    </template>

    <!-- Step 2 — details -->
    <template v-else-if="step === 2">
      <div class="font-display text-display-sm font-bold text-ink" style="font-stretch: 92%">
        Connection details
      </div>
      <p class="mb-4 mt-1 text-caption text-ink-3">{{ typeLabel }}</p>

      <template v-if="type === 'remote'">
        <label class="text-caption font-semibold text-ink-2" for="connect-address">Address</label>
        <div
          class="mt-1.5 flex items-center gap-2 rounded-chrome border border-ce bg-bath px-3 py-2.5"
        >
          <Icon name="lock" :size="14" :stroke-width="2" class="shrink-0 text-ink-3" />
          <input
            id="connect-address"
            ref="addressInput"
            v-model="address"
            data-selectable
            data-test="connect-address"
            type="text"
            placeholder="192.168.1.42:7680"
            class="data-mono min-w-0 flex-1 bg-transparent text-body text-ink outline-none placeholder:text-ink-3"
            @keydown.enter="onContinue"
          />
        </div>

        <label class="mt-4 block text-caption font-semibold text-ink-2" for="connect-name">
          Display name
        </label>
        <input
          id="connect-name"
          v-model="displayName"
          data-selectable
          data-test="connect-name"
          type="text"
          placeholder="Optional"
          class="mt-1.5 w-full rounded-chrome border border-ce bg-bath px-3 py-2.5 text-body text-ink outline-none placeholder:text-ink-3"
          @keydown.enter="onContinue"
        />

        <label class="mt-4 block text-caption font-semibold text-ink-2" for="connect-key">
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
          class="data-mono mt-1.5 w-full rounded-chrome border border-ce bg-bath px-3 py-2.5 text-body text-ink outline-none placeholder:text-ink-3"
          @keydown.enter="onContinue"
        />

        <p class="mt-4 flex items-center gap-2 text-caption text-ink-3">
          <span class="h-1.5 w-1.5 rounded-full bg-safelight" aria-hidden="true" />
          Connection is direct — no data leaves your network.
        </p>
      </template>

      <template v-else>
        <div class="flex items-center gap-2">
          <span class="text-caption text-ink-2">Nearby hosts</span>
          <div class="flex-1" />
          <button
            type="button"
            class="border-edge h-7 rounded-control border px-2.5 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
            :disabled="scanning"
            @click="scan"
          >
            {{ scanning ? "Scanning…" : "Scan again" }}
          </button>
        </div>
        <ul v-if="discovered.length" class="mt-2 flex flex-col gap-1.5">
          <li
            v-for="host in discovered"
            :key="host.url"
            data-test="connect-discovered"
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
              data-test="connect-discovered-add"
              class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
              :disabled="connecting"
              @click="pickDiscovered(host)"
            >
              Connect
            </button>
          </li>
        </ul>
        <p v-else-if="!scanning" class="mt-2 text-caption text-ink-3">
          No other mold servers found on your network.
        </p>
        <label class="mt-4 block text-caption font-semibold text-ink-2" for="connect-lan-key">
          API key
        </label>
        <input
          id="connect-lan-key"
          v-model="apiKey"
          data-selectable
          data-test="connect-key"
          type="password"
          autocomplete="off"
          placeholder="Optional — used when a picked host needs one"
          class="data-mono mt-1.5 w-full rounded-chrome border border-ce bg-bath px-3 py-2.5 text-body text-ink outline-none placeholder:text-ink-3"
        />
      </template>

      <p v-if="error" data-test="connect-error" class="mt-3 text-caption text-stop">{{ error }}</p>
    </template>

    <!-- Step 3 — confirmation -->
    <template v-else>
      <div data-test="connect-confirm" class="py-2 text-center">
        <div
          class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full text-safelight"
          style="background: color-mix(in srgb, var(--safelight) 18%, transparent)"
        >
          <Icon name="check" :size="30" :stroke-width="2.4" />
        </div>
        <div class="font-display text-display-sm font-bold text-ink" style="font-stretch: 92%">
          Machine connected
        </div>
        <p class="mt-1.5 text-caption leading-relaxed text-ink-3">
          {{ connectedName }} is online and ready. Pick it as a generation target from Machines.
        </p>
      </div>
    </template>

    <template #footer>
      <button
        v-if="step === 1"
        type="button"
        data-test="connect-cancel"
        class="border-edge rounded-chrome border px-4 py-2.5 text-body font-semibold text-ink-2 hover:text-ink"
        @click="emit('close')"
      >
        Cancel
      </button>
      <button
        v-if="canGoBack"
        type="button"
        data-test="connect-back"
        class="border-edge rounded-chrome border px-4 py-2.5 text-body font-semibold text-ink-2 hover:text-ink"
        @click="onBack"
      >
        Back
      </button>
      <div class="flex-1" />
      <button
        type="button"
        data-test="connect-continue"
        class="rounded-chrome bg-safelight px-6 py-2.5 text-body font-bold text-on-accent hover:brightness-105 active:translate-y-px disabled:opacity-50"
        :disabled="continueDisabled"
        @click="onContinue"
      >
        {{ connecting ? "Connecting…" : continueLabel }}
      </button>
    </template>
  </ModalPanel>
</template>
