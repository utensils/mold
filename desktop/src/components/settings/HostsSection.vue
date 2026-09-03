<script setup lang="ts">
/*
 * Settings ▸ Machines: this device and its key (the one machine knob that
 * belongs in Settings, because it exposes THIS device to other apps), the
 * Mold home, and where styles and pictures live. Adding, connecting, and
 * forgetting other machines all live in the Machines workspace.
 */
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import MoldHomeCard from "./MoldHomeCard.vue";
import SettingRow from "./SettingRow.vue";
import { ipc } from "../../lib/ipc";
import { useConnectionStore } from "../../stores/connection";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const router = useRouter();
const conn = useConnectionStore();
const config = useSettingsConfigStore();
const toasts = useToastStore();

const restarting = computed(() => conn.status === "starting");
/** Engine-start failures land in `error` (startLocalEngine) or `localError`
 *  (ensureLocalServer) depending on which step died — surface either. */
const localFailure = computed(() =>
  conn.localError ? conn.localError : conn.status === "error" ? conn.error : null,
);
const localKeyVisible = ref(false);
const localApiKey = computed(() => conn.localInfo?.apiKey ?? "");

async function copyLocalApiKey() {
  if (!localApiKey.value) return;
  await navigator.clipboard.writeText(localApiKey.value);
  toasts.push("This device API key copied");
}

async function restartEngine() {
  const result = await conn.restartEngine();
  if (result === "restarted") toasts.push("Engine restarted");
  else if (result === "failed" && conn.error) toasts.push(conn.error, "error");
}

const engineDot = computed(() =>
  conn.ready ? "bg-success" : conn.status === "starting" ? "bg-sapphire ms-pulse" : "bg-error",
);
</script>

<template>
  <div>
    <!-- Machines live in their own workspace — this is the doorway -->
    <SettingRow
      label="Your machines"
      help="Connect, rent, forget, and inspect machines — including RunPod and hosts on your network — in Machines."
    >
      <button
        type="button"
        data-test="open-machines"
        class="ms-toolbar-button"
        @click="router.push('/machines')"
      >
        Open Machines
      </button>
    </SettingRow>

    <!-- This device -->
    <SettingRow label="This device" :help="conn.baseUrl ?? undefined">
      <span class="flex items-center gap-2">
        <span class="h-1.5 w-1.5 rounded-full" :class="engineDot" aria-hidden="true" />
        <button
          v-if="conn.mode === 'local'"
          type="button"
          class="ms-toolbar-button"
          :disabled="restarting"
          @click="restartEngine"
        >
          {{ restarting ? "Restarting…" : "Restart engine" }}
        </button>
      </span>
    </SettingRow>

    <SettingRow
      label="This device's API key"
      help="Another mold app on your network uses this key to connect here."
    >
      <code
        class="rounded-control border border-border bg-bg px-2 py-1 font-mono text-micro text-fg-2"
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
        class="ms-toolbar-button"
        @click="localKeyVisible = !localKeyVisible"
      >
        {{ localKeyVisible ? "Hide" : "Reveal" }}
      </button>
      <button
        v-if="localApiKey"
        type="button"
        data-test="copy-local-api-key"
        class="ms-toolbar-button"
        @click="copyLocalApiKey"
      >
        Copy
      </button>
    </SettingRow>

    <div v-if="localFailure" class="border-b border-border px-3.5 py-3">
      <p class="text-xs text-error">{{ localFailure }}</p>
      <button
        type="button"
        data-test="open-local-logs"
        class="ms-toolbar-button mt-2"
        @click="ipc.openLogsDir()"
      >
        Open logs folder
      </button>
    </div>

    <MoldHomeCard />

    <!-- Where styles and pictures live (engine config) -->
    <template v-if="config.available">
      <ConfigSettingRow schema-key="models_dir" />
      <ConfigSettingRow schema-key="output_dir" />
    </template>
  </div>
</template>
