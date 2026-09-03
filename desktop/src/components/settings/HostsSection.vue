<script setup lang="ts">
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import MoldHomeCard from "./MoldHomeCard.vue";
import { ipc } from "../../lib/ipc";
import { useConnectionStore } from "../../stores/connection";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const router = useRouter();
const conn = useConnectionStore();
const config = useSettingsConfigStore();
const toasts = useToastStore();

// This-device card — the one host knob that belongs in Settings (it exposes
// THIS device to other apps). Adding, connecting, and forgetting other
// machines all moved to the Machines workspace.
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

function hostDot(status: "ready" | "connecting" | "error"): string {
  switch (status) {
    case "ready":
      return "bg-accent";
    case "connecting":
      return "bg-sapphire animate-pulse";
    default:
      return "bg-error";
  }
}
</script>

<template>
  <div class="max-w-2xl">
    <!-- Machines live in their own workspace now -->
    <div class="border-border flex items-center gap-3 rounded-window border bg-bg p-4">
      <div class="min-w-0 flex-1">
        <span class="text-sm font-medium text-fg">Machines</span>
        <p class="mt-1 text-micro text-fg-dim">
          Add, connect, forget, and inspect machines — including RunPod and hosts on your network —
          in the Machines workspace.
        </p>
      </div>
      <button
        type="button"
        data-test="open-machines"
        class="border-border-control h-8 shrink-0 rounded-control border px-3 text-sm font-semibold text-fg-2 hover:text-fg"
        @click="router.push('/machines')"
      >
        Open Machines
      </button>
    </div>

    <!-- This device -->
    <div class="border-border mt-4 rounded-window border bg-bg p-4">
      <div class="flex items-center gap-3">
        <span
          class="h-1.5 w-1.5 shrink-0 rounded-full"
          :class="
            hostDot(conn.ready ? 'ready' : conn.status === 'starting' ? 'connecting' : 'error')
          "
        />
        <span class="text-sm font-medium text-fg">This device</span>
        <span class="font-mono text-xs ml-auto text-micro text-fg-dim">{{
          conn.baseUrl ?? "—"
        }}</span>
        <button
          v-if="conn.mode === 'local'"
          type="button"
          class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg disabled:opacity-50"
          :disabled="restarting"
          @click="restartEngine"
        >
          {{ restarting ? "Restarting…" : "Restart" }}
        </button>
      </div>

      <div class="border-border mt-4 border-t pt-4">
        <span class="text-micro text-fg-2">This device API key</span>
        <p class="mt-1 text-micro text-fg-dim">
          Use this key when another Mold app connects to this device over the network.
        </p>
        <div class="mt-2 flex items-center gap-2">
          <code
            class="border-border font-mono text-xs rounded-control border bg-bg-deep px-2 py-1 text-micro text-fg-2"
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
            class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg"
            @click="localKeyVisible = !localKeyVisible"
          >
            {{ localKeyVisible ? "Hide" : "Reveal" }}
          </button>
          <button
            v-if="localApiKey"
            type="button"
            data-test="copy-local-api-key"
            class="border-border h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg"
            @click="copyLocalApiKey"
          >
            Copy
          </button>
        </div>
        <div v-if="localFailure" class="mt-2">
          <p class="text-micro text-error">{{ localFailure }}</p>
          <button
            type="button"
            data-test="open-local-logs"
            class="border-border mt-2 h-7 rounded-control border px-2.5 text-sm text-fg-2 hover:text-fg"
            @click="ipc.openLogsDir()"
          >
            Open logs folder
          </button>
        </div>
      </div>
    </div>

    <MoldHomeCard />

    <!-- Storage (engine config) -->
    <div v-if="config.available" class="mt-5">
      <ConfigSettingRow schema-key="models_dir" />
      <ConfigSettingRow schema-key="output_dir" />
    </div>
  </div>
</template>
