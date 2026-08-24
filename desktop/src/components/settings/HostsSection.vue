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
      return "bg-safelight";
    case "connecting":
      return "bg-halide animate-pulse";
    default:
      return "bg-stop";
  }
}
</script>

<template>
  <div class="max-w-2xl">
    <!-- Machines live in their own workspace now -->
    <div class="border-edge flex items-center gap-3 rounded-chrome border bg-bench p-4">
      <div class="min-w-0 flex-1">
        <span class="text-body font-medium text-ink">Machines</span>
        <p class="mt-1 text-caption text-ink-3">
          Add, connect, forget, and inspect machines — including RunPod and hosts on your network —
          in the Machines workspace.
        </p>
      </div>
      <button
        type="button"
        data-test="open-machines"
        class="border-ce h-8 shrink-0 rounded-control border px-3 text-body font-semibold text-ink-2 hover:text-ink"
        @click="router.push('/machines')"
      >
        Open Machines
      </button>
    </div>

    <!-- This device -->
    <div class="border-edge mt-4 rounded-chrome border bg-bench p-4">
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
        <div v-if="localFailure" class="mt-2">
          <p class="text-caption text-stop">{{ localFailure }}</p>
          <button
            type="button"
            data-test="open-local-logs"
            class="border-edge mt-2 h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
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
