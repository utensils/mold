<script setup lang="ts">
import { onMounted, ref } from "vue";
import SettingRow from "./SettingRow.vue";
import { apiJson } from "../../lib/api/client";
import { ipc, inTauri } from "../../lib/ipc";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";
import type { ServerStatus } from "../../lib/api/types";

const conn = useConnectionStore();
const toasts = useToastStore();
const engine = ref<ServerStatus | null>(null);
const appVersion = ref<string | null>(null);

onMounted(async () => {
  try {
    engine.value = await apiJson<ServerStatus>("/api/status");
  } catch {
    /* engine offline */
  }
  if (inTauri()) {
    const { getVersion } = await import("@tauri-apps/api/app");
    appVersion.value = await getVersion();
  }
});

async function copyDiagnostics() {
  const report = {
    app: appVersion.value,
    engine: engine.value?.version ?? null,
    git_sha: engine.value?.git_sha ?? null,
    mode: conn.mode,
    base_url: conn.baseUrl,
    models_loaded: engine.value?.models_loaded ?? [],
    queue: `${engine.value?.queue_depth ?? "—"}/${engine.value?.queue_capacity ?? "—"}`,
    platform: navigator.platform,
  };
  await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
  toasts.push("Diagnostics copied");
}
</script>

<template>
  <div class="max-w-md">
    <SettingRow label="Mold" help="Desktop app version.">
      <span class="data-mono text-body text-ink-2">{{ appVersion ?? "dev" }}</span>
    </SettingRow>
    <SettingRow label="Engine" :help="conn.baseUrl ?? undefined">
      <span class="data-mono text-body text-ink-2">
        {{ engine ? `mold ${engine.version}` : "offline" }}
      </span>
    </SettingRow>
    <SettingRow label="Processing" help="Where generations run.">
      <span class="data-mono text-body text-ink-3">Local + your hosts</span>
    </SettingRow>
    <SettingRow label="Core contributors" help="Equal project owners.">
      <span class="text-right text-body text-ink-2">James Brink · Jeffrey Dilley</span>
    </SettingRow>
    <SettingRow label="Logs" help="Engine and app logs live in ~/.mold/logs.">
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
        @click="ipc.openLogsDir()"
      >
        Open logs folder
      </button>
    </SettingRow>
    <SettingRow label="Diagnostics" help="Versions, connection, and engine state as JSON.">
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
        @click="copyDiagnostics"
      >
        Copy
      </button>
    </SettingRow>
  </div>
</template>
