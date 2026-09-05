<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import SettingRow from "./SettingRow.vue";
import { apiJson } from "../../lib/api/client";
import { ipc, inTauri } from "../../lib/ipc";
import { openExternal } from "../../lib/openExternal";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";
import type { ServerStatus } from "../../lib/api/types";

const conn = useConnectionStore();
const toasts = useToastStore();
const engine = ref<ServerStatus | null>(null);
const appVersion = ref<string | null>(null);
const PRIVACY_POLICY_URL = "https://utensils.io/mold/privacy";

/** Ask the engine again whenever the connection comes up — NOT once on
 *  mount. Launching straight into Settings (`restoreLastRoute`) mounts this
 *  before the engine is ready, and a mount-only read left Engine reading
 *  "offline" for the rest of the session. */
watch(
  () => conn.ready,
  async (ready) => {
    if (!ready) return;
    try {
      engine.value = await apiJson<ServerStatus>("/api/status");
    } catch {
      /* engine offline */
    }
  },
  { immediate: true },
);

onMounted(async () => {
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

function openPrivacyPolicy(): void {
  void openExternal(PRIVACY_POLICY_URL);
}
</script>

<template>
  <div class="w-full" data-test="about-section-content">
    <SettingRow label="Mold" help="Desktop app version.">
      <span class="font-mono text-sm text-fg-2">{{ appVersion ?? "dev" }}</span>
    </SettingRow>
    <SettingRow label="Engine" :help="conn.baseUrl ?? undefined">
      <span class="font-mono text-sm text-fg-2">
        {{ engine ? `mold ${engine.version}` : "offline" }}
      </span>
    </SettingRow>
    <SettingRow label="Processing" help="Where generations run.">
      <span class="font-mono text-sm text-fg-dim">Local + your hosts</span>
    </SettingRow>
    <SettingRow label="Core contributors">
      <span class="text-right text-sm text-fg-2">James Brink · Jeffrey Dilley</span>
    </SettingRow>
    <SettingRow label="Privacy" help="How Mold handles app and server data.">
      <button
        type="button"
        data-test="desktop-privacy-policy"
        class="ms-toolbar-button"
        @click="openPrivacyPolicy"
      >
        Privacy policy
      </button>
    </SettingRow>
    <SettingRow label="Logs" help="Engine and app logs live in the active Mold home.">
      <button type="button" class="ms-toolbar-button" @click="ipc.openLogsDir()">
        Open logs folder
      </button>
    </SettingRow>
    <SettingRow label="Diagnostics" help="Versions, connection, and engine state as JSON.">
      <button type="button" class="ms-toolbar-button" @click="copyDiagnostics">Copy</button>
    </SettingRow>
  </div>
</template>
