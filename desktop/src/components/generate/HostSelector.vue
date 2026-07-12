<script setup lang="ts">
import { computed } from "vue";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useHostsStore, type HostView } from "../../stores/hosts";

/**
 * Where the next batch runs. Only rendered with more than one live host;
 * the pick is sticky across launches (null = Auto, the least-busy host).
 */
const hosts = useHostsStore();
const prefs = useAppPrefsStore();

const value = computed(() => {
  const sel = prefs.settings?.generateTargetHost ?? null;
  // A persisted pick whose host is gone reads as Auto rather than a ghost.
  return sel && hosts.all.some((h) => h.id === sel) ? sel : "auto";
});

function onChange(event: Event) {
  const picked = (event.target as HTMLSelectElement).value;
  void prefs.update({ generateTargetHost: picked === "auto" ? null : picked });
}

function optionLabel(host: HostView): string {
  if (host.status !== "ready") return `${host.label} — offline`;
  return host.queueDepth !== null ? `${host.label} · ${host.queueDepth} queued` : host.label;
}
</script>

<template>
  <div v-if="hosts.multiHost" data-test="host-selector" class="mb-5">
    <div class="mb-2 flex items-center gap-2">
      <span class="edge-code">Host</span>
      <div class="border-edge h-px flex-1 border-t" />
    </div>
    <select
      class="border-edge h-9 w-full rounded-control border bg-bath px-2 text-body text-ink"
      :value="value"
      aria-label="Generation host"
      @change="onChange"
    >
      <option value="auto">Auto — least busy</option>
      <option v-for="h in hosts.all" :key="h.id" :value="h.id" :disabled="h.status !== 'ready'">
        {{ optionLabel(h) }}
      </option>
    </select>
  </div>
</template>
