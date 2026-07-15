<script setup lang="ts">
import { computed, onMounted } from "vue";
import { formatBytes } from "../../lib/format";
import type { UpdateChannel } from "../../lib/ipc";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useUpdaterStore } from "../../stores/updater";
import SelectControl from "./SelectControl.vue";
import SettingRow from "./SettingRow.vue";

const prefs = useAppPrefsStore();
const updater = useUpdaterStore();

const CHANNELS = [
  { value: "stable", label: "Stable" },
  { value: "nightly", label: "Nightly" },
];

const candidate = computed(() => updater.candidate);
const roundedPercent = computed(() => Math.round(updater.percent ?? 0));
const channelName = computed(() => (prefs.updateChannel === "nightly" ? "Nightly" : "Stable"));

function publishedLabel(raw: string | null): string | null {
  if (!raw) return null;
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
}

onMounted(() => void updater.init());
</script>

<template>
  <div class="max-w-2xl">
    <SettingRow
      label="Update channel"
      help="Stable follows versioned releases. Nightly follows signed builds from main."
    >
      <SelectControl
        :model-value="prefs.updateChannel"
        :options="CHANNELS"
        :disabled="updater.isBusy"
        aria-label="Update channel"
        @commit="(value) => updater.setChannel(value as UpdateChannel)"
      />
    </SettingRow>

    <p v-if="prefs.updateChannel === 'nightly'" class="mt-3 text-caption text-halide" role="status">
      Nightly builds may contain regressions. Every build is still signature-verified before Mold
      installs it.
    </p>

    <section
      class="border-edge mt-4 rounded-chrome border bg-bench p-4"
      aria-labelledby="desktop-update-status"
      :aria-busy="updater.isBusy"
    >
      <div class="flex items-start gap-4">
        <div class="min-w-0 flex-1">
          <h2 id="desktop-update-status" class="text-body-lg font-semibold text-ink">
            Desktop updates
          </h2>
          <p class="mt-0.5 text-caption text-ink-3">
            Current version
            <span class="data-mono text-ink-2">{{ updater.currentVersion ?? "dev" }}</span>
            · {{ channelName }}
          </p>
        </div>
        <button
          v-if="['idle', 'up-to-date'].includes(updater.phase)"
          type="button"
          class="border-edge h-8 shrink-0 rounded-control border px-3 text-body text-ink-2 hover:text-ink disabled:opacity-50"
          :disabled="updater.isBusy"
          @click="updater.check()"
        >
          Check for updates
        </button>
      </div>

      <p v-if="updater.phase === 'idle'" class="mt-4 text-caption text-ink-3">
        Mold checks automatically when the desktop app opens. Updates are never installed without
        your approval.
      </p>

      <p
        v-else-if="updater.phase === 'checking'"
        class="mt-4 text-caption text-ink-2"
        role="status"
        aria-live="polite"
      >
        Checking the {{ channelName }} channel…
      </p>

      <p
        v-else-if="updater.phase === 'up-to-date'"
        class="mt-4 text-caption text-ink-2"
        role="status"
        aria-live="polite"
      >
        Mold {{ updater.currentVersion }} is up to date on {{ channelName }}.
      </p>

      <div v-else-if="updater.phase === 'available' && candidate" class="mt-4">
        <p class="text-body font-medium text-ink">Mold {{ candidate.version }} is available.</p>
        <p v-if="publishedLabel(candidate.publishedAt)" class="mt-0.5 text-caption text-ink-3">
          Published {{ publishedLabel(candidate.publishedAt) }}
        </p>
        <p
          v-if="candidate.notes"
          data-test="update-notes"
          data-selectable
          class="mt-3 max-h-36 overflow-y-auto whitespace-pre-wrap text-caption text-ink-2"
        >
          {{ candidate.notes }}
        </p>
        <button
          type="button"
          data-test="install-update"
          class="mt-4 h-8 rounded-control bg-safelight px-3 text-body font-semibold text-[#141110] hover:brightness-105 active:translate-y-px"
          @click="updater.install()"
        >
          Update and restart
        </button>
      </div>

      <div v-else-if="updater.phase === 'downloading' && candidate" class="mt-4">
        <div class="flex items-center gap-3">
          <div
            class="h-1.5 flex-1 overflow-hidden rounded-full bg-bath"
            role="progressbar"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-valuenow="updater.percent === null ? undefined : roundedPercent"
            :aria-label="`Downloading Mold ${candidate.version}`"
          >
            <div
              class="h-full bg-safelight transition-[width] duration-300"
              :class="updater.percent === null ? 'grain-shimmer w-full' : ''"
              :style="updater.percent === null ? undefined : { width: `${roundedPercent}%` }"
            />
          </div>
          <span
            v-if="updater.totalBytes !== null"
            class="data-mono shrink-0 text-caption text-ink-3"
          >
            {{ formatBytes(updater.downloadedBytes) }} / {{ formatBytes(updater.totalBytes) }}
          </span>
        </div>
        <p class="mt-2 text-caption text-ink-2">Downloading Mold {{ candidate.version }}…</p>
      </div>

      <p
        v-else-if="updater.phase === 'verifying'"
        class="mt-4 text-caption text-ink-2"
        role="status"
      >
        Verifying the update signature…
      </p>
      <p v-else-if="updater.phase === 'staging'" class="mt-4 text-caption text-ink-2" role="status">
        Running complete signature, identity, Gatekeeper, and install-location checks…
      </p>
      <p
        v-else-if="updater.phase === 'installing'"
        class="mt-4 text-caption text-ink-2"
        role="status"
      >
        Installing the update and restarting Mold…
      </p>
      <div
        v-else-if="updater.phase === 'failed' && updater.error"
        class="border-stop/40 mt-4 rounded-control border bg-stop/10 p-3"
        role="alert"
      >
        <p class="text-body font-semibold text-stop">Mold couldn’t update</p>
        <p data-selectable class="mt-1 text-caption text-ink-2">{{ updater.error.message }}</p>
        <p class="mt-2 text-caption text-ink-3">
          <template v-if="updater.currentVersion">
            Mold {{ updater.currentVersion }} remains installed because the update did not complete.
          </template>
          <template v-else>The installed app was not changed.</template>
        </p>
        <div class="mt-3 flex items-center gap-3">
          <button
            v-if="updater.error.retryable"
            type="button"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="candidate ? updater.install() : updater.check()"
          >
            {{ candidate ? "Try update again" : "Check again" }}
          </button>
          <button
            type="button"
            class="h-7 px-1 text-caption text-ink-3 hover:text-ink"
            @click="updater.clearError()"
          >
            Dismiss
          </button>
        </div>
      </div>

      <p
        v-else-if="updater.phase === 'unsupported'"
        class="mt-4 text-caption text-ink-3"
        role="status"
      >
        Automatic updates are available in signed desktop builds. Browser previews and local dev
        builds stay unchanged.
      </p>
    </section>
  </div>
</template>
