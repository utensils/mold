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
  <div>
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

    <p
      v-if="prefs.updateChannel === 'nightly'"
      class="px-3.5 pt-3 text-micro text-sapphire"
      role="status"
    >
      Nightly builds may contain regressions. Every build is still signature-verified before Mold
      installs it.
    </p>

    <section
      class="border-b border-border px-3.5 py-3"
      aria-labelledby="desktop-update-status"
      :aria-busy="updater.isBusy"
    >
      <div class="flex items-start gap-4">
        <div class="min-w-0 flex-1">
          <h2 id="desktop-update-status" class="text-sm font-medium text-fg">Desktop updates</h2>
          <p class="mt-0.5 flex items-center gap-2 text-micro text-fg-dim">
            <span>
              Current version
              <span class="font-mono text-xs text-fg-2">{{ updater.currentVersion ?? "dev" }}</span>
              · {{ channelName }}
            </span>
            <span
              v-if="updater.phase === 'up-to-date'"
              data-test="update-up-to-date"
              class="inline-flex h-5 items-center rounded-inner bg-success px-[7px] font-mono text-micro font-bold text-bg-deep"
            >
              up to date
            </span>
          </p>
        </div>
        <button
          v-if="['idle', 'up-to-date'].includes(updater.phase)"
          type="button"
          class="ms-toolbar-button"
          :disabled="updater.isBusy"
          @click="updater.check()"
        >
          Check now
        </button>
      </div>

      <p v-if="updater.phase === 'idle'" class="mt-4 text-micro text-fg-dim">
        Mold checks automatically when the desktop app opens. Updates are never installed without
        your approval.
      </p>

      <p
        v-else-if="updater.phase === 'checking'"
        class="mt-4 text-micro text-fg-2"
        role="status"
        aria-live="polite"
      >
        Checking the {{ channelName }} channel…
      </p>

      <p
        v-else-if="updater.phase === 'up-to-date'"
        class="mt-4 text-micro text-fg-2"
        role="status"
        aria-live="polite"
      >
        Mold {{ updater.currentVersion }} is up to date on {{ channelName }}.
      </p>

      <div v-else-if="updater.phase === 'available' && candidate" class="mt-4">
        <p class="text-sm font-medium text-fg">Mold {{ candidate.version }} is available.</p>
        <p v-if="publishedLabel(candidate.publishedAt)" class="mt-0.5 text-micro text-fg-dim">
          Published {{ publishedLabel(candidate.publishedAt) }}
        </p>
        <p
          v-if="candidate.notes"
          data-test="update-notes"
          data-selectable
          class="mt-3 max-h-36 overflow-y-auto whitespace-pre-wrap text-micro text-fg-2"
        >
          {{ candidate.notes }}
        </p>
        <button
          type="button"
          data-test="install-update"
          class="mt-4 h-8 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent hover:brightness-105 active:translate-y-px"
          @click="updater.install()"
        >
          Update and restart
        </button>
      </div>

      <div v-else-if="updater.phase === 'downloading' && candidate" class="mt-4">
        <div class="flex items-center gap-3">
          <div
            class="h-1.5 flex-1 overflow-hidden bg-bg-deep"
            role="progressbar"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-valuenow="updater.percent === null ? undefined : roundedPercent"
            :aria-label="`Downloading Mold ${candidate.version}`"
          >
            <div
              class="h-full bg-accent transition-[width] duration-300"
              :class="updater.percent === null ? 'ms-shimmer w-full' : ''"
              :style="updater.percent === null ? undefined : { width: `${roundedPercent}%` }"
            />
          </div>
          <span
            v-if="updater.totalBytes !== null"
            class="font-mono shrink-0 text-micro text-fg-dim"
          >
            {{ formatBytes(updater.downloadedBytes) }} / {{ formatBytes(updater.totalBytes) }}
          </span>
        </div>
        <p class="mt-2 text-micro text-fg-2">Downloading Mold {{ candidate.version }}…</p>
      </div>

      <p v-else-if="updater.phase === 'verifying'" class="mt-4 text-micro text-fg-2" role="status">
        Verifying the update signature…
      </p>
      <p v-else-if="updater.phase === 'staging'" class="mt-4 text-micro text-fg-2" role="status">
        Running complete signature, identity, Gatekeeper, and install-location checks…
      </p>
      <p v-else-if="updater.phase === 'installing'" class="mt-4 text-micro text-fg-2" role="status">
        Installing the update and restarting Mold…
      </p>
      <div
        v-else-if="updater.phase === 'failed' && updater.error"
        class="border-error/40 mt-4 rounded-control border bg-error/10 p-3"
        role="alert"
      >
        <p class="text-sm font-semibold text-error">Mold couldn’t update</p>
        <p data-selectable class="mt-1 text-micro text-fg-2">{{ updater.error.message }}</p>
        <p class="mt-2 text-micro text-fg-dim">
          <template v-if="updater.currentVersion">
            Mold {{ updater.currentVersion }} remains installed because the update did not complete.
          </template>
          <template v-else>The installed app was not changed.</template>
        </p>
        <div class="mt-3 flex items-center gap-3">
          <button
            v-if="updater.error.retryable"
            type="button"
            class="ms-toolbar-button"
            @click="candidate ? updater.install() : updater.check()"
          >
            {{ candidate ? "Try update again" : "Check again" }}
          </button>
          <button
            type="button"
            class="h-[26px] px-1 text-micro text-fg-dim hover:text-fg"
            @click="updater.clearError()"
          >
            Dismiss
          </button>
        </div>
      </div>

      <p
        v-else-if="updater.phase === 'unsupported'"
        class="mt-4 text-micro text-fg-dim"
        role="status"
      >
        Automatic updates are currently available only in signed macOS builds. Linux packages,
        browser previews, and local dev builds update manually.
      </p>
    </section>
  </div>
</template>
