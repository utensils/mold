<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import SettingRow from "./SettingRow.vue";
import { ipc } from "../../lib/ipc";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useToastStore } from "../../stores/toasts";

const prefs = useAppPrefsStore();
const toasts = useToastStore();
const effectiveDirectory = ref("Loading…");
const saving = ref(false);
const usesSystemDefault = computed(() => !prefs.mediaSaveDir);

async function refresh(): Promise<void> {
  try {
    effectiveDirectory.value = await ipc.mediaSaveDirectory();
  } catch (error) {
    effectiveDirectory.value = prefs.mediaSaveDir ?? "Downloads";
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

async function chooseDirectory(): Promise<void> {
  const selected = await ipc.pickDirectory("Choose where Mold saves media");
  if (!selected) return;
  saving.value = true;
  try {
    await prefs.update({ mediaSaveDir: selected });
    await refresh();
    toasts.push("Save location updated", "info", { description: selected });
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  } finally {
    saving.value = false;
  }
}

async function useDownloads(): Promise<void> {
  saving.value = true;
  try {
    await prefs.update({ mediaSaveDir: null });
    await refresh();
    toasts.push("New saves will go to Downloads");
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  } finally {
    saving.value = false;
  }
}

onMounted(() => void refresh());
</script>

<template>
  <SettingRow
    label="Save location"
    help="Save image, Save video, and converted exports go here. Existing files are never overwritten."
  >
    <div class="flex max-w-sm flex-col items-end gap-1.5">
      <div class="flex items-center gap-2">
        <span
          class="data-mono max-w-56 truncate text-caption text-ink-2"
          :title="effectiveDirectory"
          data-test="media-save-directory"
        >
          {{ effectiveDirectory }}
        </span>
        <button
          type="button"
          class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink disabled:opacity-40"
          :disabled="saving"
          data-test="choose-media-save-directory"
          @click="chooseDirectory"
        >
          Change…
        </button>
      </div>
      <button
        v-if="!usesSystemDefault"
        type="button"
        class="text-caption text-ink-3 hover:text-ink"
        :disabled="saving"
        data-test="reset-media-save-directory"
        @click="useDownloads"
      >
        Use Downloads
      </button>
      <span v-else class="text-caption text-ink-3">System default</span>
    </div>
  </SettingRow>
</template>
