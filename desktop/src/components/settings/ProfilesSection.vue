<script setup lang="ts">
import { ref } from "vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import { useToastStore } from "../../stores/toasts";

const config = useSettingsConfigStore();
const toasts = useToastStore();
const newName = ref("");
const creating = ref(false);

async function switchTo(name: string) {
  const error = await config.switchProfile(name);
  if (error) toasts.push(error, "error");
  else toasts.push(`Switched to ${name}`);
}

async function create() {
  const name = newName.value.trim();
  if (!name) return;
  const error = await config.switchProfile(name);
  if (error) {
    toasts.push(error, "error");
    return;
  }
  toasts.push(`Created and switched to ${name}`);
  newName.value = "";
  creating.value = false;
}
</script>

<template>
  <div class="max-w-md">
    <p class="mb-3 text-caption text-ink-3">
      Profiles keep separate per-model preferences and generation settings (rows tagged ⌂ DB).
      Bootstrap settings (⛁ file) are shared.
    </p>
    <div v-if="config.profiles.length" class="flex flex-col gap-1">
      <button
        v-for="p in config.profiles"
        :key="p"
        type="button"
        class="flex h-8 items-center gap-2 rounded-control px-2.5 text-left text-body"
        :class="
          p === config.activeProfile
            ? 'bg-[color-mix(in_srgb,var(--safelight)_14%,transparent)] text-ink'
            : 'text-ink-2 hover:bg-bench hover:text-ink'
        "
        @click="switchTo(p)"
      >
        <span class="min-w-0 flex-1 truncate">{{ p }}</span>
        <span v-if="p === config.activeProfile" class="edge-code text-safelight">ACTIVE</span>
      </button>
    </div>
    <p v-else class="text-caption text-ink-3">No profiles reported by this engine.</p>

    <div class="mt-3">
      <button
        v-if="!creating"
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
        @click="creating = true"
      >
        New profile…
      </button>
      <div v-else class="flex items-center gap-2">
        <input
          v-model="newName"
          data-selectable
          type="text"
          placeholder="profile name"
          class="border-edge data-mono h-7 w-48 rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3"
          @keydown.enter="create"
          @keydown.escape="
            creating = false;
            newName = '';
          "
        />
        <button
          type="button"
          class="h-7 rounded-control bg-safelight px-2.5 text-body font-semibold text-[#141110] disabled:opacity-50"
          :disabled="!newName.trim()"
          @click="create"
        >
          Create
        </button>
      </div>
    </div>
  </div>
</template>
