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
    <p class="mb-3 text-micro text-fg-dim">
      Profiles keep separate per-model preferences and generation settings (rows tagged ⌂ DB).
      Bootstrap settings (⛁ file) are shared.
    </p>
    <div v-if="config.profiles.length" class="flex flex-col gap-1">
      <button
        v-for="p in config.profiles"
        :key="p"
        type="button"
        class="flex h-8 items-center gap-2 rounded-control px-2.5 text-left text-sm"
        :class="
          p === config.activeProfile
            ? 'bg-accent-tint text-fg'
            : 'text-fg-2 hover:bg-bg hover:text-fg'
        "
        @click="switchTo(p)"
      >
        <span class="min-w-0 flex-1 truncate">{{ p }}</span>
        <span
          v-if="p === config.activeProfile"
          class="font-mono text-micro text-fg-dim whitespace-nowrap text-accent"
          >ACTIVE</span
        >
      </button>
    </div>
    <p v-else class="text-micro text-fg-dim">No profiles reported by this engine.</p>

    <div class="mt-3">
      <button v-if="!creating" type="button" class="ms-toolbar-button" @click="creating = true">
        New profile…
      </button>
      <div v-else class="flex items-center gap-2">
        <input
          v-model="newName"
          data-selectable
          type="text"
          placeholder="profile name"
          class="border-border font-mono text-xs h-7 w-48 rounded-control border bg-bg-deep px-2 text-fg placeholder:text-fg-dim"
          @keydown.enter="create"
          @keydown.escape="
            creating = false;
            newName = '';
          "
        />
        <button
          type="button"
          class="ms-toolbar-button ms-toolbar-button--on disabled:opacity-50"
          :disabled="!newName.trim()"
          @click="create"
        >
          Create
        </button>
      </div>
    </div>
  </div>
</template>
