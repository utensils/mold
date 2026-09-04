<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ipc, type SecretName } from "../../lib/ipc";
import { useToastStore } from "../../stores/toasts";

const props = defineProps<{ name: SecretName; placeholder?: string | undefined }>();
const toasts = useToastStore();

const present = ref(false);
const editing = ref(false);
const draft = ref("");

onMounted(async () => {
  present.value = ((await ipc.secretGet(props.name)) ?? "") !== "";
});

async function save() {
  const value = draft.value.trim();
  if (!value) return;
  await ipc.secretSet(props.name, value);
  present.value = true;
  editing.value = false;
  draft.value = "";
  toasts.push("Saved");
}

async function clear() {
  await ipc.secretClear(props.name);
  present.value = false;
  editing.value = false;
  draft.value = "";
  toasts.push("Removed");
}
</script>

<template>
  <div class="flex items-center gap-2">
    <template v-if="editing">
      <input
        v-model="draft"
        type="password"
        autocomplete="off"
        data-selectable
        :placeholder="placeholder"
        class="border-border font-mono text-xs h-7 w-64 rounded-control border bg-bg-deep px-2 text-fg placeholder:text-fg-dim"
        @keydown.enter="save"
        @keydown.escape="
          editing = false;
          draft = '';
        "
      />
      <button
        type="button"
        class="ms-toolbar-button ms-toolbar-button--on disabled:opacity-50"
        :disabled="!draft.trim()"
        @click="save"
      >
        Save
      </button>
    </template>
    <template v-else>
      <span class="font-mono text-micro" :class="present ? 'text-fg-2' : 'text-fg-dim'">
        {{ present ? "••••••••  set" : "not set" }}
      </span>
      <button type="button" class="ms-toolbar-button" @click="editing = true">
        {{ present ? "Replace…" : "Set…" }}
      </button>
      <button
        v-if="present"
        type="button"
        class="h-[26px] rounded-control px-1.5 text-micro text-fg-dim hover:text-error"
        title="Remove"
        aria-label="Remove secret"
        @click="clear"
      >
        ↺
      </button>
    </template>
  </div>
</template>
