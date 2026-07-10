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
  toasts.push("Saved to Keychain");
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
        class="border-edge data-mono h-7 w-64 rounded-control border bg-bath px-2 text-ink placeholder:text-ink-3"
        @keydown.enter="save"
        @keydown.escape="
          editing = false;
          draft = '';
        "
      />
      <button
        type="button"
        class="h-7 rounded-control bg-safelight px-2.5 text-body font-semibold text-[#141110] disabled:opacity-50"
        :disabled="!draft.trim()"
        @click="save"
      >
        Save
      </button>
    </template>
    <template v-else>
      <span class="data-mono text-caption" :class="present ? 'text-ink-2' : 'text-ink-3'">
        {{ present ? "••••••••  set" : "not set" }}
      </span>
      <button
        type="button"
        class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
        @click="editing = true"
      >
        {{ present ? "Replace…" : "Set…" }}
      </button>
      <button
        v-if="present"
        type="button"
        class="h-7 rounded-control px-1.5 text-caption text-ink-3 hover:text-stop"
        title="Remove"
        @click="clear"
      >
        ↺
      </button>
    </template>
  </div>
</template>
