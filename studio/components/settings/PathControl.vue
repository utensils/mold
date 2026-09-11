<script setup lang="ts">
/*
 * A folder. The native picker is INJECTED rather than imported: it lives in
 * `desktop/src/lib/ipc` (Tauri), and `scripts/tests/frontend-architecture.sh`
 * refuses any Tauri reference under studio/. With no picker — a browser tab —
 * the path is an editable mono field, which is the only thing a tab can offer.
 */
const props = defineProps<{
  modelValue: string;
  /** Title for the native picker, and the field's accessible name. */
  title: string;
  disabled?: boolean | undefined;
  /** Native folder picker. Absent → an editable text field instead. */
  pick?: ((title: string) => Promise<string | null>) | undefined;
}>();
const emit = defineEmits<{ (e: "commit", value: string): void }>();

async function choose() {
  if (!props.pick) return;
  const directory = await props.pick(props.title);
  if (directory && directory !== props.modelValue) emit("commit", directory);
}

function commit(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  if (value !== props.modelValue) emit("commit", value);
}
</script>

<template>
  <div class="ms-path">
    <template v-if="pick">
      <span class="ms-path__value" :title="props.modelValue" dir="rtl">
        {{ props.modelValue || "—" }}
      </span>
      <button
        type="button"
        class="ms-path__button"
        :disabled="disabled"
        @click="choose"
      >
        Choose…
      </button>
    </template>
    <input
      v-else
      class="ms-path__field"
      type="text"
      data-selectable
      :value="props.modelValue"
      :disabled="disabled"
      :aria-label="title"
      @change="commit"
      @keydown.enter="($event.target as HTMLInputElement).blur()"
    />
  </div>
</template>

<style scoped>
.ms-path {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-2);
}
.ms-path__value {
  max-width: 256px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-path__button {
  height: var(--mold-ctl-md, 26px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
}
.ms-path__button:hover:not(:disabled) {
  border-color: var(--mold-border-focus);
}
.ms-path__button:disabled {
  opacity: 0.4;
  cursor: default;
}
.ms-path__field {
  width: 288px;
  height: var(--mold-ctl-lg, 32px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.ms-path__field:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-path__field:disabled {
  opacity: 0.4;
}
</style>
