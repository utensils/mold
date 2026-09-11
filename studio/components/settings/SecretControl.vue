<script setup lang="ts">
/*
 * A stored credential, fully controlled: it holds no secret of its own and
 * never renders one. The caller says whether something is stored (`present`)
 * and handles `save` / `clear` — desktop through `ipc.secret*`, web through
 * `/api/catalog/credentials`, and a config row through `secretValuePresent`,
 * because the engine answers a stored cloud key with the literal `"<set>"`.
 */
import { ref } from "vue";

const props = defineProps<{
  present: boolean;
  busy?: boolean | undefined;
  placeholder?: string | undefined;
  ariaLabel?: string | undefined;
  /** Offer removal of a stored value. */
  clearable?: boolean | undefined;
}>();
const emit = defineEmits<{
  (e: "save", value: string): void;
  (e: "clear"): void;
}>();

const editing = ref(false);
const draft = ref("");

function save() {
  const value = draft.value.trim();
  if (!value) return;
  emit("save", value);
  editing.value = false;
  draft.value = "";
}

function cancel() {
  editing.value = false;
  draft.value = "";
}

function clear() {
  emit("clear");
  cancel();
}
</script>

<template>
  <div class="ms-secret">
    <template v-if="editing">
      <input
        v-model="draft"
        class="ms-secret__field"
        type="password"
        autocomplete="off"
        data-selectable
        :placeholder="placeholder"
        :aria-label="ariaLabel"
        @keydown.enter="save"
        @keydown.escape="cancel"
      />
      <button
        type="button"
        class="ms-secret__button ms-secret__button--primary"
        data-test="secret-save"
        :disabled="busy || !draft.trim()"
        @click="save"
      >
        Save
      </button>
      <button type="button" class="ms-secret__button" @click="cancel">
        Cancel
      </button>
    </template>
    <template v-else>
      <span
        class="ms-secret__state"
        :class="{ 'ms-secret__state--set': props.present }"
      >
        {{ props.present ? "••••••••  set" : "not set" }}
      </span>
      <button
        type="button"
        class="ms-secret__button"
        data-test="secret-edit"
        :disabled="busy"
        @click="editing = true"
      >
        {{ props.present ? "Replace…" : "Set…" }}
      </button>
      <button
        v-if="props.present && clearable"
        type="button"
        class="ms-secret__remove"
        data-test="secret-clear"
        title="Remove"
        :aria-label="ariaLabel ? `Remove ${ariaLabel}` : 'Remove'"
        :disabled="busy"
        @click="clear"
      >
        ↺
      </button>
    </template>
  </div>
</template>

<style scoped>
.ms-secret {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-2);
}
.ms-secret__field {
  width: 256px;
  height: var(--mold-ctl-lg, 32px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-xs);
}
.ms-secret__field:focus {
  outline: none;
  border-color: var(--mold-border-focus);
}
.ms-secret__state {
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-secret__state--set {
  color: var(--mold-text-2);
}
.ms-secret__button {
  height: var(--mold-ctl-md, 26px);
  padding: 0 var(--mold-sp-2);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
}
.ms-secret__button--primary {
  border-color: var(--mold-blue);
  background: var(--mold-accent-tint);
}
.ms-secret__button:disabled {
  opacity: 0.5;
  cursor: default;
}
.ms-secret__remove {
  height: var(--mold-ctl-md, 26px);
  padding: 0 var(--mold-sp-1);
  border: none;
  border-radius: var(--mold-radius-2);
  background: none;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
  cursor: pointer;
}
.ms-secret__remove:hover:not(:disabled) {
  color: var(--mold-error);
}
</style>
