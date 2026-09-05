<script setup lang="ts">
import { computed } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { useLicenseAcceptance } from "../composables/useLicenseAcceptance";

const props = defineProps<{
  openExternal?: ((url: string) => void | Promise<void>) | undefined;
}>();
const licenses = useLicenseAcceptance();
// `intent: "record"` accepts the terms and downloads nothing — Settings' bare
// "Accept terms" action, and the retry a model-install takes before
// re-driving its OWN enqueue. Promising a download there would ask the user
// to consent to an operation that will not happen.
const recordOnly = computed(() => licenses.pending.value?.intent === "record");
const percent = computed(() => {
  const progress = licenses.progress.value;
  if (!progress || progress.bytesTotal <= 0) return null;
  return Math.min(
    100,
    Math.round((progress.bytesDone / progress.bytesTotal) * 100),
  );
});

const primaryLabel = computed(() => {
  if (recordOnly.value) {
    return licenses.busy.value ? "Recording…" : "Accept terms";
  }
  return licenses.busy.value ? "Downloading…" : "Accept terms and download";
});

function openTerms(event: MouseEvent, url: string) {
  if (!props.openExternal) return;
  event.preventDefault();
  void props.openExternal(url);
}
</script>

<template>
  <div
    v-if="licenses.pending.value"
    class="license-host"
    data-test="license-dialog-host"
  >
    <ModalPanel
      :open="true"
      :width="520"
      label="Review third-party model license"
      @close="licenses.cancel()"
    >
      <h2 class="license-title">Review license terms</h2>
      <p class="license-machine">
        These terms will be accepted only on
        <strong>{{ licenses.pending.value.hostLabel }}</strong
        >.
      </p>
      <div
        v-for="requirement in licenses.pending.value.requirements"
        :key="requirement.installModel"
        class="license-bundle"
      >
        <p class="license-model">
          {{ recordOnly ? "Required for" : "Required download" }}:
          {{ requirement.installModel }}
        </p>
        <article
          v-for="term in requirement.licenses"
          :key="`${term.id}:${term.sha256}`"
        >
          <h3>{{ term.name }}</h3>
          <p>{{ term.summary }}</p>
          <p class="license-links">
            <a
              :href="term.url"
              target="_blank"
              rel="noreferrer"
              @click="openTerms($event, term.url)"
              >Pinned terms</a
            >
            <a
              :href="term.canonical"
              target="_blank"
              rel="noreferrer"
              @click="openTerms($event, term.canonical)"
              >Project terms</a
            >
          </p>
        </article>
      </div>
      <p v-if="licenses.error.value" class="license-error" role="alert">
        {{ licenses.error.value }}
      </p>
      <div
        v-if="licenses.progress.value && !recordOnly"
        class="license-progress"
        aria-live="polite"
      >
        <span>
          {{
            licenses.progress.value.status === "starting"
              ? "Starting"
              : licenses.progress.value.status
          }}
          {{ licenses.progress.value.model }}
        </span>
        <span v-if="percent !== null">{{ percent }}%</span>
      </div>
      <template #footer>
        <button
          type="button"
          class="license-secondary"
          @click="licenses.cancel()"
        >
          {{
            licenses.busy.value && !recordOnly ? "Cancel download" : "Cancel"
          }}
        </button>
        <button
          type="button"
          class="license-primary"
          :disabled="licenses.busy.value"
          @click="licenses.accept()"
        >
          {{ primaryLabel }}
        </button>
      </template>
    </ModalPanel>
  </div>
</template>

<style scoped>
.license-host {
  position: fixed;
  inset: 0;
  z-index: 1000;
}
.license-title {
  margin: 0;
  color: var(--mold-text);
  font: 700 20px/1.2 var(--mold-font-sans);
}
.license-machine {
  margin: 8px 0 16px;
  color: var(--mold-text-2);
  font-size: 13px;
}
.license-bundle {
  margin-top: 12px;
  padding: 14px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
}
.license-model {
  margin: 0 0 10px;
  color: var(--mold-text-dim);
  font: 600 11px/1.4 var(--mold-font-mono);
}
article + article {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid var(--mold-border);
}
h3 {
  margin: 0;
  color: var(--mold-text);
  font-size: 15px;
}
article p {
  margin: 6px 0 0;
  color: var(--mold-text-2);
  font-size: 13px;
  line-height: 1.5;
}
.license-links {
  display: flex;
  gap: 16px;
}
a {
  color: var(--mold-blue);
}
.license-error {
  color: var(--mold-error);
  font-size: 13px;
}
.license-progress {
  display: flex;
  justify-content: space-between;
  margin-top: 14px;
  color: var(--mold-text-2);
  font: 600 12px/1.4 var(--mold-font-mono);
}
button {
  min-height: 44px;
  border-radius: var(--mold-radius-2);
  padding: 0 16px;
  font: 600 13px var(--mold-font-sans);
  cursor: pointer;
}
button:disabled {
  opacity: 0.55;
  cursor: default;
}
.license-secondary {
  margin-left: auto;
  border: 1px solid var(--mold-border-control);
  color: var(--mold-text-2);
  background: transparent;
}
.license-primary {
  border: 1px solid var(--mold-blue);
  color: var(--mold-bg-deep);
  background: var(--mold-blue);
}
</style>
