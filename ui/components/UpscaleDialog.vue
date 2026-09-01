<script setup lang="ts">
import { onBeforeUnmount, onMounted } from "vue";

type Choice = { name: string; downloaded?: boolean };
type ExecutionHost = { key: string; label: string };

const props = withDefaults(
  defineProps<{
    open: boolean;
    kind: "image" | "video";
    sourceName: string;
    models?: Choice[];
    modelValue: string;
    executionHosts?: ExecutionHost[];
    executionHostValue?: string;
    busy?: boolean;
    jobState?: string | null;
    status?: string | null;
    progress?: number | null;
    error?: string | null;
  }>(),
  {
    models: () => [],
    executionHosts: () => [],
    executionHostValue: "",
    busy: false,
    jobState: null,
    status: null,
    progress: null,
    error: null,
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: string];
  "update:executionHostValue": [value: string];
  confirm: [];
  close: [];
  pause: [];
  resume: [];
  cancel: [];
}>();

function onKeydown(event: KeyboardEvent) {
  if (!props.open || event.key !== "Escape") return;
  event.preventDefault();
  event.stopImmediatePropagation();
  emit("close");
}

onMounted(() => window.addEventListener("keydown", onKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="upscale-dialog__backdrop"
      @click.self="emit('close')"
    >
      <section
        class="upscale-dialog"
        role="dialog"
        aria-modal="true"
        :aria-label="
          kind === 'video' ? 'Framewise upscale video' : 'Upscale image'
        "
        data-test="upscale-dialog"
      >
        <header>
          <div>
            <p class="upscale-dialog__eyebrow">Library tool</p>
            <h2>
              {{
                kind === "video" ? "Framewise upscale video" : "Upscale image"
              }}
            </h2>
          </div>
          <button
            class="upscale-dialog__close"
            aria-label="Close"
            @click="emit('close')"
          >
            ×
          </button>
        </header>

        <p class="upscale-dialog__source" :title="sourceName">
          {{ sourceName }}
        </p>
        <p class="upscale-dialog__copy">
          <template v-if="kind === 'video'">
            Every frame is enhanced independently and assembled into a new
            video. Frame count, constant FPS, duration, and compatible primary
            audio are preserved. Temporal flicker may remain.
          </template>
          <template v-else>
            Creates a new upscaled Library image and keeps the original
            unchanged.
          </template>
        </p>

        <label v-if="executionHosts.length > 1" class="upscale-dialog__field">
          <span>Run on</span>
          <select
            :value="executionHostValue"
            :disabled="busy || !!jobState"
            data-test="upscale-host"
            @change="
              emit(
                'update:executionHostValue',
                ($event.target as HTMLSelectElement).value,
              )
            "
          >
            <option
              v-for="host in executionHosts"
              :key="host.key"
              :value="host.key"
            >
              {{ host.label }}
            </option>
          </select>
        </label>

        <label class="upscale-dialog__field">
          <span>Upscaler</span>
          <select
            :value="modelValue"
            :disabled="busy || !!jobState"
            data-test="upscale-model"
            @change="
              emit(
                'update:modelValue',
                ($event.target as HTMLSelectElement).value,
              )
            "
          >
            <option v-if="!models.length" :value="modelValue">
              {{ modelValue }} (downloads on first use)
            </option>
            <option
              v-for="model in models"
              :key="model.name"
              :value="model.name"
            >
              {{ model.name
              }}{{ model.downloaded ? "" : " (downloads on first use)" }}
            </option>
          </select>
        </label>

        <div
          v-if="jobState"
          class="upscale-dialog__job"
          data-test="upscale-job"
        >
          <div class="upscale-dialog__jobrow">
            <span>{{ status }}</span>
            <span v-if="progress != null"
              >{{ Math.round(progress * 100) }}%</span
            >
          </div>
          <progress
            v-if="progress != null"
            :value="progress"
            max="1"
          ></progress>
        </div>

        <p
          v-if="error"
          class="upscale-dialog__error"
          role="alert"
          data-test="upscale-error"
        >
          {{ error }}
        </p>

        <footer>
          <button class="upscale-dialog__secondary" @click="emit('close')">
            {{ jobState ? "Close" : "Cancel" }}
          </button>
          <template v-if="jobState">
            <button
              v-if="jobState === 'running' || jobState === 'queued'"
              class="upscale-dialog__secondary"
              @click="emit('pause')"
            >
              Pause
            </button>
            <button
              v-if="jobState === 'paused'"
              class="upscale-dialog__primary"
              @click="emit('resume')"
            >
              Resume
            </button>
            <button
              v-if="!['completed', 'failed', 'cancelled'].includes(jobState)"
              class="upscale-dialog__danger"
              @click="emit('cancel')"
            >
              Cancel job
            </button>
          </template>
          <button
            v-else
            class="upscale-dialog__primary"
            :disabled="busy || !modelValue"
            data-test="start-upscale"
            @click="emit('confirm')"
          >
            {{
              busy
                ? "Starting…"
                : kind === "video"
                  ? "Start Framewise upscale"
                  : "Upscale"
            }}
          </button>
        </footer>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.upscale-dialog__backdrop {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: color-mix(in srgb, #070907 68%, transparent);
  backdrop-filter: blur(10px);
}
.upscale-dialog {
  width: min(460px, 100%);
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  border: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.14));
  border-radius: 18px;
  padding: 22px;
  color: var(--text-primary, #f4f5ef);
  background: var(--surface-raised, #171a16);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
}
header,
footer,
.upscale-dialog__jobrow {
  display: flex;
  align-items: center;
  gap: 10px;
}
header {
  justify-content: space-between;
}
h2 {
  margin: 2px 0 0;
  font-size: 20px;
}
.upscale-dialog__eyebrow {
  margin: 0;
  color: var(--text-muted, #9da496);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.upscale-dialog__close {
  border: 0;
  color: inherit;
  background: transparent;
  font-size: 26px;
}
.upscale-dialog__source {
  overflow: hidden;
  margin: 20px 0 6px;
  color: var(--text-secondary, #c7cbbf);
  font-family: ui-monospace, monospace;
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.upscale-dialog__error {
  max-width: 100%;
  margin: 14px 0 0;
  overflow-wrap: anywhere;
  color: var(--danger, #ff7f72);
  font-size: 13px;
}
.upscale-dialog__copy {
  margin: 0 0 18px;
  color: var(--text-secondary, #c7cbbf);
  line-height: 1.5;
}
.upscale-dialog__field {
  display: grid;
  min-width: 0;
  box-sizing: border-box;
  gap: 7px;
  font-size: 13px;
  font-weight: 650;
}
.upscale-dialog__field + .upscale-dialog__field {
  margin-top: 14px;
}
select {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  min-height: 42px;
  border: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.14));
  border-radius: 10px;
  padding: 0 12px;
  color: inherit;
  background: var(--surface-sunken, #0f120f);
}
.upscale-dialog__job {
  margin-top: 18px;
  border-radius: 11px;
  padding: 12px;
  background: var(--surface-sunken, #0f120f);
}
.upscale-dialog__jobrow {
  justify-content: space-between;
  font-size: 13px;
}
progress {
  width: 100%;
  margin-top: 10px;
}
footer {
  justify-content: flex-end;
  margin-top: 22px;
  flex-wrap: wrap;
}
footer button {
  min-height: 38px;
  border-radius: 9px;
  padding: 0 14px;
  font-weight: 700;
}
.upscale-dialog__secondary {
  border: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.14));
  color: inherit;
  background: transparent;
}
.upscale-dialog__primary {
  border: 0;
  color: var(--button-primary-text, #11140f);
  background: var(--accent, #c8f55a);
}
.upscale-dialog__danger {
  border: 1px solid color-mix(in srgb, #e85b4d 55%, transparent);
  color: #ff9d92;
  background: transparent;
}
button:disabled,
select:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}
@media (max-width: 560px) {
  .upscale-dialog__backdrop {
    align-items: end;
    padding: 0;
  }
  .upscale-dialog {
    width: 100%;
    border-radius: 20px 20px 0 0;
    padding-bottom: calc(22px + env(safe-area-inset-bottom));
  }
}
</style>
