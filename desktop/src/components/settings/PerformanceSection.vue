<script setup lang="ts">
import { computed, reactive } from "vue";
import SettingRow from "./SettingRow.vue";
import SelectControl from "./SelectControl.vue";
import NumberControl from "./NumberControl.vue";
import { ENV_KNOB_SCHEMAS } from "../../lib/settingsSchema";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";

const prefs = useAppPrefsStore();
const conn = useConnectionStore();
const toasts = useToastStore();

/** Env knobs edit a draft; Restart engine applies them to a fresh engine. */
const dirty = reactive(new Set<string>());
const restarting = computed(() => conn.status === "starting");
const isEmbedded = computed(() => conn.mode === "local");

const envName = (key: string) => key.replace(/^env\./, "");
const valueOf = (key: string) => prefs.engineEnv[envName(key)] ?? "";

async function setKnob(key: string, value: string) {
  const name = envName(key);
  const engineEnv = { ...prefs.engineEnv };
  if (value === "") delete engineEnv[name];
  else engineEnv[name] = value;
  await prefs.update({ engineEnv });
  dirty.add(name);
}

async function restartEngine() {
  await conn.stopEngine();
  await conn.useLocal();
  if (conn.ready) {
    dirty.clear();
    toasts.push("Engine restarted — knobs applied");
  } else if (conn.error) {
    toasts.push(conn.error, "error");
  }
}
</script>

<template>
  <div>
    <p v-if="!isEmbedded" class="text-caption text-ink-3">
      Performance knobs configure the built-in engine. You're connected to a shared or remote server
      — its environment is managed where it runs.
    </p>
    <template v-else>
      <div
        v-if="dirty.size > 0"
        class="mb-3 flex items-center gap-3 rounded-chrome border border-halide bg-[color-mix(in_srgb,var(--halide)_10%,transparent)] px-3 py-2"
      >
        <span class="text-body text-ink">Knobs changed — restart the engine to apply.</span>
        <button
          type="button"
          class="ml-auto h-7 rounded-control bg-safelight px-3 text-body font-semibold text-[#141110] disabled:opacity-50"
          :disabled="restarting"
          @click="restartEngine"
        >
          {{ restarting ? "Restarting…" : "Restart engine" }}
        </button>
      </div>
      <SettingRow
        v-for="knob in ENV_KNOB_SCHEMAS"
        :key="knob.key"
        :label="knob.label"
        :help="knob.help"
        :needs-engine-restart="false"
      >
        <span class="edge-code mr-1">{{ envName(knob.key) }}</span>
        <NumberControl
          v-if="knob.editor === 'number'"
          :model-value="valueOf(knob.key) === '' ? null : Number(valueOf(knob.key))"
          :min="knob.min"
          :max="knob.max"
          placeholder="200"
          @commit="(v) => setKnob(knob.key, v === null ? '' : String(v))"
        />
        <SelectControl
          v-else
          :model-value="valueOf(knob.key)"
          :options="knob.options ?? []"
          @commit="(v) => setKnob(knob.key, v)"
        />
      </SettingRow>
    </template>
  </div>
</template>
