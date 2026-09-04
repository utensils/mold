<script setup lang="ts">
/*
 * Speed & memory. Two kinds of control, in this order:
 *
 *  1. The engine's own scheduler keys, read from `/api/config` on whatever
 *     machine is primary. They work against an embedded engine, a reused
 *     `mold serve`, and a remote host alike, so they always render (each row
 *     hides itself on an engine that does not report the key).
 *  2. The environment knobs the app applies WHEN IT STARTS the engine. Those
 *     only exist for the embedded engine, so a reused or remote server gets a
 *     sentence naming which one it is instead.
 */
import { computed, reactive } from "vue";
import ConfigSettingRow from "./ConfigSettingRow.vue";
import SettingRow from "./SettingRow.vue";
import SelectControl from "./SelectControl.vue";
import NumberControl from "./NumberControl.vue";
import { ENGINE_KEY_SCHEMAS, ENV_KNOB_SCHEMAS } from "../../lib/settingsSchema";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useConnectionStore } from "../../stores/connection";
import { useToastStore } from "../../stores/toasts";

const prefs = useAppPrefsStore();
const conn = useConnectionStore();
const toasts = useToastStore();

/** The curated engine keys this section owns (scheduler replan/warm-wait). */
const engineRows = ENGINE_KEY_SCHEMAS.filter((schema) => schema.section === "performance");

/** Env knobs edit a draft; Restart engine applies them to a fresh engine. */
const dirty = reactive(new Set<string>());
const restarting = computed(() => conn.status === "starting");
const isEmbedded = computed(() => conn.mode === "local");

/** Why the knobs are absent — named truthfully. A reused local `mold serve`
 *  is the documented normal mode and is NOT "a shared or remote server". */
const foreignEngineNote = computed(() =>
  conn.mode === "external"
    ? "Mold is using the mold server that was already running on this device, so its environment is set wherever that server was started — not here."
    : "You're connected to another machine. Its environment is managed where it runs.",
);

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
  const result = await conn.restartEngine();
  if (result === "restarted") {
    dirty.clear();
    toasts.push("Engine restarted — knobs applied");
  } else if (result === "failed" && conn.error) {
    toasts.push(conn.error, "error");
  }
}
</script>

<template>
  <div>
    <ConfigSettingRow
      v-for="schema in engineRows"
      :key="schema.key"
      :schema-key="schema.key"
      data-test="performance-engine-row"
    />

    <!-- The card has no padding of its own (rows are full-bleed, as in the
         mock), so anything that is not a row insets itself. -->
    <p v-if="!isEmbedded" class="px-3.5 py-3 text-micro text-fg-dim" data-test="performance-note">
      {{ foreignEngineNote }}
    </p>
    <template v-else>
      <div
        v-if="dirty.size > 0"
        class="mx-3.5 mt-3.5 flex items-center gap-3 rounded-control border border-sapphire bg-sapphire/10 px-3 py-2"
      >
        <span class="text-sm text-fg">Knobs changed — restart the engine to apply.</span>
        <button
          type="button"
          class="ml-auto h-7 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent transition-[filter] duration-100 hover:brightness-105 active:translate-y-px disabled:opacity-50"
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
      >
        <span class="font-mono text-micro text-fg-dim whitespace-nowrap mr-1">{{
          envName(knob.key)
        }}</span>
        <NumberControl
          v-if="knob.editor === 'number'"
          :model-value="valueOf(knob.key) === '' ? null : Number(valueOf(knob.key))"
          :min="knob.min"
          :max="knob.max"
          placeholder="200"
          :aria-label="knob.label"
          @commit="(v) => setKnob(knob.key, v === null ? '' : String(v))"
        />
        <SelectControl
          v-else
          :model-value="valueOf(knob.key)"
          :options="knob.options ?? []"
          :aria-label="knob.label"
          @commit="(v) => setKnob(knob.key, v)"
        />
      </SettingRow>
    </template>
  </div>
</template>
