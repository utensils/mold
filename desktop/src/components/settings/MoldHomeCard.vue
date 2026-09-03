<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ipc, type MoldHomeInfo } from "../../lib/ipc";

const info = ref<MoldHomeInfo | null>(null);
const editing = ref(false);
const path = ref("");
const migrate = ref(true);
const busy = ref(false);
const error = ref<string | null>(null);

const sourceLabel = computed(() => {
  if (info.value?.source === "saved") return "Saved for local Mold apps";
  if (info.value?.source === "environment") return "MOLD_HOME environment";
  if (info.value?.source === "invalid") return "Saved location needs repair";
  return "Default location";
});

const displayedPath = computed(() => info.value?.path || "No valid saved location");

onMounted(async () => {
  info.value = await ipc.getMoldHome();
  path.value = info.value?.path ?? "";
});

function beginEdit() {
  path.value = info.value?.path ?? "";
  migrate.value = info.value?.exists !== false;
  error.value = null;
  editing.value = true;
}

async function browse() {
  const selected = await ipc.pickDirectory("Choose a Mold home folder");
  if (selected) path.value = selected;
}

async function apply() {
  error.value = null;
  if (!path.value.trim()) {
    error.value = "Choose a folder or enter its full path.";
    return;
  }
  busy.value = true;
  try {
    await ipc.changeMoldHome(path.value, migrate.value);
  } catch (reason) {
    // A successful native relaunch can close the webview before invoke
    // resolves. Only surface an error while this document is still alive.
    error.value = reason instanceof Error ? reason.message : String(reason);
    busy.value = false;
  }
}
</script>

<template>
  <section
    class="border-border mt-4 rounded-window border bg-bg p-4"
    data-test="mold-home-card"
    aria-labelledby="mold-home-title"
  >
    <div class="flex items-start gap-4">
      <div class="min-w-0 flex-1">
        <h2 id="mold-home-title" class="text-sm font-medium text-fg">Mold home</h2>
        <p class="mt-1 text-micro text-fg-dim">
          The shared root for configuration, history, logs, and the default model and output
          folders.
        </p>
      </div>
      <button
        v-if="!editing"
        type="button"
        data-test="change-mold-home"
        class="border-border h-8 shrink-0 rounded-control border px-3 text-sm font-semibold text-fg-2 hover:text-fg"
        :disabled="info?.source === 'environment'"
        :title="
          info?.source === 'environment'
            ? 'Unset MOLD_HOME in the environment before changing the saved location.'
            : undefined
        "
        @click="beginEdit"
      >
        Change…
      </button>
    </div>

    <div v-if="!editing" class="border-border mt-3 rounded-control border bg-bg-deep px-3 py-2.5">
      <code
        data-selectable
        class="font-mono block truncate text-micro text-fg-2"
        :title="info?.path"
      >
        {{ info ? displayedPath : "Loading…" }}
      </code>
      <span class="mt-1 block text-micro text-fg-dim">{{ sourceLabel }}</span>
      <span v-if="info?.source === 'invalid'" class="mt-1 block text-micro text-error" role="alert">
        The saved location could not be read. Choose a folder to repair it.
      </span>
      <span v-else-if="info && !info.exists" class="mt-1 block text-micro text-error" role="alert">
        This folder is unavailable. Reconnect its drive or choose another folder before starting
        This Mac.
      </span>
      <span v-if="info?.source === 'environment'" class="mt-1 block text-micro text-sapphire">
        Unset MOLD_HOME in the launch environment to change this location here.
      </span>
    </div>

    <form v-else class="mt-4" data-test="mold-home-form" @submit.prevent="apply">
      <label for="mold-home-path" class="text-micro font-semibold text-fg-2">New folder</label>
      <div class="mt-1.5 flex gap-2">
        <input
          id="mold-home-path"
          v-model="path"
          data-test="mold-home-path"
          data-selectable
          type="text"
          spellcheck="false"
          autocomplete="off"
          placeholder="/Volumes/External/Mold"
          class="border-border font-mono h-9 min-w-0 flex-1 rounded-control border bg-bg-deep px-3 text-sm text-fg placeholder:text-fg-dim focus:border-sapphire focus:outline-none"
          :disabled="busy"
        />
        <button
          type="button"
          data-test="browse-mold-home"
          class="border-border h-9 shrink-0 rounded-control border px-3 text-sm font-semibold text-fg-2 hover:text-fg disabled:opacity-50"
          :disabled="busy"
          @click="browse"
        >
          Browse…
        </button>
      </div>

      <fieldset class="mt-4" :disabled="busy">
        <legend class="text-micro font-semibold text-fg-2">What should Mold do?</legend>
        <label
          class="border-border mt-2 flex cursor-pointer items-start gap-3 rounded-control border bg-bg-deep p-3"
          :class="[
            migrate ? 'ring-sapphire/60 ring-1' : '',
            info && !info.exists ? 'cursor-not-allowed opacity-50' : '',
          ]"
        >
          <input
            v-model="migrate"
            class="mt-0.5"
            type="radio"
            name="mold-home-action"
            :value="true"
            :disabled="info ? !info.exists : false"
          />
          <span>
            <span class="block text-sm font-medium text-fg">Copy everything</span>
            <span class="mt-0.5 block text-micro text-fg-dim">
              Recommended. Copies everything inside the current Mold home; the original stays
              untouched.
            </span>
          </span>
        </label>
        <label
          class="border-border mt-2 flex cursor-pointer items-start gap-3 rounded-control border bg-bg-deep p-3"
          :class="!migrate ? 'ring-sapphire/60 ring-1' : ''"
        >
          <input
            v-model="migrate"
            class="mt-0.5"
            type="radio"
            name="mold-home-action"
            :value="false"
          />
          <span>
            <span class="block text-sm font-medium text-fg">Use this folder as-is</span>
            <span class="mt-0.5 block text-micro text-fg-dim">
              Starts fresh in an empty folder, or opens an existing Mold home without copying.
            </span>
          </span>
        </label>
      </fieldset>

      <p v-if="migrate" class="mt-3 text-micro text-sapphire">
        Large model libraries can take a while to copy. Keep both drives connected until Mold
        restarts.
      </p>
      <p v-if="error" class="mt-3 text-micro text-error" role="alert">{{ error }}</p>

      <div class="mt-4 flex justify-end gap-2">
        <button
          type="button"
          class="border-border h-8 rounded-control border px-3 text-sm text-fg-2 hover:text-fg disabled:opacity-50"
          :disabled="busy"
          @click="editing = false"
        >
          Cancel
        </button>
        <button
          type="submit"
          data-test="apply-mold-home"
          class="h-8 rounded-control bg-accent px-3 text-sm font-semibold text-on-accent hover:brightness-105 disabled:opacity-50"
          :disabled="busy"
        >
          {{
            busy
              ? migrate
                ? "Copying…"
                : "Switching…"
              : migrate
                ? "Copy and restart"
                : "Use and restart"
          }}
        </button>
      </div>
    </form>
  </section>
</template>
