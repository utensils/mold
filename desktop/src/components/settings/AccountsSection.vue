<script setup lang="ts">
/*
 * Settings ▸ Accounts & tokens.
 *
 * These two secrets are FILE-backed on this device (`settings.json` via IPC),
 * not engine-config rows, so this section owns the reads and writes itself and
 * hands the shared control only what it needs: whether something is stored.
 * The control never holds a secret of its own and never renders one.
 */
import { onMounted, reactive } from "vue";
import SettingRow from "@studio/components/settings/SettingRow.vue";
import SecretControl from "@studio/components/settings/SecretControl.vue";
import { ipc, type SecretName } from "../../lib/ipc";
import { useToastStore } from "../../stores/toasts";

const toasts = useToastStore();

const SECRETS = [
  {
    name: "hf-token" as SecretName,
    label: "Hugging Face token",
    help: "Needed for gated models (FLUX.1-dev license, private repos). Stored only on this device and passed to the engine as HF_TOKEN.",
    placeholder: "hf_…",
  },
  {
    name: "civitai-token" as SecretName,
    label: "Civitai token",
    help: "Needed for Civitai downloads that require an API key. Passed to the engine as CIVITAI_TOKEN.",
    placeholder: "Civitai API key",
  },
];

const present = reactive<Record<string, boolean>>({});

onMounted(async () => {
  for (const secret of SECRETS) {
    present[secret.name] = ((await ipc.secretGet(secret.name)) ?? "") !== "";
  }
});

async function save(name: SecretName, value: string) {
  await ipc.secretSet(name, value);
  present[name] = true;
  toasts.push("Saved");
}

async function clear(name: SecretName) {
  await ipc.secretClear(name);
  present[name] = false;
  toasts.push("Removed");
}
</script>

<template>
  <div>
    <SettingRow
      v-for="secret in SECRETS"
      :key="secret.name"
      :label="secret.label"
      :help="secret.help"
    >
      <SecretControl
        :present="present[secret.name] === true"
        :placeholder="secret.placeholder"
        :aria-label="secret.label"
        clearable
        @save="(value) => save(secret.name, value)"
        @clear="clear(secret.name)"
      />
    </SettingRow>
    <!-- Inset like a row: the section card has no padding of its own. -->
    <p class="max-w-md px-3.5 py-3 text-micro text-fg-dim">
      Tokens apply to the built-in engine the next time it starts. The remote host API key lives in
      the Engine section.
    </p>
  </div>
</template>
