<script setup lang="ts">
import SettingRow from "./SettingRow.vue";
import SecretControl from "./SecretControl.vue";
</script>

<template>
  <div>
    <SettingRow
      label="Hugging Face token"
      help="Needed for gated models (FLUX.1-dev license, private repos). Stored only on this device and passed to the engine as HF_TOKEN."
    >
      <SecretControl name="hf-token" placeholder="hf_…" />
    </SettingRow>
    <SettingRow
      label="Civitai token"
      help="Needed for Civitai downloads that require an API key. Passed to the engine as CIVITAI_TOKEN."
    >
      <SecretControl name="civitai-token" placeholder="Civitai API key" />
    </SettingRow>
    <!-- Inset like a row: the section card has no padding of its own. -->
    <p class="max-w-md px-3.5 py-3 text-micro text-fg-dim">
      Tokens apply to the built-in engine the next time it starts. The remote host API key lives in
      the Engine section.
    </p>
  </div>
</template>
