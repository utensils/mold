<script setup lang="ts">
import { onMounted, provide, ref } from "vue";
import App from "./App.vue";
import {
  ORIGIN_ACCESS_CHANGE_KEY,
  originAuthenticatedFetch,
  originApiKey,
  setOriginApiKey,
} from "./lib/originAuth";
const ready = ref(false);
const busy = ref(false);
const key = ref("");
const error = ref("");
async function connect() {
  busy.value = true;
  error.value = "";
  try {
    const candidate = key.value.trim() || originApiKey();
    const response = await originAuthenticatedFetch("/api/status", {
      headers: candidate ? { "x-api-key": candidate } : {},
      redirect: "error",
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) {
      error.value =
        response.status === 401
          ? "Enter this machine's API key to connect."
          : `This machine returned ${response.status}. Try again.`;
      return;
    }
    if (candidate) setOriginApiKey(candidate);
    key.value = "";
    ready.value = true;
  } catch {
    error.value =
      "This machine is unavailable. Check the connection and try again.";
  } finally {
    busy.value = false;
  }
}
function changeKey() {
  ready.value = false;
  setOriginApiKey("");
  error.value = "Enter this machine's API key to connect.";
}
provide(ORIGIN_ACCESS_CHANGE_KEY, changeKey);
onMounted(connect);
</script>
<template>
  <template v-if="ready">
    <App />
  </template>
  <main v-else class="origin-access">
    <form @submit.prevent="connect">
      <h1>Connect to this machine</h1>
      <p>
        {{
          originApiKey()
            ? "Reconnect using your saved API key, or enter another key."
            : "Enter the API key configured on this machine."
        }}
      </p>
      <label for="origin-key">API key</label>
      <input
        id="origin-key"
        v-model="key"
        type="password"
        autocomplete="off"
        spellcheck="false"
        :disabled="busy"
      />
      <p v-if="error" role="alert">{{ error }}</p>
      <p>Your key stays in this tab's browser session.</p>
      <button type="submit" :disabled="busy">
        {{ busy ? "Connecting…" : "Connect" }}
      </button>
    </form>
  </main>
</template>
<style scoped>
.origin-access {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 24px;
}
.origin-access form {
  width: min(100%, 420px);
  display: grid;
  gap: 12px;
}
.origin-access input {
  padding: 12px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
  color: inherit;
}
.origin-access button {
  padding: 12px;
}
</style>
