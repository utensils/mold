<script setup lang="ts">
import { onMounted, provide, ref } from "vue";
import {
  parseMobilePairingPayload,
  claimPairingSession,
} from "@studio/api/pairing";
import { secureHostForOrigin } from "@studio/api/secureStore";
import {
  configureSecureOrigin,
  clearRelayInfoCache,
} from "@studio/api/relayTransport";
import {
  standaloneStudio,
  studioPrimary,
  setStudioPrimary,
} from "./lib/secureStudio";
import { addHost, listStoredHosts } from "./lib/hostRegistry";
import App from "./App.vue";
import { takeInitialPairingLink } from "./lib/initialPairing";
import {
  ORIGIN_ACCESS_CHANGE_KEY,
  originAuthenticatedFetch,
  originApiKey,
  setOriginApiKey,
} from "./lib/originAuth";
const hosted = standaloneStudio();
if (hosted) configureSecureOrigin(() => studioPrimary()?.url ?? null);
const invitation = ref(takeInitialPairingLink());
const remembered = ref(listStoredHosts());
const selected = ref(studioPrimary()?.id ?? remembered.value[0]?.id ?? "");
const ready = ref(false);
const requiresKey = ref(false);
const busy = ref(false);
const key = ref("");
const error = ref("");
async function connect() {
  busy.value = true;
  error.value = "";
  const candidate = hosted ? null : key.value.trim() || originApiKey();
  try {
    if (hosted) {
      if (invitation.value.trim()) {
        const payload = parseMobilePairingPayload(invitation.value.trim());
        const claim = await claimPairingSession(payload, {
          name: "Mold Studio browser",
          kind: "browser",
        });
        const host = addHost({
          url: payload.base_url,
          name: payload.name ?? "Mold machine",
          instanceId: claim.instance_id,
        });
        setStudioPrimary(host);
        selected.value = host.id;
        invitation.value = "";
        remembered.value = listStoredHosts();
      } else {
        const host =
          remembered.value.find((host) => host.id === selected.value) ??
          studioPrimary();
        if (!host || !(await secureHostForOrigin(host.url))) {
          error.value =
            "On your machine, open Remote Access and copy a new pairing link. Paste it here to connect securely.";
          return;
        }
        setStudioPrimary(host);
      }
    }
    const response = await originAuthenticatedFetch("/api/status", {
      headers: candidate ? { "x-api-key": candidate } : {},
      redirect: "error",
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) {
      requiresKey.value = !hosted && response.status === 401;
      error.value =
        !hosted && response.status === 401
          ? "Enter this machine's API key to connect."
          : `This machine returned ${response.status}. Try again.`;
      return;
    }
    if (candidate) setOriginApiKey(candidate);
    key.value = "";
    ready.value = true;
  } catch (failure) {
    if (hosted) {
      error.value =
        failure instanceof Error
          ? failure.message
          : "Could not connect securely. Create a fresh pairing link on the host and retry.";
      return;
    }
    error.value =
      "This machine is unavailable. Check the connection and try again." +
      (candidate
        ? " For remote access, open the HTTPS address directly; API-key requests cannot follow redirects."
        : "");
  } finally {
    busy.value = false;
  }
}
function changeKey() {
  ready.value = false;
  clearRelayInfoCache();
  if (hosted) {
    remembered.value = listStoredHosts();
    error.value = "";
    return;
  }
  requiresKey.value = true;
  setOriginApiKey("");
  error.value = "Enter this machine's API key to connect.";
}
provide(ORIGIN_ACCESS_CHANGE_KEY, changeKey);
onMounted(() => {
  void connect();
});
</script>
<template>
  <template v-if="ready">
    <App />
  </template>
  <main v-else class="origin-access">
    <form @submit.prevent="connect">
      <h1>{{ hosted ? "Connect to Mold" : "Connect to this machine" }}</h1>
      <template v-if="hosted">
        <p>
          Pair this browser with your machine. Your requests and media stay
          encrypted between them.
        </p>
        <label v-if="remembered.length" for="saved-machine"
          >Saved machine</label
        >
        <select
          v-if="remembered.length"
          id="saved-machine"
          v-model="selected"
          :disabled="busy"
        >
          <option v-for="host in remembered" :key="host.id" :value="host.id">
            {{ host.name }}
          </option>
        </select>
        <label for="pairing-link">Pairing link</label>
        <textarea
          id="pairing-link"
          v-model="invitation"
          autocomplete="off"
          spellcheck="false"
          :disabled="busy"
          rows="3"
          placeholder="Paste the link from Remote Access on your machine"
        />
        <p>
          A pairing link works once and expires shortly. Existing pairings from
          the previous proxy must be replaced.
        </p>
      </template>
      <p v-if="requiresKey">
        {{
          originApiKey()
            ? "Reconnect using your saved API key, or enter another key."
            : "Enter the API key configured on this machine."
        }}
      </p>
      <label v-if="requiresKey" for="origin-key">API key</label>
      <input
        v-if="requiresKey"
        id="origin-key"
        v-model="key"
        type="password"
        autocomplete="off"
        spellcheck="false"
        :disabled="busy"
      />
      <p v-if="error" role="alert">{{ error }}</p>
      <p v-if="requiresKey">Your key stays in this tab's browser session.</p>
      <button type="submit" :disabled="busy">
        {{
          busy
            ? "Connecting…"
            : hosted
              ? "Connect securely"
              : requiresKey
                ? "Connect"
                : "Try again"
        }}
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
.origin-access input,
.origin-access textarea,
.origin-access select {
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
