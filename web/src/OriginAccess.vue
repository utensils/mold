<script setup lang="ts">
import { onBeforeUnmount, onMounted, provide, ref } from "vue";
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
  suspendStudioPrimary,
  reloadStudioDocument,
  STUDIO_PRIMARY_CHANGED,
  STUDIO_PRIMARY_KEY,
  type StudioPrimary,
} from "./lib/secureStudio";
import {
  addHost,
  listStoredHosts,
  HOSTS_CHANGED_EVENT,
} from "./lib/hostRegistry";
import App from "./App.vue";
import { connectRegularHost } from "@studio/api/directConnection";
import {
  rememberConnectionMode,
  type ConnectionMode,
} from "@studio/api/connectionMode";
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
const mode = ref<ConnectionMode>("secure");
const address = ref("");
const directKey = ref("");
const remembered = ref(listStoredHosts());
const selected = ref(studioPrimary()?.id ?? remembered.value[0]?.id ?? "");
const ready = ref(false);
const requiresKey = ref(false);
const busy = ref(false);
const key = ref("");
const error = ref("");
let attempt = 0;
let controller: AbortController | undefined;
let committingSelection = false;
// App owns module singletons (models, downloads, Pinia stores). Once mounted,
// another primary needs a new document, not only a new Vue component tree.
let mountedStudio = false;
const reloading = ref(false);
const routeKey = (host: StudioPrimary | null) =>
  host
    ? `${host.url}
${host.instanceId ?? ""}
${host.connectionMode ?? "secure"}
${host.apiKey ?? ""}`
    : "";
let acceptedRoute = routeKey(studioPrimary());
function commitSelection(host: StudioPrimary | null) {
  committingSelection = true;
  try {
    setStudioPrimary(host);
  } finally {
    committingSelection = false;
  }
  acceptedRoute = routeKey(host);
}
function refreshRemembered() {
  remembered.value = listStoredHosts();
  if (!remembered.value.some((host) => host.id === selected.value))
    selected.value = remembered.value[0]?.id ?? "";
}
function suspendSelection() {
  attempt += 1;
  controller?.abort();
  ready.value = false;
  busy.value = false;
  clearRelayInfoCache();
  if (hosted) suspendStudioPrimary();
  acceptedRoute = "";
  refreshRemembered();
  error.value = "";
  if (hosted && mountedStudio && !reloading.value) {
    reloading.value = true;
    reloadStudioDocument();
  }
}
function selectionChanged() {
  if (!committingSelection && routeKey(studioPrimary()) !== acceptedRoute)
    suspendSelection();
}
function storageChanged(event: StorageEvent) {
  if (event.key === STUDIO_PRIMARY_KEY || event.key === null)
    suspendSelection();
  else if (event.key === "mold.web.hosts.v1") refreshRemembered();
}
async function connect() {
  if (reloading.value) return;
  const currentAttempt = ++attempt;
  controller?.abort();
  controller = new AbortController();
  const signal = controller.signal;
  const current = () => currentAttempt === attempt && !signal.aborted;
  busy.value = true;
  error.value = "";
  const candidate = hosted ? null : key.value.trim() || originApiKey();
  try {
    if (hosted) {
      if (invitation.value.trim()) {
        const payload = parseMobilePairingPayload(invitation.value.trim());
        if ((payload.mode ?? "secure") !== mode.value)
          throw new Error(
            `Select ${payload.mode === "regular" ? "Regular" : "Secure"} to use this pairing link.`,
          );
        const claim = await claimPairingSession(payload, {
          name: "Mold Studio browser",
          kind: "browser",
        });
        if (!current()) return;
        const host = addHost({
          url: payload.base_url,
          name: payload.name ?? "Mold machine",
          instanceId: claim.instance_id,
          connectionMode: payload.mode ?? "secure",
          ...(payload.mode === "regular" && claim.api_key
            ? { apiKey: claim.api_key }
            : {}),
        });
        commitSelection(host);
        selected.value = host.id;
        invitation.value = "";
        remembered.value = listStoredHosts();
      } else if (mode.value === "regular" && address.value.trim()) {
        const connection = await connectRegularHost(
          address.value.trim(),
          directKey.value.trim() || null,
        );
        if (!current()) return;
        const host = addHost({
          url: connection.baseUrl,
          name: connection.name,
          instanceId: connection.instanceId,
          connectionMode: "regular",
          ...(connection.apiKey ? { apiKey: connection.apiKey } : {}),
        });
        commitSelection(host);
        selected.value = host.id;
        directKey.value = "";
        refreshRemembered();
      } else {
        const host =
          remembered.value.find((host) => host.id === selected.value) ??
          studioPrimary();
        const profile = host ? await secureHostForOrigin(host.url) : null;
        if (!current()) return;
        if (!host || (!profile && host.connectionMode !== "regular")) {
          error.value =
            "On your machine, open Remote Access and copy a new pairing link. Paste it here to connect securely.";
          return;
        }
        rememberConnectionMode(host.url, host.connectionMode ?? "secure");
        commitSelection(host);
      }
    }
    const response = await originAuthenticatedFetch("/api/status", {
      headers: candidate ? { "x-api-key": candidate } : {},
      redirect: "error",
      signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]),
    });
    if (!current()) return;
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
    mountedStudio = true;
    ready.value = true;
  } catch (failure) {
    if (currentAttempt !== attempt) return;
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
    if (currentAttempt === attempt) busy.value = false;
  }
}
function changeKey() {
  if (hosted) commitSelection(null);
  suspendSelection();
  if (hosted) return;
  requiresKey.value = true;
  setOriginApiKey("");
  error.value = "Enter this machine's API key to connect.";
}
provide(ORIGIN_ACCESS_CHANGE_KEY, changeKey);
onMounted(() => {
  if (hosted) {
    window.addEventListener(STUDIO_PRIMARY_CHANGED, selectionChanged);
    window.addEventListener("storage", storageChanged);
    window.addEventListener(HOSTS_CHANGED_EVENT, refreshRemembered);
  }
  // Switch/Forget intentionally clears selection. Remembered alternatives
  // are choices at the gate, not permission to silently pick the first one.
  if (!hosted || studioPrimary() || invitation.value.trim()) void connect();
});
onBeforeUnmount(() => {
  attempt += 1;
  controller?.abort();
  window.removeEventListener(STUDIO_PRIMARY_CHANGED, selectionChanged);
  window.removeEventListener("storage", storageChanged);
  window.removeEventListener(HOSTS_CHANGED_EVENT, refreshRemembered);
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
          Pair this browser with your machine or choose a direct Regular
          connection.
        </p>
        <label v-if="remembered.length" for="saved-machine"
          >Saved machine</label
        >
        <select
          v-if="remembered.length"
          id="saved-machine"
          v-model="selected"
          :disabled="busy || reloading"
        >
          <option v-for="host in remembered" :key="host.id" :value="host.id">
            {{ host.name }}
          </option>
        </select>
        <label for="connection-mode">Connection</label>
        <select
          id="connection-mode"
          v-model="mode"
          :disabled="busy || reloading"
        >
          <option value="secure">Secure · end-to-end encrypted</option>
          <option value="regular">Regular · direct HTTP or HTTPS</option>
        </select>
        <template v-if="mode === 'regular'">
          <p>
            HTTP relies on your network or VPN for encryption. The proxy
            requires Secure.
          </p>
          <label for="regular-address"
            >Machine address (or use a pairing link below)</label
          >
          <input
            id="regular-address"
            v-model="address"
            type="url"
            placeholder="http://machine:7680"
            :disabled="busy || reloading"
          />
          <label for="regular-key">API key (if required)</label>
          <input
            id="regular-key"
            v-model="directKey"
            type="password"
            autocomplete="off"
            :disabled="busy || reloading"
          />
        </template>
        <label for="pairing-link">Pairing link</label>
        <textarea
          id="pairing-link"
          v-model="invitation"
          autocomplete="off"
          spellcheck="false"
          :disabled="busy || reloading"
          rows="3"
          placeholder="Paste the link from Remote Access on your machine"
        />
        <p>
          A pairing link works once and expires shortly. Secure connections stay
          encrypted on every route.
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
        :disabled="busy || reloading"
      />
      <p v-if="error" role="alert">{{ error }}</p>
      <p v-if="requiresKey">Your key stays in this tab's browser session.</p>
      <button type="submit" :disabled="busy || reloading">
        {{
          reloading
            ? "Reloading Studio…"
            : busy
              ? "Connecting…"
              : hosted
                ? mode === "secure"
                  ? "Connect securely"
                  : "Connect Regular"
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
