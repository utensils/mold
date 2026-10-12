<script setup lang="ts">
import { ref } from "vue";
import {
  claimPairingSession,
  parseMobilePairingPayload,
  type MobilePairingPayload,
  type PairingClaim,
  type PairingClientIdentity,
} from "../api/pairing";
import {
  connectRegularHost,
  type RegularConnection,
} from "../api/directConnection";
import type { ConnectionMode } from "../api/connectionMode";
const props = defineProps<{
  clientName: string;
  clientKind: PairingClientIdentity["kind"];
}>();
const emit = defineEmits<{
  regular: [value: RegularConnection];
  paired: [value: { payload: MobilePairingPayload; claim: PairingClaim }];
}>();
const invitation = ref("");
const mode = ref<ConnectionMode>("secure");
const address = ref("");
const apiKey = ref("");
const busy = ref(false);
const error = ref("");
async function pair() {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    if (!invitation.value.trim() && mode.value === "regular") {
      const connection = await connectRegularHost(
        address.value.trim(),
        apiKey.value.trim() || null,
      );
      apiKey.value = "";
      emit("regular", connection);
      return;
    }
    const payload = parseMobilePairingPayload(invitation.value.trim());
    if ((payload.mode ?? "secure") !== mode.value)
      throw new Error(
        `This is a ${payload.mode === "regular" ? "Regular" : "Secure"} pairing link. Select that connection type to continue.`,
      );
    const claim = await claimPairingSession(payload, {
      name: props.clientName,
      kind: props.clientKind,
    });
    invitation.value = "";
    emit("paired", { payload, claim });
  } catch (failure) {
    error.value =
      failure instanceof Error
        ? failure.message
        : "The machine could not be paired. Create a new link on the host and retry.";
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <form class="secure-pairing" @submit.prevent="pair">
    <label for="connection-mode">Connection</label>
    <select id="connection-mode" v-model="mode" :disabled="busy">
      <option value="secure">Secure · end-to-end encrypted</option>
      <option value="regular">Regular · direct HTTP or HTTPS</option>
    </select>
    <p v-if="mode === 'regular'">
      Connect directly over LAN, Tailscale, or WAN. HTTP relies on your network
      or VPN for encryption. The proxy requires Secure.
    </p>
    <p>
      On the machine you want to use, open Remote Access and copy a new pairing
      link. For a headless server, use <code>mold pairing create</code>.
    </p>
    <label for="secure-pairing-link">Pairing link</label>
    <textarea
      id="secure-pairing-link"
      v-model="invitation"
      rows="4"
      autocomplete="off"
      spellcheck="false"
      :disabled="busy"
      placeholder="Paste the pairing link or invitation JSON"
    />
    <template v-if="mode === 'regular'">
      <p>Or enter the machine's direct address and API key.</p>
      <label for="regular-address">Machine address</label>
      <input
        id="regular-address"
        v-model="address"
        type="url"
        placeholder="http://machine:7680"
        :disabled="busy"
      />
      <label for="regular-api-key">API key (if required)</label>
      <input
        id="regular-api-key"
        v-model="apiKey"
        type="password"
        autocomplete="off"
        :disabled="busy"
      />
    </template>
    <p v-if="mode === 'secure'">
      Your connection stays encrypted on the local network and through the
      proxy. Pairing links work once and expire shortly.
    </p>
    <p v-if="error" role="alert">{{ error }}</p>
    <button
      type="submit"
      :disabled="
        busy || (!invitation.trim() && (mode !== 'regular' || !address.trim()))
      "
    >
      {{
        busy
          ? "Connecting…"
          : mode === "secure"
            ? "Pair securely"
            : "Connect Regular"
      }}
    </button>
  </form>
</template>
<style scoped>
.secure-pairing {
  display: grid;
  gap: 12px;
}
.secure-pairing textarea,
.secure-pairing input,
.secure-pairing select {
  width: 100%;
  padding: 12px;
  color: inherit;
  background: var(--mold-surface);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
}
.secure-pairing button {
  min-height: 44px;
}
</style>
