<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import QRCode from "qrcode";
import type { ApiTarget } from "../api/client";
import {
  createPairingSession,
  mobilePairingUrl,
  type MobilePairingPayload,
  type PairingSession,
} from "../api/pairing";

const props = defineProps<{
  target: ApiTarget | null;
  suggestedBaseUrl: string;
  hostLabel?: string | undefined;
}>();
const emit = defineEmits<{ sessionCreated: [] }>();

const address = ref(props.suggestedBaseUrl);
const session = ref<PairingSession | null>(null);
const qrDataUrl = ref("");
const error = ref("");
const busy = ref(false);
const now = ref(Math.floor(Date.now() / 1000));
let timer: ReturnType<typeof setInterval> | null = null;

watch(
  () => props.suggestedBaseUrl,
  (next, previous) => {
    if (!session.value && (!address.value || address.value === previous))
      address.value = next;
  },
);

const secondsLeft = computed(() =>
  session.value?.expires_at
    ? Math.max(0, session.value.expires_at - now.value)
    : null,
);
const expired = computed(() => secondsLeft.value === 0);
const buttonLabel = computed(() => {
  if (busy.value) return "Creating secure code…";
  if (session.value) return "New pairing code";
  return "Create pairing code";
});

function reachableBaseUrl(input: string, hostname: string | null): string {
  const candidate = input.trim();
  let url: URL;
  try {
    url = new URL(candidate);
  } catch {
    throw new Error(
      "Enter the http:// or https:// address your mobile device uses for this host.",
    );
  }
  if (!["http:", "https:"].includes(url.protocol))
    throw new Error("Pairing requires an HTTP or HTTPS host address.");
  if (
    ["127.0.0.1", "localhost", "::1", "[::1]"].includes(
      url.hostname.toLowerCase(),
    )
  ) {
    if (!hostname)
      throw new Error(
        "Use this Mac's LAN, MagicDNS, or HTTPS address—not localhost.",
      );
    url.hostname = hostname.includes(".") ? hostname : `${hostname}.local`;
  }
  url.pathname = "";
  url.search = "";
  url.hash = "";
  return url.origin;
}

async function startPairing(): Promise<void> {
  if (!props.target || busy.value) return;
  busy.value = true;
  error.value = "";
  qrDataUrl.value = "";
  try {
    const next = await createPairingSession(props.target);
    const baseUrl = reachableBaseUrl(
      address.value || props.suggestedBaseUrl,
      next.hostname,
    );
    address.value = baseUrl;
    const payload: MobilePairingPayload = {
      type: "mold.mobile-pairing",
      version: 1,
      base_url: baseUrl,
      token: next.token,
      expires_at: next.expires_at,
      instance_id: next.instance_id,
      name: props.hostLabel || next.hostname || new URL(baseUrl).hostname,
    };
    qrDataUrl.value = await QRCode.toDataURL(mobilePairingUrl(payload), {
      width: 320,
      margin: 2,
      errorCorrectionLevel: "M",
      color: { dark: "#111111", light: "#ffffff" },
    });
    session.value = next;
    emit("sessionCreated");
    now.value = Math.floor(Date.now() / 1000);
    if (timer) clearInterval(timer);
    timer = setInterval(
      () => (now.value = Math.floor(Date.now() / 1000)),
      1_000,
    );
  } catch (reason) {
    session.value = null;
    error.value = reason instanceof Error ? reason.message : String(reason);
  } finally {
    busy.value = false;
  }
}

onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
</script>

<template>
  <div class="pairing-card" data-test="mobile-pairing-card">
    <div class="pairing-copy">
      <div class="pairing-heading">
        <span class="pairing-icon" aria-hidden="true">▦</span>
        <div>
          <h3>Mobile pairing</h3>
          <p>
            Scan once to add this host. The code expires quickly and never
            contains your API key.
          </p>
        </div>
      </div>
      <label class="pairing-address">
        <span>Address your mobile device can reach</span>
        <input
          v-model="address"
          data-selectable
          autocomplete="url"
          autocapitalize="none"
          spellcheck="false"
        />
      </label>
      <p class="pairing-hint">
        Use a LAN address, <code>.local</code> name, MagicDNS name, or HTTPS
        URL.
      </p>
      <p v-if="error" class="pairing-error" role="alert">{{ error }}</p>
      <button
        type="button"
        class="pairing-button"
        :disabled="!target || busy"
        @click="startPairing"
      >
        {{ buttonLabel }}
      </button>
    </div>
    <div
      v-if="qrDataUrl"
      class="pairing-code"
      :class="{ 'pairing-code--expired': expired }"
    >
      <img :src="qrDataUrl" alt="Mold mobile pairing QR code" />
      <div class="pairing-status">
        <span v-if="session?.auth_required === false">No key required</span>
        <span v-else-if="expired">Expired</span>
        <span v-else-if="secondsLeft !== null"
          >One use · {{ secondsLeft }}s</span
        >
        <span v-else>One use</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.pairing-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: center;
  padding: 20px;
  border: 1px solid var(--color-edge, rgba(128, 128, 128, 0.25));
  border-radius: 14px;
  background: var(--color-bench, rgba(128, 128, 128, 0.06));
}
.pairing-copy {
  min-width: 0;
}
.pairing-heading {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.pairing-icon {
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  flex: 0 0 auto;
  border-radius: 10px;
  background: color-mix(in srgb, var(--color-accent, #de6d2f) 18%, transparent);
  color: var(--color-accent, #de6d2f);
  font-size: 22px;
}
h3 {
  margin: 0;
  color: var(--color-ink, inherit);
  font-size: 15px;
  font-weight: 650;
}
p {
  margin: 4px 0 0;
  color: var(--color-ink-3, #888);
  font-size: 12px;
  line-height: 1.45;
}
.pairing-address {
  display: grid;
  gap: 6px;
  margin-top: 16px;
  color: var(--color-ink-2, inherit);
  font-size: 11px;
  font-weight: 600;
}
.pairing-address input {
  min-width: 0;
  height: 36px;
  padding: 0 10px;
  border: 1px solid var(--color-edge, #555);
  border-radius: 8px;
  background: var(--color-surface, transparent);
  color: var(--color-ink, inherit);
  font:
    500 12px ui-monospace,
    SFMono-Regular,
    monospace;
}
.pairing-hint code {
  color: var(--color-ink-2, inherit);
}
.pairing-error {
  color: var(--color-danger, #d44);
}
.pairing-button {
  min-height: 36px;
  margin-top: 14px;
  padding: 0 14px;
  border: 0;
  border-radius: 8px;
  background: var(--color-accent, #de6d2f);
  color: white;
  font-weight: 650;
  cursor: pointer;
}
.pairing-button:disabled {
  opacity: 0.5;
  cursor: default;
}
.pairing-code {
  width: 164px;
  padding: 9px;
  border-radius: 12px;
  background: white;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);
  transition: opacity 0.2s;
}
.pairing-code img {
  display: block;
  width: 100%;
  height: auto;
}
.pairing-code--expired img {
  opacity: 0.18;
}
.pairing-status {
  margin-top: 5px;
  white-space: nowrap;
  color: #333;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-align: center;
  text-transform: uppercase;
}
@media (max-width: 560px) {
  .pairing-card {
    grid-template-columns: 1fr;
  }
  .pairing-code {
    justify-self: center;
  }
}
</style>
