<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { fetchLicenseListing } from "../api/licenseAcceptance";
import type { ApiTarget } from "../api/client";
import { useLicenseAcceptance } from "../composables/useLicenseAcceptance";
import {
  licenseRequirements,
  type LicenseRequirement,
  type ThirdPartyLicenseStatus,
} from "../lib/licenseAcceptance";

const props = defineProps<{
  target: ApiTarget | null;
  hostLabel: string;
  openExternal?: ((url: string) => void | Promise<void>) | undefined;
}>();
const rows = ref<ThirdPartyLicenseStatus[]>([]);
const loading = ref(false);
const message = ref<string | null>(null);
const loadError = ref<string | null>(null);
const prompt = useLicenseAcceptance();
const requirements = computed(() =>
  licenseRequirements(
    rows.value.flatMap((license) =>
      license.accepted
        ? []
        : license.required_by.map((installModel) => ({
            install_model: installModel,
            licenses: [license],
          })),
    ),
  ),
);
let loadEpoch = 0;

async function load() {
  const epoch = ++loadEpoch;
  const target = props.target
    ? { baseUrl: props.target.baseUrl, apiKey: props.target.apiKey }
    : null;
  if (!target) {
    rows.value = [];
    return;
  }
  loading.value = true;
  rows.value = [];
  loadError.value = null;
  message.value = null;
  try {
    const listing = await fetchLicenseListing(target);
    if (epoch !== loadEpoch) return;
    rows.value = Array.isArray(listing.licenses) ? listing.licenses : [];
    loadError.value = null;
    message.value = null;
  } catch {
    if (epoch !== loadEpoch) return;
    rows.value = [];
    loadError.value = `Could not read licenses from ${props.hostLabel}.`;
  } finally {
    if (epoch === loadEpoch) loading.value = false;
  }
}

/** `intent: "record"` accepts the terms and stops there; `"download"` also
 * fetches the bundle. Consent and acquisition are different acts, and a user
 * who only wants to agree should not be made to transfer gigabytes to do it. */
async function review(
  requirement: LicenseRequirement,
  intent: "download" | "record" = "download",
) {
  if (!props.target) return;
  const { accepted } = await prompt.request({
    hostLabel: props.hostLabel,
    target: props.target,
    requirements: [requirement],
    intent,
  });
  if (accepted) {
    message.value = `Required terms accepted on ${props.hostLabel}.`;
    await load();
  }
}

function openTerms(event: MouseEvent, url: string) {
  if (!props.openExternal) return;
  event.preventDefault();
  void props.openExternal(url);
}

onMounted(load);
watch(() => [props.target?.baseUrl, props.target?.apiKey], load);
</script>

<template>
  <div class="license-settings" data-test="license-settings">
    <p class="license-settings__lede">
      Review restricted model terms for {{ hostLabel }}. Acceptance is stored on
      {{ hostLabel }} only.
    </p>
    <p v-if="loading" class="license-settings__muted">Checking licenses…</p>
    <div
      v-if="loadError && !loading"
      class="license-settings__load-error"
      role="alert"
    >
      <span>{{ loadError }}</span>
      <button type="button" @click="load">Retry</button>
    </div>
    <p v-if="message" class="license-settings__success" role="status">
      {{ message }}
    </p>
    <div
      v-if="!loadError"
      v-for="license in rows"
      :key="license.id"
      class="license-settings__row"
    >
      <div>
        <strong>{{ license.name }}</strong>
        <p>{{ license.summary }}</p>
        <a
          :href="license.url"
          target="_blank"
          rel="noreferrer"
          @click="openTerms($event, license.url)"
          >Pinned terms</a
        >
        <span> · </span>
        <a
          :href="license.canonical"
          target="_blank"
          rel="noreferrer"
          @click="openTerms($event, license.canonical)"
          >Project terms</a
        >
      </div>
      <span v-if="license.accepted" class="license-settings__accepted"
        >Accepted</span
      >
      <span v-else class="license-settings__pending">Review required</span>
    </div>
    <div
      v-if="!loadError && requirements.length"
      class="license-settings__actions"
    >
      <div
        v-for="requirement in requirements"
        :key="requirement.installModel"
        class="license-settings__action-pair"
      >
        <button type="button" @click="review(requirement, 'record')">
          Accept terms for {{ requirement.installModel }}
        </button>
        <button type="button" @click="review(requirement, 'download')">
          Review terms and download {{ requirement.installModel }}
        </button>
      </div>
    </div>
    <p
      v-if="!loading && !loadError && rows.length === 0"
      class="license-settings__muted"
    >
      This host has no third-party model licenses.
    </p>
  </div>
</template>

<style scoped>
.license-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
  color: var(--mold-text-2);
  font-size: 13px;
}
.license-settings__lede,
.license-settings__muted,
.license-settings__row p {
  margin: 0;
  line-height: 1.5;
}
.license-settings__action-pair {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.license-settings__row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
}
.license-settings__row > div:first-child {
  flex: 1;
  min-width: 0;
}
.license-settings__row strong {
  color: var(--mold-text);
}
a {
  color: var(--mold-blue);
}
.license-settings__accepted {
  color: var(--mold-success);
  font: 600 11px var(--mold-font-mono);
  text-transform: uppercase;
}
.license-settings__pending {
  color: var(--danger);
  font: 600 11px var(--mold-font-mono);
  text-transform: uppercase;
}
.license-settings__actions {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.license-settings__load-error {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--danger);
}
button {
  min-height: 44px;
  border: 1px solid var(--mold-blue);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-blue);
  padding: 0 12px;
  font: 600 12px var(--mold-font-sans);
  cursor: pointer;
}
.license-settings__success {
  color: var(--mold-success);
}
@media (max-width: 600px) {
  .license-settings__row {
    align-items: stretch;
    flex-direction: column;
  }
}
</style>
