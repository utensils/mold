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

/** Every install bundle this licence still blocks, with the OTHER terms that
 * bundle needs: one dialog answers for everything the download requires, so a
 * user never accepts one term and is stopped again by its sibling. */
function requirementsFor(
  license: ThirdPartyLicenseStatus,
): LicenseRequirement[] {
  return requirements.value.filter((requirement) =>
    requirement.licenses.some((terms) => terms.id === license.id),
  );
}

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
 * who only wants to agree should not be made to transfer gigabytes to do it —
 * so this panel only ever records, and Styles owns the acquisition door. */
async function review(requirementsToReview: LicenseRequirement[]) {
  if (!props.target || requirementsToReview.length === 0) return;
  const { accepted } = await prompt.request({
    hostLabel: props.hostLabel,
    target: props.target,
    requirements: requirementsToReview,
    intent: "record",
  });
  if (accepted) {
    message.value = `Required terms accepted on ${props.hostLabel}.`;
    await load();
  }
}

function openTerms(url: string) {
  if (!props.openExternal) {
    window.open(url, "_blank", "noreferrer");
    return;
  }
  void props.openExternal(url);
}

/** A licence with nothing outstanding has no action — its two links are the
 * whole row. A pending one opens the ONE dialog its bundle needs. */
function act(license: ThirdPartyLicenseStatus) {
  const pending = requirementsFor(license);
  if (pending.length > 0) void review(pending);
}

onMounted(load);
watch(() => [props.target?.baseUrl, props.target?.apiKey], load);
</script>

<template>
  <div class="license-settings" data-test="license-settings">
    <div class="license-settings__row">
      <p class="license-settings__lede">
        Review restricted model terms for {{ hostLabel }}. Acceptance is stored
        on {{ hostLabel }} only.
      </p>
      <slot name="machine" />
    </div>
    <p v-if="loading" class="license-settings__row license-settings__note">
      Checking licenses…
    </p>
    <div
      v-if="loadError && !loading"
      class="license-settings__row license-settings__error"
      role="alert"
    >
      <span class="license-settings__lede">{{ loadError }}</span>
      <button class="ms-toolbar-button" type="button" @click="load">
        Retry
      </button>
    </div>
    <p
      v-if="message"
      class="license-settings__row license-settings__ok"
      role="status"
    >
      {{ message }}
    </p>
    <template v-if="!loadError">
      <div
        v-for="license in rows"
        :key="license.id"
        class="license-settings__row"
      >
        <span class="license-settings__id">{{ license.id }}</span>
        <span class="license-settings__name"
          >{{ license.name }} · {{ license.summary }}</span
        >
        <!-- Both the terms this build pinned and the project's current ones:
             an agreement nobody can read is not an agreement. The shell opens
             them, so the app is never navigated away from. -->
        <a
          class="license-settings__link"
          :href="license.url"
          target="_blank"
          rel="noreferrer"
          @click.prevent="openTerms(license.url)"
          >Pinned terms</a
        >
        <a
          class="license-settings__link"
          :href="license.canonical"
          target="_blank"
          rel="noreferrer"
          @click.prevent="openTerms(license.canonical)"
          >Project terms</a
        >
        <span
          class="license-settings__state"
          :class="
            license.accepted
              ? 'license-settings__state--ok'
              : 'license-settings__state--pending'
          "
          >{{ license.accepted ? "Accepted" : "Needs your OK" }}</span
        >
        <button
          v-if="requirementsFor(license).length > 0"
          class="ms-toolbar-button"
          type="button"
          @click="act(license)"
        >
          Read &amp; accept
        </button>
      </div>
    </template>
    <p
      v-if="!loading && !loadError && rows.length === 0"
      class="license-settings__row license-settings__note"
    >
      This host has no third-party model licenses.
    </p>
  </div>
</template>

<style scoped>
.license-settings {
  display: flex;
  flex-direction: column;
  color: var(--mold-text-2);
}
.license-settings__row {
  display: flex;
  align-items: center;
  /* The phone shares this panel; a narrow row wraps rather than needing its
     own breakpoint inside a settings list. */
  flex-wrap: wrap;
  gap: 14px;
  min-height: 52px;
  margin: 0;
  padding: 13px 14px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
}
.license-settings__row:last-child {
  border-bottom: 0;
}
.license-settings__lede {
  flex: 1;
  min-width: 0;
  margin: 0;
  font-size: var(--mold-fs-sm);
  line-height: var(--mold-lh-body);
}
.license-settings__id {
  flex-shrink: 0;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  overflow-wrap: anywhere;
}
.license-settings__name {
  flex: 1;
  min-width: 0;
  font-size: var(--mold-fs-sm);
  line-height: var(--mold-lh-body);
}
.license-settings__link {
  flex-shrink: 0;
  font-size: var(--mold-fs-xs);
  color: var(--mold-blue);
  text-decoration: none;
}
.license-settings__link:hover {
  text-decoration: underline;
}
.license-settings__state {
  flex-shrink: 0;
  font-size: var(--mold-fs-xs);
  font-weight: 600;
}
.license-settings__state--ok {
  color: var(--mold-success);
}
.license-settings__state--pending {
  color: var(--mold-warning);
}
.license-settings__note {
  font-size: var(--mold-fs-sm);
  line-height: var(--mold-lh-body);
}
.license-settings__ok {
  font-size: var(--mold-fs-sm);
  color: var(--mold-success);
}
.license-settings__error {
  color: var(--mold-error);
}
</style>
