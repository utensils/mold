<script setup lang="ts">
/*
 * Settings workspace (spec §04/§06, prototype desktop Settings). Appearance
 * (theme family + dark/light on the shared lib/theme refs), account tokens
 * (browser-stored, forwarded per catalog request — see accountsStore), and an
 * About card sourced from GET /api/status. Set-once prefs only; per-generation
 * knobs stay in Create's Advanced.
 */
import { computed, ref } from "vue";
import CardSurface from "@ui/components/CardSurface.vue";
import SegmentedControl, {
  type SegmentOption,
} from "@ui/components/SegmentedControl.vue";
import { theme, themeFamily, type Theme, type ThemeFamily } from "../lib/theme";
import { toast } from "../lib/toasts";
import { useStatusPoll } from "../composables/useStatusPoll";
import {
  clearCivitaiToken,
  clearHfToken,
  getAccountTokens,
  maskToken,
  setCivitaiToken,
  setHfToken,
} from "../lib/accountsStore";

const familyOptions: SegmentOption<ThemeFamily>[] = [
  { value: "mold", label: "Mold" },
  { value: "safelight", label: "Safelight" },
];
const appearanceOptions: SegmentOption<Theme>[] = [
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];

function setFamily(value: ThemeFamily) {
  themeFamily.value = value;
}
function setAppearance(value: Theme) {
  theme.value = value;
}

// ── Account tokens (browser-stored, forwarded per catalog request) ──────
// The saved (masked) state, reflected on load and after every save/clear.
const saved = ref(getAccountTokens());
const hfMasked = computed(() => maskToken(saved.value.hfToken));
const civitaiMasked = computed(() => maskToken(saved.value.civitaiToken));

// Draft inputs, only shown while entering/replacing a token.
const hfDraft = ref("");
const civitaiDraft = ref("");
const editingHf = ref(false);
const editingCivitai = ref(false);

function saveHf() {
  const value = hfDraft.value.trim();
  if (!value) return;
  setHfToken(value);
  saved.value = getAccountTokens();
  hfDraft.value = "";
  editingHf.value = false;
  toast("success", "Hugging Face token saved in this browser");
}
function removeHf() {
  clearHfToken();
  saved.value = getAccountTokens();
  hfDraft.value = "";
  editingHf.value = false;
}
function saveCivitai() {
  const value = civitaiDraft.value.trim();
  if (!value) return;
  setCivitaiToken(value);
  saved.value = getAccountTokens();
  civitaiDraft.value = "";
  editingCivitai.value = false;
  toast("success", "Civitai token saved in this browser");
}
function removeCivitai() {
  clearCivitaiToken();
  saved.value = getAccountTokens();
  civitaiDraft.value = "";
  editingCivitai.value = false;
}

// About — server version + processing summary.
const { status } = useStatusPoll();
const version = computed(() => status.value?.version ?? "—");
</script>

<template>
  <div class="settings">
    <h1 class="settings__title">Settings</h1>

    <p class="kicker">Appearance</p>
    <CardSurface class="settings__card">
      <div class="row">
        <span class="row__label">Theme</span>
        <SegmentedControl
          data-test="seg-theme"
          :model-value="themeFamily"
          :options="familyOptions"
          label="Theme family"
          @update:model-value="setFamily"
        />
      </div>
      <div class="row">
        <span class="row__label">Appearance</span>
        <SegmentedControl
          data-test="seg-appearance"
          :model-value="theme"
          :options="appearanceOptions"
          label="Appearance"
          @update:model-value="setAppearance"
        />
      </div>
    </CardSurface>

    <p class="kicker">Accounts</p>
    <CardSurface class="settings__card">
      <!-- Hugging Face -->
      <div class="row">
        <label class="row__label" for="hf_token">Hugging Face token</label>
        <div v-if="hfMasked && !editingHf" class="field" data-test="hf-saved">
          <code class="token-mask" data-test="hf-mask">{{ hfMasked }}</code>
          <button
            type="button"
            class="btn"
            data-test="replace-hf"
            @click="editingHf = true"
          >
            Replace
          </button>
          <button
            type="button"
            class="btn btn--ghost"
            data-test="clear-hf"
            @click="removeHf"
          >
            Clear
          </button>
        </div>
        <div v-else class="field">
          <input
            id="hf_token"
            v-model="hfDraft"
            name="hf_token"
            type="password"
            placeholder="hf_…"
            autocomplete="off"
            class="input"
          />
          <button
            type="button"
            class="btn"
            data-test="save-hf"
            :disabled="!hfDraft.trim()"
            @click="saveHf"
          >
            Save
          </button>
          <button
            v-if="editingHf"
            type="button"
            class="btn btn--ghost"
            @click="editingHf = false"
          >
            Cancel
          </button>
        </div>
      </div>

      <!-- Civitai -->
      <div class="row">
        <label class="row__label" for="civitai_token">Civitai token</label>
        <div
          v-if="civitaiMasked && !editingCivitai"
          class="field"
          data-test="civitai-saved"
        >
          <code class="token-mask" data-test="civitai-mask">{{
            civitaiMasked
          }}</code>
          <button
            type="button"
            class="btn"
            data-test="replace-civitai"
            @click="editingCivitai = true"
          >
            Replace
          </button>
          <button
            type="button"
            class="btn btn--ghost"
            data-test="clear-civitai"
            @click="removeCivitai"
          >
            Clear
          </button>
        </div>
        <div v-else class="field">
          <input
            id="civitai_token"
            v-model="civitaiDraft"
            name="civitai_token"
            type="password"
            placeholder="cv_…"
            autocomplete="off"
            class="input"
          />
          <button
            type="button"
            class="btn"
            data-test="save-civitai"
            :disabled="!civitaiDraft.trim()"
            @click="saveCivitai"
          >
            Save
          </button>
          <button
            v-if="editingCivitai"
            type="button"
            class="btn btn--ghost"
            @click="editingCivitai = false"
          >
            Cancel
          </button>
        </div>
      </div>

      <p class="settings__note">
        Tokens are stored in this browser only and sent solely with catalog
        (model discovery) requests — never placed in a URL.
      </p>
    </CardSurface>

    <p class="kicker">About</p>
    <CardSurface class="settings__card settings__card--list" :padded="false">
      <div class="about-row">
        <span class="about-row__key">Version</span>
        <span class="about-row__val" data-test="about-version">{{
          version
        }}</span>
      </div>
      <div class="about-row about-row--last">
        <span class="about-row__key">Processing</span>
        <span class="about-row__val about-row__val--muted">
          local + your hosts
        </span>
      </div>
    </CardSurface>
  </div>
</template>

<style scoped>
.settings {
  max-width: 600px;
  margin: 0 auto;
  padding: 24px 20px 96px;
}

.settings__title {
  margin: 0 0 22px;
  font-family: var(--f-display);
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.kicker {
  margin: 0 0 12px;
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.settings__card {
  margin-bottom: 24px;
}

.row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.row + .row {
  margin-top: 14px;
}

.row__label {
  font-size: 13.5px;
  color: var(--rebate);
}

.field {
  display: flex;
  align-items: center;
  gap: 8px;
}

.input {
  width: 240px;
  max-width: 52vw;
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 12px;
  outline: none;
}

.input:focus-visible,
.select:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 1px;
}

.select {
  height: 34px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 12px;
  outline: none;
}

.btn {
  height: 34px;
  padding: 0 14px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}

.btn:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
}

.btn:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn--ghost {
  color: var(--ink-3);
}
.btn--ghost:hover {
  color: var(--stop);
  background: transparent;
}

.token-mask {
  flex: 1;
  min-width: 0;
  height: 34px;
  display: inline-flex;
  align-items: center;
  padding: 0 12px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--ink-2);
  font-family: var(--f-mono);
  font-size: 12.5px;
  letter-spacing: 0.04em;
}

.settings__note {
  margin: 12px 0 0;
  font-size: 11.5px;
  color: var(--ink-3);
  line-height: 1.5;
}

/* About list — rows inside an unpadded card. */
.about-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 11px 16px;
  border-bottom: 1px solid var(--edge);
  font-size: 13px;
}

.about-row--last {
  border-bottom: 0;
}

.about-row__key {
  color: var(--ink-2);
}

.about-row__val {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--rebate);
}

.about-row__val--muted {
  color: var(--ink-3);
}
</style>
