<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import CardSurface from "@ui/components/CardSurface.vue";
import Icon from "@ui/components/Icon.vue";
import type { IconName } from "@ui/icons";
import { toast } from "../lib/toasts";
import {
  CONFIG_SECTIONS,
  canResetConfig,
  fetchConfig,
  fetchProfiles,
  matchesConfigSearch,
  resetConfig,
  schemaForRow,
  switchProfile,
  writeConfig,
  type ConfigRow,
  type ConfigSchema,
  type ConfigValue,
} from "../lib/settingsConfig";
import { AUTO_TAG_SETTING_WEB } from "@studio/lib/fileUnder";
import { autoTagTitle } from "../lib/fileUnder";

/**
 * Create's "File under" auto-tag default. Unlike every other row here it is
 * a BROWSER preference, not engine config — the ghost chip is offered by
 * this client, so it is stored in `localStorage` and there is nothing to
 * save or reset on the host. It sits beside trash retention because both
 * answer "how does this machine's Library behave".
 */
const AUTO_TAG_SCHEMA: ConfigSchema = {
  key: AUTO_TAG_SETTING_WEB,
  section: "Library",
  label: "Tag new prints with their title",
  help: "Create offers each titled print its own tag as a removable chip. Off turns the ghost chip off for future prints — it never rewrites existing ones.",
  editor: "toggle",
};

const rows = ref<ConfigRow[]>([]);
const profiles = ref<string[]>([]);
const activeProfile = ref("default");
const profileName = ref("");
const search = ref("");
const unavailableMessage = ref("");
const drafts = reactive<Record<string, ConfigValue>>({});
const saving = ref<string | null>(null);
const loading = ref(false);
const loaded = ref(false);
const actionError = ref("");
const profilesError = ref("");
let loadSequence = 0;
let alive = true;
let pendingProfileReset = false;
const busy = computed(() => loading.value || saving.value !== null);

const sectionPresentation: Record<
  (typeof CONFIG_SECTIONS)[number],
  { icon: IconName; summary: string }
> = {
  "Storage & server": {
    icon: "machines",
    summary: "Paths and connection defaults for this engine",
  },
  Generation: {
    icon: "image",
    summary: "Defaults for new image and video jobs",
  },
  Library: {
    icon: "library",
    summary: "Trash retention for prints on this host",
  },
  "Prompt expansion": {
    icon: "sparkle",
    summary: "Rewrite behavior, model, and sampling controls",
  },
  Advanced: {
    icon: "settings",
    summary: "Uncommon and newly discovered engine options",
  },
};

async function load(replaceDraft: string | true | null = null) {
  const sequence = ++loadSequence;
  const current = () => alive && sequence === loadSequence;
  loading.value = true;
  try {
    const config = await fetchConfig();
    if (!current()) return;
    const previous = new Map(rows.value.map((row) => [row.key, row.value]));
    rows.value = config.filter((row) => !row.key.startsWith("tui."));
    for (const row of rows.value) {
      if (
        pendingProfileReset ||
        replaceDraft === true ||
        replaceDraft === row.key ||
        !(row.key in drafts) ||
        drafts[row.key] === previous.get(row.key)
      )
        drafts[row.key] = row.value;
    }
    unavailableMessage.value = "";
    pendingProfileReset = false;
    loaded.value = true;
  } catch (error) {
    if (current())
      unavailableMessage.value =
        error instanceof Error ? error.message : String(error);
    return;
  } finally {
    if (current()) loading.value = false;
  }
  try {
    const profileList = await fetchProfiles();
    if (!current()) return;
    profiles.value = profileList.profiles;
    activeProfile.value = profileList.active;
    profilesError.value = "";
  } catch {
    if (current())
      profilesError.value =
        "Could not refresh profiles. Existing settings remain available.";
  }
}

const showAutoTagRow = computed(() =>
  matchesConfigSearch(search.value, AUTO_TAG_SCHEMA),
);

const grouped = computed(() =>
  CONFIG_SECTIONS.map((section) => ({
    section,
    rows: rows.value
      .map((row) => ({ row, schema: schemaForRow(row) }))
      .filter(
        ({ schema }) =>
          schema.section === section &&
          matchesConfigSearch(search.value, schema),
      ),
  })).filter(
    (group) =>
      group.rows.length > 0 ||
      // Library keeps its heading for the browser-local row alone: a host
      // with no trash capability still offers the auto-tag default.
      (group.section === "Library" && showAutoTagRow.value),
  ),
);

function typedDraft(row: ConfigRow): ConfigValue {
  const value = drafts[row.key];
  const schema = schemaForRow(row);
  if (schema.editor === "number" || schema.valueType === "number") {
    if (value === "" || value === null) {
      throw new Error(`${schemaForRow(row).label} requires a number`);
    }
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      throw new Error(`${schemaForRow(row).label} requires a valid number`);
    }
    return parsed;
  }
  return value;
}

async function save(row: ConfigRow) {
  if (busy.value) return;
  actionError.value = "";
  saving.value = row.key;
  try {
    await writeConfig(row.key, typedDraft(row));
    toast("success", `${schemaForRow(row).label} saved`);
    if (alive) await load(row.key);
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : String(error);
  } finally {
    saving.value = null;
  }
}

async function reset(row: ConfigRow) {
  if (busy.value) return;
  actionError.value = "";
  saving.value = row.key;
  try {
    await resetConfig(row.key);
    toast("success", `${schemaForRow(row).label} reset`);
    if (alive) await load(row.key);
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : String(error);
  } finally {
    saving.value = null;
  }
}

async function selectProfile(name: string) {
  if (!name || name === activeProfile.value) return;
  await changeProfile(name);
}

async function createProfile() {
  if (busy.value) return;
  const name = profileName.value.trim();
  if (!name) return;
  if (await changeProfile(name)) profileName.value = "";
}

async function changeProfile(name: string) {
  if (busy.value) return false;
  saving.value = "profile";
  actionError.value = "";
  try {
    await switchProfile(name);
    if (!alive) return;
    activeProfile.value = name;
    pendingProfileReset = true;
    loaded.value = false;
    toast("success", `Profile switched to ${name}`);
    await load(true);
    return loaded.value;
  } catch (error) {
    actionError.value = error instanceof Error ? error.message : String(error);
    return false;
  } finally {
    saving.value = null;
  }
}

onMounted(() => load());
onBeforeUnmount(() => {
  alive = false;
  ++loadSequence;
});
</script>

<template>
  <section class="config-panel" aria-labelledby="engine-settings-title">
    <div class="config-panel__heading">
      <div>
        <p class="kicker">Engine configuration</p>
        <h2 id="engine-settings-title">All settings</h2>
      </div>
      <input
        v-model="search"
        class="input config-search"
        data-test="config-search"
        type="search"
        placeholder="Search settings"
        aria-label="Search settings"
      />
    </div>

    <CardSurface
      v-if="unavailableMessage"
      class="config-card"
      role="alert"
      data-test="config-load-error"
    >
      <p>Could not refresh configuration. {{ unavailableMessage }}</p>
      <button
        type="button"
        class="btn"
        :disabled="busy"
        data-test="config-retry"
        @click="load()"
      >
        Retry
      </button>
    </CardSurface>
    <p v-if="loading && !loaded" role="status">Loading configuration…</p>
    <p v-if="actionError" role="alert" data-test="config-action-error">
      {{ actionError }}
    </p>
    <template v-if="loaded">
      <CardSurface class="config-card">
        <div class="profile-row">
          <label for="profile_select">Profile</label>
          <select
            id="profile_select"
            class="input"
            data-test="profile-select"
            :value="activeProfile"
            :disabled="busy"
            @change="selectProfile(($event.target as HTMLSelectElement).value)"
          >
            <option v-for="profile in profiles" :key="profile" :value="profile">
              {{ profile }}
            </option>
          </select>
          <input
            v-model="profileName"
            class="input"
            data-test="profile-name"
            aria-label="New profile name"
            :disabled="busy"
            placeholder="New profile"
          />
          <button
            class="btn"
            data-test="profile-create"
            :disabled="busy || !profileName.trim()"
            @click="createProfile"
          >
            Create & switch
          </button>
        </div>
        <p v-if="profilesError" role="status" class="config-help">
          {{ profilesError }}
          <button type="button" class="btn" :disabled="busy" @click="load()">
            Retry profiles
          </button>
        </p>
        <p class="config-help">
          Profiles keep generation and expansion preferences separate.
        </p>
      </CardSurface>

      <section
        v-for="group in grouped"
        :key="group.section"
        class="config-group"
        data-test="config-group"
      >
        <div class="config-group__heading">
          <span class="config-group__plate" aria-hidden="true">
            <Icon :name="sectionPresentation[group.section].icon" :size="17" />
          </span>
          <div>
            <h3>{{ group.section }}</h3>
            <p class="config-group__summary">
              {{ sectionPresentation[group.section].summary }}
            </p>
          </div>
        </div>
        <CardSurface class="config-card config-card--accented" :padded="false">
          <div
            v-for="{ row, schema } in group.rows"
            :key="row.key"
            class="config-row"
          >
            <div class="config-copy">
              <label :for="`config-${row.key}`">{{ schema.label }}</label>
              <p>{{ schema.help }}</p>
              <span class="source" :data-test="`source-${row.key}`">{{
                row.source
              }}</span>
              <span v-if="row.env_var" class="env-lock"
                >Set by {{ row.env_var }}</span
              >
              <span
                v-if="schema.needsEngineRestart || row.restart_required"
                class="env-lock"
                :data-test="`restart-${row.key}`"
                >Restart server to apply</span
              >
            </div>
            <div class="config-editor">
              <label
                v-if="schema.editor === 'toggle'"
                class="config-toggle-target"
              >
                <input
                  :id="`config-${row.key}`"
                  v-model="drafts[row.key]"
                  :data-test="`config-${row.key}`"
                  type="checkbox"
                  :disabled="
                    row.source === 'env' ||
                    schema.liveReadOnly ||
                    saving === row.key ||
                    saving === 'profile'
                  "
                />
              </label>
              <select
                v-else-if="schema.editor === 'select'"
                :id="`config-${row.key}`"
                v-model="drafts[row.key]"
                class="input"
                :data-test="`config-${row.key}`"
                :disabled="
                  row.source === 'env' ||
                  schema.liveReadOnly ||
                  saving === row.key ||
                  saving === 'profile'
                "
              >
                <option
                  v-for="option in schema.options"
                  :key="option"
                  :value="option"
                >
                  {{ schema.optionLabels?.[option] ?? (option || "auto") }}
                </option>
              </select>
              <input
                v-else
                :id="`config-${row.key}`"
                v-model="drafts[row.key]"
                class="input"
                :data-test="`config-${row.key}`"
                :type="schema.editor === 'number' ? 'number' : 'text'"
                :min="schema.min"
                :max="schema.max"
                :step="schema.step"
                :disabled="
                  row.source === 'env' ||
                  schema.liveReadOnly ||
                  saving === row.key ||
                  saving === 'profile'
                "
              />
              <button
                class="btn"
                :data-test="`save-${row.key}`"
                :disabled="row.source === 'env' || schema.liveReadOnly || busy"
                @click="save(row)"
              >
                Save
              </button>
              <button
                class="btn btn--ghost"
                :data-test="`reset-${row.key}`"
                :disabled="
                  row.source === 'env' ||
                  schema.liveReadOnly ||
                  !canResetConfig(row.key) ||
                  busy
                "
                :title="
                  canResetConfig(row.key)
                    ? 'Reset to its fallback value'
                    : 'Bootstrap keys are edited directly and have no reset'
                "
                @click="reset(row)"
              >
                Reset
              </button>
            </div>
          </div>
          <!-- Browser preference, stored in this browser. No Save / Reset:
               the toggle IS the commit. -->
          <div
            v-if="group.section === 'Library' && showAutoTagRow"
            class="config-row"
            data-test="config-row-auto-tag-title"
          >
            <div class="config-copy">
              <label :for="`config-${AUTO_TAG_SCHEMA.key}`">{{
                AUTO_TAG_SCHEMA.label
              }}</label>
              <p>{{ AUTO_TAG_SCHEMA.help }}</p>
              <span class="source" data-test="source-auto-tag-title"
                >this browser</span
              >
            </div>
            <div class="config-editor">
              <label class="config-toggle-target">
                <input
                  :id="`config-${AUTO_TAG_SCHEMA.key}`"
                  v-model="autoTagTitle"
                  data-test="config-auto-tag-title"
                  type="checkbox"
                />
              </label>
            </div>
          </div>
        </CardSurface>
      </section>
      <p v-if="search && grouped.length === 0" class="empty">
        No settings match “{{ search }}”.
      </p>
    </template>
  </section>
</template>

<style scoped>
.config-toggle-target {
  width: 44px;
  min-height: 44px;
  display: grid;
  place-items: center;
  cursor: pointer;
}
.config-panel {
  margin: 32px 0 24px;
}
.config-panel__heading {
  display: flex;
  align-items: end;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}
.config-panel__heading h2 {
  margin: 2px 0 0;
  color: var(--rebate);
  font-family: var(--f-display);
  font-size: 1.1875rem;
}
.kicker {
  margin: 0;
  font: 0.875rem var(--f-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.config-search {
  width: 220px;
  max-width: 100%;
}
.config-card {
  margin-bottom: 18px;
}
.config-group__heading {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 9px;
}
.config-group__heading h3 {
  margin: 0;
  color: var(--rebate);
  font: 600 0.875rem var(--f-body);
}
.config-group__plate {
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-control);
  background: color-mix(in srgb, var(--halide) 16%, transparent);
  color: var(--halide);
}
.config-group__summary {
  margin: 2px 0 0;
  color: var(--ink-3);
  font-size: 0.875rem;
}
.config-card--accented {
  background: color-mix(in srgb, var(--halide) 3%, var(--bench));
  border-color: color-mix(in srgb, var(--halide) 22%, var(--edge));
}
.config-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--edge);
}
.config-row:last-child {
  border-bottom: 0;
}
.config-copy {
  min-width: 0;
  overflow-wrap: anywhere;
}
.config-copy label {
  color: var(--rebate);
  font-size: 0.875rem;
  font-weight: 600;
}
.config-copy p,
.config-help {
  margin: 4px 0;
  color: var(--ink-3);
  font-size: 0.875rem;
  line-height: 1.4;
}
.source,
.env-lock {
  margin-right: 7px;
  color: var(--ink-3);
  font: 0.875rem var(--f-mono);
}
.source {
  padding: 2px 5px;
  border: 1px solid var(--edge);
  border-radius: 4px;
}
.config-editor,
.profile-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 8px;
}
.profile-row {
  justify-content: flex-start;
  flex-wrap: wrap;
}
.input {
  min-width: 0;
  max-width: 100%;
  box-sizing: border-box;
  min-height: 44px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font: 0.875rem var(--f-mono);
}
.btn {
  min-height: 44px;
  padding: 0 12px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  font: 600 0.875rem var(--f-body);
  cursor: pointer;
}
.btn--ghost {
  color: var(--ink-3);
}
.btn:disabled,
.input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.empty {
  color: var(--ink-3);
  text-align: center;
}
@media (max-width: 900px) {
  .config-panel__heading,
  .config-row {
    display: flex;
    align-items: stretch;
    flex-direction: column;
  }
  .config-search {
    width: 100%;
  }
  .config-editor {
    justify-content: flex-start;
    flex-wrap: wrap;
  }
  .config-editor .input {
    min-width: 0;
    max-width: 100%;
    box-sizing: border-box;
    flex: 1 1 150px;
    min-width: 0;
  }
}
</style>
