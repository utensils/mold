<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import CardSurface from "@ui/components/CardSurface.vue";
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
  type ConfigValue,
} from "../lib/settingsConfig";

const rows = ref<ConfigRow[]>([]);
const profiles = ref<string[]>([]);
const activeProfile = ref("default");
const profileName = ref("");
const search = ref("");
const unavailable = ref(false);
const drafts = reactive<Record<string, ConfigValue>>({});
const saving = ref<string | null>(null);

async function load() {
  try {
    const [config, profileList] = await Promise.all([
      fetchConfig(),
      fetchProfiles(),
    ]);
    rows.value = config.filter((row) => !row.key.startsWith("tui."));
    profiles.value = profileList.profiles;
    activeProfile.value = profileList.active;
    for (const row of rows.value) drafts[row.key] = row.value;
    unavailable.value = false;
  } catch (error) {
    unavailable.value = true;
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

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
  })).filter((group) => group.rows.length > 0),
);

function typedDraft(row: ConfigRow): ConfigValue {
  const value = drafts[row.key];
  if (typeof row.value === "number") return Number(value);
  return value;
}

async function save(row: ConfigRow) {
  saving.value = row.key;
  try {
    await writeConfig(row.key, typedDraft(row));
    toast("success", `${schemaForRow(row).label} saved`);
    await load();
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  } finally {
    saving.value = null;
  }
}

async function reset(row: ConfigRow) {
  saving.value = row.key;
  try {
    await resetConfig(row.key);
    toast("success", `${schemaForRow(row).label} reset`);
    await load();
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  } finally {
    saving.value = null;
  }
}

async function selectProfile(name: string) {
  if (!name || name === activeProfile.value) return;
  await changeProfile(name);
}

async function createProfile() {
  const name = profileName.value.trim();
  if (!name) return;
  await changeProfile(name);
  profileName.value = "";
}

async function changeProfile(name: string) {
  try {
    await switchProfile(name);
    toast("success", `Profile switched to ${name}`);
    await load();
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}

onMounted(load);
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

    <CardSurface v-if="unavailable" class="config-card">
      This engine does not expose configuration controls. Upgrade or reconnect
      and try again.
    </CardSurface>

    <template v-else>
      <CardSurface class="config-card">
        <div class="profile-row">
          <label for="profile_select">Profile</label>
          <select
            id="profile_select"
            class="input"
            data-test="profile-select"
            :value="activeProfile"
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
            placeholder="New profile"
          />
          <button
            class="btn"
            data-test="profile-create"
            :disabled="!profileName.trim()"
            @click="createProfile"
          >
            Create & switch
          </button>
        </div>
        <p class="config-help">
          Profiles keep generation and expansion preferences separate.
        </p>
      </CardSurface>

      <section
        v-for="group in grouped"
        :key="group.section"
        class="config-group"
      >
        <h3>{{ group.section }}</h3>
        <CardSurface class="config-card" :padded="false">
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
            </div>
            <div class="config-editor">
              <input
                v-if="schema.editor === 'toggle'"
                :id="`config-${row.key}`"
                v-model="drafts[row.key]"
                :data-test="`config-${row.key}`"
                type="checkbox"
                :disabled="row.source === 'env'"
              />
              <select
                v-else-if="schema.editor === 'select'"
                :id="`config-${row.key}`"
                v-model="drafts[row.key]"
                class="input"
                :data-test="`config-${row.key}`"
                :disabled="row.source === 'env'"
              >
                <option
                  v-for="option in schema.options"
                  :key="option"
                  :value="option"
                >
                  {{ option || "auto" }}
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
                :disabled="row.source === 'env'"
              />
              <button
                class="btn"
                :data-test="`save-${row.key}`"
                :disabled="row.source === 'env' || saving === row.key"
                @click="save(row)"
              >
                Save
              </button>
              <button
                class="btn btn--ghost"
                :data-test="`reset-${row.key}`"
                :disabled="
                  row.source === 'env' ||
                  !canResetConfig(row.key) ||
                  saving === row.key
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
        </CardSurface>
      </section>
      <p v-if="search && grouped.length === 0" class="empty">
        No settings match “{{ search }}”.
      </p>
    </template>
  </section>
</template>

<style scoped>
.config-panel {
  margin: 32px 0 24px;
}
.config-panel__heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
}
.config-panel__heading h2 {
  margin: 2px 0 0;
  color: var(--rebate);
  font-family: var(--f-display);
  font-size: 19px;
}
.kicker {
  margin: 0;
  font: 10px var(--f-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.config-search {
  width: 220px;
}
.config-card {
  margin-bottom: 18px;
}
.config-group h3 {
  margin: 0 0 8px;
  color: var(--ink-2);
  font: 600 12px var(--f-body);
}
.config-row {
  display: grid;
  grid-template-columns: minmax(180px, 1fr) minmax(250px, auto);
  gap: 16px;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid var(--edge);
}
.config-row:last-child {
  border-bottom: 0;
}
.config-copy label {
  color: var(--rebate);
  font-size: 13px;
  font-weight: 600;
}
.config-copy p,
.config-help {
  margin: 4px 0;
  color: var(--ink-3);
  font-size: 11.5px;
  line-height: 1.4;
}
.source,
.env-lock {
  margin-right: 7px;
  color: var(--ink-3);
  font: 10px var(--f-mono);
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
  gap: 8px;
}
.profile-row {
  justify-content: flex-start;
  flex-wrap: wrap;
}
.input {
  min-height: 44px;
  padding: 0 11px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: var(--bath);
  color: var(--rebate);
  font: 12px var(--f-mono);
}
.btn {
  min-height: 44px;
  padding: 0 12px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--rebate);
  font: 600 12px var(--f-body);
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
@media (max-width: 640px) {
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
    flex: 1 1 150px;
    min-width: 0;
  }
}
</style>
