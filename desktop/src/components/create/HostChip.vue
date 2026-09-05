<script setup lang="ts">
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import { normalizeTargetHost } from "../../lib/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useHostsStore, type HostView } from "../../stores/hosts";

/**
 * The generation-host chip (extracted from the Create header so the chain
 * composer shares it): a live status chip that, with more than one host,
 * opens the routing menu — Auto (least busy), Most capable ("capable"
 * sentinel), or a sticky host — writing the same persisted
 * `generateTargetHost` contract as always.
 */
const hosts = useHostsStore();
const prefs = useAppPrefsStore();

/** Current routing pick; a persisted host that's gone reads as Auto. */
const target = computed(
  () => normalizeTargetHost(prefs.settings?.generateTargetHost ?? null, hosts.all) ?? "auto",
);

const targetHost = computed(() =>
  target.value === "auto" || target.value === "capable"
    ? null
    : (hosts.all.find((h) => h.id === target.value) ?? null),
);

/** The chip names the pick, not just the primary: sticky host → its label. */
const chipHost = computed(() => targetHost.value ?? hosts.primaryHost ?? null);
const chipLabel = computed(() => {
  if (target.value === "auto" && hosts.multiHost) return "Auto";
  if (target.value === "capable") return "Most capable";
  return chipHost.value?.label ?? "This device";
});
const chipReady = computed(() =>
  target.value === "auto" || target.value === "capable"
    ? hosts.all.some((h) => h.status === "ready" && !h.stale)
    : chipHost.value?.status === "ready" && !chipHost.value.stale,
);
const chipStatus = computed(() => {
  if (target.value === "auto" || target.value === "capable") {
    if (hosts.all.some((host) => host.status === "ready" && host.stale))
      return chipReady.value ? "ready" : "reconnecting";
    return chipReady.value ? "ready" : "connecting";
  }
  const status = chipHost.value?.status;
  if (chipHost.value?.stale) return "reconnecting";
  if (status === "ready") return "ready";
  if (status === "error") return "offline";
  return status ?? "connecting";
});

function pick(id: string) {
  void prefs.update({ generateTargetHost: id === "auto" ? null : id });
  popoverOpen.value = false;
}

function hostLine(host: HostView): string {
  if (host.stale) return "reconnecting";
  if (host.status === "connecting") return "connecting";
  if (host.status !== "ready") return "offline";
  return host.queueDepth !== null ? `queue ${host.queueDepth}` : "ready";
}

/*
 * The menu is the SHARED Popover, which teleports its panel to <body> and
 * positions it from the trigger's viewport rect. Rendered in place it was an
 * absolutely positioned box inside `.ms-inspector__scroll` (`overflow-y:
 * auto`), so it contributed to that ancestor's scrollable area instead of
 * overlaying it: opening "Where it runs" — which sits near the bottom of the
 * Settings list — grew the inspector's scroll height and pushed the options
 * below the fold. Escape and outside-pointerdown dismissal come with it.
 */
const popoverOpen = ref(false);
function toggle() {
  if (hosts.multiHost) popoverOpen.value = !popoverOpen.value;
}
</script>

<template>
  <Popover
    v-model:open="popoverOpen"
    class="ms-hostchip"
    placement="bottom-end"
    label="Where it runs"
  >
    <template #trigger>
      <button
        type="button"
        data-test="host-chip"
        class="ms-hostchip__chip"
        :class="{ 'ms-hostchip__chip--button': hosts.multiHost }"
        :aria-expanded="hosts.multiHost ? popoverOpen : undefined"
        :aria-haspopup="hosts.multiHost ? 'menu' : undefined"
        :tabindex="hosts.multiHost ? 0 : -1"
        @click="toggle"
      >
        <span
          class="ms-hostchip__dot"
          :class="chipReady ? 'ms-hostchip__dot--ready' : 'ms-hostchip__dot--wait'"
        />
        {{ chipLabel }} · {{ chipStatus }}
        <Icon v-if="hosts.multiHost" name="chevron-down" :size="12" class="ms-hostchip__chev" />
      </button>
    </template>
    <div data-test="host-menu" role="menu" aria-label="Where it runs" class="ms-hostchip__menu">
      <div class="ms-hostchip__kicker font-mono text-xs">run on</div>
      <button
        type="button"
        role="menuitemradio"
        data-test="host-option-auto"
        class="ms-hostchip__row"
        :aria-checked="target === 'auto'"
        @click="pick('auto')"
      >
        <span class="ms-hostchip__row-label">Auto</span>
        <span class="ms-hostchip__row-sub font-mono text-xs">least busy</span>
        <span v-if="target === 'auto'" class="ms-hostchip__check">✓</span>
      </button>
      <button
        type="button"
        role="menuitemradio"
        data-test="host-option-capable"
        class="ms-hostchip__row"
        :aria-checked="target === 'capable'"
        @click="pick('capable')"
      >
        <span class="ms-hostchip__row-label">Most capable</span>
        <span class="ms-hostchip__row-sub font-mono text-xs">strongest gpu</span>
        <span v-if="target === 'capable'" class="ms-hostchip__check">✓</span>
      </button>
      <div class="ms-hostchip__rule" />
      <button
        v-for="h in hosts.all"
        :key="h.id"
        type="button"
        role="menuitemradio"
        :data-test="`host-option-${h.id}`"
        class="ms-hostchip__row"
        :disabled="h.status !== 'ready'"
        :aria-checked="target === h.id"
        @click="pick(h.id)"
      >
        <span
          class="ms-hostchip__dot"
          :class="
            h.status === 'ready' && !h.stale ? 'ms-hostchip__dot--ready' : 'ms-hostchip__dot--wait'
          "
        />
        <span class="ms-hostchip__row-label">{{ h.label }}</span>
        <span class="ms-hostchip__row-sub font-mono text-xs">{{ hostLine(h) }}</span>
        <span v-if="target === h.id" class="ms-hostchip__check">✓</span>
      </button>
    </div>
  </Popover>
</template>

<style scoped>
/* The class lands on the shared Popover's root. Only the row behaviour is
   ours — never `display` or `position`, which the Popover sets at the same
   specificity and would win or lose by chunk order. */
.ms-hostchip {
  flex-shrink: 0;
}
.ms-hostchip__chip {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  display: flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 0;
  padding: 4px 0;
}
.ms-hostchip__chip--button {
  cursor: pointer;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  padding: 4px 10px;
  transition:
    color var(--mold-dur-quick) var(--mold-ease-out),
    border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-hostchip__chip--button:hover {
  color: var(--mold-text);
  border-color: var(--mold-border-control);
}
.ms-hostchip__chev {
  color: var(--mold-text-dim);
}
.ms-hostchip__dot {
  width: 6px;
  height: 6px;
  flex: 0 0 6px;
  border-radius: 50%;
}
.ms-hostchip__dot--ready {
  background: var(--mold-success);
}
.ms-hostchip__dot--wait {
  background: var(--mold-text-dim);
}
/* The panel's ground, border, radius and shadow are the shared Popover's;
   this only sets how wide the routing list wants to be. */
.ms-hostchip__menu {
  width: 264px;
}
.ms-hostchip__kicker {
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
  padding: 4px 8px 6px;
}
.ms-hostchip__row {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  color: var(--mold-text);
  padding: 8px;
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  text-align: left;
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-hostchip__row:hover:not(:disabled) {
  background: var(--mold-surface);
}
.ms-hostchip__row:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.ms-hostchip__row-label {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ms-hostchip__row-sub {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-hostchip__check {
  color: var(--mold-blue);
  font-size: var(--mold-fs-xs);
}
.ms-hostchip__rule {
  height: 1px;
  background: var(--mold-border);
  margin: 4px 8px;
}
</style>
