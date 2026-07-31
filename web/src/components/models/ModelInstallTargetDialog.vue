<script setup lang="ts">
/*
 * Machine picker for a model install. Shown only when the install plan is
 * genuinely ambiguous — more than one connected machine could take the model —
 * and it names what happens on each one, because "Install" and "Repair" are
 * different outcomes for the same click.
 *
 * ModalPanel is `position: absolute; inset: 0`, so it needs a viewport-fixed
 * host on this long scrolling page (same reason the detail drawer has one). It
 * sits above that drawer, which the picker can be opened from.
 */
import ModalPanel from "@ui/components/ModalPanel.vue";
import Icon from "@ui/components/Icon.vue";
import { useModelInstallTargets } from "../../composables/useModelInstallTargets";

const targets = useModelInstallTargets();
</script>

<template>
  <div
    v-if="targets.pending.value"
    class="fixed inset-0 z-50"
    data-test="install-target-host"
  >
    <ModalPanel
      :open="true"
      :width="420"
      label="Choose a machine"
      @close="targets.cancel()"
    >
      <h2 class="pick__title">
        Install {{ targets.pending.value.displayName }}
      </h2>
      <p class="pick__lede">Choose which machine to send it to.</p>
      <div class="pick__list">
        <button
          v-for="target in targets.pending.value.targets"
          :key="target.host.id"
          type="button"
          class="pick__option"
          data-test="install-target-option"
          :data-host="target.host.id"
          @click="targets.choose(target)"
        >
          <span class="pick__host">{{ target.host.label }}</span>
          <span class="pick__action" :data-action="target.action">
            {{ target.action === "install" ? "Install" : "Repair" }}
          </span>
          <Icon name="chevron-right" :size="15" class="pick__chevron" />
        </button>
      </div>
      <template #footer>
        <button
          type="button"
          class="pick__cancel"
          data-test="install-target-cancel"
          @click="targets.cancel()"
        >
          Cancel
        </button>
      </template>
    </ModalPanel>
  </div>
</template>

<style scoped>
.pick__title {
  margin: 0;
  font-family: var(--f-display);
  font-size: 18px;
  font-weight: 700;
  color: var(--rebate);
  word-break: break-word;
}

.pick__lede {
  margin: 6px 0 16px;
  font-size: 12.5px;
  color: var(--ink-3);
}

.pick__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.pick__option {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  box-sizing: border-box;
  text-align: left;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  padding: 12px 14px;
  color: var(--rebate);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.pick__option:hover {
  border-color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 8%, transparent);
}

.pick__option:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.pick__host {
  flex: 1;
  min-width: 0;
  font-family: var(--f-mono);
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pick__action {
  font-family: var(--f-mono);
  font-size: 9.5px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--safelight);
  border: 1px solid color-mix(in srgb, var(--safelight) 40%, transparent);
  padding: 2px 8px;
  border-radius: var(--radius-pill);
  white-space: nowrap;
}

.pick__action[data-action="repair"] {
  color: var(--ink-3);
  border-color: var(--ce);
}

.pick__chevron {
  flex: 0 0 auto;
  color: var(--ink-3);
}

.pick__cancel {
  margin-left: auto;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 9px 16px;
  border-radius: var(--radius-control);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}

.pick__cancel:hover {
  border-color: var(--ink-3);
  color: var(--rebate);
}

.pick__cancel:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
