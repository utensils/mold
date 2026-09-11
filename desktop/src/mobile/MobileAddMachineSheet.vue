<script setup lang="ts">
/*
 * Add a machine — the phone's one door onto a new machine, as a bottom sheet.
 *
 * It used to be a `<details>` wedged under the machine list, which meant the
 * three ways in (a pairing code, nearby discovery, an address typed by hand)
 * either pushed every card off the screen or hid behind a summary row that
 * looked like one more machine. It also auto-opened on an empty fleet, so the
 * first thing a new install showed was a form.
 *
 * The sheet takes the shared `.mobile-sheet-*` chrome: grabber, scrim, centred
 * iOS header, and `useSheetDismiss` for the drag. The machine state stays in
 * MobileApp — this component holds the flow's presentation and nothing else.
 */
import { ref, toRef } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";
import { useMobileBack } from "./useMobileBack";
import { useSheetDismiss } from "./useSheetDismiss";
import { useSheetFocus } from "./useSheetFocus";
import type { DiscoveredHost } from "./hosts";

const props = defineProps<{
  open: boolean;
  /** True while the native camera is opening for a pairing scan. */
  pairing: boolean;
  /** True while a discovery sweep is running. */
  discovering: boolean;
  discovered: readonly DiscoveredHost[];
  /** The discovered machine awaiting its API key, if any. */
  selectedDiscovered: DiscoveredHost | null;
  /** Why the last attempt failed. Shown HERE: a fixed overlay with a scrim
   *  would otherwise render its own failure on the screen behind it. */
  error?: string | null;
}>();

const emit = defineEmits<{
  close: [];
  "scan-pairing": [];
  discover: [];
  "pick-discovered": [host: DiscoveredHost];
  "clear-discovered": [];
  /** Save the machine the discovered form names. */
  "connect-discovered": [];
  /** Save the machine the manual form names. */
  "connect-manual": [];
}>();

const name = defineModel<string>("name", { required: true });
const address = defineModel<string>("address", { required: true });
const apiKey = defineModel<string>("apiKey", { required: true });

useMobileBack(toRef(props, "open"), () => emit("close"));
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-add-machine");
const panel = ref<HTMLElement | null>(null);
const body = ref<HTMLElement | null>(null);
const discoveredApiKeyInput = ref<HTMLInputElement | null>(null);
const hostAddressInput = ref<HTMLInputElement | null>(null);
const hostApiKeyInput = ref<HTMLInputElement | null>(null);

const { dragging, panelStyle, backdropStyle, beginDismiss, moveDismiss, finishDismiss, resetDrag } =
  useSheetDismiss({ body, onDismiss: () => emit("close") });

/* `aria-modal="true"` over a background that is not inert is a promise only
 * this keeps: focus in on open, Tab held inside, Escape out, focus restored. */
const { onKeydown } = useSheetFocus({
  panel,
  open: () => props.open,
  isTop,
  onClose: () => emit("close"),
  onBeforeClose: resetDrag,
});

/* MobileApp drives the focus when a discovered machine is chosen, because it
 * is what knows the choice landed. */
defineExpose({ focusDiscoveredApiKey: () => discoveredApiKeyInput.value?.focus() });
</script>

<template>
  <div
    class="mobile-sheet mobile-add-machine-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-modal="true"
    :inert="!open"
    aria-label="Add a machine"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-add-machine"
    @keydown="onKeydown"
  >
    <button
      class="mobile-sheet-scrim"
      type="button"
      data-sheet-close
      aria-label="Close Add a machine"
      :style="backdropStyle"
      @click="emit('close')"
    />
    <div
      ref="panel"
      class="mobile-sheet-panel"
      :class="{ 'is-dragging': dragging }"
      :style="panelStyle"
      tabindex="-1"
      @touchstart="beginDismiss"
      @touchmove="moveDismiss"
      @touchend="finishDismiss"
      @touchcancel="resetDrag"
    >
      <span class="mobile-sheet-grabber" aria-hidden="true" />
      <header class="mobile-sheet-head">
        <span class="mobile-sheet-head-slot" aria-hidden="true" />
        <div class="mobile-sheet-heading">
          <h2 class="mobile-sheet-title">Add a machine</h2>
        </div>
        <div class="mobile-sheet-head-slot">
          <button
            class="mobile-sheet-action is-strong"
            type="button"
            data-sheet-close
            data-test="mobile-add-machine-done"
            @click="emit('close')"
          >
            Done
          </button>
        </div>
      </header>
      <div ref="body" class="mobile-sheet-body mobile-add-machine-body">
        <p
          v-if="error"
          class="status-line error-text"
          role="alert"
          data-test="mobile-add-machine-error"
        >
          {{ error }}
        </p>
        <button
          class="primary-button mobile-pair-button"
          type="button"
          :disabled="pairing"
          data-test="mobile-scan-pairing"
          @click="emit('scan-pairing')"
        >
          <span aria-hidden="true">▦</span>
          {{ pairing ? "Opening camera…" : "Scan pairing code" }}
        </button>
        <p class="mobile-pair-note">On your machine, open Settings → Mobile pairing.</p>
        <button
          class="secondary-button"
          type="button"
          :disabled="discovering"
          data-test="mobile-discover-hosts"
          @click="emit('discover')"
        >
          {{ discovering ? "Scanning…" : "Discover nearby" }}
        </button>
        <div
          v-for="host in discovered"
          :key="`${host.host}:${host.port}`"
          class="host-row"
          data-test="mobile-discovered-host"
        >
          <div class="host-row-head">
            <div>
              <div class="host-name">{{ host.name }}</div>
              <div class="host-url">{{ host.host }}:{{ host.port }}</div>
            </div>
            <button class="secondary-button" type="button" @click="emit('pick-discovered', host)">
              Connect
            </button>
          </div>
        </div>
        <form
          v-if="selectedDiscovered"
          class="mobile-discovered-form"
          data-test="mobile-discovered-key-prompt"
          @submit.prevent="emit('connect-discovered')"
        >
          <div class="host-row">
            <div class="host-name">{{ selectedDiscovered.name }}</div>
            <div class="host-url">{{ selectedDiscovered.host }}:{{ selectedDiscovered.port }}</div>
          </div>
          <label class="field"
            ><span>API key</span
            ><input
              ref="discoveredApiKeyInput"
              v-model="apiKey"
              autocapitalize="none"
              :spellcheck="false"
              enterkeyhint="done"
              class="control"
              type="password"
              placeholder="Required by this machine"
              autocomplete="off"
              data-test="mobile-discovered-api-key"
              required
          /></label>
          <p class="section-note">This machine requires its own API key.</p>
          <div class="mobile-inline-actions">
            <button class="secondary-button" type="button" @click="emit('clear-discovered')">
              Choose another
            </button>
            <button class="primary-button" type="submit">Test and save</button>
          </div>
        </form>
        <form v-else class="mobile-host-form" @submit.prevent="emit('connect-manual')">
          <label class="field"
            ><span>Name</span
            ><input
              v-model="name"
              enterkeyhint="next"
              class="control"
              placeholder="Studio Mac (optional)"
              autocomplete="off"
              @keydown.enter.prevent="hostAddressInput?.focus()"
          /></label>
          <label class="field"
            ><span>Address or MagicDNS name</span
            ><input
              ref="hostAddressInput"
              v-model="address"
              inputmode="url"
              :spellcheck="false"
              enterkeyhint="next"
              class="control"
              placeholder="studio.tailnet.ts.net or 192.168.1.20"
              autocapitalize="none"
              autocomplete="url"
              required
              @keydown.enter.prevent="hostApiKeyInput?.focus()"
          /></label>
          <label class="field"
            ><span>API key</span
            ><input
              ref="hostApiKeyInput"
              v-model="apiKey"
              autocapitalize="none"
              :spellcheck="false"
              enterkeyhint="done"
              class="control"
              type="password"
              placeholder="If required"
              autocomplete="off"
          /></label>
          <button class="primary-button" type="submit">Test and save</button>
        </form>
      </div>
    </div>
  </div>
</template>
