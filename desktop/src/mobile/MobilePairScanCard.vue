<script setup lang="ts">
/*
 * Pair with your desktop — Settings' door onto the pairing scanner.
 *
 * The only way to pair used to be a button buried inside the Machines tab's
 * "Add a machine" disclosure, which is not where anyone looks after installing
 * the app. This is the same machinery (`scanPairingCode` → the native camera →
 * `POST /api/pairing/claim`), reached from the screen that answers "how do I
 * set this thing up".
 *
 * The tile holds the literal word QR and nothing else: the phone SCANS the
 * code the desktop shows, so there is no credential here to render or leak.
 */
const props = withDefaults(
  defineProps<{
    scanning?: boolean;
    /** Why the last scan failed. Said here, where the scan was started. */
    error?: string | null;
  }>(),
  { scanning: false, error: null },
);

const emit = defineEmits<{ scan: [] }>();
</script>

<template>
  <div class="mobile-pair-scan-card" data-test="mobile-pair-scan-card">
    <span class="mobile-pair-scan-glyph" aria-hidden="true">QR</span>
    <div class="mobile-pair-scan-copy">
      <strong>Pair with your desktop</strong>
      <!-- Never "the same network": pairing works over Tailscale too, and
           saying otherwise sends a working setup looking for a fault. -->
      <span
        >Scan the code shown in Mold Studio on your Mac. Open Settings → Mobile pairing on that
        machine.</span
      >
      <button
        class="mobile-pair-scan-button"
        type="button"
        data-test="mobile-pair-scan"
        :disabled="props.scanning"
        @click="emit('scan')"
      >
        Scan pairing code
      </button>
      <span
        v-if="props.error"
        class="mobile-pair-scan-error error-text"
        role="alert"
        data-test="mobile-pair-scan-error"
        >{{ props.error }}</span
      >
    </div>
  </div>
</template>
