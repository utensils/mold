<script setup lang="ts">
// Always-on telemetry strip. Live values arrive with the embedded engine in
// M1 (/api/resources/stream at 1 Hz); until then it shows the engine-off state.
</script>

<template>
  <footer class="border-edge flex items-center gap-4 border-t bg-bath px-3">
    <span class="data-mono text-ink-3">⌁ engine off</span>
    <div class="flex-1" />
    <span class="edge-code">QUEUE —</span>
  </footer>
</template>
