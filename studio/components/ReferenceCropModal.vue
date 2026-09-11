<script setup lang="ts">
/*
 * The dialog around the shared ReferenceCropEditor, for every surface.
 *
 * Both copies of this were a host element, a ModalPanel-shaped box and the
 * editor. The only thing they could not agree on was WHERE the host element
 * goes: the browser's Create is a long scrolling column, so the overlay takes
 * a viewport host of its own; the desktop app opens the crop from inside the
 * inspector, whose own containing block would otherwise trap an overlay that
 * is meant to cover the window. Both are the same markup in a different
 * place, so the place is `teleport` and nothing else differs.
 *
 * Tab trapping, background-scroll locking and returning focus to the opener
 * are the host surface's rules — `ui/` deliberately leaves them to the
 * caller — so a surface that has them hands its composable in as `useFocus`.
 * Without one, ModalPanel's own document-level Escape and Tab handling still
 * stands.
 */
import { computed, ref, type Ref } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import ReferenceCropEditor from "./ReferenceCropEditor.vue";
import type { ReferenceCrop } from "../lib/referenceCrop";

const props = defineProps<{
  open: boolean;
  title: string;
  image: {
    data: string;
    mimeType: string;
    width: number;
    height: number;
  } | null;
  crop: ReferenceCrop | null;
  /** `true` sends the host to `<body>`; a string names its own target. */
  teleport?: boolean | string;
  /** The surface's overlay-focus composable, called once during setup. */
  useFocus?: (
    open: Ref<boolean>,
    host: Ref<HTMLElement | null>,
    close?: () => void,
  ) => { onKeydown: (event: KeyboardEvent) => void };
}>();

const emit = defineEmits<{
  apply: [crop: ReferenceCrop | null];
  close: [];
}>();

const host = ref<HTMLElement | null>(null);
const isOpen = computed(() => props.open && props.image !== null);
const teleportTo = computed(() =>
  typeof props.teleport === "string" ? props.teleport : "body",
);
const focus = props.useFocus?.(isOpen, host, () => emit("close"));
const onKeydown = (event: KeyboardEvent) => focus?.onKeydown(event);
</script>

<template>
  <Teleport :to="teleportTo" :disabled="!teleport">
    <div
      v-if="isOpen && image"
      ref="host"
      class="rc-host"
      data-test="reference-crop-host"
      @keydown="onKeydown"
    >
      <ModalPanel
        :open="true"
        :width="720"
        :label="title"
        :title="title"
        @close="emit('close')"
      >
        <div class="rc-body">
          <ReferenceCropEditor
            :image="image"
            :crop="crop"
            @apply="emit('apply', $event)"
            @cancel="emit('close')"
          />
        </div>
      </ModalPanel>
    </div>
  </Teleport>
</template>

<style scoped>
.rc-host {
  position: fixed;
  inset: 0;
  z-index: var(--mold-z-modal, 100);
}
/* A tall reference must not push the dialog's own edges off the window. */
.rc-body {
  max-height: 72vh;
  overflow-y: auto;
}
</style>
