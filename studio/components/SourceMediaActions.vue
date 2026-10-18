<script setup lang="ts">
import { computed, inject, onBeforeUnmount, ref, watch } from "vue";
import { SOURCE_ATTACHMENT_CHOICES } from "../lib/sourceAttachmentChoices";
import { sourceMemberKind } from "../lib/sourceMediaBrowser";
import { copyTextToClipboard } from "../lib/notificationClipboard";
import { retainedSourceMediaContent } from "../api/sourceMediaBrowser";
import {
  sourceAddress,
  sourceCommand,
  sourceFileName,
  NATIVE_SOURCE_FILE_ACTIONS,
  sourceMimeHint,
} from "../lib/sourceMediaActions";
import {
  sourceOwnerIdentity,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
import { observeSourceInvalidation } from "../lib/sourceMediaResources";
const props = defineProps<{
  selection: SourceMediaSelection;
  allowOpenOwner?: boolean;
  allowOpenSource?: boolean;
}>();
const emit = defineEmits<{
  use: [selection: SourceMediaSelection];
  openOwner: [owner: SourceMediaSelection["owner"]];
  open: [];
}>();
const choices = inject(
  SOURCE_ATTACHMENT_CHOICES,
  computed(() => []),
);
const error = ref("");
const busy = ref(false);
const preparedShare = ref<File | null>(null);
const clipboard = true;
const nativeFiles = inject(NATIVE_SOURCE_FILE_ACTIONS, null);
const canCopy = computed(() => {
  if (nativeFiles && sourceMemberKind(props.selection.member) === "image")
    return true;
  const mime = sourceMimeHint(props.selection);
  return (
    !!mime &&
    typeof ClipboardItem !== "undefined" &&
    typeof navigator.clipboard?.write === "function" &&
    (typeof ClipboardItem.supports !== "function" ||
      ClipboardItem.supports(mime))
  );
});
const canShare =
  typeof navigator !== "undefined" &&
  typeof navigator.canShare === "function" &&
  typeof navigator.share === "function";
const address = computed(() => sourceAddress(props.selection));
const digest = computed(() =>
  /^[a-f0-9]{64}$/.test(props.selection.member.content_sha256 ?? "")
    ? props.selection.member.content_sha256
    : null,
);
const command = computed(() => {
  try {
    return sourceCommand(props.selection);
  } catch {
    return null;
  }
});
const menu = ref<{ x: number; y: number }>();
function open(event: MouseEvent) {
  event.preventDefault();
  event.stopPropagation();
  menu.value = {
    x: Math.min(event.clientX, Math.max(8, window.innerWidth - 280)),
    y: Math.min(event.clientY, Math.max(8, window.innerHeight - 400)),
  };
}
defineExpose({ open });
let epoch = 0;
let controller: AbortController | undefined;
let stop: (() => void) | undefined;
const urls = new Set<string>();
function invalidate() {
  epoch++;
  menu.value = undefined;
  controller?.abort();
  for (const url of urls) URL.revokeObjectURL(url);
  urls.clear();
  busy.value = false;
  preparedShare.value = null;
}
watch(
  () => [
    sourceOwnerIdentity(props.selection.owner),
    props.selection.member.member_id,
    props.selection.member.content_sha256,
  ],
  () => {
    invalidate();
    stop?.();
    stop = observeSourceInvalidation(
      sourceOwnerIdentity(props.selection.owner),
      invalidate,
    );
  },
  { immediate: true },
);
onBeforeUnmount(() => {
  invalidate();
  stop?.();
});
async function copyText(value: string) {
  try {
    if (!(await copyTextToClipboard(value)))
      throw new Error("Could not copy to the clipboard.");
    error.value = "";
  } catch (failure) {
    error.value = failure instanceof Error ? failure.message : String(failure);
  }
}
async function publish(action: "save" | "copy" | "share") {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  const token = epoch;
  const selection = props.selection;
  controller = new AbortController();
  try {
    const readOriginal = async () => {
      const blob = await retainedSourceMediaContent(
        selection.owner.filename,
        selection.member,
        selection.owner.target,
        controller!.signal,
      );
      if (token !== epoch)
        throw new DOMException("The source changed.", "AbortError");
      return blob;
    };
    if (
      action === "copy" &&
      !(nativeFiles && sourceMemberKind(selection.member) === "image")
    ) {
      const mime = sourceMimeHint(selection);
      if (!mime || !canCopy.value)
        throw new Error(
          "This browser cannot copy this file type. Use Save a Copy.",
        );
      // Safari requires write() during the click. ClipboardItem accepts a
      // promise, so loading the exact original does not consume activation.
      const original = readOriginal().then(
        (blob) => new Blob([blob], { type: mime }),
      );
      // A denied clipboard request may reject before consuming its promise.
      void original.catch(() => undefined);
      await navigator.clipboard.write([
        new ClipboardItem({ [mime]: original }),
      ]);
      return;
    }
    if (action === "share" && preparedShare.value) {
      // Sharing has no promised-file API: use a fresh, deliberate click once
      // the bounded original is ready, with no download await before share().
      await navigator.share({ files: [preparedShare.value] });
      preparedShare.value = null;
      return;
    }
    const blob = await readOriginal();
    if (token !== epoch) return;
    const filename = sourceFileName(selection);
    if (
      action === "copy" &&
      nativeFiles &&
      sourceMemberKind(selection.member) === "image"
    ) {
      await nativeFiles.copyImage(blob, () => token === epoch);
    } else if (action === "share") {
      const file = new File([blob], filename, {
        type: sourceMimeHint(selection) ?? blob.type,
      });
      if (!navigator.canShare({ files: [file] }))
        throw new Error(
          "This browser cannot share this file type. Use Save a Copy.",
        );
      preparedShare.value = file;
    } else if (nativeFiles) {
      await nativeFiles.saveFile(blob, filename, () => token === epoch);
    } else {
      const url = URL.createObjectURL(blob);
      urls.add(url);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      // Keep the URL alive long enough for the browser to take the download;
      // authority invalidation/unmount also revokes every pending URL.
      setTimeout(() => {
        URL.revokeObjectURL(url);
        urls.delete(url);
      }, 1000);
    }
  } catch (failure) {
    if (token === epoch && !controller.signal.aborted)
      error.value =
        failure instanceof Error ? failure.message : String(failure);
  } finally {
    if (token === epoch) busy.value = false;
  }
}
</script>
<template>
  <div
    v-if="menu"
    class="source-actions__scrim"
    @pointerdown="menu = undefined"
  />
  <div
    class="source-actions"
    :class="{ 'source-actions--menu': menu }"
    :style="menu ? { left: `${menu.x}px`, top: `${menu.y}px` } : {}"
    :role="menu ? 'menu' : undefined"
    aria-label="Source actions"
    @keydown.esc.stop="menu = undefined"
    @click="menu = undefined"
  >
    <button
      v-if="
        menu &&
        ['image', 'video', 'audio'].includes(sourceMemberKind(selection.member))
      "
      type="button"
      @click="emit('use', selection)"
    >
      Use as source
    </button>
    <button v-if="allowOpenSource" type="button" @click="emit('open')">
      Open Source
    </button>
    <button
      v-if="allowOpenOwner"
      type="button"
      @click="emit('openOwner', selection.owner)"
    >
      Open Print
    </button>
    <button
      v-for="intent in sourceMemberKind(selection.member) === 'image'
        ? choices
        : []"
      :key="intent"
      type="button"
      @click="emit('use', { ...selection, attachment: intent })"
    >
      Use as {{ intent }} frame
    </button>
    <code class="source-actions__id">{{ address }}</code>
    <button v-if="clipboard" type="button" @click="copyText(address)">
      Copy Source ID
    </button>
    <button
      v-if="clipboard && command"
      type="button"
      @click="copyText(command)"
    >
      Copy CLI Command
    </button>
    <button
      v-if="clipboard"
      type="button"
      @click="copyText(selection.member.display_name)"
    >
      Copy File Name
    </button>
    <template v-if="digest"
      ><code class="source-actions__id">SHA-256: {{ digest }}</code
      ><button v-if="clipboard" type="button" @click="copyText(digest!)">
        Copy SHA-256
      </button></template
    >
    <button type="button" :disabled="busy" @click="publish('save')">
      Save a Copy
    </button>
    <button
      v-if="canCopy"
      type="button"
      :disabled="busy"
      @click="publish('copy')"
    >
      Copy File
    </button>
    <button
      v-if="canShare"
      type="button"
      :disabled="busy"
      @click="publish('share')"
    >
      {{ preparedShare ? "Share" : "Prepare to Share" }}
    </button>
    <span v-if="busy" role="status"
      >Loading original from {{ selection.owner.hostLabel }}…</span
    >
    <span v-if="error" role="alert">{{ error }}</span>
  </div>
</template>
<style scoped>
.source-actions__scrim {
  position: fixed;
  inset: 0;
  z-index: 2000;
}
.source-actions--menu {
  position: fixed;
  z-index: 2001;
  width: 270px;
  max-height: min(400px, 90dvh);
  overflow: auto;
  padding: 10px;
  background: var(--mold-surface, #222);
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-2);
  box-shadow: 0 8px 28px #0006;
}
.source-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  font-size: var(--mold-fs-xs);
}
.source-actions__id {
  width: 100%;
  user-select: text;
  overflow-wrap: anywhere;
}
.source-actions button {
  min-height: 44px;
  padding: 6px;
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: inherit;
  cursor: pointer;
}
</style>
