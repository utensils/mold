<script setup lang="ts">
import SourceMediaActions from "./SourceMediaActions.vue";
import { computed, ref, watch, onBeforeUnmount } from "vue";
import { imageDimensionsFromBytes } from "../lib/imageDimensions";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { retainedSourceMediaContent } from "../api/sourceMediaBrowser";
import {
  sourceOriginals,
  observeSourceInvalidation,
} from "../lib/sourceMediaResources";
import {
  sourceMemberKind,
  sourceMemberLabel,
  sourceOwnerIdentity,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const actions = ref<InstanceType<typeof SourceMediaActions>>();
const props = defineProps<{
  selection: SourceMediaSelection;
  usages: SourceMediaSelection[];
  allowOpenOwner?: boolean;
}>();
const emit = defineEmits<{
  close: [];
  use: [selection: SourceMediaSelection];
  openOwner: [owner: SourceMediaSelection["owner"]];
  choose: [selection: SourceMediaSelection];
}>();
const opener = typeof document !== "undefined" ? document.activeElement : null;
onBeforeUnmount(() => {
  if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
});
const url = ref(""),
  error = ref(""),
  loading = ref(false),
  page = ref(0),
  retry = ref(0);
const actualSize = ref(false);
const player = ref<HTMLMediaElement>();
function stopPlayback() {
  if (player.value) {
    player.value.pause();
    player.value.removeAttribute("src");
    player.value.load();
  }
}
onBeforeUnmount(stopPlayback);
const kind = computed(() => sourceMemberKind(props.selection.member));
const usageKey = (s: SourceMediaSelection) =>
  `${sourceOwnerIdentity(s.owner)}\n${s.member.member_id}`;
watch(
  () => props.usages.length,
  () => {
    page.value = Math.min(
      page.value,
      Math.max(0, Math.ceil(props.usages.length / 20) - 1),
    );
  },
);
watch(
  [
    () => usageKey(props.selection),
    () => props.selection.member.content_sha256,
    () => props.selection.member,
    retry,
  ],
  async (_, previous, cleanup) => {
    const { owner, member } = props.selection;
    let cancelled = false;
    let ownedURL = "";
    const key = sourceOwnerIdentity(owner);
    const resourceKey = `${key}\n${member.member_id}\n${member.content_sha256 ?? "unknown"}`;
    url.value = "";
    actualSize.value = false;
    error.value = "";
    loading.value = true;
    if (sourceMemberKind(member) === "unknown") {
      loading.value = false;
      return;
    }
    if (
      retry.value ||
      (!member.content_sha256 && previous?.[2] && previous[2] !== member)
    )
      sourceOriginals.invalidate(resourceKey);
    const lease = sourceOriginals.acquire(resourceKey, (signal) =>
      retainedSourceMediaContent(owner.filename, member, owner.target, signal),
    );
    const stop = observeSourceInvalidation(key, () => {
      cancelled = true;
      url.value = "";
      if (ownedURL) URL.revokeObjectURL(ownedURL);
      ownedURL = "";
      lease.release();
      emit("close");
    });
    cleanup(() => {
      stopPlayback();
      cancelled = true;
      lease.release();
      stop();
      if (ownedURL) URL.revokeObjectURL(ownedURL);
    });
    try {
      const blob = await lease.promise;
      if (cancelled) return;
      if (sourceMemberKind(member) === "image") {
        const dimensions = imageDimensionsFromBytes(
          new Uint8Array(await blob.slice(0, 1024 * 1024).arrayBuffer()),
          ["png", "jpeg", "webp"],
        );
        if (
          !dimensions ||
          dimensions.width * dimensions.height > 32_000_000 ||
          Math.max(dimensions.width, dimensions.height) > 8192
        )
          throw new Error(
            "Original image cannot be safely previewed (PNG, JPEG or WebP, up to 32 megapixels and 8192 pixels per side). You can still use the source through the normal attachment checks.",
          );
      }
      if (cancelled) return;
      ownedURL = URL.createObjectURL(blob);
      url.value = ownedURL;
    } catch (failure) {
      if (!cancelled)
        error.value =
          failure instanceof Error
            ? failure.message.includes("exceeds the attachment limit")
              ? "Original preview exceeds the 64 MiB viewing limit."
              : failure.message
            : String(failure);
    } finally {
      if (!cancelled) loading.value = false;
    }
  },
  { immediate: true },
);
</script>
<template>
  <Teleport to="body"
    ><div class="source-viewer-host">
      <ModalPanel
        :open="true"
        :width="1040"
        title="Source media"
        label="Source media viewer"
        @close="emit('close')"
      >
        <div class="source-viewer" @contextmenu="actions?.open($event)">
          <div class="source-viewer__media">
            <p v-if="loading" role="status">Loading original…</p>
            <div v-else-if="error" role="alert">
              {{ error }} <button type="button" @click="retry++">Retry</button>
            </div>
            <div
              v-else-if="url && kind === 'image'"
              class="source-viewer__image"
            >
              <button
                type="button"
                :aria-pressed="actualSize"
                @click="actualSize = !actualSize"
              >
                {{ actualSize ? "Fit image" : "Actual size" }}
              </button>
              <div
                class="source-viewer__canvas"
                :class="{ 'source-viewer__canvas--actual': actualSize }"
              >
                <img :src="url" :alt="selection.member.display_name" />
              </div>
            </div>
            <video
              ref="player"
              v-else-if="url && kind === 'video'"
              :src="url"
              controls
              @error="
                error =
                  'This browser could not play the original media. The source can still be reused.'
              "
              playsinline
              preload="metadata"
            />
            <audio
              ref="player"
              v-else-if="url && kind === 'audio'"
              :src="url"
              controls
              @error="
                error =
                  'This browser could not play the original media. The source can still be reused.'
              "
              preload="metadata"
            />
            <p v-else>No viewer is available for this file type.</p>
          </div>
          <aside>
            <h3>{{ selection.member.display_name }}</h3>
            <dl>
              <dt>Role</dt>
              <dd>{{ sourceMemberLabel(selection.member) }}</dd>
              <template v-if="selection.member.frame_index != null"
                ><dt>Frame index</dt>
                <dd>{{ selection.member.frame_index }}</dd></template
              >
              <dt>Media</dt>
              <dd>{{ kind }}</dd>
              <dt>Size</dt>
              <dd>{{ selection.member.size_bytes.toLocaleString() }} bytes</dd>
              <dt>Machine</dt>
              <dd>{{ selection.owner.hostLabel }}</dd>
              <dt>Used by</dt>
              <dd>{{ selection.owner.title }}</dd>
            </dl>
            <button
              v-if="['image', 'video', 'audio'].includes(kind)"
              type="button"
              data-test="source-viewer-reuse"
              @click="emit('use', selection)"
            >
              Use as source
            </button>
            <SourceMediaActions
              ref="actions"
              :selection="selection"
              :allow-open-owner="allowOpenOwner"
              @use="emit('use', $event)"
              @open-owner="emit('openOwner', $event)"
            />
            <h3>Where used · {{ usages.length }}</h3>
            <ul>
              <li
                v-for="usage in usages.slice(page * 20, page * 20 + 20)"
                :key="usageKey(usage)"
              >
                <button
                  type="button"
                  class="source-viewer__usage"
                  :aria-pressed="usageKey(usage) === usageKey(selection)"
                  @click="emit('choose', usage)"
                >
                  {{ sourceMemberLabel(usage.member)
                  }}<template v-if="usage.member.frame_index != null">
                    · frame {{ usage.member.frame_index }}</template
                  >
                  · {{ usage.owner.title }} ·
                  {{ usage.owner.hostLabel }}</button
                ><button
                  v-if="allowOpenOwner"
                  type="button"
                  class="source-viewer__open-print"
                  :aria-label="`Open print ${usage.owner.title} on ${usage.owner.hostLabel}`"
                  @click="emit('openOwner', usage.owner)"
                >
                  Open print
                </button>
              </li>
            </ul>
            <div class="source-viewer__pages">
              <button v-if="page > 0" type="button" @click="page--">
                Previous usages</button
              ><button
                v-if="(page + 1) * 20 < usages.length"
                type="button"
                @click="page++"
              >
                Next usages
              </button>
            </div>
            <p class="source-viewer__note">
              Retained by the prints that use it. Manage those prints in your
              library; reusing a source does not change their retention.
            </p>
          </aside>
        </div>
        <template #footer
          ><button type="button" @click="emit('close')">Done</button></template
        >
      </ModalPanel>
    </div></Teleport
  >
</template>
<style scoped>
.source-viewer-host {
  position: fixed;
  inset: 0;
  z-index: 1000;
  color: var(--mold-text);
}
.source-viewer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 320px);
  gap: 20px;
  max-height: 75dvh;
  overflow: auto;
  overscroll-behavior: contain;
  padding: 20px;
}
.source-viewer__media {
  display: grid;
  place-items: center;
  min-height: 200px;
  min-width: 0;
  background: var(--mold-bg);
  border-radius: var(--mold-radius-3);
}
.source-viewer img,
.source-viewer video {
  width: 100%;
  max-height: 64dvh;
  object-fit: contain;
}
.source-viewer__image {
  width: 100%;
  min-width: 0;
}
.source-viewer__canvas {
  overflow: auto;
  max-height: 60dvh;
  overscroll-behavior: contain;
}
.source-viewer__canvas--actual img {
  width: auto;
  max-width: none;
  max-height: none;
}
.source-viewer audio {
  width: 100%;
}
.source-viewer aside {
  min-width: 0;
  overflow-wrap: anywhere;
}
.source-viewer h3 {
  font-size: var(--mold-fs-md);
  margin: 0 0 12px;
}
.source-viewer dl {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 6px 12px;
  font-size: var(--mold-fs-sm);
}
.source-viewer dd {
  margin: 0;
}
.source-viewer dt,
.source-viewer__note {
  color: var(--mold-text-dim);
}
.source-viewer button {
  min-height: 44px;
  padding: 8px 12px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
  color: inherit;
  cursor: pointer;
  text-align: left;
  overflow-wrap: anywhere;
}
.source-viewer ul {
  list-style: none;
  padding: 0;
  display: grid;
  gap: 8px;
}
.source-viewer li {
  display: flex;
  gap: 4px;
  align-items: center;
  border-bottom: 1px solid var(--mold-border);
}
.source-viewer button.source-viewer__usage {
  flex: 1;
  min-width: 0;
  border: 0;
  background: transparent;
  padding: 8px 0;
}
.source-viewer button.source-viewer__usage[aria-pressed="true"] {
  color: var(--mold-accent);
}
.source-viewer button.source-viewer__open-print {
  flex: 0 0 auto;
  padding: 8px;
  border: 0;
  background: transparent;
  text-decoration: underline;
  font-size: var(--mold-fs-sm);
}
.source-viewer [data-test="source-viewer-reuse"] {
  margin-block: 12px 20px;
}

.source-viewer__note {
  font-size: var(--mold-fs-sm);
}
.source-viewer__pages {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
@media (max-width: 650px) {
  .source-viewer-host :deep(.ms-modal) {
    padding: 12px;
  }
  .source-viewer-host :deep(.ms-modal__panel) {
    max-width: 100%;
  }
  .source-viewer-host :deep(.ms-modal__body) {
    padding: 0;
  }
  .source-viewer {
    grid-template-columns: minmax(0, 1fr);
    padding: 12px;
  }
  .source-viewer__media {
    min-height: 140px;
  }
}
</style>
