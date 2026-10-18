<script setup lang="ts">
import SourceMediaActions from "./SourceMediaActions.vue";
import SourceMediaViewer from "./SourceMediaViewer.vue";
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  sourcePreviews,
  observeSourceInvalidation,
} from "../lib/sourceMediaResources";
import { retainedSourceMediaThumbnail } from "../api/sourceMediaBrowser";
import {
  sourceMemberKind,
  sourceMemberLabel,
  sourceOwnerIdentity,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const actions = ref<InstanceType<typeof SourceMediaActions>>();
const props = defineProps<{
  selection: SourceMediaSelection;
  usages?: SourceMediaSelection[];
  showOwner?: boolean;
}>();
const viewer = ref<SourceMediaSelection>();
const usageLimit = ref(20);
const usageOpen = ref(false);
function toggleUsages(event: Event) {
  usageOpen.value = (event.target as HTMLDetailsElement).open;
}
const chosen = ref("");
const previewEpoch = ref(0);
const usages = computed(() => props.usages ?? [props.selection]);
watch(
  () => usages.value.length,
  (count) => {
    usageLimit.value = Math.min(
      usageLimit.value,
      Math.max(20, Math.ceil(count / 20) * 20),
    );
  },
  { flush: "sync" },
);
const usageKey = (usage: SourceMediaSelection) =>
  `${sourceOwnerIdentity(usage.owner)}\n${usage.member.member_id}`;
const selection = computed(
  () =>
    usages.value.find((usage) => usageKey(usage) === chosen.value) ??
    usages.value.find(
      (usage) => sourceMemberKind(usage.member) !== "unknown",
    ) ??
    props.selection,
);
watch(
  usages,
  () => {
    if (!viewer.value) return;
    const current = usages.value.find(
      (usage) =>
        usageKey(usage) === usageKey(viewer.value!) &&
        usage.member.content_sha256 === viewer.value!.member.content_sha256,
    );
    viewer.value = current;
  },
  { flush: "sync" },
);
const roles = computed(() =>
  [...new Set(usages.value.map((usage) => sourceMemberLabel(usage.member)))]
    .slice(0, 4)
    .join(" · "),
);
const emit = defineEmits<{
  use: [selection: SourceMediaSelection];
  openOwner: [owner: SourceMediaSelection["owner"]];
}>();
const root = ref<HTMLElement>();
const visible = ref(false);
const url = ref<string>();
const kind = computed(() => sourceMemberKind(selection.value.member));
const label = computed(() => sourceMemberLabel(selection.value.member));
let observer: IntersectionObserver | undefined;
onMounted(() => {
  if (typeof IntersectionObserver === "undefined") {
    visible.value = true;
    return;
  }
  observer = new IntersectionObserver(([entry]) => {
    if (entry?.isIntersecting) {
      visible.value = true;
      observer?.disconnect();
    }
  });
  if (root.value) observer.observe(root.value);
});
watch(
  [
    () => sourceOwnerIdentity(selection.value.owner),
    () => selection.value.member.member_id,
    () => selection.value.member,
    () => selection.value.member.content_sha256,
    () => selection.value.member.media_kind,
    visible,
    previewEpoch,
  ],
  async (_, previous, cleanup) => {
    let cancelled = false;
    let release: (() => void) | undefined;
    let unobserve: (() => void) | undefined;
    let ownedURL: string | undefined;
    const clear = () => {
      cancelled = true;
      url.value = undefined;
      if (ownedURL) URL.revokeObjectURL(ownedURL);
      ownedURL = undefined;
      release?.();
    };
    url.value = undefined;
    cleanup(() => {
      clear();
      unobserve?.();
    });
    if (!visible.value || kind.value !== "image") return;
    const { owner, member } = selection.value;
    try {
      const key = sourceOwnerIdentity(owner);
      const digest =
        member.content_sha256 && /^[a-f0-9]{64}$/.test(member.content_sha256)
          ? member.content_sha256
          : "unknown";
      const resourceKey = `${key}\n${member.member_id}\n${digest}`;
      if (digest === "unknown" && previous?.[2] && previous[2] !== member)
        sourcePreviews.invalidate(resourceKey);
      const lease = sourcePreviews.acquire(resourceKey, (signal) =>
        retainedSourceMediaThumbnail(
          owner.filename,
          member,
          owner.target,
          signal,
        ),
      );
      release = lease.release;
      unobserve = observeSourceInvalidation(key, (refresh) => {
        clear();
        if (refresh) previewEpoch.value++;
      });
      const blob = await lease.promise;
      if (cancelled) return;
      ownedURL = URL.createObjectURL(blob);
      url.value = ownedURL;
    } catch {
      /* The role tile remains usable even when its preview is unavailable. */
    }
  },
  { immediate: true },
);
onBeforeUnmount(() => observer?.disconnect());
</script>
<template>
  <article
    ref="root"
    class="source-card"
    data-test="source-media-card"
    @contextmenu="actions?.open($event)"
  >
    <button
      type="button"
      class="source-card__preview"
      :aria-label="`View original ${selection.member.display_name}`"
      @click="viewer = selection"
    >
      <img v-if="url" :src="url" :alt="label" /><span v-else>{{
        kind === "image"
          ? "Preview unavailable"
          : kind === "unknown"
            ? "Original file"
            : kind
      }}</span>
    </button>
    <strong>{{ label }}</strong>
    <span v-if="usages.length > 1" class="source-card__notice"
      >{{ usages.length }} usages · {{ roles }}</span
    >
    <button
      v-if="showOwner"
      type="button"
      class="source-card__owner"
      @click="emit('openOwner', selection.owner)"
    >
      {{ selection.owner.title }} · {{ selection.owner.hostLabel }}
    </button>
    <details v-if="usages.length > 1" @toggle="toggleUsages">
      <summary>Choose usage ({{ usages.length }})</summary>
      <ul v-if="usageOpen" class="source-card__usages">
        <li
          v-for="(usage, index) in usages.slice(usageLimit - 20, usageLimit)"
          :key="index"
        >
          <button
            type="button"
            :aria-pressed="usageKey(usage) === usageKey(selection)"
            @click="chosen = usageKey(usage)"
          >
            {{ sourceMemberLabel(usage.member) }} · {{ usage.owner.title }} ·
            {{ usage.owner.hostLabel }}
          </button>
        </li>
      </ul>
      <button
        v-if="usageOpen && usageLimit > 20"
        type="button"
        @click="usageLimit -= 20"
      >
        Previous usages
      </button>
      <button
        v-if="usageOpen && usages.length > usageLimit"
        type="button"
        @click="usageLimit += 20"
      >
        Next usages
      </button>
    </details>
    <span class="source-card__name" :title="selection.member.display_name">{{
      selection.member.display_name
    }}</span>
    <span
      v-if="selection.member.role.startsWith('matting_processed')"
      class="source-card__notice"
      >Processed image. Reuse adds it as a new input.</span
    >
    <button
      v-if="['image', 'video', 'audio'].includes(kind)"
      type="button"
      class="source-card__reuse"
      @click="emit('use', selection)"
    >
      Use as source
    </button>
    <span v-else class="source-card__notice"
      >This file type cannot be attached to a new creation.</span
    >
    <SourceMediaActions
      ref="actions"
      :selection="selection"
      :allow-open-owner="showOwner"
      allow-open-source
      @use="emit('use', $event)"
      @open-owner="emit('openOwner', $event)"
      @open="viewer = selection"
    />
    <SourceMediaViewer
      v-if="viewer"
      :selection="viewer"
      :usages="usages"
      :allow-open-owner="showOwner"
      @close="viewer = undefined"
      @choose="viewer = $event"
      @use="
        emit('use', $event);
        viewer = undefined;
      "
      @open-owner="
        emit('openOwner', $event);
        viewer = undefined;
      "
    />
  </article>
</template>
<style scoped>
.source-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-3);
  padding: 10px;
  background: var(--mold-surface, #8881);
  font-size: var(--mold-fs-sm);
}
.source-card button.source-card__preview {
  padding: 0;
  width: 100%;
  aspect-ratio: 4/3;
  display: grid;
  place-items: center;
  background: var(--mold-bg, #8881);
  border-radius: var(--mold-radius-2);
  overflow: hidden;
  color: var(--mold-text-dim, #888);
}
.source-card img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.source-card__name {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--mold-text-dim, #888);
}
.source-card button {
  min-height: 44px;
  border: 1px solid var(--mold-border, #7775);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: inherit;
  cursor: pointer;
  padding: 8px;
  overflow-wrap: anywhere;
}
.source-card__reuse {
  margin-top: auto;
}
.source-card button.source-card__owner {
  border: 0;
  padding: 0;
  min-height: 44px;
}
.source-card__owner {
  text-align: left;
  overflow-wrap: anywhere;
}
.source-card__usages {
  padding-left: 16px;
  max-height: 240px;
  overflow: auto;
}
.source-card__usages button {
  text-align: left;
  width: 100%;
}
.source-card summary {
  min-height: 44px;
  cursor: pointer;
  padding-block: 8px;
}
.source-card button:hover {
  background: var(--mold-accent-muted, #8882);
}
.source-card__notice {
  color: var(--mold-text-dim, #888);
}
</style>
