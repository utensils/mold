<script setup lang="ts">
import type { NamedViewsProfile } from "../lib/generated/generationProfileV1";
import {
  NAMED_VIEW_LABELS,
  type NamedViewRole,
  type NamedViewsState,
} from "../lib/namedViews";
import ImageDropWell from "./ImageDropWell.vue";

defineProps<{
  profile: NamedViewsProfile;
  modelValue?: NamedViewsState | null | undefined;
  disabled?: boolean;
  error?: string | null;
  touchFriendly?: boolean;
  gallery?: boolean;
}>();

const emit = defineEmits<{
  file: [role: NamedViewRole, file: File];
  gallery: [role: NamedViewRole];
  clear: [role: NamedViewRole];
}>();
</script>

<template>
  <section class="named-views" data-test="named-views-panel">
    <div class="named-views__head">
      <span>Object views</span>
      <span class="named-views__count"
        >{{ profile.min_count }}–{{ profile.max_count }} required</span
      >
    </div>
    <p class="named-views__hint">
      Add any available angles. Each image keeps its camera position even when
      other slots are empty.
    </p>
    <div class="named-views__grid">
      <div v-for="role in profile.roles" :key="role" class="named-views__slot">
        <span class="named-views__label">{{ NAMED_VIEW_LABELS[role] }}</span>
        <ImageDropWell
          :image="modelValue?.[role]?.base64 || null"
          :mime-type="modelValue?.[role]?.mimeType || null"
          :filename="modelValue?.[role]?.filename || null"
          :placeholder="`Drop ${NAMED_VIEW_LABELS[role].toLowerCase()} view or click to pick`"
          accept="image/png,image/jpeg"
          :disabled="disabled"
          :required="profile.min_count > 0"
          :gallery="gallery !== false"
          :touch-friendly="touchFriendly"
          :alt="`${NAMED_VIEW_LABELS[role]} object view`"
          :test-id="`named-view-${role}`"
          :drop-target="`named-view-${role}`"
          @file="emit('file', role, $event)"
          @gallery="emit('gallery', role)"
          @clear="emit('clear', role)"
        />
      </div>
    </div>
    <p
      v-if="error"
      class="named-views__error"
      role="alert"
      data-test="named-views-error"
    >
      {{ error }}
    </p>
  </section>
</template>

<style scoped>
.named-views {
  display: grid;
  gap: 8px;
  min-width: 0;
}
.named-views__head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--mold-text-dim, #737373);
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: var(--mold-fs-micro);
  line-height: 1.3;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.named-views__count {
  opacity: 0.75;
  letter-spacing: 0;
  text-transform: none;
}
.named-views__hint,
.named-views__error {
  margin: 0;
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-xs);
  line-height: 1.45;
}
.named-views__error {
  color: var(--mold-danger, #b91c1c);
}
.named-views__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.named-views__slot {
  display: grid;
  gap: 5px;
  min-width: 0;
}
.named-views__label {
  color: var(--mold-text-dim, #737373);
  font-size: var(--mold-fs-micro);
}
@media (max-width: 430px) {
  .named-views__grid {
    grid-template-columns: 1fr;
  }
}
</style>
