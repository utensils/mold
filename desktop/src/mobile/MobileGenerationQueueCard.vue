<script setup lang="ts">
/*
 * One queue row, for Make's queue and a machine's own queue alike.
 *
 * It used to be a title, a subtitle and a status code, which meant the screen
 * that exists to answer "is my picture coming" showed no picture, no progress
 * and no way to tell one sibling of a batch from another. Everything it needs
 * was already being carried past it: the live latent preview Make paints on,
 * the step counter the engine reports, and the batch index on the request.
 *
 * Every added part is optional and drawn only when supplied, so a shared
 * fleet row — which has a phase and a count but no pixels — keeps the plain
 * shape it always had, and a queued row stands its place in line where the
 * picture will be.
 */
import { computed } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";

const props = withDefaults(
  defineProps<{
    title: string;
    subtitle: string;
    status: string;
    detail?: string | null;
    cancelling?: boolean;
    ariaLabel?: string;
    /** The live latent preview, while there is one. */
    thumbnailUrl?: string | null;
    /** 0–100 while the engine is reporting steps. */
    progress?: number | null;
    /** The mono line under the meter: "image 2 of 4 · studio-rack". */
    meta?: string | null;
    /** Place in line, for a row with no pixels yet. */
    position?: string | null;
    tone?: "neutral" | "warning";
    /** Identifies THIS row, where the card's own data-test names the kind. */
    rowTestId?: string | null;
    /**
     * True while the host is actually working on this print. A running row
     * says what it is doing in plain words; every other row says a code.
     */
    running?: boolean;
  }>(),
  {
    detail: null,
    cancelling: false,
    thumbnailUrl: null,
    progress: null,
    meta: null,
    position: null,
    tone: "neutral",
    rowTestId: null,
    running: false,
  },
);

const emit = defineEmits<{
  activate: [];
}>();

const displayTitle = computed(() => props.title.trim() || props.subtitle);
/**
 * A long CODE has to wrap onto its own row. A running row's sentence already
 * lives in the copy column, so it never needs the full-width layout.
 */
const detailedStatus = computed(
  () => !props.running && (props.status.length > 18 || Boolean(props.detail)),
);
/** A row is "active" once it has pixels or a step count to show. */
const active = computed(() => Boolean(props.thumbnailUrl) || props.progress !== null);
</script>

<template>
  <div
    class="mobile-generation-job"
    :class="{
      'mobile-generation-job--detailed-status': detailedStatus,
      'mobile-generation-job--active': active,
      'mobile-generation-job--warning': tone === 'warning',
      'mobile-generation-job--leading': Boolean(thumbnailUrl) || Boolean(position),
    }"
    role="button"
    tabindex="0"
    :aria-label="[ariaLabel?.trim() || displayTitle, status, detail].filter(Boolean).join('. ')"
    data-test="mobile-generation-queue-card"
    :data-row-test="rowTestId ?? undefined"
    @click="emit('activate')"
    @keydown.enter.prevent="emit('activate')"
    @keydown.space.prevent="emit('activate')"
  >
    <span
      v-if="thumbnailUrl"
      class="mobile-generation-job-thumb"
      data-test="mobile-generation-job-thumb"
    >
      <img :src="thumbnailUrl" alt="" decoding="async" />
    </span>
    <span
      v-else-if="position"
      class="mobile-generation-job-position"
      data-test="mobile-generation-job-position"
      aria-hidden="true"
      >{{ position }}</span
    >
    <div class="mobile-generation-job-copy">
      <p>{{ displayTitle }}</p>
      <span v-if="title.trim() && subtitle.trim()">{{ subtitle }}</span>
      <!-- What is happening, in the host's own sentence. The same element
           carries the uppercase code in the trailing column when the row is
           not running, so there is only ever one status to read. -->
      <span
        v-if="running"
        class="mobile-generation-job-sentence"
        data-test="mobile-generation-status"
        >{{ status }}</span
      >
      <ProgressBar
        v-if="progress !== null"
        class="mobile-generation-job-meter"
        data-test="mobile-generation-job-meter"
        :value="progress"
        :height="7"
        :tone="tone === 'warning' ? 'warning' : 'accent'"
        :label="`${displayTitle} progress`"
      />
      <span v-if="meta" class="mobile-generation-job-meta" data-test="mobile-generation-job-meta">{{
        meta
      }}</span>
      <p
        v-if="detail"
        class="mobile-generation-held-error"
        data-test="mobile-generation-held-error"
      >
        {{ detail }}
      </p>
    </div>
    <div class="mobile-generation-job-action">
      <span v-if="!running" data-test="mobile-generation-status">{{ status }}</span>
      <span v-if="cancelling" data-test="mobile-generation-cancelling"> Cancelling… </span>
    </div>
  </div>
</template>
