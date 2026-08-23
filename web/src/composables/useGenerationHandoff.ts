import { ref, type Ref } from "vue";
import type { OutputMetadata } from "../types";
import type { SelectedQueuePreviewSource } from "@studio/api/generationSelection";

export interface GenerationHandoff {
  metadata: OutputMetadata;
  seedPinned: boolean | null;
  queueSelection?: SelectedQueuePreviewSource;
}

const pending = ref<GenerationHandoff | null>(null);

export function setGenerationHandoff(handoff: GenerationHandoff): void {
  pending.value = handoff;
}

export function takeGenerationHandoff(): GenerationHandoff | null {
  const handoff = pending.value;
  pending.value = null;
  return handoff;
}

export function pendingGenerationHandoff(): Ref<GenerationHandoff | null> {
  return pending;
}
