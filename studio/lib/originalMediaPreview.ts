import { shallowRef } from "vue";

/** Private transient intent. Never persisted or used as attachment authority. */
export interface OriginalMediaPreview {
  owner: symbol;
  id: number;
  data?: string | null | undefined;
  mimeType?: string | null | undefined;
  label?: string | undefined;
  load?: ((signal: AbortSignal) => Promise<Blob>) | undefined;
  authority?: string | undefined;
  opener: HTMLElement;
}
export const originalMediaPreview = shallowRef<OriginalMediaPreview | null>(
  null,
);
let nextId = 0;
export function inspectOriginalMedia(
  selection: Omit<OriginalMediaPreview, "id">,
) {
  originalMediaPreview.value = { ...selection, id: ++nextId };
}
export function dismissOriginalMedia(owner?: symbol) {
  if (owner && originalMediaPreview.value?.owner !== owner) return;
  const opener = originalMediaPreview.value?.opener;
  originalMediaPreview.value = null;
  if (opener?.isConnected) queueMicrotask(() => opener.focus());
}
