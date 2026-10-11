import { onScopeDispose, ref, type Ref } from "vue";
import {
  MediaFileReader,
  type MediaFileField,
  type ReadMediaFile,
} from "../lib/mediaFileReader";

/** One reader per mounted authoring surface. File inputs remain reusable after
 * a refusal, and stale reads never change either the form or its error. */
export function useMediaFileReader(
  options: ConstructorParameters<typeof MediaFileReader>[0] = {},
  error: Ref<string> = ref(""),
) {
  const reader = new MediaFileReader(options);
  onScopeDispose(() => reader.dispose());
  return {
    error,
    invalidate: (field?: MediaFileField) => reader.invalidate(field),
    async read(
      field: MediaFileField,
      event: Event,
      commit: (media: ReadMediaFile) => void,
    ): Promise<void> {
      const input = event.target as HTMLInputElement;
      const file = input.files?.[0];
      if (!file) return;
      input.value = "";
      const result = await reader.read(field, file, commit);
      if (result.kind !== "stale")
        error.value = result.kind === "refused" ? result.message : "";
    },
  };
}
