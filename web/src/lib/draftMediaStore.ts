import { createUuid } from "@studio/lib/id";
import {
  clearMemoryDraftsForTest,
  getDraftMedia,
  putDraftMedia,
} from "@studio/lib/draftMediaStore";

export function newDraftId(): string {
  return createUuid();
}

export { clearMemoryDraftsForTest, getDraftMedia, putDraftMedia };
