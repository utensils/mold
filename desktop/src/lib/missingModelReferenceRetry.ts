import type { GenerateRequest } from "./api/types";
import type { BatchRequestOptions } from "../stores/generation";
/** A normalized request is one scoped attempt. A download offer cannot retain
 * its handles or promise to replay it; the live draft still owns original bytes. */
export function missingModelReferenceRetry<
  T extends {
    request: GenerateRequest;
    requestOptions: BatchRequestOptions;
    resumeAfterPull?: boolean;
  },
>(submission: T): T & { resumeAfterPull?: boolean } {
  if (!submission.requestOptions.preparedReferenceLease) return submission;
  const { references: _, ...request } = submission.request;
  return { ...submission, request, requestOptions: {}, resumeAfterPull: false };
}
