import { createRetainedSourceMediaReuseSession } from "./gallerySourceMedia";
import {
  prepareReferenceUploadBatch,
  requestShouldUseReferenceUploads,
  type ReferenceUploadCapabilities,
  type ReferenceUploadLease,
  type ReferenceUploadRequest,
} from "./referenceUploads";
import type { VideoJobOwner, PreparedDurationVideo } from "./videoJobs";
import type { VideoJobRequest } from "../lib/api/videoJobTypes";

/** Reuses ordinary input preparation after the shell has fitted source pixels.
 * One sibling owns one lease/session across full validation and creation. */
export async function prepareDurationVideo<
  Generation extends ReferenceUploadRequest,
>(
  owner: VideoJobOwner,
  request: VideoJobRequest<Generation>,
  options: {
    referenceUploads?: ReferenceUploadCapabilities | null | undefined;
    preparedReferenceLease?: ReferenceUploadLease<Generation> | undefined;
    retainedSource?:
      { filename: string; memberIds: readonly string[] } | undefined;
    isCurrent?: (() => boolean) | undefined;
    signal?: AbortSignal | undefined;
  } = {},
): Promise<PreparedDurationVideo<Generation>> {
  const fresh = () =>
    !options.signal?.aborted && options.isCurrent?.() !== false;
  let generation = request.generation;
  let release: (() => Promise<void>) | undefined =
    options.preparedReferenceLease?.cancel;
  try {
    if (!fresh())
      throw new Error("The video or machine changed before input preparation.");
    if (options.preparedReferenceLease) {
      if (
        options.preparedReferenceLease.request !== generation ||
        options.preparedReferenceLease.expiresAtMs <= Date.now()
      )
        throw new Error(
          "Prepared references changed or expired. Prepare this video again from the originals.",
        );
    } else if (
      requestShouldUseReferenceUploads(
        generation,
        owner.target,
        options.referenceUploads,
      )
    ) {
      const staged = await prepareReferenceUploadBatch({
        target: owner.target,
        expectedInstanceId: owner.instanceId!,
        capabilities: options.referenceUploads,
        requests: [generation],
        ...(options.signal ? { signal: options.signal } : {}),
      });
      release = staged.release;
      if (staged.requests.length !== 1)
        throw new Error(
          "This machine returned an incompatible prepared video.",
        );
      generation = staged.requests[0]!;
    }
    if (!fresh())
      throw new Error("The video or machine changed during input preparation.");
    const session = options.retainedSource
      ? await createRetainedSourceMediaReuseSession(
          options.retainedSource.filename,
          options.retainedSource.memberIds,
          generation,
          owner.target,
        )
      : null;
    if (!fresh())
      throw new Error("The video or machine changed during input preparation.");
    return {
      request:
        generation === request.generation
          ? request
          : { ...request, generation },
      ...(session
        ? { media: { retainedMediaSession: session.session_handle } }
        : {}),
      ...(release ? { release } : {}),
    };
  } catch (cause) {
    await release?.().catch(() => undefined);
    throw cause;
  }
}
