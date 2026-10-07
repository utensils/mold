import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "./client";

const ceiling = 2 * 1024 * 1024;
/** Retained input from the owning host, independent of the submitting client. */
export async function queueSourceThumbnail(
  target: ApiTarget,
  jobId: string,
  signal?: AbortSignal,
  index?: number,
): Promise<Blob> {
  const response = await apiFetchTo(
    target,
    `/api/queue/${encodeURIComponent(jobId)}/input-thumbnail${index === undefined ? "" : `?index=${index}`}`,
    signal ? { signal } : {},
  );
  const type = response.headers.get("content-type")?.split(";")[0] ?? "";
  if (!["image/png", "image/jpeg", "image/webp"].includes(type)) {
    await response.body?.cancel();
    throw new Error("Queue source thumbnail is not an image");
  }
  if (Number(response.headers.get("content-length")) > ceiling) {
    await response.body?.cancel();
    throw new Error("Queue source thumbnail is too large");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Queue source thumbnail has no image body");
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > ceiling)
        throw new Error("Queue source thumbnail is too large");
      chunks.push(new Uint8Array(value));
    }
  } catch (error) {
    await reader.cancel();
    throw error;
  } finally {
    reader.releaseLock();
  }
  return new Blob(chunks, { type });
}

export interface QueueInput {
  index?: number;
  label: string;
  preview: boolean;
}

/** Missing additive route on an older host retains its singular preview. */
export async function queueInputs(
  target: ApiTarget,
  jobId: string,
  signal?: AbortSignal,
): Promise<QueueInput[]> {
  try {
    const items = await apiJsonTo<QueueInput[]>(
      target,
      `/api/queue/${encodeURIComponent(jobId)}/inputs`,
      signal ? { signal } : {},
    );
    if (
      !Array.isArray(items) ||
      items.length > 256 ||
      items.some(
        (item) =>
          !Number.isSafeInteger(item.index) ||
          item.index! < 0 ||
          typeof item.label !== "string" ||
          typeof item.preview !== "boolean",
      )
    )
      throw new Error("Invalid queue input list");
    return items;
  } catch (error) {
    if (error instanceof ApiError && [404, 405].includes(error.status))
      return [{ label: "Source", preview: true }];
    throw error;
  }
}
