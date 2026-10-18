import { ApiError, apiFetchTo, type ApiTarget } from "./client";
import {
  retainedSourceMediaInventory,
  parseRetainedSourceMediaInventory,
  type RetainedSourceMediaInventory,
  type RetainedSourceMediaMember,
} from "./gallerySourceMedia";

const memberPath = (filename: string, member: RetainedSourceMediaMember) =>
  `/api/gallery/source-media/${encodeURIComponent(filename)}/${encodeURIComponent(member.member_id)}`;
async function boundedBlob(
  response: Response,
  ceiling: number,
  signal?: AbortSignal,
): Promise<Blob> {
  if (Number(response.headers.get("content-length")) > ceiling) {
    await response.body?.cancel();
    throw new Error("Source media exceeds the attachment limit");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Source media response is empty");
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > ceiling)
        throw new Error("Source media exceeds the attachment limit");
      chunks.push(new Uint8Array(value));
    }
  } catch (error) {
    await reader.cancel();
    throw error;
  } finally {
    reader.releaseLock();
  }
  return new Blob(chunks, {
    type:
      response.headers.get("content-type")?.split(";")[0] ||
      "application/octet-stream",
  });
}
export async function retainedSourceMediaThumbnail(
  filename: string,
  member: RetainedSourceMediaMember,
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await apiFetchTo(
    target,
    `${memberPath(filename, member)}/thumbnail`,
    { signal: signal ?? null },
  );
  if (
    !/^image\/(png|jpeg|webp)(;|$)/.test(
      response.headers.get("content-type") ?? "",
    )
  ) {
    await response.body?.cancel();
    throw new Error("Preview unavailable");
  }
  return boundedBlob(response, 2 * 1024 * 1024, signal);
}
export async function retainedSourceMediaContent(
  filename: string,
  member: RetainedSourceMediaMember,
  target: ApiTarget,
  signal?: AbortSignal,
): Promise<Blob> {
  let response: Response;
  try {
    response = await apiFetchTo(
      target,
      `${memberPath(filename, member)}/content`,
      { signal: signal ?? null },
    );
  } catch (error) {
    if (!(error instanceof ApiError) || ![404, 405].includes(error.status))
      throw error;
    if (member.role === "keyframes")
      throw new Error("Update this machine to reuse individual keyframes.");
    response = await apiFetchTo(target, memberPath(filename, member), {
      signal: signal ?? null,
    });
  }
  return boundedBlob(response, 64 * 1024 * 1024, signal);
}
export interface SourceMediaInventories {
  instance_id?: string;
  inventories: Record<string, RetainedSourceMediaInventory>;
  errors?: Record<string, string>;
}
export async function sourceMediaInventories(
  filenames: string[],
  target: ApiTarget,
  signal?: AbortSignal,
  expectedInstance?: string,
): Promise<SourceMediaInventories> {
  if (filenames.length > 100)
    throw new Error("Source inventory is limited to 100 owners per page");
  if (!filenames.length) return { inventories: {} };
  let response: Response;
  try {
    response = await apiFetchTo(target, "/api/gallery/source-media/inventory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filenames }),
      signal: signal ?? null,
    });
  } catch (error) {
    if (!(error instanceof ApiError) || ![404, 405].includes(error.status))
      throw error;
    const result: SourceMediaInventories = { inventories: {}, errors: {} };
    let next = 0;
    async function worker() {
      while (next < filenames.length) {
        signal?.throwIfAborted();
        const filename = filenames[next++]!;
        try {
          result.inventories[filename] = await retainedSourceMediaInventory(
            filename,
            target,
            signal,
          );
        } catch (failure) {
          signal?.throwIfAborted();
          result.errors![filename] =
            failure instanceof Error ? failure.message : String(failure);
        }
      }
    }
    await Promise.all(
      Array.from({ length: Math.min(4, filenames.length) }, worker),
    );
    return result;
  }
  const result = JSON.parse(
    await (await boundedBlob(response, 8 * 1024 * 1024, signal)).text(),
  ) as SourceMediaInventories;
  if (!result || typeof result.inventories !== "object" || !result.inventories)
    throw new Error("Invalid source inventory response");
  if (expectedInstance && result.instance_id !== expectedInstance)
    throw new Error(
      "The source machine changed. Refresh the library before reusing media.",
    );
  for (const filename of filenames) {
    const inventory = result.inventories[filename];
    try {
      if (!inventory) throw new Error("Incomplete source inventory response");
      result.inventories[filename] =
        parseRetainedSourceMediaInventory(inventory);
    } catch (failure) {
      delete result.inventories[filename];
      result.errors ??= {};
      result.errors[filename] =
        failure instanceof Error ? failure.message : String(failure);
    }
  }
  return result;
}
