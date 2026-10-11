/** Stream originals into private memory with the same queue input budget. */
export async function originalMediaBlob(response: Response): Promise<Blob> {
  const limit = 64 * 1024 * 1024;
  const type = response.headers.get("content-type")?.split(";")[0] ?? "";
  if (
    !response.ok ||
    ![
      "image/png",
      "image/jpeg",
      "image/webp",
      "video/mp4",
      "audio/wav",
      "audio/x-wav",
    ].includes(type) ||
    Number(response.headers.get("content-length")) > limit
  ) {
    await response.body?.cancel();
    throw new Error("Original media unavailable or too large");
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Original media unavailable");
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) throw new Error("Original media is too large");
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
