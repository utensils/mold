import { beforeEach, describe, expect, it } from "vitest";
import {
  __testing__,
  persistGenerationSourceMedia,
  restoreGenerationSourceMedia,
  sha256HexOfBase64,
  type GenerationSourceMedia,
  type GenerationSourceMediaPersistence,
} from "./generationSourceMedia";

function memoryPersistence() {
  const records = new Map<string, GenerationSourceMedia>();
  const persistence: GenerationSourceMediaPersistence = {
    async put(record) {
      records.set(record.draftId, structuredClone(record));
      return true;
    },
    async get(id) {
      return records.get(id) ?? null;
    },
    async delete(id) {
      records.delete(id);
    },
  };
  return { persistence, records };
}

describe("generation source media", () => {
  const storage = new Map<string, string>();
  beforeEach(() => {
    storage.clear();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => storage.get(key) ?? null,
        setItem: (key: string, value: string) => storage.set(key, value),
        clear: () => storage.clear(),
      },
    });
  });

  it("maps effective generation bytes back to the editable original and all source attributes", async () => {
    const { persistence } = memoryPersistence();
    const sha = await persistGenerationSourceMedia(
      "RklUVEVE",
      {
        base64: "T1JJR0lOQUw=",
        filename: "portrait.jpg",
        kind: "gallery",
        width: 1600,
        height: 900,
        mime: "image/jpeg",
        sourceFit: { mode: "crop-fill", alignX: "right", alignY: "top" },
      },
      persistence,
    );

    expect(sha).toBe(await sha256HexOfBase64("RklUVEVE"));
    await expect(
      restoreGenerationSourceMedia(sha, persistence),
    ).resolves.toMatchObject({
      base64: "T1JJR0lOQUw=",
      filename: "portrait.jpg",
      kind: "gallery",
      width: 1600,
      height: 900,
      mime: "image/jpeg",
      sourceFit: { mode: "crop-fill", alignX: "right", alignY: "top" },
    });
  });

  it("keeps only the latest bounded set", async () => {
    const { persistence, records } = memoryPersistence();
    for (let index = 0; index <= __testing__.MAX_RECORDS; index += 1) {
      await persistGenerationSourceMedia(
        btoa(`effective-${index}`),
        {
          base64: btoa(`original-${index}`),
          filename: `${index}.png`,
          sourceFit: { mode: "lanczos-resize" },
        },
        persistence,
      );
    }
    expect(records.size).toBe(__testing__.MAX_RECORDS);
  });
});
