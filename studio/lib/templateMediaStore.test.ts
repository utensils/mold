import { describe, expect, it } from "vitest";
import {
  deleteGenerationTemplateMedia,
  hydrateGenerationTemplateMedia,
  persistGenerationTemplateMedia,
  type StoredTemplateMedia,
  type TemplateMediaPersistence,
} from "./templateMediaStore";

function memoryPersistence(failAt = -1): TemplateMediaPersistence & {
  records: Map<string, StoredTemplateMedia>;
} {
  const records = new Map<string, StoredTemplateMedia>();
  let writes = 0;
  return {
    records,
    async put(record) {
      if (writes++ === failAt) return false;
      records.set(record.assetId, structuredClone(record));
      return true;
    },
    async get(assetId) {
      const record = records.get(assetId);
      return record ? structuredClone(record) : null;
    },
    async delete(assetId) {
      records.delete(assetId);
    },
  };
}

describe("template media store", () => {
  it("round-trips local and gallery source bytes in stable order", async () => {
    const persistence = memoryPersistence();
    const assets = await persistGenerationTemplateMedia(
      "template-a",
      [
        {
          field: "imageAttachments",
          index: 0,
          filename: "local.png",
          kind: "upload",
          base64: "LOCAL_BYTES",
        },
        {
          field: "imageAttachments",
          index: 1,
          filename: "remote-gallery.jpg",
          kind: "gallery",
          base64: "REMOTE_BYTES",
        },
      ],
      persistence,
    );

    expect(assets.map((asset) => asset.assetId)).toEqual([
      "template:template-a:attachment-0",
      "template:template-a:attachment-1",
    ]);
    expect(JSON.stringify(assets)).not.toContain("_BYTES");
    await expect(
      hydrateGenerationTemplateMedia(assets, persistence),
    ).resolves.toMatchObject({
      media: [
        { filename: "local.png", kind: "upload", base64: "LOCAL_BYTES" },
        {
          filename: "remote-gallery.jpg",
          kind: "gallery",
          base64: "REMOTE_BYTES",
        },
      ],
      missing: [],
    });
  });

  it("rolls back earlier assets when a durable write fails", async () => {
    const persistence = memoryPersistence(1);
    await expect(
      persistGenerationTemplateMedia(
        "template-b",
        [
          { field: "sourceImage", filename: "one.png", base64: "ONE" },
          {
            field: "imageAttachments",
            index: 0,
            filename: "two.png",
            base64: "TWO",
          },
        ],
        persistence,
      ),
    ).rejects.toThrow("storage is unavailable");
    expect(persistence.records.size).toBe(0);
  });

  it("reports missing assets and removes owned media on deletion", async () => {
    const persistence = memoryPersistence();
    const assets = await persistGenerationTemplateMedia(
      "template-c",
      [{ field: "sourceImage", filename: "source.png", base64: "SOURCE" }],
      persistence,
    );
    persistence.records.clear();
    await expect(
      hydrateGenerationTemplateMedia(assets, persistence),
    ).resolves.toEqual({
      media: [],
      missing: assets,
    });

    await persistGenerationTemplateMedia(
      "template-c",
      [{ field: "sourceImage", filename: "source.png", base64: "SOURCE" }],
      persistence,
    );
    await deleteGenerationTemplateMedia(assets, persistence);
    expect(persistence.records.size).toBe(0);
  });
});

it("keeps original pending-video blobs outside template metadata and restores them without canonical facts", async () => {
  const records = new Map<string, StoredTemplateMedia>();
  const persistence: TemplateMediaPersistence = {
    put: async (record) => {
      records.set(record.assetId, record);
      return true;
    },
    get: async (id) => records.get(id) ?? null,
    delete: async (id) => {
      records.delete(id);
    },
  };
  const blob = new Blob(["original-video"]);
  const assets = await persistGenerationTemplateMedia(
    "pending",
    [
      {
        field: "h3References",
        index: 0,
        filename: "clip.mov",
        base64: "",
        blob,
        reference: {
          kind: "fresh_video",
          original: {
            mime_type: "video/quicktime",
            name: "clip.mov",
            byte_length: blob.size,
            sha256: "a".repeat(64),
          },
        },
      },
    ],
    persistence,
  );
  expect(assets[0]).not.toHaveProperty("blob");
  const hydrated = await hydrateGenerationTemplateMedia(assets, persistence);
  expect(hydrated.missing).toEqual([]);
  expect(await hydrated.media[0]!.blob!.text()).toBe("original-video");
  expect(hydrated.media[0]!.reference).not.toHaveProperty("duration_ms");
});
