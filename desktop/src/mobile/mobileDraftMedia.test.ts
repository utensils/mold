import { describe, expect, it, vi } from "vitest";
import { newGenerateForm } from "../lib/generateForm";
import {
  hydrateMobileDraftMedia,
  partitionMobileDraftMedia,
  MOBILE_DRAFT_MEDIA_PREFIX,
} from "./mobileDraftMedia";

function authoredForm() {
  const form = newGenerateForm();
  form.prompt = "Keep my unfinished print";
  form.model = "temporarily-unavailable-model";
  form.title = "Unfinished";
  form.sourceImage = "SOURCE_BYTES";
  form.maskImage = "MASK_BYTES";
  form.controlImage = "CONTROL_BYTES";
  form.imageAttachments = ["REFERENCE_0_BYTES", "REFERENCE_1_BYTES"];
  for (const name of [
    "endFrame",
    "identityImage",
    "sourceVideo",
    "extendVideo",
    "audioFile",
  ] as const) {
    form[name] = { filename: name, base64: `${name}_BYTES` };
  }
  form.keyframes = [0, 12].map((frame) => ({
    frame,
    image: { filename: `frame-${frame}`, base64: `KEYFRAME_${frame}_BYTES` },
  }));
  for (const role of ["front", "left", "back", "right"] as const) {
    form.namedViews![role] = {
      filename: role,
      base64: `${role}_BYTES`,
      mimeType: "image/png",
      width: 64,
      height: 64,
    };
  }
  for (const name of ["firstFrame", "lastFrame"] as const) {
    form.h3Authoring![name] = {
      filename: name,
      data: `${name}_BYTES`,
      mimeType: "image/png",
      width: 64,
      height: 64,
    };
  }
  form.h3Authoring!.references = [
    {
      reference: {
        kind: "image",
        media: { authority: "inline", data: "H3_REFERENCE_BYTES" },
        mime_type: "image/png",
        width: 64,
        height: 64,
      },
    },
  ];
  return form;
}

describe("mobile draft media", () => {
  it("round-trips every authored and parked well without retaining bytes in metadata", async () => {
    const form = authoredForm();
    const original = JSON.stringify(form);
    const partition = partitionMobileDraftMedia(form, "revision-1");
    expect(partition.media).toHaveLength(19);
    const serialized = JSON.stringify(partition.form);
    for (const media of partition.media) {
      expect(serialized).not.toContain(media.base64);
      expect(media.draftId).toMatch(/^mold\.mobile\.composer\/revision-1\//);
    }
    expect(JSON.stringify(form)).toBe(original);
    const records = new Map(partition.media.map((record) => [record.draftId, record]));
    const restored = await hydrateMobileDraftMedia(
      JSON.parse(serialized),
      "revision-1",
      async (id) => records.get(id) ?? null,
    );
    expect(restored.missing).toEqual([]);
    expect(restored.form).toEqual(form);
    expect(JSON.stringify(partition.form)).toBe(serialized);
  });

  it("retains missing media descriptors and reports every affected slot", async () => {
    const partition = partitionMobileDraftMedia(authoredForm(), "missing");
    const restored = await hydrateMobileDraftMedia(partition.form, "missing", async () => null);
    expect(restored.missing).toHaveLength(19);
    expect(restored.missing).toContain("identityImage");
    expect(restored.missing).toContain("imageAttachments/1");
    expect(restored.form.imageAttachments).toEqual(["", ""]);
    expect(restored.form.identityImage).toEqual({ filename: "identityImage", base64: "" });
    expect(restored.form.sourceImage).toBe("");
    expect(restored.form.model).toBe("temporarily-unavailable-model");
    expect(restored.form.prompt).toBe("Keep my unfinished print");
  });

  it("reports an unreadable media store without dropping the draft", async () => {
    const form = newGenerateForm();
    form.sourceImage = "SOURCE_BYTES";
    const partition = partitionMobileDraftMedia(form, "read-error");
    const restored = await hydrateMobileDraftMedia(partition.form, "read-error", async () => {
      throw new Error("Storage unavailable");
    });
    expect(restored.missing).toEqual(["sourceImage"]);
  });

  it("never persists request-scoped upload handles or server paths", async () => {
    const form = newGenerateForm();
    form.h3Authoring!.references = [
      {
        reference: {
          kind: "image",
          media: { authority: "upload", handle: "PRIVATE_HANDLE" },
          mime_type: "image/png",
          width: 64,
          height: 64,
        },
      },
      {
        reference: {
          kind: "image",
          media: { authority: "server_path", path: "PRIVATE_PATH" },
          mime_type: "image/png",
          width: 64,
          height: 64,
        },
      },
    ];
    const partition = partitionMobileDraftMedia(form, "scope");
    expect(partition.media).toEqual([]);
    expect(JSON.stringify(partition.form)).not.toMatch(/PRIVATE_HANDLE|PRIVATE_PATH/);
    const restored = await hydrateMobileDraftMedia(partition.form, "scope", async () => null);
    expect(restored.missing).toEqual(["h3Authoring/references/0", "h3Authoring/references/1"]);
    expect(form.h3Authoring!.references[0]!.reference.media.authority).toBe("upload");
  });

  it("separates H3 payloads without JSON-copying them", () => {
    const form = authoredForm();
    const stringify = vi.spyOn(JSON, "stringify");
    try {
      partitionMobileDraftMedia(form, "no-json-media-copy");
      expect(stringify).not.toHaveBeenCalled();
    } finally {
      stringify.mockRestore();
    }
  });

  it("reports non-string stored payloads as missing", async () => {
    const form = newGenerateForm();
    form.sourceImage = "";
    const restored = await hydrateMobileDraftMedia(form, "bad-bytes", async () => ({
      base64: 42 as unknown as string,
    }));
    expect(restored.missing).toEqual(["sourceImage"]);
    expect(restored.form.sourceImage).toBe("");
  });

  it("uses isolated revision keys and rejects invalid namespaces", () => {
    const form = newGenerateForm();
    form.sourceImage = "SOURCE_BYTES";
    const first = partitionMobileDraftMedia(form, "first");
    const second = partitionMobileDraftMedia(form, "second");
    expect(first.media[0]!.draftId).toBe(`${MOBILE_DRAFT_MEDIA_PREFIX}first/sourceImage`);
    expect(second.media[0]!.draftId).not.toBe(first.media[0]!.draftId);
    expect(() => partitionMobileDraftMedia(form, "../other-draft")).toThrow(
      "Invalid mobile draft revision",
    );
  });
});
