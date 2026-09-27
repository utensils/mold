import { describe, expect, it } from "vitest";

import {
  fileMatchesImageInputFormats,
  imageInputFormatForName,
  imageInputFormatOfBase64,
  imageInputFormatsSentence,
  LEGACY_REFERENCE_IMAGE_FORMATS,
  referenceImageMimeTypes,
  referenceImagesFromProfile,
  sniffImageInputFormat,
} from "./referenceImagesProfile";

describe("referenceImagesFromProfile containers", () => {
  const block = {
    mode: "adjustable" as const,
    required: false,
    max_count: 10,
    primary_is_target: false,
    source_relation: "replaces" as const,
  };

  it("reads an empty or absent list as the legacy PNG/JPEG pair", () => {
    expect(referenceImagesFromProfile(block)?.formats).toEqual(["png", "jpeg"]);
    expect(
      referenceImagesFromProfile({ ...block, formats: [] })?.formats,
    ).toEqual([...LEGACY_REFERENCE_IMAGE_FORMATS]);
    expect(referenceImagesFromProfile(block)?.canvas).toBeNull();
  });

  it("keeps an advertised list and canvas rule verbatim", () => {
    expect(
      referenceImagesFromProfile({
        ...block,
        canvas: "last-reference",
        formats: ["png", "jpeg", "webp"],
      }),
    ).toMatchObject({
      canvas: "last-reference",
      formats: ["png", "jpeg", "webp"],
    });
  });
});

describe("referenceImageMimeTypes", () => {
  it("maps each container to the MIME type a picker accepts", () => {
    expect(referenceImageMimeTypes(["png", "jpeg"])).toEqual([
      "image/png",
      "image/jpeg",
    ]);
    expect(referenceImageMimeTypes(["png", "jpeg", "webp"])).toContain(
      "image/webp",
    );
  });
});

describe("sniffImageInputFormat", () => {
  const bytes = (values: number[]) => Uint8Array.from(values);

  it("mirrors mold_core's magic-byte sniff", () => {
    expect(sniffImageInputFormat(bytes([0x89, 0x50, 0x4e, 0x47]))).toBe("png");
    expect(sniffImageInputFormat(bytes([0xff, 0xd8, 0xff]))).toBe("jpeg");
    expect(
      sniffImageInputFormat(
        bytes([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]),
      ),
    ).toBe("webp");
  });

  it("refuses everything else, including a RIFF that is not WebP", () => {
    expect(sniffImageInputFormat(bytes([0x47, 0x49, 0x46, 0x38]))).toBeNull();
    expect(
      sniffImageInputFormat(
        bytes([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45]),
      ),
    ).toBeNull();
    expect(sniffImageInputFormat(bytes([]))).toBeNull();
  });
});

describe("picker wording and file matching", () => {
  it("names the accepted containers in one sentence", () => {
    expect(imageInputFormatsSentence(["png", "jpeg"])).toBe("PNG or JPEG");
    expect(imageInputFormatsSentence(["png", "jpeg", "webp"])).toBe(
      "PNG, JPEG, or WebP",
    );
  });

  it("matches a file by MIME type, or by extension when the type is blank", () => {
    const legacy = ["png", "jpeg"] as const;
    const qwen21 = ["png", "jpeg", "webp"] as const;
    const webp = { type: "image/webp", name: "cutout.webp" };
    expect(fileMatchesImageInputFormats(webp, legacy)).toBe(false);
    expect(fileMatchesImageInputFormats(webp, qwen21)).toBe(true);
    expect(
      fileMatchesImageInputFormats({ type: "", name: "Layer.WEBP" }, qwen21),
    ).toBe(true);
    expect(
      fileMatchesImageInputFormats({ type: "", name: "photo.jpeg" }, legacy),
    ).toBe(true);
    expect(
      fileMatchesImageInputFormats(
        { type: "image/gif", name: "a.gif" },
        qwen21,
      ),
    ).toBe(false);
    expect(imageInputFormatForName("clip.mp4")).toBeNull();
  });
});

describe("imageInputFormatOfBase64", () => {
  it("reads the container from the payload's first bytes", () => {
    expect(imageInputFormatOfBase64("iVBORw0KGgoAAAANSUhEUg==")).toBe("png");
    expect(imageInputFormatOfBase64("/9j/4AAQSkZJRgABAQ==")).toBe("jpeg");
    expect(imageInputFormatOfBase64("UklGRjAHAABXRUJQVlA4IA==")).toBe("webp");
    expect(
      imageInputFormatOfBase64(
        "data:image/webp;base64,UklGRjAHAABXRUJQVlA4IA==",
      ),
    ).toBe("webp");
    expect(imageInputFormatOfBase64("R0lGODlhAQABAA==")).toBeNull();
    expect(imageInputFormatOfBase64("not base64!")).toBeNull();
  });
});
