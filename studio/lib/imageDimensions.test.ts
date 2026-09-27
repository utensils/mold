import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  imageDimensionsFromBase64,
  orientedImageDimensionsFromBase64,
} from "./imageDimensions";

function base64(bytes: number[]): string {
  return btoa(String.fromCharCode(...bytes));
}

describe("imageDimensionsFromBase64", () => {
  it("reads PNG IHDR dimensions from raw base64 or a data URL", () => {
    const png = base64([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x04, 0x92, 0x00, 0x00, 0x09, 0xe4,
    ]);

    expect(imageDimensionsFromBase64(png)).toEqual({
      width: 1170,
      height: 2532,
    });
    expect(imageDimensionsFromBase64(`data:image/png;base64,${png}`)).toEqual({
      width: 1170,
      height: 2532,
    });
  });

  it("walks JPEG metadata segments to a progressive SOF marker", () => {
    const jpeg = base64([
      0xff, 0xd8,
      // APP1 segment: length includes its own two bytes.
      0xff, 0xe1, 0x00, 0x06, 0x45, 0x78, 0x69, 0x66,
      // SOF2, 8-bit precision, 896 x 1152.
      0xff, 0xc2, 0x00, 0x11, 0x08, 0x04, 0x80, 0x03, 0x80, 0x03, 0x01, 0x11,
      0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00, 0xff, 0xd9,
    ]);

    expect(imageDimensionsFromBase64(jpeg)).toEqual({
      width: 896,
      height: 1152,
    });
  });

  it("rejects malformed, unsupported, and zero-sized headers", () => {
    expect(imageDimensionsFromBase64("not base64")).toBeNull();
    expect(
      imageDimensionsFromBase64(base64([0x47, 0x49, 0x46, 0x38])),
    ).toBeNull();
    expect(
      imageDimensionsFromBase64(
        base64([
          0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
          0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
          0x00, 0x01,
        ]),
      ),
    ).toBeNull();
  });
});

describe("imageDimensionsFromBase64 WebP", () => {
  const ALL = ["png", "jpeg", "webp"] as const;

  it("refuses WebP unless the caller names it (source wells stay PNG/JPEG)", () => {
    const webp = base64([
      0x52, 0x49, 0x46, 0x46, 0x30, 0x07, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0x56, 0x50, 0x38, 0x20, 0x24, 0x07, 0x00, 0x00, 0x50, 0xd6, 0x00, 0x9d,
      0x01, 0x2a, 0x92, 0x04, 0x41, 0x03,
    ]);
    expect(imageDimensionsFromBase64(webp)).toBeNull();
    expect(imageDimensionsFromBase64(webp, ["png", "jpeg"])).toBeNull();
    expect(imageDimensionsFromBase64(webp, ALL)).not.toBeNull();
  });

  // Header bytes from Pillow 12.3 (`Image.new(mode, (1170, 833)).save(…,
  // "WEBP")`): one per container a reference picker can hand over.
  it("reads a lossy VP8 frame", () => {
    const webp = base64([
      0x52, 0x49, 0x46, 0x46, 0x30, 0x07, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0x56, 0x50, 0x38, 0x20, 0x24, 0x07, 0x00, 0x00, 0x50, 0xd6, 0x00, 0x9d,
      0x01, 0x2a, 0x92, 0x04, 0x41, 0x03,
    ]);
    expect(imageDimensionsFromBase64(webp, ALL)).toEqual({
      width: 1170,
      height: 833,
    });
  });

  it("reads a lossless VP8L frame", () => {
    const webp = base64([
      0x52, 0x49, 0x46, 0x46, 0x48, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0x56, 0x50, 0x38, 0x4c, 0x3b, 0x00, 0x00, 0x00, 0x2f, 0x91, 0x04, 0xd0,
      0x10, 0x07, 0x10, 0x11, 0x11, 0x00,
    ]);
    expect(imageDimensionsFromBase64(webp, ALL)).toEqual({
      width: 1170,
      height: 833,
    });
  });

  it("reads the canvas of an extended VP8X file (alpha, EXIF)", () => {
    const webp = base64([
      0x52, 0x49, 0x46, 0x46, 0x82, 0x07, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
      0x56, 0x50, 0x38, 0x58, 0x0a, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x91, 0x04, 0x00, 0x40, 0x03, 0x00,
    ]);
    expect(imageDimensionsFromBase64(webp, ALL)).toEqual({
      width: 1170,
      height: 833,
    });
  });

  it("refuses a RIFF container that is not WebP", () => {
    const wav = base64([
      0x52, 0x49, 0x46, 0x46, 0x30, 0x07, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
      0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
      0x44, 0xac, 0x00, 0x00, 0x88, 0x58,
    ]);
    expect(imageDimensionsFromBase64(wav, ALL)).toBeNull();
  });
});

describe("orientedImageDimensionsFromBase64", () => {
  const ALL = ["png", "jpeg", "webp"] as const;
  const DIRECTORY = "crates/mold-core/testdata/reference_orientation";

  // The SAME Pillow-written files `mold_core::reference_image`'s
  // `pillow_written_fixtures_read_upright` reads (96x48 stored pixels,
  // big-endian EXIF), so this reader and the engine's `image` crate reader
  // are pinned to one answer per orientation and per container.
  function fixture(name: string): string {
    let directory = process.cwd();
    for (;;) {
      const candidate = resolve(directory, DIRECTORY, name);
      if (existsSync(candidate)) {
        return readFileSync(candidate).toString("base64");
      }
      const parent = dirname(directory);
      if (parent === directory) throw new Error(`missing ${name}`);
      directory = parent;
    }
  }

  it("swaps the sides for every transposing JPEG orientation", () => {
    for (let value = 1; value <= 8; value += 1) {
      const bytes = fixture(`landscape_96x48_orientation${value}.jpg`);
      expect(
        orientedImageDimensionsFromBase64(bytes, ALL),
        `orientation ${value}`,
      ).toEqual(
        value >= 5 ? { width: 48, height: 96 } : { width: 96, height: 48 },
      );
      // The unoriented reader keeps the stored header for source wells.
      expect(imageDimensionsFromBase64(bytes, ALL)).toEqual({
        width: 96,
        height: 48,
      });
    }
  });

  it("reads a PNG eXIf chunk and a WebP EXIF chunk", () => {
    for (const name of [
      "landscape_96x48_orientation6.png",
      "landscape_96x48_orientation6.webp",
    ]) {
      expect(
        orientedImageDimensionsFromBase64(fixture(name), ALL),
        name,
      ).toEqual({ width: 48, height: 96 });
    }
  });

  it("finds a WebP EXIF chunk that lies beyond the header prefix", () => {
    // VP8X (EXIF flag) + a 2 MiB unknown chunk + a little-endian EXIF chunk
    // carrying Orientation = 8.
    const tiff = [
      0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x12, 0x01,
      0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00,
    ];
    const filler = 2 * 1024 * 1024;
    const bytes = new Uint8Array(12 + 18 + 8 + filler + 8 + tiff.length);
    const text = (offset: number, value: string) => {
      for (let i = 0; i < value.length; i += 1) {
        bytes[offset + i] = value.charCodeAt(i);
      }
    };
    const le32 = (offset: number, value: number) =>
      new DataView(bytes.buffer).setUint32(offset, value, true);
    text(0, "RIFF");
    le32(4, bytes.length - 8);
    text(8, "WEBP");
    text(12, "VP8X");
    le32(16, 10);
    bytes[20] = 0x08;
    // Canvas 1170x833, stored minus one in 24 bits.
    bytes[24] = 0x91;
    bytes[25] = 0x04;
    bytes[27] = 0x40;
    bytes[28] = 0x03;
    text(30, "JUNK");
    le32(34, filler);
    text(38 + filler, "EXIF");
    le32(42 + filler, tiff.length);
    bytes.set(tiff, 46 + filler);
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    expect(orientedImageDimensionsFromBase64(btoa(binary), ALL)).toEqual({
      width: 833,
      height: 1170,
    });
  });

  it("treats a malformed or foreign EXIF block as upright", () => {
    // APP1 "Exif\0\0" with no TIFF magic, then SOF0 96x48.
    const jpeg = base64([
      0xff, 0xd8, 0xff, 0xe1, 0x00, 0x0c, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x30, 0x00,
      0x60, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00, 0xff,
      0xd9,
    ]);
    expect(orientedImageDimensionsFromBase64(jpeg)).toEqual({
      width: 96,
      height: 48,
    });
  });
});
