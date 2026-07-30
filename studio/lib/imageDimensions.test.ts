import { describe, expect, it } from "vitest";
import { imageDimensionsFromBase64 } from "./imageDimensions";

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
