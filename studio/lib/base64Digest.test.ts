import { describe, expect, it } from "vitest";
import {
  Base64DigestError,
  decodePaddedBase64,
  sha256PaddedBase64,
} from "./base64Digest";

describe("strict base64 content digests", () => {
  it("hashes the exact decoded bytes", async () => {
    await expect(sha256PaddedBase64("aW1hZ2Utb25l")).resolves.toBe(
      "8f81413241884229c9135da4ae01c0753131bf403587455763d667ee025cb129",
    );
  });

  it.each(["YQ", "YW Jj", "YWJj\n", "YWJj_===", ""])(
    "rejects noncanonical wire input %j",
    (value) => {
      expect(() => decodePaddedBase64(value)).toThrow(Base64DigestError);
    },
  );
});
