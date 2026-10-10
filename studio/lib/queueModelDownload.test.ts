import { describe, expect, it } from "vitest";
import { queueModelDownloadAvailability } from "./queueModelDownload";
const held = { state: "held", retryable: true, model: "h3-turbo" };
describe("verified queue model download hint", () => {
  it("offers exact-model repair without requiring an error code", () => {
    expect(
      queueModelDownloadAvailability({
        ...held,
        model_download: { model: held.model },
      }),
    ).toBe(true);
  });
  it("uses legacy lookup only when the hint is absent", () => {
    expect(queueModelDownloadAvailability(held)).toBeUndefined();
    expect(
      queueModelDownloadAvailability({
        ...held,
        model: " ",
        model_download: { model: " " },
      }),
    ).toBe(false);
    for (const hint of [
      null,
      {},
      { model: "other" },
      { model: "" },
      "h3-turbo",
    ])
      expect(
        queueModelDownloadAvailability({ ...held, model_download: hint }),
      ).toBe(false);
  });
  it("requires a held retryable row", () => {
    expect(
      queueModelDownloadAvailability({
        ...held,
        state: "running",
        model_download: { model: held.model },
      }),
    ).toBe(false);
    expect(
      queueModelDownloadAvailability({
        ...held,
        retryable: false,
        model_download: { model: held.model },
      }),
    ).toBe(false);
  });
});
