import { afterEach, describe, expect, it, vi } from "vitest";
import { sequenceStageMediaPath, sequenceStageMediaUrl } from "./sequenceMedia";

afterEach(() => vi.unstubAllGlobals());

describe("sequence stage media", () => {
  it("builds an encoded stage path", () => {
    expect(sequenceStageMediaPath("job/1", 2)).toBe(
      "/api/chain-jobs/job%2F1/stages/2/media",
    );
  });

  it("uses the direct Range-friendly URL on a keyless host", async () => {
    await expect(
      sequenceStageMediaUrl("job-1", 0, {
        baseUrl: "http://studio:7680/",
      }),
    ).resolves.toBe("http://studio:7680/api/chain-jobs/job-1/stages/0/media");
  });

  it("exchanges a durable key for an exact-path stage ticket", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        token: "short-lived",
        expires_at: 12345,
        auth_required: true,
      }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const url = await sequenceStageMediaUrl("job-1", 3, {
      baseUrl: "http://studio:7680",
      apiKey: "durable",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://studio:7680/api/chain-jobs/job-1/stages/3/media-token",
      {
        method: "POST",
        headers: { "x-api-key": "durable" },
      },
    );
    expect(url).toBe(
      "http://studio:7680/api/chain-jobs/job-1/stages/3/media?media_token=short-lived&expires=12345",
    );
  });
});
