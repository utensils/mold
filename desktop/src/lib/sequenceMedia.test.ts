import { beforeEach, describe, expect, it, vi } from "vitest";
import { sequenceStageMediaPath, sequenceStageMediaUrl } from "./sequenceMedia";
import { apiJsonTo } from "./api/client";

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return { ...actual, apiJsonTo: vi.fn() };
});

describe("desktop sequence stage media", () => {
  beforeEach(() => vi.mocked(apiJsonTo).mockReset());

  it("builds an encoded stage path", () => {
    expect(sequenceStageMediaPath("job/1", 4)).toBe("/api/chain-jobs/job%2F1/stages/4/media");
  });

  it("uses a direct URL when the host is keyless", async () => {
    await expect(
      sequenceStageMediaUrl({ baseUrl: "http://studio:7680", apiKey: "" }, "job-1", 0),
    ).resolves.toBe("http://studio:7680/api/chain-jobs/job-1/stages/0/media");
  });

  it("mints a scoped ticket for authenticated playback", async () => {
    vi.mocked(apiJsonTo).mockResolvedValue({
      token: "ticket",
      expires_at: 67890,
      auth_required: true,
    });
    const target = { baseUrl: "http://studio:7680", apiKey: "durable" };
    await expect(sequenceStageMediaUrl(target, "job-1", 2)).resolves.toBe(
      "http://studio:7680/api/chain-jobs/job-1/stages/2/media?media_token=ticket&expires=67890",
    );
    expect(apiJsonTo).toHaveBeenCalledWith(target, "/api/chain-jobs/job-1/stages/2/media-token", {
      method: "POST",
    });
  });
});
