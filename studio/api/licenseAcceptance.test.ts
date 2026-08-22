import { afterEach, describe, expect, it, vi } from "vitest";
import { acceptAndDownload, fetchLicenseListing } from "./licenseAcceptance";
import type { LicenseRequirement } from "../lib/licenseAcceptance";

const target = { baseUrl: "http://hal9000:7680", apiKey: "host-secret" };
const requirement: LicenseRequirement = {
  installModel: "future-face-adapter",
  licenses: [
    {
      id: "future-research-weights",
      name: "Future research weights",
      url: "https://example.test/pinned",
      canonical: "https://example.test/project",
      sha256: "a".repeat(64),
      summary: "Research use only.",
    },
  ],
};

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("host-scoped license API", () => {
  it("reads acceptance from the exact authenticated host", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        Response.json({
          licenses: [
            { ...requirement.licenses[0], accepted: false, required_by: [] },
          ],
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchLicenseListing(target)).resolves.toMatchObject({
      licenses: [{ id: "future-research-weights", accepted: false }],
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://hal9000:7680/api/licenses",
      expect.objectContaining({ headers: expect.any(Headers) }),
    );
    expect(
      (fetchMock.mock.calls[0]![1]!.headers as Headers).get("x-api-key"),
    ).toBe("host-secret");
  });

  it("sends only the exact reviewed terms and waits for terminal completion", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ id: "job-1", position: 0 }))
      .mockResolvedValueOnce(
        Response.json({
          active_jobs: [],
          queued: [],
          history: [
            {
              id: "job-1",
              model: "future-face-adapter",
              status: "completed",
              bytes_done: 10,
              bytes_total: 10,
            },
          ],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);
    const progress = vi.fn();

    await acceptAndDownload(target, requirement, progress);

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://hal9000:7680/api/downloads");
    expect(JSON.parse(String(init?.body))).toEqual({
      model: "future-face-adapter",
      accept_licenses: [
        {
          id: "future-research-weights",
          url: "https://example.test/pinned",
          sha256: "a".repeat(64),
        },
      ],
    });
    expect(progress).toHaveBeenLastCalledWith({
      model: "future-face-adapter",
      status: "completed",
      bytesDone: 10,
      bytesTotal: 10,
    });
  });

  it("joins a matching in-progress download returned as 409", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        Response.json(
          { id: "existing-job", position: 0 },
          { status: 409, statusText: "Conflict" },
        ),
      )
      .mockResolvedValueOnce(
        Response.json({
          active_jobs: [],
          queued: [],
          history: [
            {
              id: "existing-job",
              model: requirement.installModel,
              status: "completed",
              bytes_done: 4,
              bytes_total: 4,
            },
          ],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      acceptAndDownload(target, requirement, vi.fn()),
    ).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it.each(["failed", "cancelled"] as const)(
    "surfaces a %s terminal job without resuming",
    async (status) => {
      const fetchMock = vi
        .fn()
        .mockResolvedValueOnce(Response.json({ id: "job-1", position: 0 }))
        .mockResolvedValueOnce(
          Response.json({
            active_jobs: [],
            queued: [],
            history: [
              {
                id: "job-1",
                model: requirement.installModel,
                status,
                bytes_done: 1,
                bytes_total: 4,
                error: `${status} on host`,
              },
            ],
          }),
        );
      vi.stubGlobal("fetch", fetchMock);

      await expect(
        acceptAndDownload(target, requirement, vi.fn()),
      ).rejects.toThrow(`${status} on host`);
    },
  );

  it("fails closed when a restarted host loses the accepted job", async () => {
    vi.useFakeTimers();
    const empty = {
      active_jobs: [],
      queued: [],
      history: [],
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ id: "lost-job", position: 0 }));
    for (let index = 0; index < 10; index += 1) {
      fetchMock.mockResolvedValueOnce(Response.json(empty));
    }
    vi.stubGlobal("fetch", fetchMock);

    const result = acceptAndDownload(target, requirement, vi.fn());
    const rejection = expect(result).rejects.toThrow(
      "no longer reports download 'lost-job'",
    );
    await vi.advanceTimersByTimeAsync(5_000);
    await rejection;
  });

  it("cancels the exact host job when consent is withdrawn during download", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ id: "job-1", position: 0 }))
      .mockResolvedValueOnce(
        Response.json({
          active_jobs: [],
          queued: [
            {
              id: "job-1",
              model: requirement.installModel,
              status: "queued",
              bytes_done: 0,
              bytes_total: 4,
            },
          ],
          history: [],
        }),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    const result = acceptAndDownload(
      target,
      requirement,
      vi.fn(),
      controller.signal,
    );
    await vi.advanceTimersByTimeAsync(0);
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://hal9000:7680/api/downloads/job-1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
