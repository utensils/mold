import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import {
  runWithLicenseConsent,
  useLicenseAcceptance,
} from "./useLicenseAcceptance";

// Fictional throughout: the consent stack must never know a real model or
// license id, and a test naming one would hide a regression that hard-codes it.
const target = { baseUrl: "http://hal9000:7680", apiKey: "host-secret" };
const terms = {
  id: "future-research-weights",
  name: "Future research weights",
  url: "https://example.test/pinned",
  canonical: "https://example.test/project",
  sha256: "a".repeat(64),
  summary: "Research use only.",
};

function refusal() {
  return new ApiError("license not accepted", 403, {
    error: "license not accepted",
    code: "LICENSE_NOT_ACCEPTED",
    license: terms,
  });
}

/** Settle whatever prompt the wrapper opened, the way a user would. */
async function answerPrompt(accept: boolean) {
  const prompt = useLicenseAcceptance();
  for (let i = 0; i < 50 && !prompt.pending.value; i += 1) {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  expect(prompt.pending.value).not.toBeNull();
  if (accept) await prompt.accept();
  else prompt.cancel();
}

afterEach(() => {
  vi.unstubAllGlobals();
  const prompt = useLicenseAcceptance();
  if (prompt.pending.value) prompt.cancel();
});

describe("runWithLicenseConsent", () => {
  it("dismisses after an older host queues the accepted download", async () => {
    const fetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/api/licenses/accept")) {
          return Response.json({ error: "not found" }, { status: 404 });
        }
        if (url.endsWith("/api/downloads") && init?.method === "POST") {
          return Response.json({ id: "job-legacy", position: 0 });
        }
        if (url.endsWith("/api/downloads")) {
          return Response.json({
            active_jobs: [],
            queued: [],
            history: [
              {
                id: "job-legacy",
                model: "future-face-adapter",
                status: "completed",
                bytes_done: 1,
                bytes_total: 1,
              },
            ],
          });
        }
        throw new Error(`unexpected request: ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetch);
    const start = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(refusal())
      .mockRejectedValueOnce(refusal());

    const running = runWithLicenseConsent({
      hostLabel: "legacy-host",
      target,
      installModel: "future-face-adapter",
      start,
    });
    await answerPrompt(true);

    await expect(running).resolves.toEqual({ kind: "accepted" });
    expect(useLicenseAcceptance().pending.value).toBeNull();
    expect(
      fetch.mock.calls.filter(
        ([input, init]) =>
          String(input).endsWith("/api/downloads") && !init?.method,
      ),
    ).toHaveLength(0);
  });

  it("takes consent and re-drives the caller's own enqueue exactly once", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json({ licenses: [] })),
    );
    // Refuse, refuse (the in-gate re-drive), then succeed after acceptance.
    const start = vi
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(refusal())
      .mockRejectedValueOnce(refusal())
      .mockResolvedValueOnce("job-1");

    const running = runWithLicenseConsent({
      hostLabel: "hal9000",
      target,
      installModel: "future-face-adapter",
      start,
    });
    await answerPrompt(true);

    await expect(running).resolves.toEqual({ kind: "ok", value: "job-1" });
    expect(start).toHaveBeenCalledTimes(3);
  });

  it("declines without enqueueing anything", async () => {
    const start = vi.fn<() => Promise<string>>().mockRejectedValue(refusal());

    const running = runWithLicenseConsent({
      hostLabel: "hal9000",
      target,
      installModel: "future-face-adapter",
      start,
    });
    await answerPrompt(false);

    await expect(running).resolves.toEqual({ kind: "declined" });
  });

  it("rethrows a refusal that is not about a license", async () => {
    const conflict = new ApiError("already queued", 409, { id: "job-9" });
    const start = vi.fn<() => Promise<string>>().mockRejectedValue(conflict);

    await expect(
      runWithLicenseConsent({
        hostLabel: "hal9000",
        target,
        installModel: "future-face-adapter",
        start,
      }),
    ).rejects.toBe(conflict);
    expect(start).toHaveBeenCalledTimes(1);
  });

  it("never prompts when the host raises no objection", async () => {
    const start = vi.fn<() => Promise<string>>().mockResolvedValue("job-2");
    await expect(
      runWithLicenseConsent({
        hostLabel: "hal9000",
        target,
        installModel: "future-face-adapter",
        start,
      }),
    ).resolves.toEqual({ kind: "ok", value: "job-2" });
    expect(useLicenseAcceptance().pending.value).toBeNull();
  });
});
