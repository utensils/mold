import { describe, expect, it } from "vitest";
import type { CompleteEvent, GenerateRequest } from "../lib/api/types";
import { newJob, type Job } from "../lib/generationJob";
import {
  mobileCompletionSummary,
  summarizeMobileGenerationOutcome,
} from "./mobileGenerationOutcome";

function job(overrides: Partial<Job> = {}): Job {
  const request: GenerateRequest = {
    prompt: "portrait",
    model: "flux:test",
    width: 1024,
    height: 1024,
    steps: 4,
  };
  return Object.assign(newJob(request), overrides);
}

function result(overrides: Partial<CompleteEvent> = {}): CompleteEvent {
  return {
    image: "",
    seed_used: 42,
    generation_time_ms: 1500,
    width: 1024,
    height: 1024,
    model: "flux:test",
    format: "png",
    ...overrides,
  };
}

describe("mobile generation terminal outcome", () => {
  it("summarizes a successful single print", () => {
    const complete = job({ status: "complete", result: result(), requestWarnings: ["notice"] });
    const outcome = summarizeMobileGenerationOutcome([complete], {
      hostLabel: "Studio",
      prepared: false,
    });

    expect(outcome.status).toEqual({ message: "1.5s · seed 42", isError: false });
    expect(outcome.announcement).toBe("Generation completed.");
    expect(outcome.advisories).toEqual(["notice"]);
    expect(outcome.latestCompleted).toBe(complete);
  });

  it("reports a missing latest preview as a completed generation error", () => {
    const outcome = summarizeMobileGenerationOutcome(
      [job({ status: "complete", result: result(), resultError: "ticket expired" })],
      { hostLabel: "Studio", prepared: false },
    );

    expect(outcome.status).toEqual({ message: "ticket expired", isError: true });
    expect(outcome.announcement).toContain("latest preview is unavailable");
  });

  it("names failed prepared variations alongside completed siblings", () => {
    const outcome = summarizeMobileGenerationOutcome(
      [
        job({ status: "complete", result: result() }),
        job({ status: "error", prompt: "alternate", error: "host refused" }),
      ],
      { hostLabel: "Studio", prepared: true },
    );

    expect(outcome.status?.isError).toBe(true);
    expect(outcome.announcement).toContain("Variation 2, “alternate”, failed: host refused");
  });

  it("summarizes a fully failed ordinary submission", () => {
    const outcome = summarizeMobileGenerationOutcome(
      [job({ status: "error", error: "host refused" })],
      { hostLabel: "Studio", prepared: false },
    );

    expect(outcome.status).toEqual({ message: "host refused", isError: true });
    expect(outcome.announcement).toBe("Generation failed. host refused");
  });

  it("keeps cancellation copy singular and timing formatting stable", () => {
    const outcome = summarizeMobileGenerationOutcome(
      [job({ status: "error", error: "cancelled" })],
      { hostLabel: "Studio", prepared: false },
    );

    expect(outcome.status).toEqual({ message: "Cancelled", isError: false });
    expect(outcome.announcement).toBe("1 generation cancelled.");
    expect(mobileCompletionSummary(result({ generation_time_ms: 0 }))).toBe("seed 42");
  });
});
