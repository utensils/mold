import { describe, expect, it, vi } from "vitest";
import { OwnPrintPreviewWatchers, previewDataUrl } from "./ownPrintPreview";

const target = { baseUrl: "http://studio.test", apiKey: null };

function fakeWatch() {
  const stops: Array<ReturnType<typeof vi.fn>> = [];
  const calls: Array<{ jobId: string; onEnded: (() => void) | undefined }> = [];
  const watch = vi.fn(
    (
      _t: unknown,
      jobId: string,
      _on: unknown,
      _ms: number,
      onEnded?: () => void,
    ) => {
      const stop = vi.fn();
      stops.push(stop);
      calls.push({ jobId, onEnded });
      return stop;
    },
  );
  return { watch, stops, calls };
}

describe("OwnPrintPreviewWatchers", () => {
  it("polls one server job per client job and treats a repeat as a no-op", () => {
    const { watch, stops } = fakeWatch();
    const watchers = new OwnPrintPreviewWatchers(watch as never);
    const apply = vi.fn();
    watchers.ensure("client-1", target, "job-a", apply);
    watchers.ensure("client-1", target, "job-a", apply);
    expect(watch).toHaveBeenCalledTimes(1);
    expect(stops[0]).not.toHaveBeenCalled();
    expect(watchers.has("client-1")).toBe(true);
  });

  it("replaces the watcher when the client job maps to a new server job", () => {
    const { watch, stops } = fakeWatch();
    const watchers = new OwnPrintPreviewWatchers(watch as never);
    watchers.ensure("client-1", target, "job-a", vi.fn());
    watchers.ensure("client-1", target, "job-b", vi.fn());
    expect(watch).toHaveBeenCalledTimes(2);
    expect(stops[0]).toHaveBeenCalledTimes(1);
    expect(stops[1]).not.toHaveBeenCalled();
  });

  it("stops polling when the child leaves the running state, and on stopAll", () => {
    const { watch, stops } = fakeWatch();
    const watchers = new OwnPrintPreviewWatchers(watch as never);
    watchers.ensure("client-1", target, "job-a", vi.fn());
    watchers.ensure("client-2", target, "job-b", vi.fn());
    watchers.stop("client-1");
    expect(stops[0]).toHaveBeenCalledTimes(1);
    expect(watchers.has("client-1")).toBe(false);
    watchers.stop("client-1");
    expect(stops[0]).toHaveBeenCalledTimes(1);
    watchers.stopAll();
    expect(stops[1]).toHaveBeenCalledTimes(1);
    expect(watchers.has("client-2")).toBe(false);
  });

  it("forgets a watcher the host ended so a later running snapshot re-arms it", () => {
    const { watch, calls } = fakeWatch();
    const watchers = new OwnPrintPreviewWatchers(watch as never);
    watchers.ensure("client-1", target, "job-a", vi.fn());
    calls[0]!.onEnded?.();
    expect(watchers.has("client-1")).toBe(false);
    watchers.ensure("client-1", target, "job-a", vi.fn());
    expect(watch).toHaveBeenCalledTimes(2);
  });

  it("renders a preview as a PNG data URL", () => {
    expect(previewDataUrl({ image: "QUJD", step: 3, total: 8 })).toBe(
      "data:image/png;base64,QUJD",
    );
  });
});
