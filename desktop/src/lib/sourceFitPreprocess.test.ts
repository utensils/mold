import { describe, expect, it, vi } from "vitest";
import type { Rect, SourceFitTransform } from "./sourceFit";
import {
  applySourceFitPreprocess,
  drawableFitPolicy,
  type SourceFitCanvasOps,
} from "./sourceFitPreprocess";

/** Canvas ops fake: pure string transforms so assertions stay exact. */
function fakeOps(size: { width: number; height: number }): SourceFitCanvasOps & {
  fitCalls: Array<{ base64: string; transform: SourceFitTransform }>;
  maskCalls: Array<{ existing: string | null; padRects: Rect[] }>;
} {
  const fitCalls: Array<{ base64: string; transform: SourceFitTransform }> = [];
  const maskCalls: Array<{ existing: string | null; padRects: Rect[] }> = [];
  return {
    fitCalls,
    maskCalls,
    imageSize: () => Promise.resolve(size),
    fitImage: (base64, transform) => {
      fitCalls.push({ base64, transform });
      return Promise.resolve(`fit(${base64})`);
    },
    buildMask: (existing, _transform, padRects) => {
      maskCalls.push({ existing, padRects });
      return Promise.resolve(`mask(${existing ?? "none"})`);
    },
  };
}

const TARGET = { width: 1024, height: 1024 };

describe("drawableFitPolicy", () => {
  it("falls back to pad-repaint for missing and upscale-then-fit policies", () => {
    expect(drawableFitPolicy(undefined)).toEqual({ mode: "pad-repaint" });
    expect(
      drawableFitPolicy({
        mode: "upscale-then-fit",
        upscalerModel: "real-esrgan-x2plus:fp16",
        fit: { mode: "crop-fill" },
      }),
    ).toEqual({ mode: "pad-repaint" });
    expect(drawableFitPolicy({ mode: "crop-fill" })).toEqual({ mode: "crop-fill" });
  });
});

describe("applySourceFitPreprocess", () => {
  it("is a no-op when there is no source image", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const result = await applySourceFitPreprocess(
      { source: null, mask: null, policy: { mode: "pad-repaint" }, target: TARGET },
      { ops },
    );
    expect(result).toEqual({ source: null, mask: null, changed: false });
    expect(ops.fitCalls).toHaveLength(0);
  });

  it("is a no-op when the source already matches the target dimensions", async () => {
    const ops = fakeOps({ width: 1024, height: 1024 });
    const result = await applySourceFitPreprocess(
      { source: "SRC", mask: "MASK", policy: { mode: "pad-repaint" }, target: TARGET },
      { ops },
    );
    expect(result).toEqual({ source: "SRC", mask: "MASK", changed: false });
    expect(ops.fitCalls).toHaveLength(0);
    expect(ops.maskCalls).toHaveLength(0);
  });

  it("pad-repaint fits the source and attaches a generated pad mask", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const result = await applySourceFitPreprocess(
      { source: "SRC", mask: null, policy: { mode: "pad-repaint" }, target: TARGET },
      { ops },
    );

    expect(result.changed).toBe(true);
    expect(result.source).toBe("fit(SRC)");
    expect(result.mask).toBe("mask(none)");
    expect(ops.fitCalls[0]).toMatchObject({
      base64: "SRC",
      transform: { drawWidth: 1024, drawHeight: 768, offsetY: 128, maskPaddedPixels: true },
    });
    // Top and bottom letterbox bands get repainted (white in the mask).
    expect(ops.maskCalls[0]!.padRects).toEqual([
      { x: 0, y: 0, width: 1024, height: 128 },
      { x: 0, y: 896, width: 1024, height: 128 },
    ]);
  });

  it("crop-fill leaves a missing mask absent but refits an existing mask", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const noMask = await applySourceFitPreprocess(
      { source: "SRC", mask: null, policy: { mode: "crop-fill" }, target: TARGET },
      { ops },
    );
    expect(noMask.mask).toBeNull();
    expect(ops.maskCalls).toHaveLength(0);

    const withMask = await applySourceFitPreprocess(
      { source: "SRC", mask: "MASK", policy: { mode: "crop-fill" }, target: TARGET },
      { ops },
    );
    // The user's painted mask must track the fitted source — no pad rects.
    expect(withMask.mask).toBe("mask(MASK)");
    expect(ops.maskCalls[0]).toMatchObject({ existing: "MASK", padRects: [] });
  });

  it("upscale-then-fit rewrites the source through the upscaler before fitting", async () => {
    const sizes = new Map([
      ["SRC", { width: 640, height: 480 }],
      ["up(SRC)", { width: 1280, height: 960 }],
    ]);
    const ops = fakeOps({ width: 0, height: 0 });
    ops.imageSize = (b64: string) => Promise.resolve(sizes.get(b64)!);
    const upscale = vi.fn((image: string, _model: string) => Promise.resolve(`up(${image})`));

    const result = await applySourceFitPreprocess(
      {
        source: "SRC",
        mask: null,
        policy: {
          mode: "upscale-then-fit",
          upscalerModel: "real-esrgan-x2plus:fp16",
          fit: { mode: "pad-repaint" },
        },
        target: TARGET,
      },
      { ops, upscale },
    );

    expect(upscale).toHaveBeenCalledWith("SRC", "real-esrgan-x2plus:fp16");
    // The FIT step consumed the upscaled bytes, not the original.
    expect(ops.fitCalls[0]!.base64).toBe("up(SRC)");
    expect(result.source).toBe("fit(up(SRC))");
    expect(result.changed).toBe(true);
    // drawableFitPolicy maps upscale-then-fit to pad-repaint (web parity),
    // so the pad mask is generated.
    expect(result.mask).toBe("mask(none)");
  });

  it("upscale-then-fit without an upscaler model skips the upscale but still fits", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const upscale = vi.fn();
    const result = await applySourceFitPreprocess(
      {
        source: "SRC",
        mask: null,
        policy: { mode: "upscale-then-fit", upscalerModel: "", fit: { mode: "pad-repaint" } },
        target: TARGET,
      },
      { ops, upscale },
    );
    expect(upscale).not.toHaveBeenCalled();
    expect(result.source).toBe("fit(SRC)");
  });

  it("propagates upscaler failures without touching the form fields", async () => {
    const ops = fakeOps({ width: 640, height: 480 });
    const upscale = vi.fn(() => Promise.reject(new Error("no such model")));
    await expect(
      applySourceFitPreprocess(
        {
          source: "SRC",
          mask: null,
          policy: {
            mode: "upscale-then-fit",
            upscalerModel: "real-esrgan-x2plus:fp16",
            fit: { mode: "pad-repaint" },
          },
          target: TARGET,
        },
        { ops, upscale },
      ),
    ).rejects.toThrow("no such model");
    expect(ops.fitCalls).toHaveLength(0);
  });

  it("reports progress for the upscale stage", async () => {
    const ops = fakeOps({ width: 1024, height: 1024 });
    const statuses: string[] = [];
    await applySourceFitPreprocess(
      {
        source: "SRC",
        mask: null,
        policy: {
          mode: "upscale-then-fit",
          upscalerModel: "real-esrgan-x2plus:fp16",
          fit: { mode: "pad-repaint" },
        },
        target: TARGET,
      },
      {
        ops,
        upscale: (image) => Promise.resolve(image),
        onStatus: (msg) => statuses.push(msg),
      },
    );
    expect(statuses.some((s) => s.includes("real-esrgan-x2plus:fp16"))).toBe(true);
  });
});
