import { describe, expect, it } from "vitest";
import { isNativeAndroidRuntime, isNativeIOSRuntime } from "./platform";

describe("mobile platform detection", () => {
  it("distinguishes native Android and iOS runtimes", () => {
    expect(isNativeAndroidRuntime("android")).toBe(true);
    expect(isNativeAndroidRuntime("ios")).toBe(false);
    expect(isNativeIOSRuntime("ios")).toBe(true);
    expect(isNativeIOSRuntime("android")).toBe(false);
  });
});
