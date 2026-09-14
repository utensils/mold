import { describe, expect, it } from "vitest";
import { formatMemoryGB, formatMemoryGBPair } from "./formatMemory";

const GIB = 1024 ** 3;

describe("formatMemoryGB", () => {
  /*
   * The bug this module exists to close: a 24 GiB RTX 4090 rendered as
   * "25.8 GB" on the Machines meter (decimal, `bytes / 1e9`) and as
   * "24.0 GB" one card below it (binary, `bytes / 1024 ** 3`), both spelled
   * "GB". Memory is a binary quantity everywhere a person meets it — the
   * card's box, nvidia-smi, Task Manager, Activity Monitor — so the binary
   * reading is the one that matches the hardware, and it is now the only one.
   */
  it("reads a 24 GiB card as the 24 GB it is sold as", () => {
    expect(formatMemoryGB(24 * GIB)).toBe("24.0 GB");
  });

  it("never reports the decimal inflation that split the two surfaces", () => {
    expect(formatMemoryGB(24 * GIB)).not.toBe("25.8 GB");
  });

  it("keeps one decimal place", () => {
    expect(formatMemoryGB(1.75 * GIB)).toBe("1.8 GB");
    expect(formatMemoryGB(0)).toBe("0.0 GB");
  });

  it("renders an absent or nonsense reading as an em dash, never a zero", () => {
    expect(formatMemoryGB(null)).toBe("—");
    expect(formatMemoryGB(undefined)).toBe("—");
    expect(formatMemoryGB(Number.NaN)).toBe("—");
    expect(formatMemoryGB(-1)).toBe("—");
  });
});

describe("formatMemoryGBPair", () => {
  it("shares one unit across the pair and names it once", () => {
    expect(formatMemoryGBPair(1.75 * GIB, 24 * GIB)).toBe("1.8 / 24.0 GB");
  });

  it("agrees with the single-value reading of the same total", () => {
    const pair = formatMemoryGBPair(1.75 * GIB, 24 * GIB);
    expect(pair.endsWith(formatMemoryGB(24 * GIB))).toBe(true);
  });

  it("dashes the half it cannot read rather than fabricating a number", () => {
    expect(formatMemoryGBPair(null, 24 * GIB)).toBe("— / 24.0 GB");
    expect(formatMemoryGBPair(1.75 * GIB, null)).toBe("1.8 / —");
  });
});
