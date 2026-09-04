/**
 * The Machines list and the machine pane both say one sentence about a box.
 * They used to build it twice: the list called a 4× L40S machine `4× L40S`
 * and the pane called the same box `L40S`, and one said "This machine" where
 * the other said "this device".
 */
import { describe, expect, it } from "vitest";
import { machineHardware, machineLocation, machineSentence } from "./machineSentence";
import type { GpuSnapshot } from "./api/types";

const gpu = (name: string, backend: string | null = "cuda"): GpuSnapshot =>
  ({ ordinal: 0, name, backend, vram_used: 0, vram_total: 0 }) as GpuSnapshot;

describe("machineHardware", () => {
  it("names every card in the box, not just the first", () => {
    expect(machineHardware([gpu("L40S"), gpu("L40S"), gpu("L40S"), gpu("L40S")])).toBe(
      "4× L40S · CUDA",
    );
    expect(machineHardware([gpu("RTX 4090"), gpu("B200")])).toBe("RTX 4090 + B200 · CUDA");
  });

  it("infers the backend from the card when the host does not report one", () => {
    expect(machineHardware([gpu("Apple M3 Max", null)])).toBe("Apple M3 Max · METAL");
  });

  it("says nothing at all before telemetry has arrived", () => {
    expect(machineHardware([])).toBe("");
  });
});

describe("machineLocation", () => {
  it("uses the lexicon's word for this device", () => {
    expect(machineLocation({ kind: "local" })).toBe("this device");
  });

  it("names a rented GPU by what it costs to keep", () => {
    expect(machineLocation({ kind: "remote", baseUrl: "https://abc-8188.proxy.runpod.net" })).toBe(
      "rented cloud GPU",
    );
  });

  it("carries the address only where the caller asks for it", () => {
    const host = { kind: "remote", baseUrl: "http://hal9000:7680" } as const;
    expect(machineLocation(host)).toBe("on your network");
    expect(machineLocation(host, true)).toBe("on your network at hal9000:7680");
  });
});

describe("machineSentence", () => {
  it("is the card's sentence plus the pane's uptime, from one builder", () => {
    const host = { kind: "remote", baseUrl: "http://plato:7680" } as const;
    const gpus = [gpu("L40S"), gpu("L40S")];
    expect(machineSentence(host, gpus, { address: true })).toBe(
      "2× L40S · CUDA · on your network at plato:7680",
    );
    expect(machineSentence(host, gpus, { uptimeSeconds: 5 })).toBe(
      "2× L40S · CUDA · on your network · up 5s",
    );
  });

  it("starts at where it is when nothing has reported hardware", () => {
    expect(machineSentence({ kind: "local" }, [])).toBe("this device");
  });
});
