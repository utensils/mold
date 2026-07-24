// happy-dom v20 ships a file-backed localStorage that requires a CLI flag
// (`--localstorage-file`) to initialize. Under vitest we just want an
// in-memory store, so we replace the global with a minimal shim that
// matches the Storage interface used by the app.
class MemoryStorage implements Storage {
  private map = new Map<string, string>();
  get length(): number {
    return this.map.size;
  }
  clear(): void {
    this.map.clear();
  }
  getItem(key: string): string | null {
    return this.map.has(key) ? (this.map.get(key) as string) : null;
  }
  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }
  removeItem(key: string): void {
    this.map.delete(key);
  }
  setItem(key: string, value: string): void {
    this.map.set(key, String(value));
  }
}

Object.defineProperty(globalThis, "localStorage", {
  value: new MemoryStorage(),
  writable: true,
  configurable: true,
});

// happy-dom has no canvas implementation (`getContext` returns null), but the
// shared DevelopCanvas primitive (@ui) paints the develop grain on a real 2D
// context at mount. Provide a minimal no-op context so components that layer
// the grain can mount in tests without stubbing the component everywhere.
function stubContext2d(): object {
  return {
    imageSmoothingEnabled: true,
    globalAlpha: 1,
    globalCompositeOperation: "source-over",
    fillStyle: "",
    clearRect() {},
    fillRect() {},
    drawImage() {},
    putImageData() {},
    createPattern() {
      return null;
    },
  };
}

const canvasProto = (
  globalThis as unknown as {
    HTMLCanvasElement?: { prototype: { getContext: unknown } };
  }
).HTMLCanvasElement?.prototype;
if (canvasProto) {
  Object.defineProperty(canvasProto, "getContext", {
    value: () => stubContext2d(),
    writable: true,
    configurable: true,
  });
}

// happy-dom ships a ResizeObserver, but keep a guard for environments that
// don't — DevelopCanvas sizes its backing store from one.
if (typeof globalThis.ResizeObserver === "undefined") {
  class NoopResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  Object.defineProperty(globalThis, "ResizeObserver", {
    value: NoopResizeObserver,
    writable: true,
    configurable: true,
  });
}
