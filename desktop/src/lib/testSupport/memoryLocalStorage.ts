/**
 * Test-only in-memory `localStorage`. happy-dom v20 (the desktop vitest
 * environment) does not implement the Web Storage API, but the real Tauri
 * webview does — so code that persists to `localStorage` needs this shim under
 * test. Idempotent: safe to call from multiple test files.
 */
export function installMemoryLocalStorage(): void {
  const g = globalThis as { localStorage?: Storage };
  if (g.localStorage) return;

  const store = new Map<string, string>();
  const impl: Storage = {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string) {
      return store.has(key) ? store.get(key)! : null;
    },
    key(index: number) {
      return Array.from(store.keys())[index] ?? null;
    },
    removeItem(key: string) {
      store.delete(key);
    },
    setItem(key: string, value: string) {
      store.set(key, String(value));
    },
  };
  g.localStorage = impl;
}
