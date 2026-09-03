/**
 * `studio/components/MeshViewer.vue` is the one gallery component that loads
 * its media with `fetch()` (every other viewer points an `<img>`/`<video>`
 * element straight at a URL), and CSP governs a `fetch()` load through
 * `connect-src`, not `img-src`/`media-src`. On desktop, `fullSizeMediaUrl`
 * (`desktop/src/lib/gallery/media.ts`) hands the viewer a `blob:` object URL
 * for a mesh print's bytes fetched over the native IPC bridge — but WebKit
 * does not match `blob:` against `'self'`, and `connect-src` never carried
 * `blob:`, so a packaged build's Library lightbox and Create canvas mesh
 * viewer always fell through to "The 3-D view couldn't start, so here's the
 * poster." `desktop-dev` applies no CSP at all (`beforeDevCommand` has no
 * `devCsp`), so this was invisible outside a real bundle — this test reads
 * the shipped policy directly rather than relying on a packaged-build UAT.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath, URL as NodeURL } from "node:url";
import path from "node:path";

// happy-dom's global URL resolves a relative constructor arg against its
// fake `http://localhost:3000/` document location rather than the given
// base, so this reads the file relative to this test file using Node's URL
// explicitly (import.meta.url itself is a real file: URL either way).
const configPath = fileURLToPath(new NodeURL("../src-tauri/tauri.conf.json", import.meta.url));

function parseCsp(csp: string): Map<string, Set<string>> {
  const directives = new Map<string, Set<string>>();
  for (const clause of csp.split(";")) {
    const tokens = clause.trim().split(/\s+/).filter(Boolean);
    if (tokens.length === 0) continue;
    const [name, ...sources] = tokens;
    directives.set(name!, new Set(sources));
  }
  return directives;
}

describe("desktop tauri.conf.json CSP", () => {
  const raw = readFileSync(configPath, "utf-8");
  const config = JSON.parse(raw) as { app: { security: { csp: string } } };
  const directives = parseCsp(config.app.security.csp);

  it(`reads the config from ${path.basename(configPath)}`, () => {
    expect(directives.size).toBeGreaterThan(0);
  });

  // Every scheme a `fetch()`-loading component (MeshViewer) can be handed by
  // `fullSizeMediaUrl`/`streamableMediaUrl`: `blob:` for the native-IPC mesh
  // route, `http:`/`https:` for a direct or ticketed remote stream, and
  // `mold-local:` for a byte on this device served over the restricted
  // native protocol. `'self'` covers the app's own origin.
  it("connect-src allows every scheme fullSizeMediaUrl can hand a fetch()-loading component", () => {
    const connectSrc = directives.get("connect-src");
    expect(connectSrc).toBeDefined();
    for (const scheme of ["'self'", "blob:", "http:", "https:", "mold-local:"]) {
      expect(connectSrc, `connect-src should allow ${scheme}`).toContain(scheme);
    }
  });

  // The policy is a full description here, not a one-liner: img-src and
  // media-src already carried blob: (an <img>/<video> element can be pointed
  // straight at a blob: URL under 'self' in a way fetch() cannot), and that
  // must stay true alongside the connect-src fix above.
  it("keeps blob: on img-src and media-src for <img>/<video> elements", () => {
    const imgSrc = directives.get("img-src");
    const mediaSrc = directives.get("media-src");
    expect(imgSrc).toContain("blob:");
    expect(mediaSrc).toContain("blob:");
  });
});
