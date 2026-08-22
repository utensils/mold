/**
 * Read advisories attached to an accepted browser request.
 *
 * Each header value is one opaque advisory. In particular, never split on
 * `; `: advisory prose may contain semicolons. The optional `getAll` branch
 * preserves one-header-per-advisory responses in runtimes that expose it.
 * Standard browser `Headers` irreversibly combines repeated response fields,
 * so the server must keep emitting one joined field until the wire gains a
 * structured encoding; browsers deliberately surface that combined value as
 * one advisory rather than guessing at prose delimiters.
 */
export function requestWarningsFromHeaders(
  headers: Pick<Headers, "get"> & { getAll?: (name: string) => string[] },
): string[] {
  let all: string[] = [];
  try {
    all = headers.getAll?.("x-mold-request-warning") ?? [];
  } catch {
    // Bun and some browser-compatible runtimes expose getAll only for
    // Set-Cookie. Their ordinary response fields still remain available via get().
  }
  const values = all.length > 0 ? all : [headers.get("x-mold-request-warning")];
  return values.flatMap((value) => {
    const warning = value?.trim();
    return warning ? [warning] : [];
  });
}

/**
 * Read advisories off an SSE completion event.
 *
 * A streaming render has no response headers to carry them: the only headers
 * a caller ever sees arrived before the job ran, and an advisory the render
 * itself produced — which of several faces the identity extractor conditioned
 * on — is decided long after that. The server therefore repeats them in the
 * completion payload, and this is the one reader.
 *
 * Defensive about the shape because the field is additive: an older server
 * omits it entirely, and nothing here may throw on a completion event that
 * otherwise delivered a perfectly good print.
 */
export function requestWarningsFromCompleteEvent(event: unknown): string[] {
  if (typeof event !== "object" || event === null) return [];
  const raw = (event as { request_warnings?: unknown }).request_warnings;
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((value) => {
    const warning = typeof value === "string" ? value.trim() : "";
    return warning ? [warning] : [];
  });
}
