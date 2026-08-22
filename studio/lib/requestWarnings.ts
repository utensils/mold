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
