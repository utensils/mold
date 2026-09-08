/** Share only supported view state. Never copy arbitrary query parameters,
 * credentials, or authenticated media URLs into a link. Host IDs refer only
 * to machines already connected by the person opening the link. */
export function libraryLink(
  origin: string,
  query: Record<string, unknown>,
): string {
  const url = new URL("/library", origin);
  for (const key of [
    "scope",
    "c",
    "tag",
    "fav",
    "q",
    "type",
    "host",
    "print",
    "printHost",
  ]) {
    const value = query[key];
    if (typeof value === "string" && value) url.searchParams.set(key, value);
  }
  return url.href;
}
