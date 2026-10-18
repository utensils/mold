/** Recipe comparisons preserve JSON integer tokens, including Rust u64 seeds. */
function fields(text: string): Map<string, string> {
  let cursor = 0;
  const skip = () => {
    while (/\s/.test(text[cursor] ?? "") && cursor < text.length) cursor++;
  };
  function string(): string {
    const start = cursor++;
    while (cursor < text.length) {
      const char = text[cursor++]!;
      if (char === "\\") cursor++;
      else if (char === '"') break;
    }
    return text.slice(start, cursor);
  }
  function value(): string {
    skip();
    if (text[cursor] === '"') return JSON.stringify(JSON.parse(string()));
    if (text[cursor] === "{" || text[cursor] === "[") {
      const object = text[cursor++] === "{";
      const values: string[] = [];
      skip();
      while (cursor < text.length && text[cursor] !== (object ? "}" : "]")) {
        if (object) {
          const key = JSON.stringify(JSON.parse(string()));
          skip();
          cursor++;
          values.push(key + ":" + value());
        } else values.push(value());
        skip();
        if (text[cursor] !== ",") break;
        cursor++;
        skip();
      }
      cursor++;
      return object
        ? "{" + values.sort().join(",") + "}"
        : "[" + values.join(",") + "]";
    }
    const start = cursor;
    while (cursor < text.length && !/[\s,}\]]/.test(text[cursor]!)) cursor++;
    const token = text.slice(start, cursor);
    if (/^-?\d+$/.test(token)) return BigInt(token).toString();
    if (/^-?\d/.test(token)) return JSON.stringify(Number(token));
    return token;
  }
  JSON.parse(text); // Validate before the bounded canonical scan.
  skip();
  if (text[cursor++] !== "{")
    throw new Error("Recipe metadata must be an object.");
  const result = new Map<string, string>();
  skip();
  while (text[cursor] !== "}" && cursor < text.length) {
    const key = JSON.parse(string()) as string;
    skip();
    cursor++;
    result.set(key, value());
    skip();
    if (text[cursor] !== ",") break;
    cursor++;
    skip();
  }
  return result;
}
export function mirrorMetadataMatches(
  first: string,
  second: string,
  verifiedEmbedded?: string,
): boolean {
  const left = fields(first),
    right = fields(second);
  const missing = (map: Map<string, string>, key: string) =>
    !map.has(key) || map.get(key) === "null";
  for (const key of ["job_id", "generation_time_ms", "has_alpha"]) {
    if (missing(left, key) || missing(right, key)) {
      left.delete(key);
      right.delete(key);
    }
  }
  if (left.has("version") && right.has("version")) {
    const a = JSON.parse(left.get("version")!),
      b = JSON.parse(right.get("version")!);
    if (typeof a === "string" && typeof b === "string") {
      const short = (version: string) =>
        version.replace(/ \([0-9a-f]{7,40} \d{4}-\d{2}-\d{2}\)$/, "");
      if (short(a) === short(b) && (a === short(a) || b === short(b))) {
        left.set("version", JSON.stringify(short(a)));
        right.set("version", JSON.stringify(short(b)));
      }
    }
  }
  if (verifiedEmbedded) {
    const embedded = fields(verifiedEmbedded);
    for (const key of ["scheduler", "transparent_background"]) {
      if (missing(left, key) !== missing(right, key)) {
        const present = missing(left, key) ? right.get(key) : left.get(key);
        if (!missing(embedded, key) && embedded.get(key) === present) {
          left.delete(key);
          right.delete(key);
        }
      }
    }
  }
  return (
    left.size === right.size &&
    [...left].every(([key, value]) => right.get(key) === value)
  );
}

/** Port of MoldClient.EmbeddedPrintMetadata; caller must first hash exact output bytes. */
export function embeddedMirrorMetadata(bytes: Uint8Array): string | undefined {
  const decoder = new TextDecoder(),
    limit = 2 * 1024 * 1024;
  const text = (payload: Uint8Array, prefix = ""): string | undefined => {
    if (payload.length > limit) return;
    const value = decoder.decode(payload);
    if (!value.startsWith(prefix)) return;
    const json = value.slice(prefix.length);
    try {
      const parsed = JSON.parse(json);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed))
        return json;
    } catch {
      /* not Mold metadata */
    }
  };
  if (bytes[0] === 137 && decoder.decode(bytes.subarray(1, 4)) === "PNG") {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let cursor = 8; cursor + 12 <= bytes.length;) {
      const length = view.getUint32(cursor);
      if (length > bytes.length - cursor - 12) return;
      const kind = decoder.decode(bytes.subarray(cursor + 4, cursor + 8));
      const payload = bytes.subarray(cursor + 8, cursor + 8 + length);
      const separator = payload.indexOf(0);
      if (
        length <= limit &&
        separator >= 0 &&
        decoder.decode(payload.subarray(0, separator)) === "mold:parameters"
      ) {
        if (kind === "tEXt") return text(payload.subarray(separator + 1));
        if (kind === "iTXt" && payload[separator + 1] === 0) {
          const languageEnd = payload.indexOf(0, separator + 3),
            translatedEnd = payload.indexOf(0, languageEnd + 1);
          if (languageEnd >= 0 && translatedEnd >= 0)
            return text(payload.subarray(translatedEnd + 1));
        }
      }
      cursor += length + 12;
    }
  } else if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    for (
      let cursor = 2;
      cursor + 4 <= bytes.length && bytes[cursor] === 0xff;
    ) {
      const marker = bytes[cursor + 1]!;
      if (marker === 0xd9 || marker === 0xda) return;
      if (
        marker === 0xd8 ||
        marker === 0x01 ||
        (marker >= 0xd0 && marker <= 0xd7)
      ) {
        cursor += 2;
        continue;
      }
      const length = bytes[cursor + 2]! * 256 + bytes[cursor + 3]!;
      if (length < 2 || length > bytes.length - cursor - 2) return;
      if (marker === 0xfe) {
        const result = text(
          bytes.subarray(cursor + 4, cursor + 2 + length),
          "mold:parameters ",
        );
        if (result) return result;
      }
      cursor += length + 2;
    }
  } else if (decoder.decode(bytes.subarray(0, 4)) === "GIF8") {
    for (let cursor = 6; cursor + 2 < bytes.length; cursor++) {
      if (bytes[cursor] !== 0x21 || bytes[cursor + 1] !== 0xfe) continue;
      cursor += 2;
      const chunks: Uint8Array[] = [];
      let count = 0;
      while (cursor < bytes.length) {
        const size = bytes[cursor++]!;
        if (!size) break;
        if (size > bytes.length - cursor || (count += size) > limit) return;
        chunks.push(bytes.subarray(cursor, cursor + size));
        cursor += size;
      }
      const payload = new Uint8Array(count);
      let offset = 0;
      for (const chunk of chunks) {
        payload.set(chunk, offset);
        offset += chunk.length;
      }
      const result = text(payload, "mold:parameters ");
      if (result) return result;
    }
  }
}
