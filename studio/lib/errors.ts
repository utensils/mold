/**
 * Turn host/API failures into copy that helps a person recover.
 *
 * Callers should keep the original error for diagnostics and clipboard copy;
 * this function is the presentation boundary shared by web, desktop, and
 * iPhone. Domain-authored messages pass through unless they match a transport
 * or runtime signature that would expose implementation jargon.
 */

const TRANSPORT_MESSAGE = new RegExp(
  [
    "^load failed$",
    "^fetch failed$",
    "^failed to fetch$",
    "^network error$",
    "networkerror when attempting to fetch resource",
    "network connection was lost",
    "^the request timed out\\.?$",
    "hostname could not be found",
    "internet connection appears to be offline",
    "could not connect to the server",
    "software caused connection abort",
    "socket is not connected",
    "stream closed before completion",
    "sse response has no body",
  ].join("|"),
  "i",
);

const SSE_HTTP_FAILURE = /^SSE request failed with HTTP (\d+)$/i;
const PLAIN_HTTP_FAILURE = /^(?:HTTP\s+)?(\d{3})(?::\s*(.*))?$/i;
const ENDPOINT_HTTP_FAILURE =
  /^(?:GET|POST|PUT|PATCH|DELETE)\s+\S+\s+failed(?::|\s+with)?\s*(\d{3})(?:\s+[^:]*:\s*(.*))?$/i;

interface StructuredFailure {
  status: number;
  message?: string;
  body?: unknown;
}

function hostLabel(hostName: string | null | undefined): string {
  return hostName?.trim() || "the host";
}

export function errorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  return error == null ? "" : String(error);
}

function structuredFailure(error: unknown): StructuredFailure | null {
  if (typeof error !== "object" || error === null || !("status" in error))
    return null;
  const status = Number((error as { status: unknown }).status);
  if (!Number.isFinite(status)) return null;
  return {
    status,
    message: errorMessage(error),
    body: (error as { body?: unknown }).body,
  };
}

function bodyValue(body: unknown, key: string): string {
  if (typeof body !== "object" || body === null || !(key in body)) return "";
  const value = (body as Record<string, unknown>)[key];
  return typeof value === "string" ? value.trim() : "";
}

function diagnosticCode(failure: StructuredFailure): string {
  return bodyValue(failure.body, "code").toUpperCase();
}

function diagnosticDetail(failure: StructuredFailure): string {
  return (
    bodyValue(failure.body, "error") ||
    bodyValue(failure.body, "message") ||
    failure.message?.trim() ||
    ""
  );
}

function memoryFailure(message: string, host: string): string | null {
  if (
    /(?:out of memory|insufficient memory|allocation failed|memory allocation|cuda.*oom|\boom\b)/i.test(
      message,
    )
  ) {
    return `${host} ran out of memory. Try a smaller model, image size, or batch.`;
  }
  return null;
}

function describeHttpFailure(
  status: number,
  detail: string,
  host: string,
  code = "",
): string {
  const memory =
    code === "INSUFFICIENT_MEMORY"
      ? memoryFailure("out of memory", host)
      : null;
  if (memory) return memory;
  if (status === 401 || status === 403) {
    return `${host} didn’t accept the API key. Update it in Machines and try again.`;
  }
  if (status === 404) {
    return `${host} couldn’t find that model or job. It may have been removed or the host may have restarted.`;
  }
  if (status === 408 || status === 504) {
    return `${host} took too long to respond. Try again.`;
  }
  if (status === 413) {
    return `That input is too large for ${host}. Try a smaller source file or output size.`;
  }
  if (status === 426) {
    return `${host} needs a newer version of Mold before it can do that.`;
  }
  if (status === 429 || code === "QUEUE_FULL") {
    return `${host} is busy right now. Wait a moment and try again.`;
  }
  if (status === 503 && !detail.trim()) {
    return `${host} is unavailable or still starting. Try again shortly.`;
  }
  if (status >= 500 && !detail.trim()) {
    return `${host} hit an internal error. Try again.`;
  }
  return detail.trim() || `${host} couldn’t complete the request.`;
}

/** Whether a failure is the connection itself dying rather than an HTTP reply. */
export function isTransportFailure(error: unknown): boolean {
  if (structuredFailure(error)) return false;
  if (error instanceof TypeError) return true;
  return TRANSPORT_MESSAGE.test(errorMessage(error).trim());
}

export function describeTransportError(
  error: unknown,
  hostName?: string | null,
): string {
  const host = hostLabel(hostName);
  const structured = structuredFailure(error);
  if (structured) {
    const detail = diagnosticDetail(structured);
    const memory = memoryFailure(detail, host);
    if (memory) return memory;
    return describeHttpFailure(
      structured.status,
      detail,
      host,
      diagnosticCode(structured),
    );
  }

  if ((error as { name?: unknown } | null | undefined)?.name === "AbortError") {
    return "Cancelled";
  }

  const message = errorMessage(error).trim();
  const memory = memoryFailure(message, host);
  if (memory) return memory;

  const sse = SSE_HTTP_FAILURE.exec(message);
  if (sse) return describeHttpFailure(Number(sse[1]), "", host);
  const endpoint = ENDPOINT_HTTP_FAILURE.exec(message);
  if (endpoint) {
    return describeHttpFailure(Number(endpoint[1]), endpoint[2] ?? "", host);
  }
  const plain = PLAIN_HTTP_FAILURE.exec(message);
  if (plain) return describeHttpFailure(Number(plain[1]), plain[2] ?? "", host);

  if (isTransportFailure(error)) {
    return `Couldn’t reach ${host}. Check the connection and try again.`;
  }
  if (
    error instanceof SyntaxError ||
    /JSON Parse error|Unexpected token|unexpected end of JSON/i.test(message)
  ) {
    return `${host} sent an unexpected reply.`;
  }
  if (message) return message;
  return `Something went wrong talking to ${host}.`;
}

/** Include the raw diagnostic only when it adds information to the friendly copy. */
export function copyableError(
  error: unknown,
  friendly: string = describeTransportError(error),
): string {
  const raw = errorMessage(error).trim();
  return raw && raw !== friendly
    ? `${friendly}\n\nTechnical details: ${raw}`
    : friendly;
}
