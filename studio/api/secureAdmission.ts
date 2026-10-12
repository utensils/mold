/** A recognized gateway pressure envelope before session admission. */
export class PreAdmissionProxyBusy extends Error {
  constructor() {
    super("The secure proxy is temporarily busy. Try connecting again.");
  }
}
/** Browser WebSockets hide upgrade HTTP status; retry only before socket open. */
export class PreOpenSocketFailure extends Error {
  constructor() {
    super("Could not reach this host securely.");
  }
}
function backoff(signal: AbortSignal, delay: number): Promise<void> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const finish = () => {
      signal.removeEventListener("abort", abort);
      resolve();
    };
    const timer = setTimeout(finish, delay);
    const abort = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      reject(signal.reason);
    };
    signal.addEventListener("abort", abort, { once: true });
  });
}
/** The callback admits a fresh channel; no HTTP request is ever replayed here. */
export async function retrySecureAdmission<T>(
  open: (signal: AbortSignal) => Promise<T>,
  parent?: AbortSignal,
): Promise<T> {
  const deadline = new AbortController();
  const timer = setTimeout(
    () => deadline.abort(new Error("Secure admission deadline exceeded.")),
    15000,
  );
  const signal = parent
    ? AbortSignal.any([parent, deadline.signal])
    : deadline.signal;
  try {
    for (let attempt = 0; ; attempt++) {
      signal.throwIfAborted();
      try {
        return await open(signal);
      } catch (error) {
        signal.throwIfAborted();
        if (
          !(
            error instanceof PreAdmissionProxyBusy ||
            error instanceof PreOpenSocketFailure
          ) ||
          attempt >= 2
        )
          throw error;
        await backoff(
          signal,
          200 * (attempt + 1) + Math.floor(Math.random() * 100),
        );
      }
    }
  } finally {
    clearTimeout(timer);
  }
}
