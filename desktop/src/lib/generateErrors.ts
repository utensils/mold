/**
 * Classifies a failed generation's error string. A 404 from the generate
 * endpoints means the routed host doesn't know the model (it was never
 * pulled there, or is a catalog model saved under a name the host lacks) —
 * the Generate view offers pull-and-resume instead of the raw error.
 */
export function isMissingModelError(error: string | null | undefined): boolean {
  if (!error) return false;
  return /HTTP 404\b/.test(error) || /model .+ (?:not found|unknown)/i.test(error);
}
