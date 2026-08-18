/**
 * Runtime-resolved pipeline metadata is provenance, not always an authored
 * override. An automatically chained One shot records the pipeline used by
 * its stages, but restoring that value as an explicit selector disables the
 * same chain route and collapses the duration control to "1 generation".
 */
export function pipelineForSettingsReuse<T>(metadata: {
  pipeline?: T | null;
  output_mode?: "one-shot" | "sequence" | null;
  chain?: unknown | null;
  chain_job_id?: string | null;
}): T | null {
  const automaticChain =
    Boolean(metadata.chain) &&
    (metadata.output_mode === "one-shot" ||
      (metadata.output_mode !== "sequence" && !metadata.chain_job_id));
  if (automaticChain) return null;
  return metadata.pipeline ?? null;
}
