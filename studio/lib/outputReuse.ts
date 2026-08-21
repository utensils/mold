/**
 * Runtime-resolved pipeline metadata is provenance, not always an authored
 * override. An automatically chained One shot records the pipeline used by
 * its stages, but restoring that value as an explicit selector disables the
 * same chain route and collapses the duration control to "1 generation".
 */
export function pipelineForSettingsReuse<T>(metadata: {
  pipeline?: T | null;
  pipeline_requested?: boolean | null;
  output_mode?: "one-shot" | "sequence" | null;
  chain?: unknown | null;
  chain_job_id?: string | null;
}): T | null {
  if (metadata.pipeline_requested === true) return metadata.pipeline ?? null;
  if (metadata.pipeline_requested === false) return null;

  // Before `pipeline_requested` was added, ordinary LTX outputs also stamped
  // the runtime-resolved default into `pipeline`. There is no authored/runtime
  // discriminator in those rows, so treating a legacy One shot value as an
  // override recreates the wrong request and disables duration chaining.
  // A canonical authored Sequence is the only legacy shape whose pipeline is
  // safe to retain; durable pre-output_mode sequences are identified by their
  // chain job. Every ordinary legacy output must fall back to Auto.
  const authoredSequence =
    metadata.output_mode === "sequence" ||
    (metadata.output_mode == null && Boolean(metadata.chain_job_id));
  return authoredSequence ? (metadata.pipeline ?? null) : null;
}
