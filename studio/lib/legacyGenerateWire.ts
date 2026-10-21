/** Legacy GenerateRequest endpoints decode shell newline escapes once. Protect
 * literal backslashes only on a wire copy; new /video-jobs JSON stays literal. */
export function protectLegacyGenerateWire<T>(request: T): T {
  if (!request || typeof request !== "object" || Array.isArray(request))
    return request;
  const wire: Record<string, unknown> = {
    ...(request as Record<string, unknown>),
  };
  for (const field of ["prompt", "negative_prompt", "original_prompt"]) {
    if (typeof wire[field] === "string")
      wire[field] = wire[field].replaceAll("\\", "\\\\");
  }
  const source = wire.prompt_transform;
  if (source && typeof source === "object" && !Array.isArray(source)) {
    const transform = { ...source } as Record<string, unknown>;
    for (const field of ["source_prompt", "root_prompt"]) {
      if (typeof transform[field] === "string")
        transform[field] = transform[field].replaceAll("\\", "\\\\");
    }
    wire.prompt_transform = transform;
  }
  return wire as T;
}
