/**
 * Whether a completion is a 3-D mesh. Keys on `mesh_vertices` exactly as the
 * server's own probe does; every surface tests this BEFORE
 * `isAudioCompletion` and the video probes, because a mesh carries neither
 * frames nor samples and a wider test running first would classify it as a
 * still and try to draw glTF bytes.
 */
export type MeshCompletionProbe = { mesh_vertices?: number | null };

export function isMeshCompletion(
  result: MeshCompletionProbe | null | undefined,
): boolean {
  return result?.mesh_vertices != null;
}
