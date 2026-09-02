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

/** A completion as the viewer question reads it: facts, or the container. */
export type MeshArtifactProbe = MeshCompletionProbe & {
  format?: string | null;
};

/**
 * Whether a completion's stored ARTIFACT is binary glTF.
 *
 * A live SSE completion carries the server's own mesh facts, so
 * {@link isMeshCompletion} answers for it. A completion a client SYNTHESIZES
 * from a durable batch child carries only what that child names — a filename
 * and the requested container — because the batch route reports no vertex
 * counts at all. Keying the viewer on the facts therefore read a finished 3-D
 * print as a still and handed the `<img>` arm a `.glb`, which is the broken
 * resource icon the Create canvas showed after every durable mesh generation
 * while the same print opened correctly from the Library.
 *
 * The container is the other thing the server is authoritative about, and it
 * is present on BOTH paths, so it answers whenever the facts are absent. Ask
 * this wherever the question is "which viewer does this print get";
 * `isMeshCompletion` remains "did the server report mesh facts".
 */
export function isMeshArtifact(
  result: MeshArtifactProbe | null | undefined,
): boolean {
  return (
    isMeshCompletion(result) ||
    (result?.format ?? "").trim().toLowerCase() === "glb"
  );
}
