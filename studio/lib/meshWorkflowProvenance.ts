/** Historical server-minted mesh workflow metadata; no app authoring or navigation. */
export interface MeshWorkflowProvenance {
  job_id: string;
  /** `text_to_mesh` | `mesh_roundtrip` | `mesh_texture`, or a newer host's. */
  mode: string;
  /** `generated_image` | `matted_image` | `delighted_image` | `final_glb`. */
  role: string;
  stage_index: number;
}
