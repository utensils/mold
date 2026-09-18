import Foundation

/// The durable 3-D run a print belongs to, and the part it plays in it
/// (`MeshWorkflowProvenance`, `mesh_workflow.rs:176-190`).
///
/// Server-minted and refused on every generate door, so it is read-only
/// provenance like the rest of this file's neighbours. It is in its own file
/// rather than beside `MeshProvenance` because a workflow is not one of the
/// controls a mesh ran with -- it is which RUN the print came out of.
public struct MeshWorkflowProvenance: Codable, Hashable, Sendable {
    public let jobId: String?
    /// `text_to_mesh`, `mesh_roundtrip` or `mesh_texture`. A mode this build
    /// has never heard of is still displayable, which is why it is a String.
    public let mode: String?
    /// `generated_image`, `matted_image`, `delighted_image`, `final_glb`.
    /// `final_glb` is the run's LEAD -- the print a client shows when it
    /// collapses the run into one item.
    public let role: String?
    /// Zero-based, so a client can order a run's members without knowing the
    /// stage graph.
    public let stageIndex: Int?
}

public extension OutputMetadata {
    /// Every identity photograph's LABEL, in request order -- the names twin
    /// of `identityDigests`, and the same rule: the plural is recorded only
    /// for a multi-photograph print, so the singular stands in for the
    /// one-photograph case.
    var identityPhotoNames: [String] {
        if let plural = idImageNames, !plural.isEmpty { return plural }
        return idImageName.map { [$0] } ?? []
    }
}
