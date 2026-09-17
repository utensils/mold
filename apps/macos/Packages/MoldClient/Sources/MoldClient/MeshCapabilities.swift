import Foundation

/// What a host can do with 3-D artifacts, as `GET /api/capabilities` reports
/// it (`crates/mold-core/src/types.rs:11956-11996`).
///
/// ABSENT means a host with no mesh family at all — one that is too old to
/// carry the block, or one built without the manifests. A client reading an
/// absent block must not offer 3-D exports, because such a host has nothing
/// to export.
public struct MeshCapabilities: Codable, Hashable, Sendable {
    /// A mesh family has a runnable ENGINE here. False on a host that carries
    /// the manifests and the contract but no engine arm, which is a real
    /// state and not "this server is too old".
    public let generation: Bool
    /// Formats this host will STORE. GLB only today.
    public let formats: [String]
    /// Formats `POST /api/gallery/export/:filename` transcodes a stored mesh
    /// into, with the stored `glb` listed FIRST so a client can see what it
    /// holds. The server has already dropped any name IT does not know
    /// (`known_mesh_export_formats`, `types.rs:11999-12009`).
    public let exportFormats: [String]?
    /// Whether generated PBR textures are available.
    public let textures: Bool?
    /// The geometry controls a geometry export accepts. Absence is the ONLY
    /// gate — see `MeshExportGeometryCapabilities`.
    public let exportGeometry: MeshExportGeometryCapabilities?

    public init(generation: Bool, formats: [String] = [], exportFormats: [String]? = nil,
                textures: Bool? = nil,
                exportGeometry: MeshExportGeometryCapabilities? = nil) {
        self.generation = generation
        self.formats = formats
        self.exportFormats = exportFormats
        self.textures = textures
        self.exportGeometry = exportGeometry
    }
}

public extension Capabilities {
    /// The split export menu for a mesh print on this host, built from what
    /// the host advertises and NEVER from a list this app carries.
    ///
    /// Empty on a host with no `mesh` block: absence there is "no mesh family
    /// here", so there is nothing to convert.
    var meshExports: MeshExport.Split {
        MeshExport.split(mesh?.exportFormats)
    }

    /// The geometry knobs to OFFER for one container, or nil to post the bare
    /// format. Nil covers three different hosts and all three want the same
    /// request: one with no mesh block, one that predates `export_geometry`,
    /// and one that does not scale this container.
    func meshGeometryDefaults(for format: String) -> MeshExportGeometry? {
        MeshExportGeometry.defaults(mesh?.exportGeometry, format: format)
    }
}
