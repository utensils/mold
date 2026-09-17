import Foundation

/// What can be done to a mesh that is on screen.
///
/// The view controls, the tile's contextual menu and the Library's own menu
/// draw ONE list, so a name learned in one place is found in the others.
public enum MeshViewAction: Hashable, Sendable {
    case resetView
    case toggleWireframe
    case toggleAutoRotate
    /// A one-click transcode into a container the HOST advertised.
    case exportFile(format: String)
    /// The animated containers share one entry, which opens the sheet.
    case exportTurntable
    case save
    case showInLibrary
}

/// What to offer for the mesh on screen.
public struct MeshViewMenuPlan: Sendable {
    /// The host's advertised export list, already split.
    public let exports: MeshExport.Split
    /// False for a mesh whose every triangle is degenerate: nothing to outline.
    public let hasEdges: Bool
    public let isWireframe: Bool
    public let isAutoRotating: Bool
    /// The tour is only ever offered where the surface wants one AND this
    /// view has not been handled yet -- an interaction is never taken back.
    public let offersAutoRotate: Bool
    public let canSave: Bool
    /// Absent on the surface that IS the Library.
    public let canShowInLibrary: Bool

    public init(exports: MeshExport.Split, hasEdges: Bool, isWireframe: Bool,
                isAutoRotating: Bool, offersAutoRotate: Bool, canSave: Bool,
                canShowInLibrary: Bool) {
        self.exports = exports
        self.hasEdges = hasEdges
        self.isWireframe = isWireframe
        self.isAutoRotating = isAutoRotating
        self.offersAutoRotate = offersAutoRotate
        self.canSave = canSave
        self.canShowInLibrary = canShowInLibrary
    }

    public typealias Item = RowAction<MeshViewAction>

    /// The items, in the order they are shown.
    ///
    /// Nothing here is destructive, so the list is flat and `rendered` leaves
    /// its order alone but still drops an Export submenu with nothing in it --
    /// which is exactly what a host advertising no exports leaves behind.
    public var items: [Item] {
        var items: [Item] = [Item(kind: .resetView, title: "Reset View")]
        if hasEdges {
            items.append(Item(kind: .toggleWireframe,
                              title: isWireframe ? "Hide Wireframe" : "Show Wireframe"))
        }
        if offersAutoRotate {
            items.append(Item(kind: .toggleAutoRotate,
                              title: isAutoRotating ? "Stop Turning" : "Turn Slowly"))
        }
        var exportItems = exports.files.map {
            Item(kind: .exportFile(format: $0), title: $0.uppercased())
        }
        if !exports.animations.isEmpty {
            exportItems.append(Item(kind: .exportTurntable, title: "Turntable…"))
        }
        items.append(Item(title: "Export", children: exportItems))
        if canSave { items.append(Item(kind: .save, title: "Save a Copy…")) }
        if canShowInLibrary {
            items.append(Item(kind: .showInLibrary, title: "Show in Library"))
        }
        return RowAction.rendered(items)
    }
}
