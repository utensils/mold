import Foundation
import Testing

@testable import MoldClient

/// The one list the mesh view's controls, its contextual menu and the
/// Library's menu all draw.
///
/// **Fails today**: a mesh print had no menu at all — the viewer showed a
/// poster and a sentence, and Export's formats came from a hard-coded client
/// set rather than from the host.
@Suite struct MeshViewMenuSuite {

    private func plan(exports: [String] = ["glb", "obj", "stl", "gif", "webp"],
                      hasEdges: Bool = true, isWireframe: Bool = false,
                      isAutoRotating: Bool = false, offersAutoRotate: Bool = true,
                      canSave: Bool = true,
                      canShowInLibrary: Bool = true) -> MeshViewMenuPlan {
        MeshViewMenuPlan(exports: MeshExport.split(exports), hasEdges: hasEdges,
                         isWireframe: isWireframe, isAutoRotating: isAutoRotating,
                         offersAutoRotate: offersAutoRotate, canSave: canSave,
                         canShowInLibrary: canShowInLibrary)
    }

    @Test func offersTheViewControlsThenTheHostsExportsThenTheFileActions() {
        let items = plan().items
        #expect(items.map(\.title) == ["Reset View", "Show Wireframe", "Turn Slowly",
                                       "Export", "Save a Copy…", "Show in Library"])
        let exports = items.first { $0.title == "Export" }?.children ?? []
        // The host's order, the stored `glb` dropped, and the two animated
        // containers collapsed into ONE entry that opens the sheet.
        #expect(exports.map(\.title) == ["OBJ", "STL", "Turntable…"])
        #expect(exports.first?.kind == .exportFile(format: "obj"))
        #expect(exports.last?.kind == .exportTurntable)
    }

    /// A submenu with nothing in it is a dead end, so a host that advertises
    /// no exports leaves no Export row behind at all.
    @Test func dropsExportEntirelyForAHostThatAdvertisesNone() {
        #expect(!plan(exports: []).items.contains { $0.title == "Export" })
        #expect(!plan(exports: ["glb"]).items.contains { $0.title == "Export" })
    }

    /// The wording says what the control will DO, not what is true now.
    @Test func namesTheStateEachToggleWillLeaveBehind() {
        #expect(plan(isWireframe: true).items.map(\.title).contains("Hide Wireframe"))
        #expect(plan(isAutoRotating: true).items.map(\.title).contains("Stop Turning"))
    }

    /// A mesh whose every triangle is degenerate has nothing to outline, so
    /// the control is ABSENT rather than present and inert.
    @Test func offersNoWireframeForAMeshWithNoEdges() {
        #expect(!plan(hasEdges: false).items.contains { $0.kind == .toggleWireframe })
    }

    /// The tour is a property of the surface: the Library's viewer offers it
    /// and a Generate result does not, and an interaction retires it.
    @Test func offersTheTourOnlyWhereTheSurfaceWantsOne() {
        #expect(!plan(offersAutoRotate: false).items.contains { $0.kind == .toggleAutoRotate })
    }

    /// The surface that IS the Library does not offer to show you the Library.
    @Test func dropsTheActionsASurfaceCannotPerform() {
        let bare = plan(canSave: false, canShowInLibrary: false).items
        #expect(bare.map(\.title) == ["Reset View", "Show Wireframe", "Turn Slowly", "Export"])
    }

    /// Nothing here can be taken back, so the list carries no divider and no
    /// destructive row -- and it is never empty, because Reset View always
    /// applies to a mesh that is on screen.
    @Test func alwaysOffersAMenu() {
        #expect(RowAction.offersMenu(plan(exports: [], hasEdges: false,
                                          offersAutoRotate: false, canSave: false,
                                          canShowInLibrary: false).items))
        #expect(!plan().items.contains { $0.isDestructive || $0.isSeparator })
    }
}
