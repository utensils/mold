import AppKit
import MoldClient
import SwiftUI

// A mesh in the viewer. Split from the rest for size, and because this is the
// seam the interactive viewer replaced.
extension LibraryViewer {

    /// A mesh, drawn.
    ///
    /// `MeshCanvas` is the native port of `studio/components/MeshViewer.vue`:
    /// the same orthographic camera, the same key/fill/rim shading, and a home
    /// view that IS the poster the gallery tile already shows. The poster
    /// stays underneath it and is what a failure of any kind lands on -- no
    /// Metal device, a refused download, a corrupt file -- with one line
    /// saying why. It used to fall through to the image arm, where
    /// `NSImage(data:)` returns nil for a GLB, so the poster sat at 0.55 and
    /// `.interpolation(.low)` -- this app's own visual language for "still
    /// loading" -- forever, with no error and no way to tell it from a slow
    /// download.
    @ViewBuilder var mesh: some View {
        MeshCanvas(
            printID: "\(entry.id.host)#\(entry.id.filename)",
            fetch: { try await actions.meshBytes(for: entry) },
            poster: placeholder,
            alt: entry.print.displayName,
            // The Library tours a mesh it is showing; the first drag, key or
            // wheel ends it, and Reduce Motion never starts it.
            offersAutoRotate: true,
            exports: actions.meshExports(for: entry),
            canSave: true,
            // This IS the Library; offering to show it here would be a door
            // back into the room you are standing in.
            canShowInLibrary: false,
            perform: perform)
        .padding(24)
    }

    /// The file actions the canvas cannot perform itself, answered by the
    /// Library's own one door so a mesh's Export and Save are the same two
    /// things the tile's menu offers.
    private func perform(_ action: MeshViewAction) {
        switch action {
        case let .exportFile(format):
            actions.requestExport(entry, as: format)
        case .exportTurntable:
            actions.requestTurntable(entry)
        case .save:
            actions.save([entry])
        case .resetView, .toggleWireframe, .toggleAutoRotate, .showInLibrary:
            // The canvas performs its own view controls, and this IS the
            // Library -- neither reaches here.
            break
        }
    }
}
