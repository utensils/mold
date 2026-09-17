import AppKit
import MoldClient
import SwiftUI

// A mesh in the viewer. Split from the rest for size, and because this is the
// seam the interactive viewer replaces.
extension LibraryViewer {

    /// A mesh, as far as this app can show one today.
    ///
    /// The server renders a poster for every GLB and this draws it, at FULL
    /// opacity and with a line saying what it is. It used to fall through to
    /// the image arm, where `NSImage(data:)` returns nil for a GLB, so `full`
    /// stayed nil and the poster sat at 0.55 and `.interpolation(.low)` --
    /// this app's own visual language for "still loading" -- forever, with no
    /// error and no way to tell it from a slow download. Quick Look does not
    /// rescue it either: macOS ships no GLB preview generator.
    ///
    /// **The seam for the interactive viewer (M6/F1)**: an `MTKView` port of
    /// `studio/components/MeshViewer.vue` replaces the `posterOrGlyph` below,
    /// taking the GLB bytes from `actions.data(for:)` and falling back to
    /// exactly this on any failure. Its home view IS this poster, by
    /// construction -- `raster::sweep_fit_for` frames both.
    @ViewBuilder var mesh: some View {
        VStack(spacing: 10) {
            posterOrGlyph
            Text("3-D object · Save a copy or File ▸ Export As to open it elsewhere")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(24)
    }

    @ViewBuilder private var posterOrGlyph: some View {
        if let placeholder {
            Image(nsImage: placeholder)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fit)
        } else {
            Image(systemName: "cube")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
                .accessibilityLabel("3-D object")
        }
    }
}
