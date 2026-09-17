import AppKit
import MoldStyle
import SwiftUI

/// One reference thumbnail, decoded once per encoded string rather than once
/// per keystroke -- `body` re-runs on every draft edit (a slider drag fires
/// many), and `Data(base64Encoded:)` plus `NSImage(data:)` were both inside
/// it. The pattern `RunCanvas` uses for its own preview and result images.
struct ReferenceWell: View {
    let encoded: String

    @State private var image: NSImage?

    var body: some View {
        ZStack {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Chrome.wellFill
            }
        }
        .frame(width: 52, height: 52)
        .clipShape(RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous))
        .task(id: encoded) { image = await PicturePreview.decode(encoded) }
    }
}
