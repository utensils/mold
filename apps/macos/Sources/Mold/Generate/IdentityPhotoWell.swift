import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// One staged identity photograph.
///
/// Decoded once per encoded string rather than once per keystroke -- `body`
/// re-runs on every draft edit, and a photo restored by `RenderDraft+Park`
/// arrives as a plain encoded string with no cached `NSImage` behind it, so
/// decoding can never happen only at pick time. The same pattern
/// `ReferenceWell` uses for its own thumbnail.
struct IdentityPhotoWell: View {
    let photo: IdentityPhoto
    let onRemove: () -> Void

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
        .overlay(alignment: .topTrailing) { removeButton }
        .task(id: photo.encoded) { image = await PicturePreview.decode(photo.encoded) }
        .help(photo.name)
    }

    private var removeButton: some View {
        Button(action: onRemove) {
            Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white, Chrome.badgeBackdrop)
        .padding(2)
        .help("Remove this photograph")
    }
}
