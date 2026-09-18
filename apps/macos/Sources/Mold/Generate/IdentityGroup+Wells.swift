import MoldClient
import MoldStyle
import SwiftUI

// The group's two kinds of square -- a staged photograph and the add well --
// each a `PictureWell` with its own destination. Split from the group's own
// shape purely for size.
extension IdentityGroup {
    /// A staged photograph keeps its ✕, so it does not open its menu on a
    /// plain click -- a `Menu` label would swallow that button's taps.
    func photoWell(_ photo: IdentityPhoto) -> some View {
        PictureWell(
            rows: GenerateMenus.identityPhoto(),
            picture: photo.encoded,
            accepting: PictureImport.identityReadable,
            size: Self.photoSize,
            opensOnClick: false,
            label: photo.name,
            pick: { draft.media = Self.replacing(photo, with: $0, in: draft.media) },
            perform: { perform($0, on: photo) })
            .overlay(alignment: .topTrailing) { removeButton(photo) }
    }

    var addWell: some View {
        PictureWell(
            rows: GenerateMenus.identityAdd(canPaste: PicturePaste.hasPicture),
            placeholder: "person.crop.circle.badge.plus",
            accepting: PictureImport.identityReadable,
            allowsMultiple: true,
            size: Self.photoSize,
            caption: WellCaption.identityAdd,
            label: "Add a photograph of the face to preserve",
            pick: { draft.media = Self.staging($0, in: draft.media, maxPhotos: maxPhotos) })
    }

    private func removeButton(_ photo: IdentityPhoto) -> some View {
        Button {
            remove(photo)
        } label: {
            Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white, Chrome.badgeBackdrop)
        .padding(2)
        .help("Remove this photograph")
    }
}
