import MoldClient
import SwiftUI

// Where a picked photograph goes, and what one offers on a right-click. Split
// from the group's own shape purely for size.
//
// The two staging rules are VALUES, not view methods: the host's own photo
// limit has to be checked against the list a picture is JOINING rather than
// the one the pick started with -- a four-file drop onto a two-photo group
// must not stage four -- and a replacement has to keep its slot.
extension IdentityGroup {
    /// One photograph staged, if there is room for it.
    static func staging(
        _ picked: ImportedPicture, in media: DraftMedia, maxPhotos: Int
    ) -> DraftMedia {
        var media = media
        var conditioning = media.identity ?? IdentityConditioning(photos: [])
        guard conditioning.photos.count < maxPhotos else { return media }
        conditioning.photos.append(IdentityPhoto(encoded: picked.encoded, name: picked.name))
        media.identity = conditioning
        return media
    }

    /// One photograph replaced IN PLACE. A remove-then-add sent the
    /// replacement to the end of the group; one that is no longer staged --
    /// removed while the panel was open -- changes nothing rather than
    /// appending a stranger.
    static func replacing(
        _ photo: IdentityPhoto, with picked: ImportedPicture, in media: DraftMedia
    ) -> DraftMedia {
        var media = media
        guard var conditioning = media.identity,
              let index = conditioning.photos.firstIndex(where: { $0.id == photo.id })
        else { return media }
        conditioning.photos[index] = IdentityPhoto(encoded: picked.encoded, name: picked.name)
        media.identity = conditioning
        return media
    }

    /// A drop onto the group -- rather than onto one of its wells -- appends,
    /// through the same pipeline every well uses.
    func stage(_ drops: [PictureDrop]) {
        importTask?.cancel()
        importTask = PictureIntake(
            accepting: Self.accepting, hosts: hosts, library: library,
            deliver: { draft.media = Self.staging($0, in: draft.media, maxPhotos: maxPhotos) },
            report: { importFailure = $0 }
        ).drops(drops)
    }

    /// Only the rows the chooser does not own reach here.
    func perform(_ action: GenerateAction, on photo: IdentityPhoto) {
        guard action == .removePhoto else { return }
        remove(photo)
    }
}
