import MoldClient
import SwiftUI

// Getting a photograph in, and what one offers on a right-click. Split from
// the group's own shape purely for size.
extension IdentityGroup {
    /// Reads, conforms to PNG/JPEG and encodes off the main actor. A HEIC or
    /// TIFF photograph is TRANSCODED rather than refused -- it is the likeliest
    /// picture of a face on this Mac, and re-encoding it is the whole fix.
    ///
    /// Awaited in SEQUENCE: a task per file let a slow one's `importFailure =
    /// nil` clear a newer file's error message (review 06, medium).
    func append(_ urls: [URL]) {
        importTask?.cancel()
        importTask = Task {
            for url in urls {
                guard !Task.isCancelled, photos.count < maxPhotos else { return }
                do {
                    let picked = try await PictureImport.load(
                        url, accepting: PictureImport.identityReadable)
                    guard !Task.isCancelled, photos.count < maxPhotos else { return }
                    var conditioning = draft.media.identity ?? IdentityConditioning(photos: [])
                    conditioning.photos.append(
                        IdentityPhoto(encoded: picked.encoded, name: picked.name))
                    draft.media.identity = conditioning
                    importFailure = nil
                } catch is CancellationError {
                    return
                } catch {
                    importFailure = error.reasonSentence
                }
            }
        }
    }

    /// One photograph's menu. Replace is Choose File… aimed at this slot;
    /// Remove is the well's own ✕ (`GenerateMenus.identityPhoto`).
    var photoMenu: [GenerateMenus.Row] { GenerateMenus.identityPhoto() }

    func perform(_ action: GenerateAction, on photo: IdentityPhoto) {
        switch action {
        case .replacePhoto:
            remove(photo)
            choose()
        case .removePhoto:
            remove(photo)
        default:
            break
        }
    }
}
