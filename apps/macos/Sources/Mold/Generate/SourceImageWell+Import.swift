import AppKit
import MoldClient
import SwiftUI

// Getting a picture INTO the well, and out again. Split from the well's
// own shape purely for size.
extension SourceImageWell {
    func handle(_ drop: PictureDrop) {
        Task {
            do {
                apply(try await PictureSource.bytes(of: drop, hosts: hosts, library: library))
            } catch {
                if case let .print(id) = drop {
                    hosts.report(error, on: id.host, doing: "fetch that picture")
                } else {
                    importFailure = error.reasonSentence
                }
            }
        }
    }

    /// The panel runs on the main actor -- it has to -- but the read, the
    /// transcode and the base64 do not (finding 02#10).
    func chooseFile() {
        guard let url = PictureSource.chooseFile() else { return }
        Task {
            do {
                apply(try await PictureImport.load(url, accepting: PictureImport.engineReadable))
            } catch {
                importFailure = error.reasonSentence
            }
        }
    }

    func clear() {
        draft.media.sourceImage = nil
        draft.media.sourceImageName = nil
        importFailure = nil
        preview = nil
    }

    /// `PictureImport` has already read, conformed and base64'd it off the
    /// main actor, so the draft holds exactly what will be sent.
    func apply(_ picked: ImportedPicture) {
        draft.media.sourceImage = picked.encoded
        draft.media.sourceImageName = picked.name
        // Last write wins on an EXCLUSIVE recipe: attaching here parks the
        // reference strip rather than refusing the drop (`ExclusiveWells`).
        draft.media.lastExclusiveWrite = .source
        importFailure = nil
    }
}
