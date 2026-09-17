import AppKit
import MoldClient
import SwiftUI

// Getting a picture INTO the well, and out again. Split from the well's
// own shape purely for size.
//
// Every path goes through `startImport`, which CANCELS the one before it.
// Reading, transcoding and encoding moved off the main actor (02#10), and an
// untracked task per pick meant a 48 MP HEIC chosen first could finish last
// and overwrite the small PNG chosen after it. A newer pick always wins.
extension SourceImageWell {
    func handle(_ drop: PictureDrop) {
        startImport {
            do {
                return try await PictureSource.bytes(of: drop, hosts: hosts, library: library)
            } catch {
                if case let .print(id) = drop {
                    hosts.report(error, on: id.host, doing: "fetch that picture")
                    return nil
                }
                throw error
            }
        }
    }

    /// The panel runs on the main actor -- it has to -- but the read, the
    /// transcode and the base64 do not (finding 02#10).
    func chooseFile() {
        guard let url = PictureSource.chooseFile() else { return }
        startImport {
            try await PictureImport.load(url, accepting: PictureImport.engineReadable)
        }
    }

    /// One import at a time, newest wins. `nil` from `fetch` means it already
    /// reported itself and there is nothing to apply.
    private func startImport(_ fetch: @escaping () async throws -> ImportedPicture?) {
        importTask?.cancel()
        importTask = Task {
            do {
                let picked = try await fetch()
                guard !Task.isCancelled, let picked else { return }
                apply(picked)
            } catch is CancellationError {
                // Superseded by a newer pick, not a failure.
            } catch {
                guard !Task.isCancelled else { return }
                importFailure = error.reasonSentence
            }
        }
    }

    /// Paste is the same import path with the pasteboard as its source.
    func pasteFromPasteboard() {
        let data = PicturePaste.pasteboardData()
        startImport { try await PicturePaste.read(data) }
    }

    func clear() {
        importTask?.cancel()
        importTask = nil
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
