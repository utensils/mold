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
        draft.media.sourceImageOriginal = nil
        draft.media.sourceImageOriginalName = nil
        importFailure = nil
        preview = nil
    }

    /// `PictureImport` has already read, conformed and base64'd it off the
    /// main actor, so the draft holds exactly what will be sent.
    func apply(_ picked: ImportedPicture) {
        // Studio's own predicate: the bytes are not the bytes that were
        // there, which a FIRST picture satisfies too (`CreatePage.vue:1151`).
        let replaced = draft.media.sourceImageOriginal != picked.encoded
        // The UNFITTED copy is what every later re-fit starts from: fitting an
        // already-fitted picture crops a crop. `sourceImage` below is the
        // first, unfitted showing of it; `refittingSource` replaces it the
        // moment the canvas is known.
        draft.media.sourceImageOriginal = picked.encoded
        draft.media.sourceImageOriginalName = picked.name
        draft.media.sourceImage = picked.encoded
        draft.media.sourceImageName = picked.name
        if let size = PictureImport.pixelSize(of: picked.data) {
            draft.attachSourceShape(size, recipe: recipe, replaced: replaced)
        }
        // Last write wins on an EXCLUSIVE recipe: attaching here parks the
        // reference strip rather than refusing the drop (`ExclusiveWells`).
        draft.media.lastExclusiveWrite = .source
        importFailure = nil
    }
}
