import AppKit
import MoldClient
import SwiftUI

// Getting reference pictures in, in the order they were given, and the menus
// that ask for them. Split from the strip's own shape purely for size.
extension ReferenceStrip {
    /// A drop is ORDERED, and so is the strip: on a `primaryIsTarget` recipe
    /// index 0 is the picture being EDITED. A task per drop appended in
    /// COMPLETION order, so a small local file could overtake a Library print
    /// fetched over the network and the render edited the wrong picture. They
    /// are awaited in sequence instead (review 06, medium).
    func handle(_ drops: [PictureDrop]) {
        importTask?.cancel()
        importTask = Task {
            for drop in drops {
                guard !Task.isCancelled else { return }
                do {
                    append(try await PictureSource.bytes(
                        of: drop, hosts: hosts, library: library))
                } catch is CancellationError {
                    return
                } catch {
                    if case let .print(id) = drop {
                        hosts.report(error, on: id.host, doing: "fetch that picture")
                    } else {
                        importFailure = error.reasonSentence
                    }
                }
            }
        }
    }

    /// The panel runs on the main actor -- it has to -- but the read, the
    /// transcode and the base64 do not (finding 02#10).
    func chooseFile() {
        guard let url = PictureSource.chooseFile() else { return }
        importTask?.cancel()
        importTask = Task {
            do {
                append(try await PictureImport.load(
                    url, accepting: PictureImport.engineReadable))
            } catch is CancellationError {
                // Superseded by a newer pick.
            } catch {
                guard !Task.isCancelled else { return }
                importFailure = error.reasonSentence
            }
        }
    }

    func append(_ picked: ImportedPicture) {
        guard capability.hasRoom(for: draft.media.editImages.count) else { return }
        draft.media.editImages.append(picked.encoded)
        // Last write wins on an EXCLUSIVE recipe (`ExclusiveWells`).
        draft.media.lastExclusiveWrite = .references
        importFailure = nil
    }

    /// Replace swaps ONE slot in place, so the strip's order -- and Qwen's
    /// Target at index 0 -- is untouched.
    func replace(at index: Int) {
        guard let url = PictureSource.chooseFile() else { return }
        importTask?.cancel()
        importTask = Task {
            do {
                let picked = try await PictureImport.load(
                    url, accepting: PictureImport.engineReadable)
                guard !Task.isCancelled,
                      draft.media.editImages.indices.contains(index) else { return }
                draft.media.editImages[index] = picked.encoded
                draft.media.lastExclusiveWrite = .references
                importFailure = nil
            } catch is CancellationError {
                // Superseded by a newer pick.
            } catch {
                importFailure = error.reasonSentence
            }
        }
    }

    func pasteReference() {
        let data = PicturePaste.pasteboardData()
        importTask?.cancel()
        importTask = Task {
            do {
                guard let picked = try await PicturePaste.read(data),
                      !Task.isCancelled else { return }
                append(picked)
            } catch is CancellationError {
                // Superseded by a newer pick.
            } catch {
                importFailure = error.reasonSentence
            }
        }
    }
}
