import AppKit
import Foundation
import SwiftUI

/// Getting picked pictures IN: the one pipeline every picture well shares.
///
/// Four wells had four copies of this -- read, conform, encode, deliver, and
/// cancel the one before -- and they disagreed. The strip awaited a multi-file
/// drop in SEQUENCE while the identity group's per-file task let a slow file's
/// `importFailure = nil` wipe a newer file's message; the source well cancelled
/// its predecessor and the ControlNet well did not conform a Library print at
/// all because it had no Library door. It is a value rather than a view so the
/// group-level drop `IdentityGroup` accepts and the chooser inside each well
/// are provably the same code.
///
/// `deliver` is called once per picture, in PICK order, on the main actor:
/// order is the whole point on a `primaryIsTarget` strip, where index 0 is the
/// picture being edited.
@MainActor
struct PictureIntake {
    /// What the machine can read from the well this intake feeds
    /// (`PictureImport.engineReadable` / `.identityReadable`). Applied to
    /// every door, the Library included -- a print is bytes mold made, but
    /// not necessarily bytes THIS well's path can decode.
    let accepting: Set<String>
    let hosts: HostStore
    let library: LibraryStore
    let deliver: (ImportedPicture) -> Void
    /// The sentence beside the well, or `nil` to clear it. Beside the control
    /// that collected the file rather than in a 422 after the upload (02#7).
    let report: (String?) -> Void

    /// A drop is ORDERED, so its files are awaited in sequence: a task per
    /// file appended in COMPLETION order, and a small local file could
    /// overtake a Library print fetched over the network.
    func drops(_ drops: [PictureDrop]) -> Task<Void, Never> {
        Task {
            for drop in drops {
                guard !Task.isCancelled else { return }
                do {
                    deliver(try await PictureSource.bytes(
                        of: drop, accepting: accepting, hosts: hosts, library: library))
                    report(nil)
                } catch is CancellationError {
                    return
                } catch {
                    // A print that would not come off its machine is that
                    // machine's failure, reported where every other one is.
                    if case let .print(id) = drop {
                        hosts.report(error, on: id.host, doing: "fetch that picture")
                    } else {
                        report(error.reasonSentence)
                    }
                }
            }
        }
    }

    /// The panel runs on the main actor -- it has to -- but the read, the
    /// transcode and the base64 do not (02#10).
    func files(_ urls: [URL]) -> Task<Void, Never> {
        Task {
            for url in urls {
                guard !Task.isCancelled else { return }
                do {
                    deliver(try await PictureImport.load(url, accepting: accepting))
                    report(nil)
                } catch is CancellationError {
                    return
                } catch {
                    report(error.reasonSentence)
                }
            }
        }
    }

    /// Paste is the same pipeline with the pasteboard as its source. The bytes
    /// are read on the main actor because `NSPasteboard` is not thread-safe;
    /// everything after that is not.
    func paste() -> Task<Void, Never> {
        let data = PicturePaste.pasteboardData()
        return Task {
            do {
                guard let picked = try await PicturePaste.read(data, accepting: accepting),
                      !Task.isCancelled else { return }
                deliver(picked)
                report(nil)
            } catch is CancellationError {
                return
            } catch {
                report(error.reasonSentence)
            }
        }
    }
}
