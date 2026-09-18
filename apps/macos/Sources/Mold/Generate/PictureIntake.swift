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

    /// A drop is ORDERED, and so is the strip: a task per file appended in
    /// COMPLETION order, and a small local file could overtake a Library print
    /// fetched over the network, so a render edited the wrong picture. They are
    /// awaited in sequence instead.
    func drops(_ drops: [PictureDrop]) -> Task<Void, Never> {
        run(drops) { drop in
            do {
                deliver(try await PictureSource.bytes(
                    of: drop, accepting: accepting, hosts: hosts, library: library))
                return nil
            } catch {
                // A print that would not come off its machine is that
                // machine's failure, reported where every other one is.
                guard case let .print(id) = drop else { throw error }
                hosts.report(error, on: id.host, doing: "fetch that picture")
                return nil
            }
        }
    }

    /// The panel runs on the main actor -- it has to -- but the read, the
    /// transcode and the base64 do not (02#10).
    func files(_ urls: [URL]) -> Task<Void, Never> {
        run(urls) { url in
            deliver(try await PictureImport.load(url, accepting: accepting))
            return nil
        }
    }

    /// Paste is the same pipeline with the pasteboard as its source. The bytes
    /// are read on the main actor because `NSPasteboard` is not thread-safe;
    /// everything after that is not.
    func paste() -> Task<Void, Never> {
        let data = PicturePaste.pasteboardData()
        return run([data]) { data in
            guard let picked = try await PicturePaste.read(data, accepting: accepting)
            else { return nil }
            deliver(picked)
            return nil
        }
    }

    /// One pick, however many pictures it brought.
    ///
    /// The sentence is cleared ONCE, at the start, and never again inside the
    /// loop: clearing it per delivery meant three files where the first two
    /// could not be read and the third could ended with nothing said at all.
    /// And a file that fails does not end the pick -- the rest are still
    /// attempted, the `return`-instead-of-`continue` that lost nine of ten
    /// files in the Library's own import (03-M-ish, `LibraryActions.send`).
    private func run<Item>(
        _ items: [Item], each: @escaping (Item) async throws -> String?
    ) -> Task<Void, Never> {
        Task {
            report(nil)
            var failures: [String] = []
            for item in items {
                guard !Task.isCancelled else { return }
                do {
                    if let refusal = try await each(item) { failures.append(refusal) }
                } catch is CancellationError {
                    return
                } catch {
                    failures.append(error.failureSentence)
                }
            }
            guard !Task.isCancelled else { return }
            report(Self.summary(of: failures))
        }
    }

    /// What one pick's failures say. Several are one line that says HOW MANY,
    /// the way the Library's own import reports a batch -- naming only the
    /// first would under-report what did not arrive.
    static func summary(of failures: [String]) -> String? {
        switch failures.count {
        case 0: nil
        case 1: failures[0]
        default: "\(failures.count) of those files couldn't be used."
        }
    }
}
