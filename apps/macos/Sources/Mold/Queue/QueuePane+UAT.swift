import Foundation
import MoldClient

// The UAT-only seed for the whole pane -- `MOLD_NATIVE_QUEUE_FIXTURE`, the
// same class of hook as `MOLD_NATIVE_SOURCE_IMAGE`
// (`GeneratePane+UAT.swift:6`) and `MOLD_NATIVE_DESTINATION`
// (`MachinesSettings.swift`). Split from the main file purely for size.
extension QueuePane {
    /// Reads `MOLD_NATIVE_QUEUE_FIXTURE`'s file and decodes it as one
    /// machine-keyed fixture (design M6 decision 27) -- `nil` when the
    /// variable is unset, the file is missing, or it does not parse, so a
    /// broken path fails open to an ordinary live refresh rather than
    /// silently showing nothing.
    static func fixtureIfRequested() -> QueueStore.Fixture? {
        guard let path = ProcessInfo.processInfo.environment["MOLD_NATIVE_QUEUE_FIXTURE"],
              let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        else { return nil }
        return try? MoldJSON.decoder.decode(QueueStore.Fixture.self, from: data)
    }
}
