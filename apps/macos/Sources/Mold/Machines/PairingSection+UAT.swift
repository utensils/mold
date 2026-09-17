import Foundation
import MoldClient

// The UAT-only seed for the whole section -- `MOLD_NATIVE_PAIRING_FIXTURE`,
// the same class of hook as `MOLD_NATIVE_QUEUE_FIXTURE`
// (`QueuePane+UAT.swift`). Split from the main file purely for size.
extension PairingSection {
    /// Reads `MOLD_NATIVE_PAIRING_FIXTURE`'s file and decodes it as one
    /// machine-keyed fixture (design decision 25) -- `nil` when the variable
    /// is unset, the file is missing, or it does not parse, so a broken path
    /// fails open to an ordinary live refresh rather than silently showing
    /// nothing.
    static func fixtureIfRequested() -> PairingStore.Fixture? {
        guard let path = ProcessInfo.processInfo.environment["MOLD_NATIVE_PAIRING_FIXTURE"],
              let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        else { return nil }
        return try? MoldJSON.decoder.decode(PairingStore.Fixture.self, from: data)
    }
}
