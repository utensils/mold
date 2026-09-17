import Foundation
import MoldClient

// The UAT-only seed for the whole section -- `MOLD_NATIVE_PAIRING_FIXTURE`,
// the same class of hook as `MOLD_NATIVE_QUEUE_FIXTURE`
// (`QueuePane+UAT.swift`) -- and the one load the Machines pane calls.
extension PairingStore {
    /// Fixture first, the wire otherwise; nothing twice. Called from the
    /// Machines pane's own per-host task, NOT from the section: the section
    /// draws `EmptyView` until it has an answer, and a `.task` hung off an
    /// `EmptyView` never fires -- which is how the section was dead on every
    /// machine until M7 UAT read the page (M7 S7 fix).
    func load(on host: MoldHost.ID, fixture: Fixture? = PairingStore.fixtureIfRequested()) async {
        guard !isSeeded else { return }
        if let fixture {
            seed(from: fixture)
        } else {
            await refresh(on: host)
        }
    }

    /// Reads `MOLD_NATIVE_PAIRING_FIXTURE`'s file and decodes it as one
    /// machine-keyed fixture (design decision 25) -- `nil` when the variable
    /// is unset, the file is missing, or it does not parse, so a broken path
    /// fails open to an ordinary live refresh rather than silently showing
    /// nothing.
    static func fixtureIfRequested() -> PairingStore.Fixture? {
        guard let path = NativeUAT.pairingFixture.value(),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        else { return nil }
        return try? MoldJSON.decoder.decode(PairingStore.Fixture.self, from: data)
    }
}
