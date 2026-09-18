import Foundation
import MoldClient
import Testing

@testable import Mold

/// 05-H6: removing a machine is irreversible on this Mac -- it takes the
/// stored API key with it -- so it asks first, in a plain dialog with a danger
/// button and no typed phrase.
@MainActor
struct MachinesSettingsTests {
    private func host(_ name: String, apiKey: String? = nil) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name):7680")!, apiKey: apiKey)
    }

    /// **Fails today**: `MachinesSettings.remove(_:)` called `hosts.remove`
    /// straight from a borderless icon-only "−" next to "+". There was no
    /// `Destruction` for it at all.
    @Test func removingAMachineAsksFirstAndOnlyThenRemovesIt() {
        var removed = false
        let destruction = MachineRemoval.destruction(of: host("workstation")) { removed = true }

        #expect(destruction.title == "Remove “workstation”?")
        #expect(destruction.verb == "Remove")
        #expect(!removed, "building the question must not be the answer")

        destruction.perform()
        #expect(removed)
    }

    /// The consequence the web counterpart names
    /// (`web/src/pages/MachinesPage.vue:203-217`), stated only where it is
    /// true: a keyless machine has no credential to lose.
    @Test func theQuestionNamesTheKeyItDestroys() {
        let keyed = MachineRemoval.message(for: host("workstation", apiKey: "k"))
        #expect(keyed.contains("API key"))
        #expect(keyed.contains("workstation:7680"))

        let keyless = MachineRemoval.message(for: host("hal9000"))
        #expect(!keyless.contains("API key"))
        #expect(keyless.contains("hal9000:7680"))
    }
}
