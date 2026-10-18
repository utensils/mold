import Foundation
import Testing
@testable import MoldClient

struct SourceMediaActionsTests {
    @Test func commandsKeepOwningHostAndQuoteEveryUntrustedArgument() {
        let member = RetainedSourceMedia.Member(memberId: "abc12345", role: "source_image", displayName: "input $(id).png", sizeBytes: 1)
        let command = SourceMediaActions.command(filename: "print $(id).png", member: member, host: URL(string: "https://example.invalid:7680")!)
        #expect(command == "mold --host 'https://example.invalid:7680' library source-media 'print $(id).png' --member 'abc12345' --output 'input $(id).png'")
        #expect(SourceMediaActions.identifier(filename: "a.png", member: member) == "source:a.png#abc12345")
        #expect(SourceMediaActions.shellQuote("a'b") == "'a'\"'\"'b'")
    }
    @Test func textOnlyLegacyInventoryHasNoRetainedMedia() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data("{}".utf8))
        #expect(SourceMediaState(inventory: .init(availability: .unavailableLegacy), failed: false, metadata: metadata) == .none)
        #expect(SourceMediaState(inventory: nil, failed: true, metadata: metadata) == .failed)
    }
    @Test func noInputsRequiresAnAuthoritativeInventory() {
        #expect(SourceMediaState(inventory: .init(availability: .available), failed: false) == .none)
        #expect(SourceMediaState(inventory: .init(availability: .unavailableLegacy), failed: false) == .unavailable)
        #expect(SourceMediaState(inventory: nil, failed: true) == .failed)
        #expect(SourceMediaState(inventory: nil, failed: false) == .loading)
        #expect(!SourceMediaState.none.canRetry)
        #expect(SourceMediaState.unavailable.canRetry)
    }
}
