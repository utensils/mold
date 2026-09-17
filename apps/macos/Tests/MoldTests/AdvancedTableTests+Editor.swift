import Foundation
import MoldClient
import Testing

@testable import Mold

/// `ConfigValueField.resolve` -- what one row draws, from the entry alone,
/// the same pure-resolver idiom `DiscoverRow.resolve` uses for the Models
/// table's State column. Split from `AdvancedTableTests.swift` to keep both
/// files under the size floor.
@MainActor
struct AdvancedTableEditorTests {
    @Test func anEnvRowDrawsItsVariableAndNoField() {
        let entry = FakeFixtures.configEntry(
            "models_dir", value: .string("/data/models"), source: "env", envVar: "MOLD_MODELS_DIR")

        let plan = ConfigValueField.resolve(entry)

        #expect(plan.kind == .envOwned)
        #expect(plan.envVar == "MOLD_MODELS_DIR")
    }

    @Test func aSecretRowOffersABlankFieldAndTheWordSet() {
        let entry = FakeFixtures.configEntry("runpod.api_key", value: .string("<set>"), source: "db")

        let plan = ConfigValueField.resolve(entry)

        #expect(plan.kind == .secret)
        #expect(entry.secretState == "Set")
        #expect(entry.editableText == "")
    }

    @Test func anUnsetSecretReadsNotSet() {
        let entry = FakeFixtures.configEntry("lambda.api_key", value: .null, source: "default")

        #expect(ConfigValueField.resolve(entry).kind == .secret)
        #expect(entry.secretState == "Not set")
    }

    @Test func onlyADbRowDrawsAReset() {
        let sources = ["db", "file", "env", "default"]
        let plans = sources.map { source in
            ConfigValueField.resolve(FakeFixtures.configEntry("gallery.trash_retention_days", value: .number(30), source: source))
        }

        #expect(plans.map(\.showsReset) == [true, false, false, false])
    }

    @Test func everyOtherEditorKindMapsFromTheValuesOwnType() {
        #expect(ConfigValueField.resolve(FakeFixtures.configEntry("k", value: .bool(true), source: "db")).kind == .toggle)
        #expect(ConfigValueField.resolve(FakeFixtures.configEntry("k", value: .number(1), source: "db")).kind == .number)
        #expect(ConfigValueField.resolve(FakeFixtures.configEntry("k", value: .string("x"), source: "db")).kind == .text)
        #expect(ConfigValueField.resolve(FakeFixtures.configEntry("k", value: .null, source: "default")).kind == .unset)
    }

    /// The one trap the editor must not fall into: a secret's field starts
    /// EMPTY (the "<set>" is a mask), and the parser turns an empty string
    /// into `.null` -- so a commit that went through unconditionally would
    /// clear the API key on the machine.
    ///
    /// **Fails today** on the Return case: a blur already commited only what
    /// changed, but Return commited whatever was in the field, and an empty
    /// secret field is indistinguishable from an untouched one. Tab into
    /// `runpod.api_key`, hesitate, press Return, and the credential was gone
    /// (review 05-M12). Clearing is the row's own Clear button now.
    @Test func anEmptySecretFieldNeverClearsTheStoredCredential() {
        let entry = FakeFixtures.configEntry("runpod.api_key", value: .string("<set>"), source: "db")

        #expect(ConfigValueField.commitScalar(text: "", entry: entry, onBlur: true) == nil)
        #expect(ConfigValueField.commitScalar(text: "", entry: entry, onBlur: false) == nil)
        #expect(ConfigValueField.commitScalar(text: "rp-1", entry: entry, onBlur: true) == .string("rp-1"))
        // An ordinary text row is unaffected: emptying one IS how it unsets.
        let text = FakeFixtures.configEntry("models_dir", value: .string("/data"), source: "db")
        #expect(ConfigValueField.commitScalar(text: "", entry: text, onBlur: false) == .null)
    }

    @Test func aBlurWithNoChangeSendsNoWrite() {
        let entry = FakeFixtures.configEntry("expand.max_tokens", value: .number(256), source: "db")

        #expect(ConfigValueField.commitScalar(text: "256", entry: entry, onBlur: true) == nil)
        #expect(ConfigValueField.commitScalar(text: "300", entry: entry, onBlur: true) == .number(300))
        #expect(ConfigValueField.commitScalar(text: "abc", entry: entry, onBlur: false) == nil)
    }
}
