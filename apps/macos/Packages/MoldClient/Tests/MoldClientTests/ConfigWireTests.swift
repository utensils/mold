import Foundation
import Testing

@testable import MoldClient

// `Fixtures/config-workstation.json` is a live `GET /api/config` from workstation: 63
// entries, every wire shape a real row can take -- a bool (`embed_metadata`),
// a number (`server_port`), a string (`default_model`), and a null
// (`output_dir`, env-owned). `Fixtures/config-profiles-workstation.json` is
// `GET /api/config/profiles` from the same machine.

private func live() throws -> ConfigListing {
    try MoldJSON.decoder.decode(
        ConfigListing.self, from: RepoFixtures.fixture("config-workstation.json"))
}

private func row(_ key: String) throws -> ConfigEntry {
    let entries = try live().entries
    return try #require(entries.first { $0.key == key })
}

@Test func aProfileListingDecodesTheActiveNameAndTheList() throws {
    let profiles = try MoldJSON.decoder.decode(
        ConfigProfiles.self, from: RepoFixtures.fixture("config-profiles-workstation.json"))
    #expect(profiles.active == "default")
    #expect(profiles.profiles == ["default"])
}

@Test func aBoolRowDrawsAsAToggle() throws {
    #expect(try row("embed_metadata").editor == .toggle)
}

@Test func aNumberRowDrawsAsANumberField() throws {
    #expect(try row("server_port").editor == .number)
}

@Test func aStringRowDrawsAsText() throws {
    #expect(try row("default_model").editor == .text)
}

/// A present key with a null value must not read as an empty text field --
/// it means the same thing as no row at all, and the Advanced table says so
/// rather than looking like somebody cleared it.
@Test func aRowWithANullValueDrawsAsUnsetRatherThanAsEmptyText() throws {
    let output = try row("output_dir")
    #expect(output.value == .null)
    #expect(output.editor == .unset)
}

/// The trap fact 3 describes: a text field bound to `row.value` would PUT the
/// literal string `<set>` on its first blur and destroy the key. The field
/// must start empty, always, whatever the wire says.
@Test func aMaskedSecretNeverStartsAFieldWithItsMask() {
    let masked = ConfigEntry(key: "runpod.api_key", value: .string("<set>"), source: "file")
    #expect(masked.editor == .secret)
    #expect(masked.editableText == "")
    #expect(masked.secretState == "Set")

    let unset = ConfigEntry(key: "lambda.api_key", value: .null, source: "file")
    #expect(unset.editor == .secret)
    #expect(unset.editableText == "")
    #expect(unset.secretState == "Not set")
}

/// `source == "env"` is exactly PUT's own refusal gate
/// (`routes_config.rs:192-197`) -- read off the row, no key list.
@Test func anEnvOwnedRowIsReadOnlyBeforeAnybodyTriesIt() throws {
    let models = try row("models_dir")
    #expect(models.source == "env")
    #expect(models.isEnvOwned)
    #expect(try !row("default_model").isEnvOwned)
}

/// `source == "db"` is exactly DELETE's own gate (`routes_config.rs:270-276`)
/// -- Reset is never offered anywhere else.
@Test func onlyADbRowOffersAReset() throws {
    #expect(try row("default_width").source == "db")
    #expect(try row("default_width").canReset)
    #expect(try !row("default_model").canReset) // source: "file"
    #expect(try !row("models_dir").canReset) // source: "env"
}

/// Only the three `scheduler.*` keys ever carry `restart_required`; every
/// other DB row is absent, and absent must not read as `true`.
@Test func absentRestartRequiredIsNotARestart() throws {
    #expect(try row("default_width").restartRequired == nil)
    #expect(try !row("default_width").needsRestart)
    #expect(try row("scheduler.replan_debounce_ms").needsRestart)
}

/// A null body is how an optional key clears (`routes_config.rs:143`) -- an
/// emptied field must send `.null`, never the string `""`.
@Test func anEmptiedOptionalFieldSendsNullRatherThanAnEmptyString() {
    #expect(ConfigEntry.scalar(from: "", editor: .text) == .null)
    #expect(ConfigEntry.scalar(from: "", editor: .unset) == .null)
    #expect(ConfigEntry.scalar(from: "", editor: .secret) == .null)
    #expect(ConfigEntry.scalar(from: "a new value", editor: .text) == .string("a new value"))
}

@Test func aNumberFieldParsesAndEmptyClears() {
    #expect(ConfigEntry.scalar(from: "30", editor: .number) == .number(30))
    #expect(ConfigEntry.scalar(from: "", editor: .number) == .null)
    #expect(ConfigEntry.scalar(from: "not a number", editor: .number) == nil)
}

@Test func aToggleParsesExactlyTrueOrFalse() {
    #expect(ConfigEntry.scalar(from: "true", editor: .toggle) == .bool(true))
    #expect(ConfigEntry.scalar(from: "false", editor: .toggle) == .bool(false))
    #expect(ConfigEntry.scalar(from: "yes", editor: .toggle) == nil)
}
