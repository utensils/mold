import Foundation
import Testing

@testable import MoldClient

/// Pins the curated Settings arrays (`SettingKeys`) against the engine's own
/// registry -- `crates/mold-core/src/config_keys.rs` -- the same arrangement
/// `ModelFamilyContractTests` has with `manifest.rs`, and the same
/// arrangement `studio/lib/settingsSchema.ts` has with the same file.
func rustSource(_ root: URL, _ relativePath: String) throws -> String {
    try String(contentsOf: root.appending(path: relativePath), encoding: .utf8)
}

/// Just `set_static_value`'s own body -- narrow enough that a key literal
/// (`"t5_variant"`) appears exactly once, so a non-greedy search from it
/// cannot wander into a different key's arm the way searching the whole
/// file (which also has the key in `ALL_KEYS` and in `get_static_value`)
/// could.
func setStaticValueBody(_ source: String) throws -> Substring {
    let start = try #require(source.range(of: "\nfn set_static_value"))
    let end = try #require(
        source.range(of: "\nfn set_model_value", range: start.upperBound..<source.endIndex))
    return source[start.upperBound..<end.lowerBound]
}

/// Every key `ALL_KEYS` registers, literal or resolved through a
/// `pub const NAME: &str = "..."` -- most keys are inline string literals,
/// but `gallery.*`, `queue.*` and `generate.auto_tag_title` are declared as
/// named constants and referenced by identifier (`config_keys.rs:145-168`),
/// the same split `ModelFamilyContractTests.rustFamilies` already resolves
/// for `manifest.rs`.
func rustAllKeys(_ source: String) throws -> Set<String> {
    let start = try #require(source.range(of: "pub const ALL_KEYS: &[ConfigKeyInfo] = &["))
    let end = try #require(source.range(of: "\n];", range: start.upperBound..<source.endIndex))
    let body = source[start.upperBound..<end.lowerBound]

    var keys: Set<String> = []
    for match in body.matches(of: /key:\s*"([^"]+)",/) {
        keys.insert(String(match.1))
    }
    for match in body.matches(of: /key:\s*([A-Z][A-Z0-9_]+),/) {
        let name = String(match.1)
        let declaration = "pub const \(name): &str = \""
        let declStart = try #require(source.range(of: declaration))
        let declEnd = try #require(source.range(of: "\"", range: declStart.upperBound..<source.endIndex))
        keys.insert(String(source[declStart.upperBound..<declEnd.lowerBound]))
    }
    return keys
}

func curatedSettingKeys() -> [String] { SettingKeys.all.flatMap { $0.map(\.key) } }

func curatedSetting(_ key: String) -> SettingKey? {
    SettingKeys.all.flatMap { $0 }.first { $0.key == key }
}

@Test func everyCuratedKeyIsInTheEnginesRegistry() throws {
    let root = try #require(RepoFixtures.repoRoot, "mold checkout not found above the tests")
    let registry = try rustAllKeys(rustSource(root, "crates/mold-core/src/config_keys.rs"))
    for key in curatedSettingKeys() {
        #expect(registry.contains(key), "\(key) is curated but missing from ALL_KEYS")
    }
}

@Test func everyChoiceListIsValidateEnumsOwnList() throws {
    let root = try #require(RepoFixtures.repoRoot)
    let body = try setStaticValueBody(rustSource(root, "crates/mold-core/src/config_keys.rs"))

    let t5Match = try #require(
        body.firstMatch(of: /"t5_variant"[\s\S]*?validate_enum\(v, &\[([^\]]+)\], key\)/))
    #expect(curatedSetting("t5_variant")?.editor == .choice(choiceList(t5Match.1)))

    let qwenMatch = try #require(
        body.firstMatch(of: /"qwen3_variant"[\s\S]*?validate_enum\(v, &\[([^\]]+)\], key\)/))
    #expect(curatedSetting("qwen3_variant")?.editor == .choice(choiceList(qwenMatch.1)))
}

private func choiceList(_ raw: Substring) -> [String] {
    raw.split(separator: ",").map {
        $0.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }
}

/// The trap fact 3 describes: `"<set>"` is the ONLY signal a masked key
/// exists on the wire at all, so a third one appearing upstream must fail
/// here rather than ship a secret in plaintext through a curated pane that
/// binds a text field straight to the row.
@Test func theMaskedKeysAreTheOnlyTwoTheEngineMasks() throws {
    let root = try #require(RepoFixtures.repoRoot)
    let source = try rustSource(root, "crates/mold-core/src/config_keys.rs")
    let occurrences = source.components(separatedBy: "\"<set>\"").count - 1
    #expect(occurrences == 2, "a third masked key appeared in the engine")
    #expect(ConfigEntry.secretKeys == ["runpod.api_key", "lambda.api_key"])
}

/// `umt5_variant` is registered but has no getter/setter arm --
/// `list_config`'s `if let Ok` silently drops it from every listing (design
/// fact 2) -- so a curated pane must never name it: the machine will never
/// answer with a row for it to bind to.
@Test func aKeyTheEngineNeverListsIsNeverCurated() throws {
    let root = try #require(RepoFixtures.repoRoot)
    let registry = try rustAllKeys(rustSource(root, "crates/mold-core/src/config_keys.rs"))
    #expect(registry.contains("umt5_variant"))
    #expect(!curatedSettingKeys().contains("umt5_variant"))
}

@Test func noKeyIsCuratedTwice() {
    let keys = curatedSettingKeys()
    #expect(keys.count == Set(keys).count)
}
