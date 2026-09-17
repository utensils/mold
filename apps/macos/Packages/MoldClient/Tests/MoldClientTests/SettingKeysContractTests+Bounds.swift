import Foundation
import Testing

@testable import MoldClient

// Numeric bounds, split from the main contract file for size (S4a): a
// curated `.number` key's bound must be the one `set_static_value` actually
// enforces, and the two families of SHARED bounds (`scheduler.*`,
// the two retention keys) must still be one named constant each rather than
// each arm re-declaring its own copy of the same number.

/// One-line arms only: `"<key>" => config.<field> = parse_(u16|u32|f64)(raw,
/// MIN, MAX, key)?,`. `scheduler.*` and the two retention keys are NOT
/// one-liners -- they route through a shared symbolic constant instead of a
/// literal (`theThreeSchedulerBoundsAreTheSharedConstant`,
/// `theTwoRetentionBoundsAreTheSharedConstant`) -- so this only ever needs to
/// answer for the keys S4a curates.
private func oneLineNumericBounds(_ body: Substring) -> [String: (min: Double, max: Double)] {
    var result: [String: (min: Double, max: Double)] = [:]
    let pattern = /"([a-zA-Z0-9_.]+)"\s*=>\s*config\.[\w.]+\s*=\s*parse_(u16|u32|f64)\(raw,\s*([\d.]+),\s*([\d.]+),\s*key\)\?,/
    for match in body.matches(of: pattern) {
        result[String(match.1)] = (Double(match.3) ?? 0, Double(match.4) ?? 0)
    }
    return result
}

@Test func everyNumericBoundIsTheOneTheSetterEnforces() throws {
    let root = try #require(RepoFixtures.repoRoot)
    let body = try setStaticValueBody(rustSource(root, "crates/mold-core/src/config_keys.rs"))
    let bounds = oneLineNumericBounds(body)

    for setting in curatedSettingKeys().compactMap(curatedSetting) {
        guard case let .number(min, max, _) = setting.editor else { continue }
        let found = try #require(bounds[setting.key], "no one-line parse_ arm found for \(setting.key)")
        #expect(found.min == min, "\(setting.key) min")
        #expect(found.max == max, "\(setting.key) max")
    }
}

private func rustU32Constant(_ source: String, _ name: String) throws -> UInt32 {
    let declaration = "pub const \(name): u32 = "
    let start = try #require(source.range(of: declaration))
    let end = try #require(source.range(of: ";", range: start.upperBound..<source.endIndex))
    let raw = source[start.upperBound..<end.lowerBound].replacingOccurrences(of: "_", with: "")
    return try #require(UInt32(raw))
}

/// `SCHEDULER_TIMING_MAX_MS` (`config.rs:809`) backs all three
/// `scheduler.*` keys through ONE shared match arm
/// (`config_keys.rs:687-689`), not three copies of the same number --
/// counting the reference is the whole test, since a literal 30_000 typed
/// three times would drift the day only one of them changed.
@Test func theThreeSchedulerBoundsAreTheSharedConstant() throws {
    let root = try #require(RepoFixtures.repoRoot)
    let configSource = try rustSource(root, "crates/mold-core/src/config.rs")
    #expect(try rustU32Constant(configSource, "SCHEDULER_TIMING_MAX_MS") == 30_000)

    let keysSource = try rustSource(root, "crates/mold-core/src/config_keys.rs")
    let occurrences = keysSource.components(separatedBy: "crate::config::SCHEDULER_TIMING_MAX_MS").count - 1
    #expect(occurrences == 1, "the three scheduler.* keys should share one parse_u32 call")
}

/// `GALLERY_TRASH_RETENTION_MAX_DAYS` (`config.rs:812`) backs
/// `gallery.trash_retention_days` AND `queue.held_retention_days`, each its
/// own arm but citing the same named constant rather than each hard-coding
/// `3650`.
@Test func theTwoRetentionBoundsAreTheSharedConstant() throws {
    let root = try #require(RepoFixtures.repoRoot)
    let configSource = try rustSource(root, "crates/mold-core/src/config.rs")
    #expect(try rustU32Constant(configSource, "GALLERY_TRASH_RETENTION_MAX_DAYS") == 3650)

    let keysSource = try rustSource(root, "crates/mold-core/src/config_keys.rs")
    let occurrences =
        keysSource.components(separatedBy: "crate::config::GALLERY_TRASH_RETENTION_MAX_DAYS").count - 1
    #expect(occurrences == 2, "gallery.trash_retention_days and queue.held_retention_days should each cite it")
}
